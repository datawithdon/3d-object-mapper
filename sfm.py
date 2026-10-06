"""
sfm.py - A minimal incremental Structure-from-Motion (SfM) pipeline.

Given a sequence of photos of an object taken from different angles
(assumed to roughly circle the object, e.g. a phone held and walked
around it, or a turntable), this module:

  1. Detects SIFT features in every photo.
  2. Matches features between consecutive photos.
  3. Recovers the relative camera pose between the first pair via the
     essential matrix, and triangulates an initial 3D point cloud.
  4. For every following photo, uses already-triangulated 3D points to
     solve the new camera's pose (PnP), then triangulates fresh points
     between it and the previous photo.
  5. Merges everything into one colored point cloud and removes
     statistical outliers.

This is a deliberately compact, dependency-light implementation (just
OpenCV + NumPy) rather than a full bundle-adjusted pipeline like COLMAP.
It will not be perfectly metric/scaled and works best with:
  - 10-30 photos of the same object, same camera/resolution
  - good overlap (~60-80%) between consecutive shots
  - decent lighting and surface texture (SIFT needs texture to match)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_IMAGE_DIM = 1200       # resize longest edge to this before processing
MIN_MATCHES = 20           # below this, two images are considered non-overlapping
RATIO_TEST = 0.75          # SIFT descriptor ratio test threshold
MIN_PNP_CORRESPONDENCES = 8


@dataclass
class ReconstructionResult:
    points: list              # list of [x, y, z, r, g, b]
    num_points: int
    num_images_used: int
    warnings: list = field(default_factory=list)


def _load_and_resize(path: str, max_dim: int = MAX_IMAGE_DIM) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {os.path.basename(path)}")
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return img


def _intrinsics_for(width: int, height: int) -> np.ndarray:
    """Heuristic camera intrinsics for an uncalibrated camera.

    Assumes a roughly 50-60 degree horizontal field of view, which is a
    reasonable approximation for most phone cameras. Without real
    calibration the reconstruction's absolute scale/shape has some
    distortion, but relative structure is usually recognizable.
    """
    focal = 1.2 * max(width, height)
    return np.array([
        [focal, 0, width / 2.0],
        [0, focal, height / 2.0],
        [0, 0, 1.0],
    ], dtype=np.float64)


def _detect_features(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=4000)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def _match(des_a, des_b):
    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return []
    bf = cv2.BFMatcher()
    raw = bf.knnMatch(des_a, des_b, k=2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < RATIO_TEST * n.distance:
            good.append(m)
    return good


def _sample_color(img: np.ndarray, pt) -> list:
    h, w = img.shape[:2]
    x = int(round(min(max(pt[0], 0), w - 1)))
    y = int(round(min(max(pt[1], 0), h - 1)))
    b, g, r = img[y, x].tolist()
    return [r, g, b]


def _remove_outliers(points: np.ndarray, k: int = 12, std_ratio: float = 2.0) -> np.ndarray:
    """Simple statistical outlier filter (no Open3D dependency).

    For each point, compute the mean distance to its k nearest
    neighbours; drop points whose mean distance is more than
    `std_ratio` standard deviations above the global average.
    """
    n = len(points)
    if n < k + 2:
        return np.ones(n, dtype=bool)

    xyz = points[:, :3]
    mean_dists = np.empty(n)
    # Brute-force kNN is fine here: SfM point clouds from a handful of
    # photos are small (hundreds to low thousands of points).
    for i in range(n):
        d = np.linalg.norm(xyz - xyz[i], axis=1)
        d.sort()
        mean_dists[i] = d[1:k + 1].mean()  # skip self (distance 0)

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std
    return mean_dists <= threshold


def reconstruct(image_paths: list) -> dict:
    """Run incremental SfM over `image_paths` (assumed in capture order).

    Returns a dict: {points, num_points, num_images_used, warnings}
    where `points` is a list of [x, y, z, r, g, b].
    """
    warnings: list = []
    images = []
    for p in sorted(image_paths):
        try:
            images.append(_load_and_resize(p))
        except ValueError as e:
            warnings.append(str(e))

    if len(images) < 3:
        return {"points": [], "num_points": 0, "num_images_used": len(images), "warnings": warnings}

    # Make every image the same size as the first so a single intrinsics
    # matrix is valid for all of them (keeps the pipeline uncalibrated
    # but internally consistent).
    h0, w0 = images[0].shape[:2]
    for i in range(1, len(images)):
        h, w = images[i].shape[:2]
        if (h, w) != (h0, w0):
            images[i] = cv2.resize(images[i], (w0, h0), interpolation=cv2.INTER_AREA)
    K = _intrinsics_for(w0, h0)

    feats = [_detect_features(img) for img in images]

    all_points = []   # list of [x, y, z, r, g, b]
    # track[i] maps keypoint index in image i -> index into all_points
    track = [dict() for _ in images]

    # ---- Initialize with the first pair ----
    cur_pose = None  # (R, t) of the "previous" successfully-posed image
    poses = [None] * len(images)
    used_images = 0

    kp0, des0 = feats[0]
    init_ok = False
    for j in range(1, len(images)):
        kp1, des1 = feats[j]
        matches = _match(des0, des1)
        if len(matches) < MIN_MATCHES:
            warnings.append(f"Image 0 and {j}: only {len(matches)} matches, skipping as init pair.")
            continue

        pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            continue
        inlier_mask = mask.ravel().astype(bool)
        if inlier_mask.sum() < MIN_PNP_CORRESPONDENCES:
            continue

        _, R, t, pose_mask = cv2.recoverPose(E, pts0[inlier_mask], pts1[inlier_mask], K)

        P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = K @ np.hstack([R, t])

        good_idx = np.where(inlier_mask)[0]
        pts0_in = pts0[good_idx]
        pts1_in = pts1[good_idx]
        matches_in = [matches[k] for k in good_idx]

        pts4d = cv2.triangulatePoints(P0, P1, pts0_in.T, pts1_in.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        for idx, (m, X) in enumerate(zip(matches_in, pts3d)):
            if not np.all(np.isfinite(X)):
                continue
            color = _sample_color(images[j], pts1_in[idx])
            point_id = len(all_points)
            all_points.append([float(X[0]), float(X[1]), float(X[2])] + color)
            track[0][m.queryIdx] = point_id
            track[j][m.trainIdx] = point_id

        poses[0] = (np.eye(3), np.zeros((3, 1)))
        poses[j] = (R, t)
        cur_pose = (R, t)
        prev_idx = j
        used_images = 2
        init_ok = True
        break

    if not init_ok:
        warnings.append("Could not find any two images with enough overlap to start reconstruction.")
        return {"points": [], "num_points": 0, "num_images_used": 0, "warnings": warnings}

    # ---- Incrementally register the remaining images ----
    remaining = [i for i in range(len(images)) if i not in (0, prev_idx)]
    for i in remaining:
        kp_prev, des_prev = feats[prev_idx]
        kp_i, des_i = feats[i]
        matches = _match(des_prev, des_i)
        if len(matches) < MIN_MATCHES:
            warnings.append(f"Image {prev_idx} and {i}: only {len(matches)} matches, skipping image {i}.")
            continue

        pts3d_for_pnp = []
        pts2d_for_pnp = []
        new_matches = []
        for m in matches:
            pid = track[prev_idx].get(m.queryIdx)
            if pid is not None:
                pts3d_for_pnp.append(all_points[pid][:3])
                pts2d_for_pnp.append(kp_i[m.trainIdx].pt)
            else:
                new_matches.append(m)

        R_i, t_i = None, None
        if len(pts3d_for_pnp) >= MIN_PNP_CORRESPONDENCES:
            obj_pts = np.array(pts3d_for_pnp, dtype=np.float64)
            img_pts = np.array(pts2d_for_pnp, dtype=np.float64)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj_pts, img_pts, K, None,
                                                          reprojectionError=8.0, confidence=0.99)
            if ok:
                R_i, _ = cv2.Rodrigues(rvec)
                t_i = tvec

        if R_i is None:
            # Fall back to essential-matrix pose estimation relative to prev image.
            pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches])
            pts_i = np.float32([kp_i[m.trainIdx].pt for m in matches])
            E, mask = cv2.findEssentialMat(pts_prev, pts_i, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is None:
                warnings.append(f"Could not estimate pose for image {i}, skipping.")
                continue
            inlier_mask = mask.ravel().astype(bool)
            _, R_rel, t_rel, _ = cv2.recoverPose(E, pts_prev[inlier_mask], pts_i[inlier_mask], K)
            R_prev, t_prev = poses[prev_idx]
            R_i = R_rel @ R_prev
            t_i = R_rel @ t_prev + t_rel

        poses[i] = (R_i, t_i)
        used_images += 1

        # Triangulate brand-new points from matches that had no existing 3D point.
        if new_matches:
            R_prev, t_prev = poses[prev_idx]
            P_prev = K @ np.hstack([R_prev, t_prev])
            P_i = K @ np.hstack([R_i, t_i])
            pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in new_matches])
            pts_i = np.float32([kp_i[m.trainIdx].pt for m in new_matches])
            pts4d = cv2.triangulatePoints(P_prev, P_i, pts_prev.T, pts_i.T)
            pts3d = (pts4d[:3] / pts4d[3]).T
            for m, X, px in zip(new_matches, pts3d, pts_i):
                if not np.all(np.isfinite(X)):
                    continue
                color = _sample_color(images[i], px)
                point_id = len(all_points)
                all_points.append([float(X[0]), float(X[1]), float(X[2])] + color)
                track[prev_idx][m.queryIdx] = point_id
                track[i][m.trainIdx] = point_id

        prev_idx = i

    if not all_points:
        return {"points": [], "num_points": 0, "num_images_used": used_images, "warnings": warnings}

    arr = np.array(all_points, dtype=np.float64)
    keep_mask = _remove_outliers(arr)
    arr = arr[keep_mask]

    # Center and scale to a friendly viewing range for the frontend viewer.
    xyz = arr[:, :3]
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    scale = np.abs(xyz).max()
    if scale > 1e-9:
        xyz = xyz / scale * 2.0
    arr[:, :3] = xyz

    points_out = arr.tolist()
    return {
        "points": points_out,
        "num_points": len(points_out),
        "num_images_used": used_images,
        "warnings": warnings,
    }
