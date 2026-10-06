import numpy as np

from ply_io import write_ply
from sfm import _detect_features, _intrinsics_for, _remove_outliers


def test_intrinsics_shape_and_values():
    K = _intrinsics_for(800, 600)
    assert K.shape == (3, 3)
    assert K[0, 2] == 400.0
    assert K[1, 2] == 300.0
    assert K[0, 0] == K[1, 1]
    assert K[0, 0] > 0


def test_detect_features_runs_on_textured_image():
    # Synthetic checkerboard-ish image so SIFT has something to find.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(300, 300, 3), dtype=np.uint8)
    kp, des = _detect_features(img)
    assert isinstance(kp, (list, tuple))
    # Random noise usually yields at least a handful of keypoints.
    assert des is None or des.shape[0] == len(kp)


def test_remove_outliers_flags_far_point():
    cluster = np.random.default_rng(1).normal(0, 0.05, size=(30, 6))
    outlier = np.array([[50, 50, 50, 255, 255, 255]])
    points = np.vstack([cluster, outlier])
    keep = _remove_outliers(points, k=5, std_ratio=2.0)
    assert keep[-1] == False  # noqa: E712 - explicit bool comparison reads clearer here
    assert keep[:-1].sum() >= 25


def test_write_ply_header_and_row_count(tmp_path):
    points = [[0.0, 0.0, 0.0, 255, 0, 0], [1.0, 1.0, 1.0, 0, 255, 0]]
    out = tmp_path / "out.ply"
    write_ply(str(out), points)
    text = out.read_text()
    assert "element vertex 2" in text
    assert text.strip().splitlines()[-1] == "1.0 1.0 1.0 0 255 0"
