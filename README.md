# 3D Object Mapper

Upload a set of photos of a real-world object taken from different
angles, and this app reconstructs a rotatable, colored 3D point cloud
you can view in the browser and download as a `.ply` file.

It's a small end-to-end **photogrammetry** pipeline:

1. **SIFT** feature detection on every uploaded photo (OpenCV).
2. **Feature matching** between consecutive photos.
3. **Essential matrix + pose recovery** to bootstrap the first two cameras,
   then **incremental registration** (solvePnP) of every following photo.
4. **Triangulation** of matched points into 3D, merged into one colored
   point cloud with simple statistical outlier removal.
5. A **Flask** backend serves the pipeline; the frontend renders the
   result with **Three.js** (orbit/zoom controls) and offers a `.ply`
   download for use in Blender, MeshLab, CloudCompare, etc.

No paid APIs, no cloud processing - everything runs locally.

## Setup

```bash
cd 3d-object-mapper
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open **http://localhost:5050** in your browser.

## Usage

1. Take 10-30 photos of an object, moving all the way around it in small
   steps (roughly every 15-20 degrees), keeping the object centered and
   in focus.
2. Upload them all at once (click the dropzone, or drag & drop).
3. Click **Build 3D Map**. Reconstruction runs locally and usually takes
   a few seconds to a couple of minutes depending on photo count/size.
4. Drag to orbit, scroll to zoom, in the resulting point cloud.
5. Click **Download .ply** to save the point cloud for use elsewhere.

### Tips for a good scan

- Use the **same camera** for every shot in a batch (the pipeline assumes
  one shared, uncalibrated focal-length estimate for the whole set).
- Keep **~60-80% visual overlap** between consecutive photos - lots of
  small steps beats a few big jumps around the object.
- Even, diffuse **lighting** and a **textured/detailed surface** help a
  lot - SIFT matching needs visual texture to lock onto. Plain, glossy,
  or transparent objects are hard to reconstruct.
- If a scan comes out empty or very sparse, the app's status message
  will suggest what to try (more overlap, more light/texture).

## Known limitations

This is a compact, from-scratch SfM pipeline (OpenCV + NumPy only) built
for a clear demo, not a COLMAP/RealityCapture replacement:

- Camera intrinsics are a **heuristic estimate** (no calibration step),
  so absolute scale and some geometry can be distorted.
- No bundle adjustment, so pose drift can accumulate over long photo
  sequences.
- Output is a **point cloud**, not a watertight mesh.
- Works best on a single object with a reasonably simple, textured
  surface - not whole rooms or scenes.

## Project layout

```
3d-object-mapper/
├── app.py              # Flask routes (upload, reconstruct, download)
├── sfm.py              # Structure-from-motion pipeline
├── ply_io.py           # Minimal ASCII .ply writer
├── templates/index.html
├── static/js/viewer.js # Upload UI + Three.js point-cloud viewer
├── static/css/style.css
└── tests/
```

## Tests

```bash
pytest
```
