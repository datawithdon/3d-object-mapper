"""
3D Object Mapper - Flask backend.

Upload a handful of photos of an object taken from different angles and
the app reconstructs a colored 3D point cloud (via sfm.py) that you can
rotate/zoom in the browser, and download as a .ply file.
"""

import os
import shutil
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory

from ply_io import write_ply
from sfm import reconstruct

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_IMAGES = 3
MAX_IMAGES = 60

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload cap


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reconstruct", methods=["POST"])
def api_reconstruct():
    files = request.files.getlist("images")

    if len(files) > MAX_IMAGES:
        return jsonify({"error": f"Please upload at most {MAX_IMAGES} photos per job."}), 400

    job_id = uuid.uuid4().hex[:12]
    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)

    saved_paths = []
    try:
        for idx, f in enumerate(files):
            ext = os.path.splitext(f.filename or "")[1].lower()
            if ext not in ALLOWED_EXT:
                continue
            save_path = os.path.join(job_upload_dir, f"{idx:03d}{ext}")
            f.save(save_path)
            saved_paths.append(save_path)

        if len(saved_paths) < MIN_IMAGES:
            return jsonify({
                "error": f"Need at least {MIN_IMAGES} valid photos ({', '.join(sorted(ALLOWED_EXT))})."
            }), 400

        try:
            result = reconstruct(saved_paths)
        except Exception as e:  # noqa: BLE001 - surface any reconstruction failure to the caller
            return jsonify({"error": f"Reconstruction failed: {e}"}), 500

        if result["num_points"] == 0:
            return jsonify({
                "error": "Could not reconstruct any 3D points from these photos.",
                "detail": "Try photos with more overlap between angles, better lighting, "
                           "and more surface texture/detail on the object.",
                "warnings": result.get("warnings", []),
            }), 422

        ply_path = os.path.join(OUTPUT_DIR, f"{job_id}.ply")
        write_ply(ply_path, result["points"])

        return jsonify({
            "job_id": job_id,
            "num_points": result["num_points"],
            "num_images_used": result["num_images_used"],
            "points": result["points"],
            "warnings": result.get("warnings", []),
            "download_url": f"/api/download/{job_id}",
        })
    finally:
        shutil.rmtree(job_upload_dir, ignore_errors=True)


@app.route("/api/download/<job_id>")
def download(job_id):
    filename = f"{job_id}.ply"
    if not os.path.exists(os.path.join(OUTPUT_DIR, filename)):
        return jsonify({"error": "File not found."}), 404
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
