import io

import app as app_module


def make_client():
    app_module.app.testing = True
    return app_module.app.test_client()


def test_index_route_ok():
    client = make_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"3D Object Mapper" in resp.data


def test_reconstruct_requires_min_images():
    client = make_client()
    data = {
        "images": [
            (io.BytesIO(b"not a real image"), "a.jpg"),
            (io.BytesIO(b"not a real image"), "b.jpg"),
        ]
    }
    resp = client.post("/api/reconstruct", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "at least" in resp.get_json()["error"]


def test_download_missing_job_returns_404():
    client = make_client()
    resp = client.get("/api/download/doesnotexist")
    assert resp.status_code == 404
