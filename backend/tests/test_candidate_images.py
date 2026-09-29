"""Search candidates must resolve to the exact gallery images they ranked."""

import io
import zipfile

from fastapi.testclient import TestClient
from PIL import Image

import app.main as backend_module
from app.gallery_store import COMMON_GALLERY_ID


def _image(index):
    stream = io.BytesIO()
    Image.new("RGB", (20, 20), (index, 0, 0)).save(stream, format="PNG")
    return stream.getvalue()


def test_all_ranked_candidates_have_zoomable_gallery_images(tmp_path, monkeypatch):
    monkeypatch.setenv("LCT_GALLERY_STATE_DIR", str(tmp_path))

    async def build(gallery_id, job_id):
        from app.gallery_store import read_gallery, _write_json, gallery_dir
        meta = read_gallery(gallery_id)
        meta["images"].extend(meta["pending"])
        meta.update(pending=[], gallery_file="gallery_0001.npz", state="ready", job_id=None)
        _write_json(gallery_dir(gallery_id) / "meta.json", meta)

    async def search(path, image, form):
        assert form["gallery_id"] == COMMON_GALLERY_ID
        ranked = [{"gallery_id": f"car{i}", "similarity": 1 - i / 100,
                   "confidence": 1 - i / 100} for i in range(1, 11)]
        return {"status": "matched", "model_version": "fixture", "gallery_id": COMMON_GALLERY_ID,
                "ranked": ranked, "accepted": [ranked[0]], "candidates": [ranked[0]],
                "threshold": .394, "confidence_semantics": "raw cosine similarity; not a probability"}

    monkeypatch.setattr(backend_module, "build_ml_gallery", build)
    monkeypatch.setattr(backend_module, "call_ml", search)
    client = TestClient(backend_module.app)
    manifest = "filename,gallery_id,x,y,w,h\n" + "".join(
        f"car{i}.png,car{i},{2 if i == 1 else 0},{3 if i == 1 else 0},"
        f"{10 if i == 1 else 20},{11 if i == 1 else 20}\n" for i in range(1, 11)
    )
    upload = client.post("/api/common-gallery/images", files=[
        ("images", (f"car{i}.png", _image(i), "image/png")) for i in range(1, 11)
    ] + [("manifest", ("manifest.csv", manifest.encode(), "text/csv"))])
    assert upload.status_code == 202, upload.text
    response = client.post("/api/infer", files={"image": ("query.png", _image(1), "image/png")},
                           data={"x": 0, "y": 0, "w": 20, "h": 20,
                                 "gallery_id": COMMON_GALLERY_ID})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["ranked"]) == 10
    assert all(item["image_url"].endswith("/crop") for item in body["ranked"])
    assert body["accepted"][0]["image_url"] == body["ranked"][0]["image_url"]
    crop = client.get(body["ranked"][0]["image_url"])
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(crop.content)).size == (10, 11)
    assert client.get(body["ranked"][0]["source_image_url"]).content == _image(1)
    assert client.get(f"/api/galleries/{COMMON_GALLERY_ID}/images/not-in-gallery/crop").status_code == 404


def test_batch_zip_uses_full_csv_and_processes_each_query(tmp_path, monkeypatch):
    monkeypatch.setenv("LCT_GALLERY_STATE_DIR", str(tmp_path))
    from app.gallery_store import ensure_common_gallery, gallery_dir, read_gallery, _write_json
    ensure_common_gallery()
    meta = read_gallery(COMMON_GALLERY_ID)
    meta.update(gallery_file="gallery_0001.npz", state="ready", images=[
        {"gallery_id": f"car{i}", "image_key": f"key{i}.png", "filename": f"car{i}.png",
         "bbox": [0, 0, 20, 20]} for i in range(1, 11)
    ])
    _write_json(gallery_dir(COMMON_GALLERY_ID) / "meta.json", meta)
    seen = []

    async def search(path, image, form):
        seen.append((image.filename, form["x"], form["y"], form["w"], form["h"]))
        ranked = [{"gallery_id": "car1", "similarity": .9, "confidence": .9}]
        return {"status": "matched", "model_version": "test", "gallery_id": COMMON_GALLERY_ID,
                "ranked": ranked, "accepted": ranked, "candidates": ranked,
                "threshold": .394, "confidence_semantics": "cosine"}

    monkeypatch.setattr(backend_module, "call_ml", search)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as zipped:
        zipped.writestr("query/car1.png", _image(1))
        zipped.writestr("query/car2.png", _image(2))
        zipped.writestr("query/notes.csv", "unrelated")
    csv_bytes = b"image_id,x,y,w,h\nunused,0,0,20,20\ncar1,2,3,10,11\ncar2.png,0,0,20,20\n"
    client = TestClient(backend_module.app)
    response = client.post("/api/infer-batch", data={"gallery_id": COMMON_GALLERY_ID}, files={
        "archive": ("queries.zip", stream.getvalue(), "application/zip"),
        "manifest": ("all_queries.csv", csv_bytes, "text/csv"),
    })
    assert response.status_code == 200, response.text
    assert seen == [("car1.png", 2, 3, 10, 11), ("car2.png", 0, 0, 20, 20)]
    assert len(response.json()["results"]) == 2
    assert response.json()["results"][0]["payload"]["ranked"][0]["image_url"].endswith("/crop")
