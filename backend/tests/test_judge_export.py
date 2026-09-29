"""Weight-free test of the separate browser/CLI submission contract."""

from __future__ import annotations

import io
import json
import time
from threading import Lock
from types import SimpleNamespace
from zipfile import ZipFile

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import app.judge_export_routes as gateway
import app.ml.judge_export_api as worker
from app.main import app as backend_app


def organizer_zip() -> bytes:
    image = Image.new("RGB", (96, 72), (20, 70, 130))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    result = io.BytesIO()
    with ZipFile(result, "w") as archive:
        gallery = [f"{index:032x}" for index in range(10)]
        query = [f"{10:032x}"]
        for name, ids in (("test_gallery.csv", gallery), ("test_query.csv", query)):
            archive.writestr(name, "image_id,x,y,w,h\n" + "".join(f"{item},10,8,60,40\n" for item in ids))
            for item in ids:
                archive.writestr(f"images/{item}.jpg", buffer.getvalue())
    return result.getvalue()


def fixture_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("LCT_JUDGE_EXPORT_ENABLED", "1")
    monkeypatch.setenv("LCT_JUDGE_EXPORT_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(worker, "DISK_RESERVE_BYTES", 0)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "model_manifest.json").write_text(json.dumps({
        "model_version": "fixture", "model_sha256": "fixture-sha",
    }))
    monkeypatch.setenv("LCT_ML_ARTIFACT_DIR", str(artifacts))
    runtime = SimpleNamespace(model_version="fixture", model_sha256="fixture-sha",
                              encoder=object(), _inference_lock=Lock())

    def fake_build(source, artifact_dir, output, **kwargs):
        assert source.is_file() and source.read_bytes() == organizer_zip()
        assert artifact_dir == artifacts and kwargs["encoder"] is runtime.encoder
        output.joinpath("submission").mkdir(parents=True)
        for name in worker.DOWNLOADS:
            (output / "submission" / name).write_bytes(name.encode())
        kwargs["progress"]("gallery", 10, 10)
        kwargs["progress"]("query", 1, 1)
        return {"status": "completed", "scope": "full organizer test archive",
                "submission_validation": {"queries": 1, "gallery": 10,
                                          "roundtrip_top10_exact": True},
                "input_sha256": "fixture-input-sha"}

    monkeypatch.setattr(worker, "build", fake_build)
    app = FastAPI()
    app.include_router(worker.create_router(lambda: runtime))
    return app


def wait_complete(client, job_id):
    for _ in range(100):
        response = client.get(f"/internal/judge-exports/{job_id}")
        if response.json()["state"] == "completed":
            return response.json()
        time.sleep(0.01)
    raise AssertionError("Fixture export did not complete")


def test_worker_streams_zip_and_exposes_only_complete_files(tmp_path, monkeypatch):
    client = TestClient(fixture_worker(tmp_path, monkeypatch))
    response = client.post("/internal/judge-exports", content=organizer_zip(),
                           headers={"content-type": "application/zip"})
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    assert response.json()["counts"]["gallery"] == 10
    result = wait_complete(client, job_id)
    assert result["validation"]["roundtrip_top10_exact"]
    for name in worker.DOWNLOADS:
        downloaded = client.get(f"/internal/judge-exports/{job_id}/files/{name}")
        assert downloaded.status_code == 200
        assert downloaded.content == name.encode()
    assert client.get(f"/internal/judge-exports/{job_id}/files/../../archive.zip").status_code != 200


def test_worker_rejects_missing_bbox_csv(tmp_path, monkeypatch):
    client = TestClient(fixture_worker(tmp_path, monkeypatch))
    response = client.post("/internal/judge-exports", content=b"not a ZIP",
                           headers={"content-type": "application/zip"})
    assert response.status_code == 422
    assert client.post("/internal/judge-exports", content=b"abc",
                       headers={"content-type": "text/plain"}).status_code == 415


def test_backend_browser_gateway_streams_same_job(tmp_path, monkeypatch):
    ml_app = fixture_worker(tmp_path, monkeypatch)
    original = httpx.AsyncClient

    def local_client(*args, **kwargs):
        return original(transport=httpx.ASGITransport(app=ml_app), **kwargs)

    monkeypatch.setattr(gateway.httpx, "AsyncClient", local_client)
    client = TestClient(backend_app)
    archive = organizer_zip()
    response = client.post("/api/judge-export/jobs", content=archive,
                           headers={"content-type": "application/zip", "origin": "http://testserver"})
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    for _ in range(100):
        status = client.get(f"/api/judge-export/jobs/{job_id}").json()
        if status["state"] == "completed":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("Gateway export did not complete")
    assert client.get(f"/api/judge-export/jobs/{job_id}/files/embeddings.npy").content == b"embeddings.npy"
    assert client.post("/api/judge-export/jobs", content=archive,
                       headers={"content-type": "application/zip", "origin": "https://evil.test"}).status_code == 403
