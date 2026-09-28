"""Primary UI routes must expose the three real ReID workflows."""

from fastapi.testclient import TestClient

from app.main import app


def test_home_links_to_gallery_batch_and_single():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    for route in ("/gallery", "/many", "/solo"):
        assert f'href="{route}"' in response.text
    assert 'data-route="/replenishment"' not in response.text


def test_workflow_pages_render_without_ml_service():
    client = TestClient(app)
    for route, title, script in (
        ("/gallery", "Загрузить галерею", "gallery.js"),
        ("/many", "Изображения + BBox CSV", "many.js"),
        ("/solo", "ручной BBox", "solo.js"),
    ):
        response = client.get(route)
        assert response.status_code == 200
        assert title in response.text
        assert script in response.text


def test_gallery_upload_writes_to_indexing_api_not_replenishment_store():
    response = TestClient(app).get("/static/gallery.js")
    assert response.status_code == 200
    assert "`/api/galleries/${id}/images`" in response.text
    assert "/api/replenishment/" not in response.text
