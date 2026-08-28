"""
Integration tests for the FastAPI REST API.
Uses FastAPI's TestClient (sync) to make real HTTP calls against the app
without needing a running server.
"""
import sys
import os
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# The app object is rebuilt by importing app.py
import app as app_module
client = TestClient(app_module.app, raise_server_exceptions=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_png_bytes(fill=(128, 128, 128)) -> bytes:
    img = np.full((64, 64, 3), fill, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ── GET /health ────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# ── GET /metrics ───────────────────────────────────────────────────────────────

class TestMetrics:
    def test_returns_200_or_503(self):
        """If metrics.json exists, returns 200 with content; if not, 503."""
        r = client.get("/metrics")
        assert r.status_code in (200, 503)

    def test_200_has_required_keys(self):
        r = client.get("/metrics")
        if r.status_code == 200:
            data = r.json()
            assert "quality_label" in data
            assert "feature_importances" in data


# ── POST /analyze ─────────────────────────────────────────────────────────────

class TestAnalyze:
    def test_valid_image_returns_200(self):
        png = make_png_bytes()
        r = client.post("/analyze", files={"file": ("test.png", png, "image/png")})
        assert r.status_code == 200

    def test_response_shape(self):
        png = make_png_bytes()
        r = client.post("/analyze", files={"file": ("test.png", png, "image/png")})
        data = r.json()
        assert "quality_score" in data
        assert "quality_label" in data
        assert "issues" in data
        assert "confidence" in data
        assert "image_stats" in data
        assert "id" in data          # DB id must be included

    def test_quality_score_is_int(self):
        png = make_png_bytes()
        r = client.post("/analyze", files={"file": ("test.png", png, "image/png")})
        assert isinstance(r.json()["quality_score"], int)

    def test_non_image_returns_400(self):
        r = client.post(
            "/analyze",
            files={"file": ("file.txt", b"hello world", "text/plain")},
        )
        assert r.status_code == 400

    def test_empty_file_returns_400(self):
        r = client.post(
            "/analyze",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert r.status_code == 400

    def test_corrupt_image_returns_422(self):
        r = client.post(
            "/analyze",
            files={"file": ("bad.jpg", b"not image data", "image/jpeg")},
        )
        assert r.status_code == 422

    def test_result_saved_to_db(self):
        """After a successful analysis, id should be present and retrievable."""
        png = make_png_bytes()
        r = client.post("/analyze", files={"file": ("test.png", png, "image/png")})
        assert r.status_code == 200
        analysis_id = r.json()["id"]
        assert isinstance(analysis_id, int)

        detail = client.get(f"/history/{analysis_id}")
        assert detail.status_code == 200
        assert detail.json()["id"] == analysis_id


# ── GET /history ───────────────────────────────────────────────────────────────

class TestHistory:
    def test_returns_list(self):
        r = client.get("/history")
        assert r.status_code == 200
        data = r.json()
        assert "analyses" in data
        assert isinstance(data["analyses"], list)

    def test_limit_param_respected(self):
        r = client.get("/history?limit=2")
        assert r.status_code == 200
        assert len(r.json()["analyses"]) <= 2

    def test_history_items_have_required_fields(self):
        # First ensure there is at least one analysis
        png = make_png_bytes()
        client.post("/analyze", files={"file": ("t.png", png, "image/png")})

        r = client.get("/history?limit=1")
        items = r.json()["analyses"]
        if items:
            item = items[0]
            assert "id" in item
            assert "filename" in item
            assert "quality_label" in item
            assert "quality_score" in item
            assert "created_at" in item


# ── GET /history/{id} ─────────────────────────────────────────────────────────

class TestHistoryDetail:
    def test_known_id_returns_200(self):
        png = make_png_bytes()
        r = client.post("/analyze", files={"file": ("t.png", png, "image/png")})
        aid = r.json()["id"]
        detail = client.get(f"/history/{aid}")
        assert detail.status_code == 200

    def test_unknown_id_returns_404(self):
        r = client.get("/history/9999999")
        assert r.status_code == 404
