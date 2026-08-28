"""
Unit tests for backend/inference.py — analyze_image() and helpers.
Uses the pre-trained .joblib models so no training is required to run these.
"""
import sys
import os

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from inference import analyze_image, _severity_from_features


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_png_bytes(h: int = 64, w: int = 64, fill=(128, 128, 128)) -> bytes:
    """Create a minimal in-memory PNG image as raw bytes."""
    img = np.full((h, w, 3), fill, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok, "Failed to encode test PNG"
    return buf.tobytes()


def make_noisy_png_bytes(h: int = 64, w: int = 64, sigma: int = 45) -> bytes:
    rng = np.random.default_rng(99)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ── analyze_image ─────────────────────────────────────────────────────────────

class TestAnalyzeImage:
    def test_returns_required_keys(self):
        """Response must contain all keys from the assessment spec."""
        result = analyze_image(make_png_bytes())
        assert "quality_score" in result
        assert "quality_label" in result
        assert "issues" in result
        assert "confidence" in result
        assert "label_probabilities" in result
        assert "image_stats" in result

    def test_quality_score_is_int(self):
        """quality_score must be an integer (not float) per the spec example."""
        result = analyze_image(make_png_bytes())
        assert isinstance(result["quality_score"], int), \
            f"Expected int, got {type(result['quality_score'])}"

    def test_quality_score_in_range(self):
        result = analyze_image(make_png_bytes())
        assert 0 <= result["quality_score"] <= 100

    def test_quality_label_valid(self):
        result = analyze_image(make_png_bytes())
        assert result["quality_label"] in ("ACCEPTABLE", "DEGRADED", "DEFECTIVE")

    def test_confidence_in_range(self):
        result = analyze_image(make_png_bytes())
        assert 0.0 <= result["confidence"] <= 1.0

    def test_label_probabilities_sum_to_one(self):
        result = analyze_image(make_png_bytes())
        total = sum(result["label_probabilities"].values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_issues_have_correct_shape(self):
        result = analyze_image(make_png_bytes())
        for issue in result["issues"]:
            assert "type" in issue
            assert "severity" in issue
            assert "confidence" in issue
            assert issue["severity"] in ("low", "medium", "high")
            assert 0.0 <= issue["confidence"] <= 1.0

    def test_image_stats_has_width_height(self):
        result = analyze_image(make_png_bytes(h=80, w=120))
        stats = result["image_stats"]
        assert stats["width"] == 120
        assert stats["height"] == 80

    def test_invalid_bytes_raises_value_error(self):
        with pytest.raises(ValueError):
            analyze_image(b"not an image")

    def test_empty_bytes_raises_value_error(self):
        with pytest.raises(ValueError):
            analyze_image(b"")

    def test_too_small_image_raises_value_error(self):
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".png", tiny)
        with pytest.raises(ValueError):
            analyze_image(buf.tobytes())

    def test_noisy_image_analyzed(self):
        """Noisy image should still return a valid result (not crash)."""
        result = analyze_image(make_noisy_png_bytes())
        assert result["quality_label"] in ("ACCEPTABLE", "DEGRADED", "DEFECTIVE")

    def test_dark_image_returns_result(self):
        dark = make_png_bytes(fill=(5, 5, 5))
        result = analyze_image(dark)
        assert "quality_label" in result

    def test_bright_image_returns_result(self):
        bright = make_png_bytes(fill=(250, 250, 250))
        result = analyze_image(bright)
        assert "quality_label" in result


# ── _severity_from_features ───────────────────────────────────────────────────

class TestSeverityFromFeatures:
    def test_blur_high_sharpness_is_low(self):
        feats = {"sharpness": 500}
        assert _severity_from_features("blur", feats, 0.9) == "low"

    def test_blur_very_low_sharpness_is_high(self):
        feats = {"sharpness": 10}
        assert _severity_from_features("blur", feats, 0.5) == "high"

    def test_underexposure_very_dark_is_high(self):
        feats = {"mean_brightness": 20}
        assert _severity_from_features("underexposure", feats, 0.5) == "high"

    def test_overexposure_many_bright_pixels_is_high(self):
        feats = {"bright_pixel_fraction": 0.7}
        assert _severity_from_features("overexposure", feats, 0.5) == "high"

    def test_noise_low_residual_is_low(self):
        feats = {"noise": 2}
        assert _severity_from_features("noise", feats, 0.9) == "low"

    def test_corruption_high_uniformity_is_high(self):
        feats = {"block_uniformity": 0.5}
        assert _severity_from_features("corruption", feats, 0.5) == "high"

    def test_visual_defect_falls_back_to_confidence(self):
        feats = {}
        assert _severity_from_features("visual_defect", feats, 0.9) == "high"
        assert _severity_from_features("visual_defect", feats, 0.6) == "medium"
        assert _severity_from_features("visual_defect", feats, 0.3) == "low"
