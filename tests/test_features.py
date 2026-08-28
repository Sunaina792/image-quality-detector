"""
Unit tests for backend/features.py.
Tests each feature function with known synthetic numpy arrays so we can
verify the math is correct independently of any trained model.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from features import (
    block_uniformity_score,
    brightness_stats,
    contrast_score,
    edge_density,
    extract_features,
    features_to_vector,
    FEATURE_ORDER,
    noise_score,
    saturation_stats,
    sharpness_score,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_gray(h: int = 64, w: int = 64, fill: int = 128) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.uint8)


def make_bgr(h: int = 64, w: int = 64, fill=(128, 128, 128)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = fill
    return img


# ── sharpness_score ─────────────────────────────────────────────────────────

class TestSharpnessScore:
    def test_flat_image_is_blurry(self):
        """A perfectly uniform image has zero Laplacian variance → score 0."""
        gray = make_gray(fill=128)
        assert sharpness_score(gray) == pytest.approx(0.0, abs=1e-9)

    def test_checkerboard_is_sharp(self):
        """Alternating 0/255 checkerboard has maximum second-derivative → high variance."""
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[::2, ::2] = 255
        gray[1::2, 1::2] = 255
        score = sharpness_score(gray)
        assert score > 1000, f"Expected high sharpness, got {score}"

    def test_gradient_has_medium_sharpness(self):
        """A smooth gradient has moderate sharpness."""
        gray = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
        score = sharpness_score(gray)
        assert 0 < score < 5000


# ── brightness_stats ─────────────────────────────────────────────────────────

class TestBrightnessStats:
    def test_mid_gray(self):
        gray = make_gray(fill=128)
        stats = brightness_stats(gray)
        assert stats["mean_brightness"] == pytest.approx(128.0, abs=1)
        assert stats["dark_pixel_fraction"] == pytest.approx(0.0, abs=1e-9)
        assert stats["bright_pixel_fraction"] == pytest.approx(0.0, abs=1e-9)

    def test_black_image(self):
        gray = make_gray(fill=0)
        stats = brightness_stats(gray)
        assert stats["mean_brightness"] == pytest.approx(0.0)
        assert stats["dark_pixel_fraction"] == pytest.approx(1.0, abs=0.01)

    def test_white_image(self):
        gray = make_gray(fill=255)
        stats = brightness_stats(gray)
        assert stats["mean_brightness"] == pytest.approx(255.0)
        assert stats["bright_pixel_fraction"] == pytest.approx(1.0, abs=0.01)


# ── contrast_score ───────────────────────────────────────────────────────────

class TestContrastScore:
    def test_flat_image_zero_contrast(self):
        assert contrast_score(make_gray(fill=128)) == pytest.approx(0.0, abs=1e-9)

    def test_binary_image_max_contrast(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[:32, :] = 255
        # std of 50/50 split between 0 and 255 ≈ 127.5
        assert contrast_score(gray) == pytest.approx(127.5, abs=1)


# ── noise_score ───────────────────────────────────────────────────────────────

class TestNoiseScore:
    def test_flat_image_no_noise(self):
        """A uniform image has nothing for the residual to capture."""
        gray = make_gray(fill=128)
        score = noise_score(gray)
        assert score == pytest.approx(0.0, abs=0.5)

    def test_random_noise_detected(self):
        """High-sigma Gaussian noise should produce a high noise score."""
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 255, (64, 64), dtype=np.uint8)
        score = noise_score(gray)
        assert score > 10, f"Expected noisy score > 10, got {score}"


# ── block_uniformity_score ────────────────────────────────────────────────────

class TestBlockUniformity:
    def test_uniform_image_max_uniformity(self):
        gray = make_gray(fill=0)
        score = block_uniformity_score(gray)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_natural_image_low_uniformity(self):
        rng = np.random.default_rng(7)
        gray = rng.integers(0, 255, (128, 128), dtype=np.uint8)
        score = block_uniformity_score(gray)
        assert score < 0.1


# ── extract_features + features_to_vector ────────────────────────────────────

class TestExtractFeatures:
    def test_returns_all_keys(self):
        bgr = make_bgr()
        feats = extract_features(bgr)
        for k in FEATURE_ORDER:
            assert k in feats, f"Missing feature key: {k}"

    def test_vector_length_matches_feature_order(self):
        bgr = make_bgr()
        feats = extract_features(bgr)
        vec = features_to_vector(feats)
        assert vec.shape == (len(FEATURE_ORDER),)

    def test_all_finite(self):
        bgr = make_bgr()
        feats = extract_features(bgr)
        vec = features_to_vector(feats)
        assert np.all(np.isfinite(vec))
