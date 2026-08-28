"""
Synthetic degradation generator.

Takes clean images and deliberately damages them in controlled, labeled ways
(blur, over/underexposure, noise, corruption). This is our "test kitchen":
instead of hunting for already-bad photos with unreliable labels, we take
known-good photos and break them ourselves, so we know the ground truth.

Justified by assessment section 8: "If synthetic degradation is used,
describe how training and evaluation data were generated."
"""

import io
import random

import cv2
import numpy as np
from PIL import Image

ISSUE_TYPES = [
    "clean",
    "blur",
    "underexposure",
    "overexposure",
    "noise",
    "corruption",
    "visual_defect",
]


def apply_blur(bgr: np.ndarray, severity: str) -> np.ndarray:
    """Gaussian blur. severity in {low, medium, high} controls kernel size."""
    k = {"low": 5, "medium": 11, "high": 21}[severity]
    return cv2.GaussianBlur(bgr, (k, k), 0)


def apply_underexposure(bgr: np.ndarray, severity: str) -> np.ndarray:
    """Darken the image (simulates shooting in low light with no flash)."""
    factor = {"low": 0.6, "medium": 0.35, "high": 0.15}[severity]
    return np.clip(bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_overexposure(bgr: np.ndarray, severity: str) -> np.ndarray:
    """Brighten/blow out the image (simulates shooting into direct sun)."""
    add = {"low": 60, "medium": 110, "high": 170}[severity]
    return np.clip(bgr.astype(np.float32) + add, 0, 255).astype(np.uint8)


def apply_noise(bgr: np.ndarray, severity: str) -> np.ndarray:
    """Add Gaussian sensor noise (simulates high-ISO low-light shots)."""
    sigma = {"low": 10, "medium": 25, "high": 45}[severity]
    noise = np.random.normal(0, sigma, bgr.shape).astype(np.float32)
    return np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_corruption(bgr: np.ndarray, severity: str) -> np.ndarray:
    """
    Simulate corrupted/truncated JPEG data by re-encoding at very low
    quality and/or blacking out random blocks (mimics decode failures
    where chunks of the image fail to load).
    """
    quality = {"low": 25, "medium": 10, "high": 3}[severity]
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    if severity in ("medium", "high"):
        h, w = out.shape[:2]
        n_blocks = {"medium": 2, "high": 6}[severity]
        block = max(8, min(h, w) // 8)
        for _ in range(n_blocks):
            y = random.randint(0, max(0, h - block))
            x = random.randint(0, max(0, w - block))
            out[y : y + block, x : x + block] = 0
    return out


def apply_visual_defect(bgr: np.ndarray, severity: str) -> np.ndarray:
    """
    Simulate visual defects: random colored scratch lines and a semi-transparent
    tinted region (watermark / lens flare artifact). These represent
    "Potential visual defect" from the assessment requirements — defects that
    are NOT just exposure/blur/noise problems.
    """
    out = bgr.copy().astype(np.float32)
    h, w = out.shape[:2]

    # Number of scratch lines and tint intensity scale with severity
    n_scratches = {"low": 2, "medium": 5, "high": 12}[severity]
    tint_alpha = {"low": 0.15, "medium": 0.30, "high": 0.50}[severity]

    # Scratch lines — thin random diagonal lines across the image
    for _ in range(n_scratches):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        color = (
            float(random.randint(180, 255)),
            float(random.randint(180, 255)),
            float(random.randint(180, 255)),
        )
        cv2.line(out, (x1, y1), (x2, y2), color, thickness=random.randint(1, 3))

    # Tinted rectangular region (simulates lens flare / watermark / burn)
    rx1 = random.randint(0, w // 2)
    ry1 = random.randint(0, h // 2)
    rx2 = min(w, rx1 + random.randint(w // 4, w // 2))
    ry2 = min(h, ry1 + random.randint(h // 4, h // 2))
    tint_color = np.array(
        [random.randint(0, 255), random.randint(0, 255), random.randint(200, 255)],
        dtype=np.float32,
    )
    region = out[ry1:ry2, rx1:rx2]
    out[ry1:ry2, rx1:rx2] = region * (1 - tint_alpha) + tint_color * tint_alpha

    return np.clip(out, 0, 255).astype(np.uint8)


DEGRADERS = {
    "blur": apply_blur,
    "underexposure": apply_underexposure,
    "overexposure": apply_overexposure,
    "noise": apply_noise,
    "corruption": apply_corruption,
    "visual_defect": apply_visual_defect,
}


def make_labeled_sample(bgr: np.ndarray, issue_type: str = None, severity: str = None):
    """
    Produce one (degraded_image, label) pair.
    If issue_type is None, pick randomly (including 'clean' = no degradation).
    Returns (image, issue_type, severity, quality_label, quality_score)
    """
    if issue_type is None:
        issue_type = random.choice(ISSUE_TYPES)
    if severity is None and issue_type != "clean":
        severity = random.choice(["low", "medium", "high"])

    if issue_type == "clean":
        out = bgr.copy()
        severity = "none"
    else:
        out = DEGRADERS[issue_type](bgr, severity)

    # Map (issue_type, severity) -> a rough quality score / label for training targets.
    if issue_type == "clean":
        score, label = random.randint(88, 100), "ACCEPTABLE"
    else:
        severity_penalty = {"low": 20, "medium": 45, "high": 70}[severity]
        score = max(0, 95 - severity_penalty + random.randint(-5, 5))
        if score >= 70:
            label = "ACCEPTABLE"
        elif score >= 40:
            label = "DEGRADED"
        else:
            label = "DEFECTIVE"

    return out, issue_type, severity, label, score
