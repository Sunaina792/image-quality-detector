"""
Image quality feature extraction.

Every function here computes ONE interpretable signal about the image
(sharpness, brightness, noise, etc). We keep them separate and interpretable
on purpose -- this is what lets us explain *why* an image was flagged
(see explainability requirement in the assessment), not just spit out a score.
"""

import cv2
import numpy as np


def load_image_bgr(path_or_bytes):
    """Load image as BGR numpy array. Accepts a file path or raw bytes."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        if not path_or_bytes:
            return None  # Empty buffer — caller maps this to ValueError
        arr = np.frombuffer(path_or_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)
    return img  # None if decode failed -> caller must handle "corrupted"


def sharpness_score(gray: np.ndarray) -> float:
    """
    Variance of the Laplacian. Intuition: sharp edges create large second
    derivatives; blur smooths them out. Low variance -> blurry image.
    Typical thresholds (empirical, tune on your data): <100 = very blurry,
    100-300 = soft/mild blur, >300 = sharp.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_stats(gray: np.ndarray) -> dict:
    """
    Mean brightness (0-255) and the fraction of pixels that are near-black
    or near-white (clipped). A photo taken in a dark room clips toward 0;
    a photo taken facing the sun clips toward 255.
    """
    mean_brightness = float(gray.mean())
    total = gray.size
    dark_frac = float(np.sum(gray < 15) / total)
    bright_frac = float(np.sum(gray > 240) / total)
    return {
        "mean_brightness": mean_brightness,
        "dark_pixel_fraction": dark_frac,
        "bright_pixel_fraction": bright_frac,
    }


def contrast_score(gray: np.ndarray) -> float:
    """Standard deviation of pixel intensities. Low = flat/washed-out image."""
    return float(gray.std())


def noise_score(gray: np.ndarray) -> float:
    """
    Estimate noise using the difference between the image and a
    median-blurred version of itself. Median blur removes noise but
    keeps edges, so what's left over (residual) is mostly noise.
    High residual std -> noisy image (like grain in a low-light photo).
    """
    denoised = cv2.medianBlur(gray, 5)
    residual = gray.astype(np.float32) - denoised.astype(np.float32)
    return float(residual.std())


def saturation_stats(bgr: np.ndarray) -> dict:
    """Mean and std of saturation channel in HSV. Very low mean -> washed out."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    return {"mean_saturation": float(sat.mean()), "std_saturation": float(sat.std())}


def edge_density(gray: np.ndarray) -> float:
    """
    Fraction of pixels detected as edges by Canny. Very low edge density
    combined with low sharpness is a strong blur signal (a sharp image of
    a truly flat wall is rare -- most real photos have texture somewhere).
    """
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges) / edges.size)


def block_uniformity_score(gray: np.ndarray, block_size: int = 16) -> float:
    """
    Corruption/severe-degradation heuristic: split the image into blocks
    and count how many blocks are near-perfectly uniform (std ~ 0).
    Legitimate photos rarely have large flat regions of identical pixel
    values -- that pattern shows up in truncated/corrupted JPEGs where
    entire blocks fail to decode and get filled with a placeholder color.
    """
    h, w = gray.shape
    if h < block_size or w < block_size:
        return 0.0
    n_blocks = 0
    n_uniform = 0
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y : y + block_size, x : x + block_size]
            n_blocks += 1
            if block.std() < 1.0:
                n_uniform += 1
    return float(n_uniform / n_blocks) if n_blocks else 0.0


def extract_features(bgr: np.ndarray) -> dict:
    """
    Run the full feature pipeline on a loaded BGR image and return a flat
    dict. This dict is what gets: (a) fed to the ML model, and (b) shown
    to the user for explainability.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    feats = {}
    feats["sharpness"] = sharpness_score(gray)
    feats.update(brightness_stats(gray))
    feats["contrast"] = contrast_score(gray)
    feats["noise"] = noise_score(gray)
    feats.update(saturation_stats(bgr))
    feats["edge_density"] = edge_density(gray)
    feats["block_uniformity"] = block_uniformity_score(gray)
    feats["height"], feats["width"] = gray.shape

    return feats


FEATURE_ORDER = [
    "sharpness",
    "mean_brightness",
    "dark_pixel_fraction",
    "bright_pixel_fraction",
    "contrast",
    "noise",
    "mean_saturation",
    "std_saturation",
    "edge_density",
    "block_uniformity",
]


def features_to_vector(feats: dict) -> np.ndarray:
    """Convert the feature dict to a fixed-order numpy vector for the model."""
    return np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float32)
