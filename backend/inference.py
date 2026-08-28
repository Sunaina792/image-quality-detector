"""
Inference wrapper: image bytes -> structured analysis JSON.

Combines:
  - hard corruption check (can we even decode it / is it absurdly uniform)
  - engineered features
  - trained RF models (quality label, quality score, issue type)
into the final response shape specified in the assessment (section 7).
"""

import os

import joblib
import numpy as np

from features import extract_features, features_to_vector, load_image_bgr

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_label_clf = None
_score_reg = None
_issue_clf = None


def _lazy_load():
    global _label_clf, _score_reg, _issue_clf
    if _label_clf is None:
        _label_clf = joblib.load(os.path.join(MODEL_DIR, "quality_label_clf.joblib"))
        _score_reg = joblib.load(os.path.join(MODEL_DIR, "quality_score_reg.joblib"))
        _issue_clf = joblib.load(os.path.join(MODEL_DIR, "issue_type_clf.joblib"))
    return _label_clf, _score_reg, _issue_clf


def _severity_from_features(issue_type: str, feats: dict, confidence: float) -> str:
    """
    Determine severity based on the actual feature magnitude for each issue type,
    rather than using model confidence alone. Falls back to confidence-based bucketing
    for issue types without a dominant scalar feature.
    """
    if issue_type == "blur":
        s = feats.get("sharpness", 1000)
        if s < 50:
            return "high"
        if s < 150:
            return "medium"
        return "low"

    if issue_type == "underexposure":
        b = feats.get("mean_brightness", 128)
        if b < 40:
            return "high"
        if b < 80:
            return "medium"
        return "low"

    if issue_type == "overexposure":
        bf = feats.get("bright_pixel_fraction", 0)
        if bf > 0.5:
            return "high"
        if bf > 0.2:
            return "medium"
        return "low"

    if issue_type == "noise":
        n = feats.get("noise", 0)
        if n > 20:
            return "high"
        if n > 10:
            return "medium"
        return "low"

    if issue_type == "corruption":
        bu = feats.get("block_uniformity", 0)
        if bu > 0.3:
            return "high"
        if bu > 0.1:
            return "medium"
        return "low"

    # visual_defect and any future types: fall back to confidence-based
    if confidence > 0.75:
        return "high"
    if confidence > 0.5:
        return "medium"
    return "low"


def analyze_image(image_bytes: bytes) -> dict:
    """
    Returns a dict matching the assessment's expected response shape:
    {
      "quality_score": int,
      "quality_label": str,
      "issues": [{"type": str, "severity": str, "confidence": float}],
      ...extra explainability fields
    }
    Raises ValueError on unreadable/invalid images (caller maps this to HTTP 400).
    """
    bgr = load_image_bgr(image_bytes)
    if bgr is None or bgr.size == 0:
        raise ValueError("Image could not be decoded — file is invalid or unreadable.")

    h, w = bgr.shape[:2]
    if h < 10 or w < 10:
        raise ValueError("Image dimensions too small to analyze.")

    feats = extract_features(bgr)
    x = features_to_vector(feats).reshape(1, -1)

    label_clf, score_reg, issue_clf = _lazy_load()

    quality_label = label_clf.predict(x)[0]
    label_proba = dict(zip(label_clf.classes_, label_clf.predict_proba(x)[0].tolist()))
    quality_score = float(np.clip(score_reg.predict(x)[0], 0, 100))

    issue_pred = issue_clf.predict(x)[0]
    issue_proba = dict(zip(issue_clf.classes_, issue_clf.predict_proba(x)[0].tolist()))

    issues = []
    if issue_pred != "clean":
        confidence = issue_proba.get(issue_pred, 0.0)
        severity = _severity_from_features(issue_pred, feats, confidence)
        issues.append({"type": issue_pred, "severity": severity, "confidence": round(confidence, 3)})

    # Also surface any secondary issue type with meaningful probability
    # (helps catch images with more than one problem, e.g. dark AND noisy)
    for issue_type, prob in sorted(issue_proba.items(), key=lambda kv: -kv[1]):
        if issue_type in ("clean", issue_pred):
            continue
        if prob > 0.25:
            sev = _severity_from_features(issue_type, feats, prob)
            issues.append({"type": issue_type, "severity": sev, "confidence": round(prob, 3)})

    result = {
        "quality_score": int(round(quality_score)),
        "quality_label": quality_label,
        "issues": issues,
        "confidence": round(max(label_proba.values()), 3),
        "label_probabilities": {k: round(v, 3) for k, v in label_proba.items()},
        "image_stats": {
            "width": w,
            "height": h,
            "sharpness": round(feats["sharpness"], 2),
            "mean_brightness": round(feats["mean_brightness"], 2),
            "contrast": round(feats["contrast"], 2),
            "noise": round(feats["noise"], 2),
            "edge_density": round(feats["edge_density"], 4),
            "block_uniformity": round(feats["block_uniformity"], 4),
        },
    }
    return result
