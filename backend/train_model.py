"""
Train the hybrid quality classifier.

Approach: engineered CV features (sharpness, brightness, noise, etc.) ->
Random Forest classifier for quality_label (ACCEPTABLE / DEGRADED / DEFECTIVE)
and a separate regressor for the continuous quality_score, plus one
multi-label-ish classifier per issue_type (blur / exposure / noise / corruption)
so we can report *which* issues were detected, not just an overall bucket.

Why Random Forest over a raw CNN here: we only have ~10 engineered features
(not raw pixels), the relationship between them and quality is non-linear but
low-dimensional, and RF gives us built-in feature_importances_ for free --
which directly satisfies the "explainability" requirement (section 10)
without needing Grad-CAM or similar.

Usage:
    python train_model.py --csv ../data/synthetic/dataset.csv --out ../models
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="Output directory for model artifacts")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.out, exist_ok=True)

    X = df[FEATURE_ORDER].values
    y_label = df["quality_label"].values
    y_score = df["quality_score"].values
    y_issue = df["issue_type"].values

    X_train, X_test, yl_train, yl_test, ys_train, ys_test, yi_train, yi_test = train_test_split(
        X, y_label, y_score, y_issue, test_size=0.2, random_state=42, stratify=y_label
    )

    # 1) Quality label classifier (ACCEPTABLE / DEGRADED / DEFECTIVE)
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, class_weight="balanced")
    clf.fit(X_train, yl_train)
    yl_pred = clf.predict(X_test)

    acc = accuracy_score(yl_test, yl_pred)
    f1 = f1_score(yl_test, yl_pred, average="macro")
    report = classification_report(yl_test, yl_pred, output_dict=True)
    cm = confusion_matrix(yl_test, yl_pred, labels=clf.classes_).tolist()

    print("=== Quality label classifier ===")
    print(f"Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
    print(classification_report(yl_test, yl_pred))

    # 2) Quality score regressor (0-100 continuous score)
    reg = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    reg.fit(X_train, ys_train)
    ys_pred = reg.predict(X_test)
    mae = mean_absolute_error(ys_test, ys_pred)
    print(f"\n=== Quality score regressor ===\nMAE: {mae:.2f}")

    # 3) Issue-type classifier (blur / underexposure / overexposure / noise / corruption / clean)
    issue_clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, class_weight="balanced")
    issue_clf.fit(X_train, yi_train)
    yi_pred = issue_clf.predict(X_test)
    issue_acc = accuracy_score(yi_test, yi_pred)
    issue_f1 = f1_score(yi_test, yi_pred, average="macro")
    issue_report = classification_report(yi_test, yi_pred, output_dict=True)
    issue_cm = confusion_matrix(yi_test, yi_pred, labels=issue_clf.classes_).tolist()

    print("\n=== Issue type classifier ===")
    print(f"Accuracy: {issue_acc:.4f}  Macro-F1: {issue_f1:.4f}")
    print(classification_report(yi_test, yi_pred))

    # Save models
    joblib.dump(clf, os.path.join(args.out, "quality_label_clf.joblib"))
    joblib.dump(reg, os.path.join(args.out, "quality_score_reg.joblib"))
    joblib.dump(issue_clf, os.path.join(args.out, "issue_type_clf.joblib"))

    # Save feature importances for explainability endpoint
    importances = dict(zip(FEATURE_ORDER, clf.feature_importances_.tolist()))

    metrics = {
        "quality_label": {
            "accuracy": acc,
            "macro_f1": f1,
            "confusion_matrix": cm,
            "classes": clf.classes_.tolist(),
            "classification_report": report,
        },
        "quality_score_mae": mae,
        "issue_type": {
            "accuracy": issue_acc,
            "macro_f1": issue_f1,
            "confusion_matrix": issue_cm,
            "classes": issue_clf.classes_.tolist(),
            "classification_report": issue_report,
        },
        "feature_importances": importances,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_order": FEATURE_ORDER,
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved models + metrics.json to {args.out}")


if __name__ == "__main__":
    main()
