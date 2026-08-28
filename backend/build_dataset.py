"""
Build a labeled CSV dataset of engineered features from clean source images
by applying synthetic degradations.

Usage:
    python build_dataset.py --src ../data/raw --out ../data/synthetic/dataset.csv --n_per_image 6

For each clean source image we generate N degraded variants (random issue
type + severity, including some 'clean' variants) and compute engineered
features + labels for each. This CSV is what trains the classical/hybrid
model, and is also useful for exploratory analysis / evaluation plots.
"""

import argparse
import csv
import glob
import os
import sys

sys.path.append(os.path.dirname(__file__))

from features import extract_features, FEATURE_ORDER, load_image_bgr
from degrade import make_labeled_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Directory of clean source images")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--n_per_image", type=int, default=6)
    args = ap.parse_args()

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(glob.glob(os.path.join(args.src, "**", ext), recursive=True))

    if not paths:
        print(f"No images found under {args.src}")
        return

    print(f"Found {len(paths)} source images. Generating {args.n_per_image} variants each...")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = ["source_path", "issue_type", "severity", "quality_label", "quality_score"] + FEATURE_ORDER

    n_written = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, p in enumerate(paths):
            bgr = load_image_bgr(p)
            if bgr is None:
                continue
            # keep things fast: downsize very large images before degrading/featurizing
            h, w = bgr.shape[:2]
            max_dim = 640
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                import cv2

                bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

            for _ in range(args.n_per_image):
                degraded, issue_type, severity, label, score = make_labeled_sample(bgr)
                feats = extract_features(degraded)
                row = {
                    "source_path": p,
                    "issue_type": issue_type,
                    "severity": severity,
                    "quality_label": label,
                    "quality_score": score,
                }
                row.update({k: feats[k] for k in FEATURE_ORDER})
                writer.writerow(row)
                n_written += 1

            if (i + 1) % 50 == 0:
                print(f"  processed {i+1}/{len(paths)} source images -> {n_written} samples so far")

    print(f"Done. Wrote {n_written} labeled samples to {args.out}")


if __name__ == "__main__":
    main()
