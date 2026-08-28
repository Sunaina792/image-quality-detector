"""
Generate sample images for the data/samples/ directory.

Creates one representative image for each quality condition so reviewers
can test the app without needing to source their own images.

Usage:
    python generate_samples.py --out ../data/samples
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(__file__))
from degrade import (
    apply_blur,
    apply_corruption,
    apply_noise,
    apply_overexposure,
    apply_underexposure,
    apply_visual_defect,
)


def make_clean_image(h: int = 480, w: int = 640) -> np.ndarray:
    """
    Create a synthetic 'clean' source image with enough texture and
    variation that feature extraction produces realistic values.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky gradient (top 40%)
    for y in range(int(h * 0.4)):
        t = y / (h * 0.4)
        img[y, :] = [
            int(220 - 60 * t),
            int(180 - 40 * t),
            int(80 + 60 * t),
        ]

    # Ground (bottom 60%)
    for y in range(int(h * 0.4), h):
        t = (y - h * 0.4) / (h * 0.6)
        img[y, :] = [
            int(40 + 20 * t),
            int(100 - 30 * t),
            int(60 - 20 * t),
        ]

    # Building rectangle for sharp edges
    cv2.rectangle(img, (w // 4, int(h * 0.15)), (w * 3 // 4, int(h * 0.6)), (180, 160, 140), -1)
    cv2.rectangle(img, (w // 4, int(h * 0.15)), (w * 3 // 4, int(h * 0.6)), (100, 90, 80), 3)

    # Windows
    for row in range(3):
        for col in range(4):
            wx = w // 4 + 30 + col * 70
            wy = int(h * 0.2) + row * 70
            cv2.rectangle(img, (wx, wy), (wx + 40, wy + 50), (80, 120, 160), -1)
            cv2.rectangle(img, (wx, wy), (wx + 40, wy + 50), (60, 90, 120), 2)

    # Texture layer
    noise_layer = np.random.randint(0, 60, (h, w, 3), dtype=np.uint8)
    noise_layer = cv2.GaussianBlur(noise_layer, (21, 21), 0)
    img = cv2.add(img, noise_layer)

    return img


def save_sample(img: np.ndarray, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    cv2.imwrite(path, img)
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/samples", help="Output directory")
    ap.add_argument("--size", default="480x640", help="HxW of generated images")
    args = ap.parse_args()

    h, w = [int(x) for x in args.size.split("x")]
    os.makedirs(args.out, exist_ok=True)

    print("Generating clean base image...")
    clean = make_clean_image(h, w)

    samples = [
        ("clean.jpg", clean),
        ("blur_low.jpg", apply_blur(clean, "low")),
        ("blur_high.jpg", apply_blur(clean, "high")),
        ("underexposure.jpg", apply_underexposure(clean, "high")),
        ("overexposure.jpg", apply_overexposure(clean, "high")),
        ("noise.jpg", apply_noise(clean, "high")),
        ("corruption.jpg", apply_corruption(clean, "medium")),
        ("visual_defect.jpg", apply_visual_defect(clean, "medium")),
    ]

    print(f"Saving {len(samples)} sample images to {args.out}/")
    for name, img in samples:
        save_sample(img, args.out, name)

    print(f"\nDone. {len(samples)} samples saved.")


if __name__ == "__main__":
    main()
