"""
Download a small set of public-domain clean images into data/raw/
so the full training pipeline can be run from scratch.

Images are sourced from the Wikimedia Commons API (public domain / CC0).
No authentication required.

Usage:
    python download_raw_images.py --out ../data/raw --n 40
"""

import argparse
import os
import sys
import urllib.request
import urllib.error

# Public-domain images from Lorem Picsum (picsum.photos) — a reliable CDN
# specifically designed for placeholder/test images. No auth required.
# Each URL returns a different photo at 640x480.
SEED_URLS = [
    f"https://picsum.photos/seed/{seed}/640/480.jpg"
    for seed in [
        "arch", "nature", "city", "forest", "ocean", "mountain", "street",
        "building", "animal", "flower", "sunset", "bridge", "train", "food",
        "sky", "river", "beach", "park", "abstract", "texture", "macro",
        "portrait", "landscape", "urban", "winter", "autumn", "spring", "summer",
        "night", "day",
    ]
]


def download_images(out_dir: str, n: int = 30) -> None:
    os.makedirs(out_dir, exist_ok=True)
    urls = SEED_URLS[:n]

    print(f"Downloading {len(urls)} images to {out_dir}/")
    success = 0
    for i, url in enumerate(urls):
        ext = ".jpg" if ".jpg" in url.lower() else ".png"
        dest = os.path.join(out_dir, f"raw_{i:03d}{ext}")
        if os.path.exists(dest):
            print(f"  [{i+1}/{len(urls)}] Skipped (already exists): {dest}")
            success += 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            with open(dest, "wb") as f:
                f.write(data)
            print(f"  [{i+1}/{len(urls)}] Downloaded: {dest}")
            success += 1
        except Exception as e:
            print(f"  [{i+1}/{len(urls)}] FAILED ({url}): {e}")

    print(f"\nDone. {success}/{len(urls)} images in {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description="Download clean source images for training.")
    ap.add_argument("--out", default="../data/raw", help="Output directory (default: ../data/raw)")
    ap.add_argument("--n", type=int, default=30, help="Number of images to download (max 30)")
    args = ap.parse_args()
    download_images(args.out, min(args.n, len(SEED_URLS)))


if __name__ == "__main__":
    main()
