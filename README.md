# PixelGuard — AI-Powered Image Quality & Defect Detection

Full-stack application that accepts an image, analyzes it for quality issues
(blur, under/overexposure, noise, corruption, visual defects), and returns a
structured JSON result — with a polished web UI and a Gradio fallback UI.

No external AI/vision APIs are used. All inference runs on locally trained
scikit-learn models using engineered OpenCV features.

---

## Approach

**Hybrid: engineered CV features + Random Forest ensemble.**

For each image, 10 interpretable features are computed:

| Feature | Signal |
|---|---|
| `sharpness` | Laplacian variance — low = blurry |
| `mean_brightness` | Mean pixel value — low = underexposed |
| `dark_pixel_fraction` | Fraction of near-black pixels |
| `bright_pixel_fraction` | Fraction of near-white pixels |
| `contrast` | Std dev of pixel intensities |
| `noise` | Median-blur residual std — high = noisy |
| `mean_saturation` | HSV saturation — low = washed out |
| `std_saturation` | Saturation spread |
| `edge_density` | Canny edge fraction |
| `block_uniformity` | Fraction of identical-pixel blocks (corruption heuristic) |

These 10 features feed **three separate Random Forest models**:

1. `quality_label_clf` → ACCEPTABLE / DEGRADED / DEFECTIVE
2. `quality_score_reg` → continuous 0–100 score (MAE ~ 5 pts on held-out set)
3. `issue_type_clf` → blur / underexposure / overexposure / noise / corruption / visual_defect / clean

Random Forest was chosen because:
- Input is a 10-dim feature vector (not raw pixels) — RF is a perfect fit for low-dimensional non-linear classification.
- `feature_importances_` gives free, per-prediction explainability (required by the assessment).
- Training is fast (~10 s on a laptop) so the full pipeline is easily reproducible.
- Class-weighted training handles the imbalanced clean/degraded split correctly.

See `GET /metrics` for live accuracy, F1, confusion matrix, and feature importances.

---

## Detection Capabilities

| Issue Type | Detected By |
|---|---|
| Blur / insufficient sharpness | Laplacian variance + edge density |
| Underexposure | Mean brightness + dark pixel fraction |
| Overexposure | Bright pixel fraction + contrast |
| Image noise | Median-blur residual (noise_score) |
| Image corruption / severe degradation | Block uniformity heuristic |
| Visual defect (scratches, lens flare, watermarks) | Trained RF on visual_defect class |

---

## Project Structure

```
backend/
  features.py              - CV feature extraction (10 interpretable features)
  degrade.py               - Synthetic degradation generator (7 issue types)
  build_dataset.py         - Builds labeled CSV from clean source images
  download_raw_images.py   - Downloads ~30 public-domain clean images for training
  generate_samples.py      - Generates sample/demo images for data/samples/
  train_model.py           - Trains RF models + writes models/metrics.json
  inference.py             - Loads models, runs analysis pipeline on new image
  db.py                    - SQLite persistence for analysis history
  app.py                   - FastAPI REST API + custom frontend + Gradio fallback
  requirements.txt
frontend/
  index.html               - Main web UI (dark glassmorphism design)
  style.css                - Design system CSS
  app.js                   - Vanilla JS: upload, results, history, modal
models/
  quality_label_clf.joblib - Trained label classifier
  quality_score_reg.joblib - Trained score regressor
  issue_type_clf.joblib    - Trained issue-type classifier
  metrics.json             - Accuracy, F1, confusion matrix, feature importances
data/
  raw/                     - Clean source images for training (populated by download_raw_images.py)
  synthetic/               - Auto-generated labeled CSV dataset
  samples/                 - Demo images covering all quality conditions
  analyses.db              - SQLite database (auto-created at runtime)
tests/
  test_features.py         - Unit tests for feature extraction functions
  test_inference.py        - Unit tests for analyze_image() and severity logic
  test_api.py              - Integration tests for all API endpoints
notebooks/
  IIIT_H (1).ipynb         - Exploratory analysis notebook
  evaluate_generalization.md - Evaluation results, failure cases, limitations
Dockerfile
docker-compose.yml
```

---

## Setup (Local)

> **Requires Python 3.11.** The pinned OpenCV and NumPy wheels do not support Python 3.14.

```bash
# Create and activate venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r backend\requirements.txt
```

### Option A — Use Pre-Trained Models (fastest, recommended)

The `.joblib` model files are already committed. Skip steps 1–3 and go straight to step 4.

### Option B — Retrain From Scratch

```bash
cd backend

# 1. Download clean source images
python download_raw_images.py --out ../data/raw --n 30

# 2. Generate synthetic labeled dataset
python build_dataset.py --src ../data/raw --out ../data/synthetic/dataset.csv --n_per_image 8

# 3. Train models (produces models/*.joblib + models/metrics.json)
python train_model.py --csv ../data/synthetic/dataset.csv --out ../models

# 4. Generate demo sample images
python generate_samples.py --out ../data/samples
```

### Run the App

```bash
cd backend
python app.py
# -> http://localhost:7860  (custom frontend)
# -> http://localhost:7860/gradio  (Gradio fallback UI)
# -> http://localhost:7860/docs    (FastAPI auto-docs)
```

---

## Docker / Docker Compose

```bash
docker compose up --build
# -> http://localhost:7860
```

The Dockerfile uses `python:3.11-slim`, installs system deps for OpenCV (libgl1),
and bundles `backend/`, `models/`, and `frontend/` into a single image.
No data volume is needed for inference — the SQLite DB is created at runtime inside the container.

---

## Database Setup

SQLite is used for analysis history. **No manual setup is required** — the database
is auto-created at `data/analyses.db` when the app starts (`db.init_db()` is called
on startup in `app.py`).

To reset the history, simply delete `data/analyses.db`. It will be recreated empty on next startup.

To inspect the database directly:
```bash
sqlite3 data/analyses.db "SELECT id, filename, quality_label, quality_score, created_at FROM analyses ORDER BY id DESC LIMIT 10;"
```

---

## API Reference

### `GET /health`
Health check.
```json
{"status": "ok"}
```

### `POST /analyze`
Multipart file upload (`file` field). Accepts any image format (JPEG, PNG, WEBP, BMP, etc).

Returns:
```json
{
  "quality_score": 82,
  "quality_label": "ACCEPTABLE",
  "issues": [{"type": "noise", "severity": "low", "confidence": 0.71}],
  "confidence": 0.88,
  "label_probabilities": {"ACCEPTABLE": 0.88, "DEGRADED": 0.1, "DEFECTIVE": 0.02},
  "image_stats": {"width": 640, "height": 480, "sharpness": 312.4, "mean_brightness": 142.1, "contrast": 58.3, "noise": 3.2, "edge_density": 0.0821, "block_uniformity": 0.0},
  "id": 17
}
```

Error responses:
- `400` — file is not an image or is empty
- `422` — image is unreadable / corrupt bytes
- `500` — internal analysis failure

### `POST /analyze/batch`
Upload multiple images at once. Returns a list of results in the same order.
Files that fail include an `error` field instead.

```bash
curl -X POST \
  -F "files=@clean.jpg" -F "files=@blurry.jpg" \
  http://localhost:7860/analyze/batch
```

### `GET /history?limit=50`
Returns recent analyses (id, filename, label, score, timestamp).

### `GET /history/{id}`
Returns full stored result for one analysis, including the complete result JSON.

### `GET /metrics`
Returns `models/metrics.json`: accuracy, macro-F1, confusion matrix, feature importances,
and per-class precision/recall/F1 for all three models.

### `GET /health`
Returns `{"status": "ok"}`. Used as container health check.

---

### Example API Calls

```bash
# Analyze an image
curl -X POST -F "file=@data/samples/blur_high.jpg" http://localhost:7860/analyze

# Batch analysis
curl -X POST \
  -F "files=@data/samples/clean.jpg" \
  -F "files=@data/samples/noise.jpg" \
  http://localhost:7860/analyze/batch

# Get model metrics / feature importances
curl http://localhost:7860/metrics

# Get history
curl http://localhost:7860/history?limit=10
```

---

## Evaluation

Run `train_model.py` to regenerate `models/metrics.json`, which includes:
- Accuracy and macro-F1 for quality label classifier and issue type classifier
- MAE for the quality score regressor
- Confusion matrices and per-class precision/recall/F1

See [`notebooks/evaluate_generalization.md`](notebooks/evaluate_generalization.md) for:
- Full evaluation results table
- Feature importance rankings
- Known failure cases and limitations
- Generalization discussion (synthetic vs. real images)

---

## Explainability

Every `/analyze` response includes:
- **`image_stats`** — the raw feature values that drove the decision (sharpness, brightness, noise, etc.)
- **`confidence`** — the Random Forest's max class probability
- **`label_probabilities`** — full probability distribution over ACCEPTABLE/DEGRADED/DEFECTIVE
- **`issues[].severity`** — determined by feature magnitude (e.g. sharpness < 50 = "high" blur), not just confidence

`GET /metrics` exposes `feature_importances` showing which signals matter most for each model.

---

## Running Tests

```bash
cd backend
pytest ../tests/ -v
```

Tests cover:
- `test_features.py` — unit tests for all 10 feature functions
- `test_inference.py` — unit tests for `analyze_image()` and severity logic
- `test_api.py` — integration tests for all API endpoints

---

## Deployment (Hugging Face Spaces)

This app is a plain Docker-SDK Space:

1. Create a new Space → SDK = **Docker**.
2. Push this repo (including `models/*.joblib`, `models/metrics.json`, and `frontend/`)
   to the Space's git remote.
3. HF Spaces builds the `Dockerfile` and exposes port 7860 automatically.
4. The same URL serves the custom frontend (`/`), REST API (`/analyze`, `/history`, `/metrics`),
   and Gradio UI (`/gradio`).

Deployed URL: _add after deploying_

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `7860` | Port the app listens on (required by HF Spaces) |
#   i m a g e - q u a l i t y - d e t e c t o r  
 