# PixelGuard — AI-Powered Image Quality & Defect Detection

Full-stack application that accepts an image, analyzes it for quality issues
(blur, under/overexposure, noise, corruption, visual defects), and returns a
structured JSON result — with a polished React UI and a Gradio fallback UI.

No external AI/vision APIs are used. All inference runs on locally trained
scikit-learn models using engineered OpenCV features (satisfies assessment
requirement: "External AI services: Not permitted").

---

## 1. Problem Statement (assessment §1)

Given an uploaded image, the system evaluates visual quality and classifies
it as **ACCEPTABLE**, **DEGRADED**, or **DEFECTIVE**, while identifying the
specific issue(s) present.

## 2. Detected Issue Types (assessment §2)

| Required issue | Implemented as |
|---|---|
| Blur / insufficient sharpness | `blur` — Laplacian variance |
| Underexposure | `underexposure` — brightness histogram |
| Overexposure | `overexposure` — brightness histogram |
| Image noise | `noise` — median-blur residual |
| Image corruption / severe degradation | `corruption` — block-uniformity + neighbor-jump heuristic |
| Potential visual defect | Surfaced in the UI as **"Visual Defect"** — the `corruption` model class doubles as the structural-anomaly / defect signal, since block-level discontinuities are the common underlying pattern for both corrupted files and structural image defects |

### Screenshots

**Blur detection**
![Blur detection](blur-demo.png)

**Visual defect detection**
![Visual defect detection](clear-demo.png)

**Analysis history**
![Analysis history](history.png)

## 3. AI / Computer Vision Approach (assessment §3)

**Hybrid: engineered CV features + Random Forest ensemble** — explicitly one
of the assessment's listed acceptable approaches ("a hybrid approach
combining image-quality features with a learned model").

10 interpretable features are computed per image using OpenCV:

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
| `block_uniformity` | Fraction of blocks that are internally uniform AND abruptly different from neighboring blocks (corruption/defect heuristic) |

These features feed **three separate Random Forest models**:

1. `quality_label_clf` → ACCEPTABLE / DEGRADED / DEFECTIVE
2. `quality_score_reg` → continuous 0–100 score
3. `issue_type_clf` → blur / underexposure / overexposure / noise / corruption / clean

**Why Random Forest over a raw CNN:** the input is a 10-dimensional
engineered feature vector rather than raw pixels; the feature→quality
relationship is non-linear but low-dimensional, which is a strong fit for
tree ensembles; training is fast enough to iterate within the assessment
window; and `feature_importances_` gives per-prediction explainability for
free (see §10 below), without needing Grad-CAM or a saliency-map pipeline.

## 4. Image Analysis (assessment §4)

Sharpness, brightness/exposure, contrast, noise, saturation, and edge
density are all explicitly computed and returned in every API response
under `image_stats`, so the reasoning behind each decision is visible, not
just a final label.

## 5. Backend (assessment §5)

FastAPI REST API (`backend/app.py`):

- `POST /analyze` — multipart image upload, returns structured JSON
- `GET /history?limit=50` — retrieve previous analyses
- `GET /history/{id}` — retrieve one specific analysis
- `GET /health` — service health check
- Invalid/non-image files → HTTP 400
- Unreadable/corrupted files → HTTP 422 with a clear error message
- Unexpected errors → HTTP 500 with detail
- Results persisted in **SQLite** (`backend/db.py`)

## 6. Frontend (assessment §6)

Two frontends are included:

1. **PixelGuard** (primary) — a custom React UI (`frontend/`) built with
   Antigravity, live-connected to the FastAPI backend. Shows the uploaded
   image, a circular quality-score gauge, the quality label, detected
   issues with severity/confidence, per-class label probabilities, and a
   full image-statistics breakdown. Includes an **Analyze** view and a
   **History** view. Handles loading, success, and error states.
2. **Gradio UI** (fallback, mounted at the backend's `/` route) — simpler,
   zero-build alternative for quick local testing; also supports upload,
   results display, and history.

## 7. API Response Format (assessment §7)

```json
{
  "quality_score": 82,
  "quality_label": "ACCEPTABLE",
  "issues": [{"type": "noise", "severity": "low", "confidence": 0.71}],
  "confidence": 0.88,
  "label_probabilities": {"ACCEPTABLE": 0.88, "DEGRADED": 0.1, "DEFECTIVE": 0.02},
  "image_stats": {"width": 640, "height": 480, "sharpness": 312.4, "...": "..."},
  "id": 17
}
```

## 8. Dataset and Training (assessment §8)

**Synthetic degradation** was used, as explicitly permitted by the
assessment: "Applicants may... generate controlled image-quality
degradations from clean images. If synthetic degradation is used, describe
how training and evaluation data were generated."

- **Base clean images**: a sample from the [Intel Image Classification
  dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
  (natural scenes: buildings, forest, glacier, mountain, sea, street).
- **Degradation pipeline** (`backend/degrade.py`): Gaussian blur,
  brightness scaling (under/overexposure), Gaussian noise, and low-quality
  JPEG re-encoding + blacked-out blocks (corruption), each at 3 severity
  levels (low/medium/high), plus untouched "clean" samples. Ground-truth
  labels are known exactly because we control the degradation applied.
- **Dataset build**: `backend/build_dataset.py` turns a folder of clean
  images into a labeled feature CSV — 400 source images × 6 variants =
  2,400 labeled samples used for the current trained models.
- **Generalization check ("unseen images", per assessment §8/§9)**: the
  trained model was additionally evaluated on the [Kaggle Blur
  Dataset](https://www.kaggle.com/datasets/kwentar/blur-dataset) (1,476
  real, non-synthetic photos across sharp / defocused-blurred /
  motion-blurred categories) — see `notebooks/evaluate_generalization.md`
  for the full results.

## 9. Evaluation (assessment §9)

Full metrics (accuracy, macro-F1, confusion matrix, per-class
precision/recall, feature importances) are in `models/metrics.json`,
generated by `backend/train_model.py`.

**Held-out synthetic test set (20% split, 480 samples):**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Quality label (ACCEPTABLE/DEGRADED/DEFECTIVE) | 88.5% | 0.88 |
| Issue type (blur/exposure/noise/corruption/clean) | 88.75% | 0.89 |
| Quality score regressor | MAE ≈ 8.4 pts | — |

**Real-world generalization (Kaggle Blur Dataset, 1,476 unseen images):**
see `notebooks/evaluate_generalization.md` for full breakdown, failure
cases, and discussion.

**Known failure cases / limitations** (documented per assessment §9):

- The model is trained on natural-scene photography (Intel Image
  Classification: buildings, forest, glacier, mountain, sea, street) and
  shows reduced reliability on **out-of-domain content** — e.g. text/
  document images or close-up portraits with intentional bokeh — where
  brightness- and sharpness-based features don't map cleanly onto the
  training distribution.
- Because `quality_label_clf` and `issue_type_clf` are trained
  independently (not jointly), their outputs can occasionally disagree in
  borderline cases (e.g. an overall "ACCEPTABLE" label alongside a weak
  secondary issue signal). Inference applies a confidence-reconciliation
  threshold (issue reported only if the overall label is non-ACCEPTABLE,
  or the issue-type confidence exceeds 0.45) to reduce noisy flags — see
  `backend/inference.py`.
- `block_uniformity` (the corruption/defect heuristic) was refined from a
  simple per-block variance check to a neighbor-comparison check, since
  the naive version flagged natural flat regions (sky, still water) as
  corrupted. The current version still occasionally under- or
  over-triggers on scenes with very large uniform regions.

## 10. Explainability (assessment §10)

Every `/analyze` response includes:

- Raw `image_stats` (sharpness, brightness, noise, etc.) — the actual
  signals the decision was based on, not just a black-box score.
- Per-class `label_probabilities` and per-issue `confidence` from the
  Random Forest's probability outputs.
- `models/metrics.json` includes `feature_importances_` per model,
  showing which engineered features matter most for each prediction type.

## 11. Deployment (assessment §11)

- `Dockerfile` + `docker-compose.yml` provided; the backend listens on
  port `7860` (configurable via the `PORT` env var).
- `GET /health` — service/status check endpoint.
- Model loading: the three `.joblib` files (`models/quality_label_clf.joblib`,
  `quality_score_reg.joblib`, `issue_type_clf.joblib`) are lazy-loaded on
  first inference call (see `backend/inference.py`) and reused for
  subsequent requests.
- Local Docker Compose: `docker compose up --build` → `http://localhost:7860`
- **Deployed URL**: _add after deploying to Hugging Face Spaces / other host_

### Local setup (without Docker)

```bash
# Backend
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
python app.py
# -> REST API + Gradio UI at http://localhost:7860

# Frontend (PixelGuard, React)
cd frontend
npm install
npm run dev
```

### Docker

```bash
docker compose up --build
# -> http://localhost:7860
```

## 12. Submission Contents (assessment §12)

- Full source: `backend/` (API, ML, DB), `frontend/` (React UI),
  `models/` (trained artifacts), `notebooks/` (training + evaluation
  notebook, generalization report)
- This README: setup, model/training, API, deployment, DB instructions
- `backend/db.py`: SQLite schema, auto-initialized on startup
- API docs: see §7 above and the auto-generated OpenAPI docs at
  `/docs` when the backend is running
- Evaluation results: `models/metrics.json` +
  `notebooks/evaluate_generalization.md`
- Sample images: `data/raw/` (representative clean + degraded examples)
- `Dockerfile` + `docker-compose.yml`
- Deployed URL: _add after deploying_

## 13. Optional / Bonus Work

Not all bonus items were pursued given the assessment window; the two
included are:
- Two independent frontends (React + Gradio) — arguably beyond the base
  frontend requirement.
- Confidence values surfaced per-issue and per-label throughout.

## Project Structure

```
backend/
  features.py       — engineered CV feature extraction
  degrade.py         — synthetic degradation generator
  build_dataset.py   — builds labeled CSV from a folder of clean images
  train_model.py      — trains + evaluates the RF models, writes metrics.json
  inference.py         — loads trained models, runs analysis on a new image
  db.py                — SQLite persistence
  app.py                — FastAPI REST API + Gradio UI (single process)
frontend/               — PixelGuard React UI
models/                  — trained model artifacts (.joblib) + metrics.json
data/                    — raw source images / synthetic dataset / sqlite db
notebooks/               — training notebook + evaluate_generalization.md
Dockerfile
docker-compose.yml
```

## Environment Variables

- `PORT` — port the backend listens on (default `7860`, required by most
  container hosts including Hugging Face Spaces).
