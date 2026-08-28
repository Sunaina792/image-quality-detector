"""
Main application entrypoint.

- FastAPI provides the REST API (analyze, history, health, metrics) with
  proper JSON responses, status codes, and DB persistence.
- A custom HTML/CSS/JS frontend is served via StaticFiles at '/' —
  this is the main user interface.

Run locally:
    python app.py   # -> http://localhost:7860
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import db
from inference import analyze_image

db.init_db()

app = FastAPI(title="AI Image Quality & Defect Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Return model evaluation metrics and feature importances from metrics.json."""
    metrics_path = Path(__file__).parent.parent / "models" / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=503,
            detail="metrics.json not found. Run train_model.py first."
        )
    with open(metrics_path, "r") as f:
        return json.load(f)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = analyze_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    analysis_id = db.save_analysis(file.filename, result)
    result["id"] = analysis_id
    return result


@app.post("/analyze/batch")
async def analyze_batch(files: list[UploadFile] = File(...)):
    """
    Analyze multiple images in one request.
    Returns a list of results in the same order as the uploaded files.
    Files that fail analysis include an 'error' field instead of the normal result.
    """
    results = []
    for file in files:
        entry: dict = {"filename": file.filename}
        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            entry["error"] = "Not an image file."
            results.append(entry)
            continue
        raw = await file.read()
        if not raw:
            entry["error"] = "Empty file."
            results.append(entry)
            continue
        try:
            result = analyze_image(raw)
            analysis_id = db.save_analysis(file.filename, result)
            result["id"] = analysis_id
            entry.update(result)
        except ValueError as e:
            entry["error"] = str(e)
        except Exception as e:
            entry["error"] = f"Analysis failed: {e}"
        results.append(entry)
    return {"results": results, "total": len(results)}


@app.get("/history")
def history(limit: int = 50):
    return {"analyses": db.get_history(limit=limit)}


@app.get("/history/{analysis_id}")
def history_detail(analysis_id: int):
    row = db.get_analysis(analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return row


# ---------------------------------------------------------------------------
# Custom frontend — served at root '/'
# The frontend/ directory lives one level up from backend/
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index():
        """Serve the custom frontend index.html at the root URL."""
        return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))