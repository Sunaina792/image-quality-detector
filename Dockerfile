# Hugging Face Spaces Docker SDK expects the app to listen on port 7860.
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY models /app/models
COPY frontend /app/frontend

# data/ is created at runtime for the SQLite DB
RUN mkdir -p /app/data

ENV PORT=7860
EXPOSE 7860

WORKDIR /app/backend
CMD ["python", "app.py"]
