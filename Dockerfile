# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System basics (slim but helpful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && rm -rf /var/lib/apt/lists/*

# Python deps first for cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .
ENV PYTHONPATH=/app
EXPOSE 8501

# Default: ingest then serve Streamlit
CMD bash -lc "python -m scripts.ingest && python -m streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501"
