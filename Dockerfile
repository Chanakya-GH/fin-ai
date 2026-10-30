# Multi-stage Dockerfile for AI Financial Advisory Agent
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY src/ ./src/
COPY data/prepare_data.py ./data/
COPY Modelfile .
COPY README.md .

# Create cache and db directories
RUN mkdir -p chroma_db checkpoints data/processed data/raw

EXPOSE 8000

ENV DEMO_MODE=true
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
