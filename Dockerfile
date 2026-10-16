# ==============================================================================
# Production Dockerfile for Smart Campus Lost & Found Hub
# Stack: FastAPI, Jinja2, HTMX, Tailwind CSS, PyTorch, SentenceTransformers
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models_cache \
    DATABASE_URL=sqlite:////app/data/campus_hub.db \
    PORT=8000

# Install minimal OS dependencies for image processing and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entrypoint and scripts
COPY docker-entrypoint.sh /app/
COPY scripts/ /app/scripts/
RUN chmod +x /app/docker-entrypoint.sh /app/scripts/*.py

# Optional build argument to bake models into image (default: false to keep image lean)
ARG PRELOAD_MODELS=false
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
        echo "Baking ML models into image layer..."; \
        python /app/scripts/download_models.py; \
    fi

# Copy full application code
COPY app/ /app/app/

# Create persistent mount targets
RUN mkdir -p /app/uploads /app/data /app/models_cache

EXPOSE 8000

# Healthcheck to monitor web container responsiveness
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/feed || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
