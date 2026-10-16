#!/bin/bash
set -e

echo "=========================================="
echo "⚡ Starting Smart Campus Lost & Found Hub"
echo "=========================================="

# Ensure data and upload directories exist
mkdir -p /app/uploads /app/data /app/models_cache

# Set Hugging Face cache directory
export HF_HOME=${HF_HOME:-/app/models_cache}

# 1. Check if fine-tuned weights need to be synced from remote cloud storage
if [ -n "$MODELS_URL" ]; then
    echo "☁️  Remote MODELS_URL detected. Checking fine-tuned weights..."
    python scripts/sync_models.py
fi

# 2. Check if base/fallback model weights are present or need downloading
echo "🔍 Checking ML model cache in $HF_HOME..."
if [ ! -d "/app/app/ml/clip-campus-finetuned" ] && [ ! -d "$HF_HOME/hub" ] && [ ! -d "$HF_HOME/models--sentence-transformers--all-MiniLM-L6-v2" ]; then
    echo "⬇️  Model cache not found in persistent volume. Downloading base models..."
    python scripts/download_models.py
else
    echo "✅ ML model weights ready (fine-tuned or cached base models)."
fi

# 3. Run database schema migrations & location seeding
echo "🗄️  Running database migrations and seed checks..."
python -c "from app.database import init_db; init_db(); print('Database ready.')"

echo "🚀 Launching FastAPI server..."
exec "$@"
