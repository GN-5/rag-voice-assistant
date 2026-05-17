#!/bin/sh
set -e

echo "=== Speaches Entry Point ==="
echo "Checking for preloaded models..."

# Download models if PRELOAD_MODELS is set
if [ -n "$PRELOAD_MODELS" ]; then
  echo "Preloading models: $PRELOAD_MODELS"
  python3 -c "
import json, os, sys
from huggingface_hub import snapshot_download

models = json.loads(os.environ.get('PRELOAD_MODELS', '[]'))
for model_id in models:
    print(f'Downloading {model_id}...')
    try:
        snapshot_download(repo_id=model_id)
        print(f'Successfully downloaded {model_id}')
    except Exception as e:
        print(f'Warning: Failed to download {model_id}: {e}', file=sys.stderr)
"
fi

echo "Starting Speaches server..."
exec uvicorn speaches.main:create_app --host 0.0.0.0 --port 8000
