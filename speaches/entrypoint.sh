#!/bin/sh
set -e

echo "=== Speaches Entry Point ==="
echo "Checking for preloaded models..."

if [ -n "$PRELOAD_MODELS" ]; then
  echo "Preloading models: $PRELOAD_MODELS"
  python3 -c "
import json, os, sys
from pathlib import Path
from huggingface_hub import snapshot_download, try_to_load_from_cache

models = json.loads(os.environ.get('PRELOAD_MODELS', '[]'))
cache_dir = Path(os.environ.get('HF_HOME', '/root/.cache/huggingface/hub'))

for model_id in models:
    safe_name = model_id.replace('/', '--')
    model_path = cache_dir / f'models--{safe_name}'
    
    if model_path.exists() and (model_path / 'snapshots').exists():
        snapshots = list((model_path / 'snapshots').iterdir())
        if snapshots:
            print(f'Model {model_id} already cached at {model_path}, skipping download')
            continue
    
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
