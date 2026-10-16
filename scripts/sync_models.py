#!/usr/bin/env python3
"""
Sync Fine-Tuned ML Models from Cloud Storage or Remote Bucket.
Used in production when model weights (>600MB) are kept outside the Docker image.
Supports:
  - Direct HTTP/HTTPS tar.gz or zip download (e.g., S3 pre-signed URL, GCS public/signed URL, GitHub Releases)
  - Extraction directly to app/ml/ or /app/models/
"""

import os
import sys
import urllib.request
import tarfile
import zipfile

def sync_models():
    models_url = os.getenv("MODELS_URL")
    target_dir = os.getenv("MODELS_TARGET_DIR", os.path.join(os.path.dirname(__file__), "..", "app", "ml"))
    
    if not models_url:
        print("ℹ️  No MODELS_URL specified. Skipping remote model synchronization.")
        return

    os.makedirs(target_dir, exist_ok=True)
    archive_path = os.path.join(target_dir, "models_archive.tmp")
    
    print(f"📦 Downloading fine-tuned model archive from: {models_url}")
    try:
        def progress(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r   Downloading: {percent}% ({count * block_size / (1024*1024):.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(models_url, archive_path, reporthook=progress)
        print("\n✅ Download complete. Extracting archive...")
        
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=target_dir)
        elif zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        else:
            print("❌ Error: Downloaded file is neither a valid tar nor zip archive.")
            return

        if os.path.exists(archive_path):
            os.remove(archive_path)

        print(f"🎉 Custom fine-tuned models extracted to: {target_dir}")
    except Exception as e:
        print(f"⚠️ Failed to sync remote models: {e}")

if __name__ == "__main__":
    sync_models()
