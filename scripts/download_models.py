#!/usr/bin/env python3
"""
Model Management Script for Production Deployment
Downloads and verifies PyTorch and SentenceTransformer model weights.
Can be run:
  1. During Docker build (to bake models into the image)
  2. In docker-entrypoint.sh (to populate a persistent volume mount)
  3. Pre-flight check in CI/CD pipelines
"""

import os
import sys
import time

def download_models():
    print("=" * 60)
    print("🚀 Smart Campus Hub — Model Download & Warmup")
    print("=" * 60)
    
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"Target Cache Directory: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check GPU availability
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference Device Detected: {device.upper()}")
    
    from sentence_transformers import SentenceTransformer
    
    models_to_download = [
        ("all-MiniLM-L6-v2", "Semantic Text Model (SBERT)"),
        ("clip-ViT-B-32", "Vision-Language Image Model (CLIP)")
    ]
    
    for model_name, description in models_to_download:
        start_time = time.time()
        print(f"\n📦 Loading & Caching {description} ['{model_name}']...")
        try:
            model = SentenceTransformer(model_name, device=device)
            elapsed = time.time() - start_time
            print(f"✅ Successfully cached '{model_name}' in {elapsed:.2f}s")
            
            # Quick verification inference
            if "clip" in model_name:
                print("   Testing dummy text vector inference...")
                vec = model.encode(["lost blue backpack"])
                print(f"   Inference OK: Vector shape {vec.shape}")
            else:
                print("   Testing dummy description inference...")
                vec = model.encode("black leather wallet found in SJT")
                print(f"   Inference OK: Vector shape {vec.shape}")
        except Exception as e:
            print(f"❌ Failed to download/verify '{model_name}': {e}", file=sys.stderr)
            sys.exit(1)
            
    print("\n" + "=" * 60)
    print("🎉 All ML models downloaded, verified, and warmed up!")
    print("=" * 60)

if __name__ == "__main__":
    download_models()
