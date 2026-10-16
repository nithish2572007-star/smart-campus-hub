import os
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image

# Automatically use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading ML models on {device}...")

# Determine absolute paths to the fine-tuned model folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
custom_text_path = os.path.join(BASE_DIR, "text-campus-finetuned")
custom_image_path = os.path.join(BASE_DIR, "clip-campus-finetuned")

# 1. Load text model (use custom if exists, cache if available, fallback to download)
if os.path.exists(custom_text_path):
    print(f"Loading custom fine-tuned text model from: {custom_text_path}")
    text_model = SentenceTransformer(custom_text_path, device=device)
else:
    print("Custom text model not found, falling back to base 'all-MiniLM-L6-v2'...")
    try:
        text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device, local_files_only=True)
    except Exception:
        text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# 2. Load image model (use custom if exists, cache if available, fallback to download)
if os.path.exists(custom_image_path):
    print(f"Loading custom fine-tuned CLIP model from: {custom_image_path}")
    image_model = SentenceTransformer(custom_image_path, device=device)
else:
    print("Custom image model not found, falling back to base 'clip-ViT-B-32'...")
    try:
        image_model = SentenceTransformer('clip-ViT-B-32', device=device, local_files_only=True)
    except Exception:
        image_model = SentenceTransformer('clip-ViT-B-32', device=device)

def get_text_embedding(text: str) -> list[float]:
    """Generates a vector embedding for the report description."""
    return text_model.encode(text).tolist()

def get_image_embedding(image_path: str) -> list[float]:
    """Generates a vector embedding for the uploaded photo."""
    with Image.open(image_path) as img:
        return image_model.encode(img.convert("RGB")).tolist()