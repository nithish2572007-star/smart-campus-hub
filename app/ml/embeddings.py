import torch
from sentence_transformers import SentenceTransformer
from PIL import Image

# Automatically use your RTX 4060 via CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading ML models on {device}...")

# Load models once into memory when the server starts
text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
image_model = SentenceTransformer('clip-ViT-B-32', device=device)

def get_text_embedding(text: str) -> list[float]:
    """Generates a vector embedding for the report description."""
    # .tolist() converts the NumPy array to a standard Python list for JSON storage
    return text_model.encode(text).tolist()

def get_image_embedding(image_path: str) -> list[float]:
    """Generates a vector embedding for the uploaded photo."""
    image = Image.open(image_path)
    return image_model.encode(image).tolist()