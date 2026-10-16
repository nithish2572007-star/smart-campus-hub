import torch
import torch.nn.functional as F
from datetime import datetime
from app.models import Report, ReportType, ReportStatus

# Fine-tuned tuning weights for Phase 4
WEIGHTS = {
    "image": 0.40,
    "text": 0.30,
    "category": 0.15,
    "location": 0.10,
    "time": 0.05
}

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculates cosine similarity between two JSON vectors using PyTorch."""
    if not vec1 or not vec2:
        return 0.0
    t1 = torch.tensor(vec1).unsqueeze(0)
    t2 = torch.tensor(vec2).unsqueeze(0)
    return F.cosine_similarity(t1, t2).item()

def score_candidates(target_report: Report, candidates: list[Report]) -> list[dict]:
    """Scores and ranks candidate matches based on the five criteria."""
    scored_matches = []
    
    for candidate in candidates:
        # 1. Candidate Filters: Opposite type, open status, and logical time
        if candidate.type == target_report.type:
            continue
        if candidate.status != ReportStatus.OPEN:
            continue
            
        # Determine which is 'lost' and which is 'found' for time logic
        lost_time = target_report.event_time if target_report.type == ReportType.LOST else candidate.event_time
        found_time = target_report.event_time if target_report.type == ReportType.FOUND else candidate.event_time
        
        # Found time cannot be before lost time
        if found_time < lost_time:
            continue

        # 2. Compute the five individual scores
        # Image Score (CLIP) - safely retrieve vector if embedding exists
        target_clip = target_report.embedding.clip_image_vector if target_report.embedding else None
        cand_clip = candidate.embedding.clip_image_vector if candidate.embedding else None
        image_score = cosine_similarity(target_clip, cand_clip)
        
        # Text Score (Sentence-BERT) - safely retrieve vector if embedding exists
        target_sbert = target_report.embedding.sbert_text_vector if target_report.embedding else None
        cand_sbert = candidate.embedding.sbert_text_vector if candidate.embedding else None
        text_score = cosine_similarity(target_sbert, cand_sbert)
        
        # Category Score (Binary 1 or 0)
        category_score = 1.0 if target_report.category == candidate.category else 0.0
        
        # Location Score (Binary 1 or 0 for now, can be expanded to Haversine distance)
        location_score = 1.0 if target_report.location_id == candidate.location_id else 0.0
        
        # Time Score (Decays based on hours apart)
        time_diff_hours = (found_time - lost_time).total_seconds() / 3600
        time_score = max(0.0, 1.0 - (time_diff_hours / 168)) # Degrades to 0 after 1 week
        
        # 3. Weighted Fusion
        final_score = (
            (image_score * WEIGHTS["image"]) +
            (text_score * WEIGHTS["text"]) +
            (category_score * WEIGHTS["category"]) +
            (location_score * WEIGHTS["location"]) +
            (time_score * WEIGHTS["time"])
        )
        
        scored_matches.append({
            "candidate": candidate,
            "score": round(final_score * 100, 2), # Convert to percentage
            "breakdown": {
                "image": round(image_score, 2),
                "text": round(text_score, 2),
                "category": round(category_score, 2),
                "location": round(location_score, 2),
                "time": round(time_score, 2)
            }
        })
        
    # Sort highest score first
    scored_matches.sort(key=lambda x: x["score"], reverse=True)
    return scored_matches