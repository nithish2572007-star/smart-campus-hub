from typing import Optional
import os
import shutil
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.database import engine, Base, get_db
from app import models
from app.ml import embeddings
from fastapi.staticfiles import StaticFiles
from app.ml import ranking

# Initialize the SQLite database
Base.metadata.create_all(bind=engine)

# 1. Define the app variable BEFORE any routes
app = FastAPI(title="Smart Campus Lost & Found Hub")

# 2. Configure the templates directory
templates = Jinja2Templates(directory="app/templates")

# 3. Create upload directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="base.html")

@app.get("/report")
def get_report_form(request: Request):
    return templates.TemplateResponse(request=request, name="report_form.html")

@app.post("/report")
async def create_report(
    type: str = Form(...),
    category: str = Form(...),
    location: str = Form(...),
    event_time: str = Form(...),
    description: str = Form(...),
    photo: UploadFile = File(...),
    finder_secret_note: str = Form(None),
    db: Session = Depends(get_db)
):
    # 1. Save the uploaded file locally
    file_location = f"uploads/{photo.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(photo.file, file_object)
        
    # 2. Parse the datetime string from the HTML form
    parsed_time = datetime.fromisoformat(event_time)
    
    # 3. Create the database record
    new_report = models.Report(
        type=models.ReportType(type),
        category=category,
        description=description,
        photo_path=file_location,
        event_time=parsed_time,
        finder_secret_note=finder_secret_note,
        location_id=1, 
        owner_id=1 
    )
    
    db.add(new_report)
    db.flush() # Flushes to get the new_report.id without committing the transaction yet
    
    # 4. Generate Embeddings (Runs on RTX 4060 if CUDA is active)
    text_vector = embeddings.get_text_embedding(description)
    image_vector = embeddings.get_image_embedding(file_location)
    
    new_embedding = models.Embedding(
        report_id=new_report.id,
        clip_image_vector=image_vector,
        sbert_text_vector=text_vector
    )
    
    db.add(new_embedding)
    db.commit()
    db.refresh(new_report)
        
    return f"""
    <div class='p-4 bg-green-100 text-green-700 rounded-md border border-green-300'>
        {type} report (ID: {new_report.id}) saved with ML embeddings!
    </div>
    """

@app.get("/feed")
def get_feed(request: Request, db: Session = Depends(get_db)):
    # Fetch all reports initially, ordered by newest first
    reports = db.query(models.Report).order_by(models.Report.event_time.desc()).all()
    return templates.TemplateResponse(
        request=request, 
        name="feed.html", 
        context={"reports": reports}
    )

@app.get("/search")
def search_reports(
    request: Request,
    category: Optional[str] = None,
    type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.Report)
    
    # Apply filters if provided
    if category:
        query = query.filter(models.Report.category == category)
    if type:
        # Convert string to the Enum used in the database
        query = query.filter(models.Report.type == models.ReportType(type))
        
    reports = query.order_by(models.Report.event_time.desc()).all()
    
    # Return ONLY the report cards HTML so HTMX can inject it into the feed
    return templates.TemplateResponse(
        request=request, 
        name="report_cards.html", 
        context={"reports": reports}
    )
@app.get("/report/{report_id}")
def view_report(report_id: int, request: Request, db: Session = Depends(get_db)):
    # Fetch the target report
    target_report = db.query(models.Report).filter(models.Report.id == report_id).first()
    
    if not target_report:
        return {"error": "Report not found"}
        
    # Fetch all OTHER reports to act as candidates
    all_other_reports = db.query(models.Report).filter(models.Report.id != report_id).all()
    
    # Run the Phase 2 AI ranking engine
    scored_matches = ranking.score_candidates(target_report, all_other_reports)
    
    return templates.TemplateResponse(
        request=request, 
        name="report_detail.html", 
        context={
            "report": target_report,
            "matches": scored_matches
        }
    )