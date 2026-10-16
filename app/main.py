from typing import Optional
import os
import shutil
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.database import engine, Base, get_db
from app import models

# Initialize the SQLite database
Base.metadata.create_all(bind=engine)

# 1. Define the app variable BEFORE any routes
app = FastAPI(title="Smart Campus Lost & Found Hub")

# 2. Configure the templates directory
templates = Jinja2Templates(directory="app/templates")

# 3. Create upload directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

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
    # Save the uploaded file locally
    file_location = f"uploads/{photo.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(photo.file, file_object)
        
    # Parse the datetime string from the HTML form
    parsed_time = datetime.fromisoformat(event_time)
    
    # Create the database record
    # Note: Hardcoding location_id=1 and owner_id=1 temporarily until Auth is added
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
    db.commit()
    db.refresh(new_report)
        
    return f"""
    <div class='p-4 bg-green-100 text-green-700 rounded-md border border-green-300'>
        {type} report (ID: {new_report.id}) saved successfully to the database!
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