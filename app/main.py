import asyncio
from datetime import datetime
import os
import shutil
from typing import Optional
import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app import models
from app.database import Base, engine, get_db, init_db
from app.ml import embeddings, ranking
from app.services.broadcaster import broadcaster

# 1. Initialize tables, migrations, and seed baseline locations/users
init_db()

# 2. Define FastAPI application
app = FastAPI(title="Smart Campus Lost & Found Hub")

# 3. Configure templates directory and static uploads
templates = Jinja2Templates(directory="app/templates")

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# --- Navigation & Feed Routes ---

@app.get("/")
def read_root():
    """Redirect root path directly to the campus reports feed."""
    return RedirectResponse(url="/feed", status_code=303)


@app.get("/feed")
def get_feed(request: Request, db: Session = Depends(get_db)):
    """Displays all campus reports, ordered from newest to oldest."""
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
    """HTMX dynamic search endpoint returning partial report cards."""
    query = db.query(models.Report)
    
    if category:
        query = query.filter(models.Report.category == category)
    if type:
        query = query.filter(models.Report.type == models.ReportType(type))
        
    reports = query.order_by(models.Report.event_time.desc()).all()
    
    return templates.TemplateResponse(
        request=request, 
        name="report_cards.html", 
        context={"reports": reports}
    )


# --- Server-Sent Events (SSE) Real-Time Alerts ---

@app.get("/alerts/sse")
async def alerts_sse(request: Request):
    """Streams real-time Server-Sent Events (SSE) for high-probability AI match alerts."""
    async def event_generator():
        q = broadcaster.connect()
        try:
            # Initial ping/connection handshake
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield message
                except asyncio.TimeoutError:
                    # Keepalive ping
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            broadcaster.disconnect(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# --- Report Filing & AI Matching ---

@app.get("/report")
def get_report_form(request: Request):
    """Displays the report filing form."""
    return templates.TemplateResponse(request=request, name="report_form.html")


@app.post("/report")
async def create_report(
    request: Request,
    type: str = Form(...),
    category: str = Form(...),
    location: str = Form(...),
    event_time: str = Form(...),
    description: str = Form(...),
    photo: UploadFile = File(...),
    finder_verification_question: Optional[str] = Form(None),
    finder_secret_note: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Submits a report, saves photo, generates GPU embeddings, and checks for high-match alerts."""
    # 1. Save uploaded photo with a sanitized UUID prefix to prevent collisions
    safe_filename = f"{uuid.uuid4().hex[:8]}_{photo.filename.replace(' ', '_')}"
    file_location = f"uploads/{safe_filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(photo.file, file_object)
        
    # 2. Parse event timestamp
    parsed_time = datetime.fromisoformat(event_time)
    
    # 3. Dynamic Location resolution (lookup existing or create baseline)
    loc_record = db.query(models.Location).filter(models.Location.building_name == location).first()
    if not loc_record:
        loc_record = models.Location(building_name=location)
        db.add(loc_record)
        db.flush()
        
    # 4. Default user resolution
    user_record = db.query(models.User).first()
    if not user_record:
        user_record = models.User(email="campus.user@university.edu")
        db.add(user_record)
        db.flush()
        
    # 5. Create database record
    new_report = models.Report(
        type=models.ReportType(type),
        category=category,
        description=description,
        photo_path=file_location,
        event_time=parsed_time,
        finder_verification_question=finder_verification_question,
        finder_secret_note=finder_secret_note,
        location_id=loc_record.id, 
        owner_id=user_record.id,
        status=models.ReportStatus.OPEN
    )
    
    db.add(new_report)
    db.flush()
    
    # 6. Generate ML Embeddings (CLIP image vector + SBERT text vector)
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
    
    # 7. Real-Time High-Probability Match Detection (Threshold >= 80%)
    opposite_candidates = db.query(models.Report).filter(
        models.Report.id != new_report.id,
        models.Report.type != new_report.type,
        models.Report.status == models.ReportStatus.OPEN
    ).all()
    
    scored_matches = ranking.score_candidates(new_report, opposite_candidates)
    high_matches = [m for m in scored_matches if m["score"] >= 80.0]
    
    # If high matches found, broadcast real-time SSE toast to connected clients
    if high_matches:
        top_match = high_matches[0]
        toast_html = templates.get_template("partials/match_toast.html").render({
            "target": new_report,
            "candidate": top_match["candidate"],
            "score": top_match["score"],
            "breakdown": top_match["breakdown"]
        })
        await broadcaster.broadcast("match-alert", toast_html)
        
    # Build success response message
    high_match_banner = ""
    if high_matches:
        top = high_matches[0]
        high_match_banner = f"""
        <div class="mt-3 p-3 bg-emerald-50 border border-emerald-300 rounded-lg text-emerald-800 text-xs flex items-center justify-between">
            <div>
                <strong>⚡ Instant AI Match Found ({top['score']}%)!</strong>
                Matches {top['candidate'].type.value} report #{top['candidate'].id} ({top['candidate'].category}).
            </div>
            <a href="/report/{new_report.id}" class="px-3 py-1 bg-emerald-600 text-white rounded font-bold hover:bg-emerald-700">View Match</a>
        </div>
        """
        
    return f"""
    <div class='p-4 bg-green-100 text-green-800 rounded-xl border border-green-300 shadow-sm'>
        <p class='font-bold'>{type} report (ID: #{new_report.id}) successfully filed with AI embeddings!</p>
        <p class='text-xs text-green-700 mt-1'>Category: <span class='capitalize font-medium'>{category}</span> | Location: {location} | Timestamp: {parsed_time.strftime('%Y-%m-%d %H:%M')}</p>
        {high_match_banner}
        <div class='mt-3 flex gap-2'>
            <a href='/report/{new_report.id}' class='text-xs font-semibold px-3 py-1.5 bg-green-600 text-white rounded-md hover:bg-green-700 transition'>Inspect Report & Matches</a>
            <a href='/feed' class='text-xs font-semibold px-3 py-1.5 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 transition'>Return to Feed</a>
        </div>
    </div>
    """


@app.get("/report/{report_id}")
def view_report(report_id: int, request: Request, db: Session = Depends(get_db)):
    """Displays detail page for a specific report along with AI candidate rankings and claims."""
    target_report = db.query(models.Report).filter(models.Report.id == report_id).first()
    
    if not target_report:
        return templates.TemplateResponse(
            request=request, 
            name="feed.html", 
            context={"reports": [], "error": f"Report #{report_id} not found"}
        )
        
    # Fetch candidate reports for ranking
    all_other_reports = db.query(models.Report).filter(models.Report.id != report_id).all()
    scored_matches = ranking.score_candidates(target_report, all_other_reports)
    
    # Fetch any claims associated with this item
    claims = db.query(models.Claim).filter(models.Claim.report_id == report_id).all()
    
    return templates.TemplateResponse(
        request=request, 
        name="report_detail.html", 
        context={
            "report": target_report,
            "matches": scored_matches,
            "claims": claims
        }
    )


# --- The Claim Vetting Flow ---

@app.get("/report/{report_id}/claim-modal")
def get_claim_modal(report_id: int, request: Request, db: Session = Depends(get_db)):
    """Returns the HTMX claim modal prompting the claimant for proof answer."""
    report = db.query(models.Report).filter(models.Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
        
    return templates.TemplateResponse(
        request=request,
        name="partials/claim_modal.html",
        context={"report": report}
    )


@app.post("/report/{report_id}/claim")
async def submit_claim(
    report_id: int,
    request: Request,
    claimant_name: str = Form(...),
    claimant_email: str = Form(...),
    claimant_reg_no: str = Form(...),
    proof_answer: str = Form(...),
    db: Session = Depends(get_db)
):
    """Submits proof of ownership for a found item, creating a pending Claim record."""
    report = db.query(models.Report).filter(models.Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
        
    new_claim = models.Claim(
        report_id=report.id,
        claimant_name=claimant_name,
        claimant_email=claimant_email,
        claimant_reg_no=claimant_reg_no.upper().strip(),
        proof_answer=proof_answer,
        status=models.ClaimStatus.PENDING
    )
    
    db.add(new_claim)
    db.commit()
    db.refresh(new_claim)
    
    return templates.TemplateResponse(
        request=request,
        name="partials/claim_success_modal.html",
        context={"claim": new_claim}
    )


# --- Approval & Status Management ---

@app.get("/claims")
def list_claims(
    request: Request,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Admin / Finder dashboard to review submitted claims, compare answers, and transition status."""
    query = db.query(models.Claim)
    
    if status:
        query = query.filter(models.Claim.status == models.ClaimStatus(status))
        
    claims = query.order_by(models.Claim.created_at.desc()).all()
    
    # Calculate counters for navigation badges
    all_claims = db.query(models.Claim).all()
    stats = {
        "all": len(all_claims),
        "pending": sum(1 for c in all_claims if c.status == models.ClaimStatus.PENDING),
        "approved": sum(1 for c in all_claims if c.status == models.ClaimStatus.APPROVED),
        "resolved": sum(1 for c in all_claims if c.status == models.ClaimStatus.RESOLVED),
        "rejected": sum(1 for c in all_claims if c.status == models.ClaimStatus.REJECTED),
    }
    
    return templates.TemplateResponse(
        request=request,
        name="claims.html",
        context={
            "claims": claims,
            "current_status": status,
            "stats": stats
        }
    )


@app.post("/claims/{claim_id}/approve")
def approve_claim(claim_id: int, request: Request, db: Session = Depends(get_db)):
    """Approves a claim: transitions claim to APPROVED and report to CLAIM_PENDING."""
    claim = db.query(models.Claim).filter(models.Claim.id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
        
    claim.status = models.ClaimStatus.APPROVED
    claim.reviewed_at = datetime.utcnow()
    
    # Transition report status to CLAIM_PENDING
    if claim.report:
        claim.report.status = models.ReportStatus.CLAIM_PENDING
        
    db.commit()
    db.refresh(claim)
    
    return templates.TemplateResponse(
        request=request,
        name="partials/claim_card.html",
        context={"claim": claim}
    )


@app.post("/claims/{claim_id}/reject")
def reject_claim(claim_id: int, request: Request, db: Session = Depends(get_db)):
    """Rejects a claim: transitions claim to REJECTED."""
    claim = db.query(models.Claim).filter(models.Claim.id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
        
    claim.status = models.ClaimStatus.REJECTED
    claim.reviewed_at = datetime.utcnow()
    
    db.commit()
    db.refresh(claim)
    
    return templates.TemplateResponse(
        request=request,
        name="partials/claim_card.html",
        context={"claim": claim}
    )


# --- Handover Protocol & Handover Audit Log ---

@app.get("/claims/{claim_id}/handover-modal")
def get_handover_modal(claim_id: int, request: Request, db: Session = Depends(get_db)):
    """Returns the modal dialog to record physical item handover and registration number."""
    claim = db.query(models.Claim).filter(models.Claim.id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
        
    return templates.TemplateResponse(
        request=request,
        name="partials/handover_modal.html",
        context={"claim": claim}
    )


@app.post("/claims/{claim_id}/handover")
def complete_handover(
    claim_id: int,
    request: Request,
    claimant_reg_no: str = Form(...),
    claimant_name: str = Form(...),
    verified_by: str = Form(...),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Executes the physical handover, transitioning the item to RESOLVED and creating an Audit Log."""
    claim = db.query(models.Claim).filter(models.Claim.id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
        
    # 1. Transition statuses
    claim.status = models.ClaimStatus.RESOLVED
    if claim.report:
        claim.report.status = models.ReportStatus.RESOLVED
        
    # 2. Create immutable Handover Audit Log
    handover_log = models.HandoverLog(
        report_id=claim.report_id,
        claim_id=claim.id,
        claimant_reg_no=claimant_reg_no.upper().strip(),
        claimant_name=claimant_name.strip(),
        verified_by=verified_by.strip(),
        timestamp=datetime.utcnow(),
        notes=notes
    )
    
    db.add(handover_log)
    db.commit()
    db.refresh(handover_log)
    
    return templates.TemplateResponse(
        request=request,
        name="partials/handover_success_modal.html",
        context={"log": handover_log}
    )


@app.get("/audit-logs")
def view_audit_logs(request: Request, db: Session = Depends(get_db)):
    """Displays the institution's official Handover Audit Trail."""
    logs = db.query(models.HandoverLog).order_by(models.HandoverLog.timestamp.desc()).all()
    return templates.TemplateResponse(
        request=request,
        name="audit_logs.html",
        context={"logs": logs}
    )