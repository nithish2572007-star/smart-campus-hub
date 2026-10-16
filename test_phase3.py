import os
import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from fastapi.testclient import TestClient
from app.main import app, alerts_sse
from app.database import SessionLocal, init_db
from app import models
from app.services.broadcaster import broadcaster

client = TestClient(app)

def test_phase3_suite():
    print("=== STARTING PHASE 3 TEST SUITE ===")
    
    db = SessionLocal()
    
    # Ensure baseline setup
    init_db()
    
    # 1. Create test user and location
    user = db.query(models.User).first()
    if not user:
        user = models.User(email="test.student@campus.edu")
        db.add(user)
        db.commit()
        db.refresh(user)

    location = db.query(models.Location).filter(models.Location.building_name == "SJT").first()
    if not location:
        location = models.Location(building_name="SJT", latitude=12.9716, longitude=79.1589)
        db.add(location)
        db.commit()
        db.refresh(location)
        
    print(f"✓ Baseline Database ready. User ID: {user.id}, Location: {location.building_name}")

    # 2. Test SSE Endpoint and Streaming Generator
    async def test_sse_endpoint():
        mock_req = MagicMock()
        mock_req.is_disconnected = AsyncMock(side_effect=[False, True])
        sse_response = await alerts_sse(mock_req)
        assert sse_response.media_type == "text/event-stream"
        assert sse_response.headers["Cache-Control"] == "no-cache"
        
        gen = sse_response.body_iterator
        first_event = await gen.__anext__()
        assert "connected" in first_event

    asyncio.run(test_sse_endpoint())
    print("✓ Test 1 Passed: /alerts/sse generator streams valid text/event-stream with handshake.")

    # 3. Create a test Found report with secret note and verification question
    found_report = models.Report(
        type=models.ReportType.FOUND,
        category="wallet",
        description="Black leather Tommy Hilfiger wallet with blue stripe",
        photo_path="uploads/test_wallet.png",
        event_time=datetime(2026, 10, 16, 14, 0),
        status=models.ReportStatus.OPEN,
        finder_verification_question="What unique family photo or card is inside the flap?",
        finder_secret_note="A photo of two golden retrievers at the beach and a metro card ending in 8892",
        owner_id=user.id,
        location_id=location.id
    )
    db.add(found_report)
    db.commit()
    db.refresh(found_report)
    print(f"✓ Test 2 Passed: Found report #{found_report.id} created with secret note & verification question.")

    # 4. Test Claim Vetting Modal Route
    claim_modal_res = client.get(f"/report/{found_report.id}/claim-modal")
    assert claim_modal_res.status_code == 200
    assert "Ownership Proof Question" in claim_modal_res.text
    assert "What unique family photo or card is inside the flap?" in claim_modal_res.text
    print("✓ Test 3 Passed: Claim modal renders proof question properly.")

    # 5. Test Submitting a Claim (The Claim Vetting Flow)
    claim_payload = {
        "claimant_name": "Alex Mercer",
        "claimant_email": "alex.mercer@campus.edu",
        "claimant_reg_no": "21BCE0492",
        "proof_answer": "Photo of two golden retrievers at the beach and Delhi metro card ending in 8892"
    }
    claim_res = client.post(f"/report/{found_report.id}/claim", data=claim_payload)
    assert claim_res.status_code == 200
    assert "Claim Submitted Successfully" in claim_res.text
    assert "21BCE0492" in claim_res.text

    # Verify claim record in database
    claim = db.query(models.Claim).filter(models.Claim.report_id == found_report.id).order_by(models.Claim.id.desc()).first()
    assert claim is not None
    assert claim.claimant_reg_no == "21BCE0492"
    assert claim.status == models.ClaimStatus.PENDING
    print(f"✓ Test 4 Passed: Claim #{claim.id} recorded with PENDING status.")

    # 6. Test Claims Review Dashboard
    claims_list_res = client.get("/claims")
    assert claims_list_res.status_code == 200
    assert "Claim Review" in claims_list_res.text
    assert "Finder" in claims_list_res.text and "Secret" in claims_list_res.text
    assert "Claimant" in claims_list_res.text and "Proof" in claims_list_res.text
    print("✓ Test 5 Passed: Claims dashboard displays confidential comparison view.")

    # 7. Test Claim Approval & Status Transition (OPEN -> CLAIM_PENDING)
    approve_res = client.post(f"/claims/{claim.id}/approve")
    assert approve_res.status_code == 200
    
    db.refresh(claim)
    db.refresh(found_report)
    assert claim.status == models.ClaimStatus.APPROVED
    assert found_report.status == models.ReportStatus.CLAIM_PENDING
    print(f"✓ Test 6 Passed: Claim approved and report transitioned to {found_report.status.value}.")

    # 8. Test Handover Modal
    handover_modal_res = client.get(f"/claims/{claim.id}/handover-modal")
    assert handover_modal_res.status_code == 200
    assert "Handover" in handover_modal_res.text
    assert "21BCE0492" in handover_modal_res.text
    print("✓ Test 7 Passed: Handover modal prefilled with student registration number.")

    # 9. Test Handover Execution & Audit Logging (CLAIM_PENDING -> RESOLVED)
    handover_payload = {
        "claimant_reg_no": "21BCE0492",
        "claimant_name": "Alex Mercer",
        "verified_by": "Officer Rajesh K. (SJT Security Desk)",
        "notes": "Verified student ID card #21BCE0492. Claimant confirmed photo in wallet. Handover completed."
    }
    handover_res = client.post(f"/claims/{claim.id}/handover", data=handover_payload)
    assert handover_res.status_code == 200
    assert "Handover Complete" in handover_res.text

    db.refresh(claim)
    db.refresh(found_report)
    assert claim.status == models.ClaimStatus.RESOLVED
    assert found_report.status == models.ReportStatus.RESOLVED

    # Verify HandoverLog entry in DB
    audit_log = db.query(models.HandoverLog).filter(models.HandoverLog.claim_id == claim.id).first()
    assert audit_log is not None
    assert audit_log.claimant_reg_no == "21BCE0492"
    assert audit_log.verified_by == "Officer Rajesh K. (SJT Security Desk)"
    print(f"✓ Test 8 Passed: Handover finalized, report marked RESOLVED, AuditLog #HL-{audit_log.id:04d} created.")

    # 10. Test Handover Audit Trail Page
    audit_page_res = client.get("/audit-logs")
    assert audit_page_res.status_code == 200
    assert "Handover Audit Trail" in audit_page_res.text
    assert "21BCE0492" in audit_page_res.text
    assert "Officer Rajesh K." in audit_page_res.text
    print("✓ Test 9 Passed: Audit trail page displays recorded handover with registration number.")

    # 11. Test SSE Real-Time Match Alert Notification
    async def test_broadcast():
        q = broadcaster.connect()
        test_toast = "<div class='toast'>Match detected!</div>"
        await broadcaster.broadcast("match-alert", test_toast)
        msg = await asyncio.wait_for(q.get(), timeout=2.0)
        broadcaster.disconnect(q)
        assert "event: match-alert" in msg
        assert "Match detected!" in msg

    asyncio.run(test_broadcast())
    print("✓ Test 10 Passed: SSE broadcaster successfully dispatches match-alert events to subscribers.")

    db.close()
    print("\n🎉 ALL PHASE 3 TESTS PASSED SUCCESSFULLY! 🎉")

if __name__ == "__main__":
    test_phase3_suite()
