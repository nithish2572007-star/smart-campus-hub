#!/usr/bin/env python3
"""
Full Stack UI/UX and Backend Test Suite for Smart Campus Hub
Validates:
  1. Backend API routes, database models, ML embeddings, and ranking fusion
  2. HTML template rendering, HTMX attributes, indicators, and DOM targets
  3. UI/UX accessibility, navigation links, mobile responsiveness, and form validation
"""

import asyncio
from datetime import datetime
from html.parser import HTMLParser
import os
import re
import sys
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

from fastapi.testclient import TestClient

from app.database import SessionLocal, init_db
from app.main import alerts_sse, app
from app.ml import embeddings, ranking
from app import models
from app.services.broadcaster import broadcaster

client = TestClient(app)

# ==============================================================================
# Helper HTML & DOM Parser for UI/UX Testing
# ==============================================================================
class DOMValidator(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.classes = set()
        self.links = []
        self.forms = []
        self.inputs = []
        self.images = []
        self.hx_targets = []
        self.hx_indicators = []
        self.hx_posts = []
        self.hx_gets = []
        self.hx_disabled_elts = []
        self.sse_swaps = []
        self.has_viewport = False
        self.current_form = None

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        
        # Track IDs and classes
        if 'id' in attr_dict:
            self.ids.add(attr_dict['id'])
        if 'class' in attr_dict:
            self.classes.update(attr_dict['class'].split())

        # Check viewport
        if tag == 'meta' and attr_dict.get('name') == 'viewport':
            self.has_viewport = True

        # Track links
        if tag == 'a' and 'href' in attr_dict:
            self.links.append(attr_dict['href'])

        # Track images
        if tag == 'img':
            self.images.append({
                'src': attr_dict.get('src', ''),
                'alt': attr_dict.get('alt', ''),
                'has_alt': 'alt' in attr_dict
            })

        # Track forms & inputs
        if tag == 'form':
            self.current_form = attr_dict
            self.forms.append(attr_dict)

        if tag in ('input', 'select', 'textarea'):
            self.inputs.append({
                'tag': tag,
                'name': attr_dict.get('name'),
                'type': attr_dict.get('type'),
                'required': 'required' in attr_dict,
                'accept': attr_dict.get('accept')
            })

        # Track HTMX attributes
        if 'hx-target' in attr_dict:
            self.hx_targets.append(attr_dict['hx-target'])
        if 'hx-indicator' in attr_dict:
            self.hx_indicators.append(attr_dict['hx-indicator'])
        if 'hx-post' in attr_dict:
            self.hx_posts.append(attr_dict['hx-post'])
        if 'hx-get' in attr_dict:
            self.hx_gets.append(attr_dict['hx-get'])
        if 'hx-disabled-elt' in attr_dict:
            self.hx_disabled_elts.append(attr_dict['hx-disabled-elt'])
        if 'sse-swap' in attr_dict:
            self.sse_swaps.append(attr_dict['sse-swap'])

    def handle_endtag(self, tag):
        if tag == 'form':
            self.current_form = None


def parse_dom(html_content: str) -> DOMValidator:
    validator = DOMValidator()
    validator.feed(html_content)
    return validator


# ==============================================================================
# MAIN TEST SUITE
# ==============================================================================
def run_full_suite():
    passed_tests = 0
    failed_tests = 0
    test_results = []

    def record_test(name: str, passed: bool, details: str = ""):
        nonlocal passed_tests, failed_tests
        if passed:
            passed_tests += 1
            print(f"  ✅ PASS: {name}")
            test_results.append((name, True, details))
        else:
            failed_tests += 1
            print(f"  ❌ FAIL: {name} — {details}")
            test_results.append((name, False, details))

    print("\n" + "=" * 75)
    print("🚀 SMART CAMPUS HUB — FULL BACKEND & UI/UX TEST SUITE")
    print("=" * 75)

    db = SessionLocal()
    init_db()

    # --------------------------------------------------------------------------
    # PART 1: BACKEND ARCHITECTURE & DATA LAYER
    # --------------------------------------------------------------------------
    print("\n📦 [PART 1] BACKEND ARCHITECTURE & DATA LAYER TESTS")
    print("-" * 75)

    # Test 1.1: Database Seeding & Models
    locations = db.query(models.Location).all()
    user = db.query(models.User).first()
    has_seed = len(locations) >= 5 and user is not None
    record_test("Database Seeding & Locations", has_seed, f"Locations: {len(locations)}, User: {getattr(user, 'email', None)}")

    # Test 1.2: Location Model Relationship
    test_loc = locations[0]
    dummy_report = models.Report(
        type=models.ReportType.LOST,
        category="wallet",
        description="Relationship test item",
        photo_path="uploads/test_wallet.png",
        event_time=datetime.now(),
        location_id=test_loc.id,
        owner_id=user.id,
        status=models.ReportStatus.OPEN
    )
    db.add(dummy_report)
    db.commit()
    db.refresh(dummy_report)
    has_rel = dummy_report.location is not None and dummy_report.location.building_name == test_loc.building_name
    record_test("Report -> Location ORM Relationship", has_rel, f"Report location: {getattr(dummy_report.location, 'building_name', None)}")

    # Test 1.3: Text ML Embedding Vector Shape
    text_vec = embeddings.get_text_embedding("Lost iPhone 14 Pro in Central Library")
    is_valid_text = isinstance(text_vec, list) and len(text_vec) == 384 and all(isinstance(x, float) for x in text_vec)
    record_test("SBERT Text Embedding (384-dim vector)", is_valid_text, f"Vector length: {len(text_vec)}")

    # Test 1.4: Image ML Embedding Vector Shape
    sample_img = "uploads/ChatGPT Image Oct 12, 2026, 08_19_35 PM.png"
    img_vec = embeddings.get_image_embedding(sample_img)
    is_valid_img = isinstance(img_vec, list) and len(img_vec) == 512 and all(isinstance(x, float) for x in img_vec)
    record_test("CLIP Image Embedding (512-dim vector)", is_valid_img, f"Vector length: {len(img_vec)}")

    # Test 1.5: Cosine Similarity Edge Cases (Identical, Orthogonal, Null)
    sim_identical = ranking.cosine_similarity(text_vec, text_vec)
    sim_null = ranking.cosine_similarity(None, text_vec)
    orthogonal_vec = [-x for x in text_vec]
    sim_ortho = ranking.cosine_similarity(text_vec, orthogonal_vec)
    is_cos_correct = (abs(sim_identical - 1.0) < 1e-4) and (sim_null == 0.0) and (sim_ortho < -0.99)
    record_test("Cosine Similarity Math (Identical=1.0, Null=0.0, Opp=-1.0)", is_cos_correct, 
                f"Identical: {sim_identical:.2f}, Null: {sim_null}, Ortho: {sim_ortho:.2f}")

    # Test 1.6: Temporal Causality Filter in Ranking
    t_lost = datetime(2026, 10, 16, 12, 0)
    t_found_early = datetime(2026, 10, 16, 10, 0) # Found before lost! (Impossible)
    t_found_valid = datetime(2026, 10, 16, 14, 0) # Found after lost

    rep_lost = models.Report(type=models.ReportType.LOST, category="bottle", description="Milton bottle", event_time=t_lost, status=models.ReportStatus.OPEN, location_id=test_loc.id, owner_id=user.id)
    rep_found_early = models.Report(type=models.ReportType.FOUND, category="bottle", description="Milton bottle", event_time=t_found_early, status=models.ReportStatus.OPEN, location_id=test_loc.id, owner_id=user.id)
    rep_found_valid = models.Report(type=models.ReportType.FOUND, category="bottle", description="Milton bottle", event_time=t_found_valid, status=models.ReportStatus.OPEN, location_id=test_loc.id, owner_id=user.id)
    
    rep_lost.embedding = models.Embedding(clip_image_vector=img_vec, sbert_text_vector=text_vec)
    rep_found_early.embedding = models.Embedding(clip_image_vector=img_vec, sbert_text_vector=text_vec)
    rep_found_valid.embedding = models.Embedding(clip_image_vector=img_vec, sbert_text_vector=text_vec)

    scores_early = ranking.score_candidates(rep_lost, [rep_found_early])
    scores_valid = ranking.score_candidates(rep_lost, [rep_found_valid])
    causality_holds = len(scores_early) == 0 and len(scores_valid) == 1
    record_test("Ranking Temporal Causality (Found time >= Lost time)", causality_holds, 
                f"Early filtered: {len(scores_early)==0}, Valid accepted: {len(scores_valid)==1}")

    # Test 1.7: Five-Factor Weighted Fusion Accuracy
    valid_score = scores_valid[0]["score"]
    # Image (0.40*1.0) + Text (0.30*1.0) + Cat (0.15*1.0) + Loc (0.10*1.0) + Time (~0.05*0.988) = ~99-100%
    fusion_accurate = 95.0 <= valid_score <= 100.0
    record_test("Ranking 5-Factor Weighted Score Arithmetic", fusion_accurate, f"Calculated Score: {valid_score}%")


    # --------------------------------------------------------------------------
    # PART 2: REST & HTMX API ROUTE TESTS
    # --------------------------------------------------------------------------
    print("\n🌐 [PART 2] REST & HTMX API ROUTE TESTS")
    print("-" * 75)

    # Test 2.1: Root path redirect
    res_root = client.get("/", follow_redirects=False)
    is_root_redirect = res_root.status_code == 303 and res_root.headers.get("location") == "/feed"
    record_test("GET / -> 303 Redirect to /feed", is_root_redirect, f"Status: {res_root.status_code}, Location: {res_root.headers.get('location')}")

    # Test 2.2: Feed route
    res_feed = client.get("/feed")
    is_feed_ok = res_feed.status_code == 200 and "Recent Campus Reports" in res_feed.text
    record_test("GET /feed -> 200 OK with Feed UI", is_feed_ok, f"Status: {res_feed.status_code}")

    # Test 2.3: Dynamic HTMX Search route
    res_search = client.get("/search?category=wallet&type=Lost")
    is_search_ok = res_search.status_code == 200 and "wallet" in res_search.text.lower()
    record_test("GET /search?category=wallet&type=Lost -> 200 OK (Partial Swap)", is_search_ok, f"Status: {res_search.status_code}")

    # Test 2.4: Dynamic HTMX Search empty query
    res_empty_search = client.get("/search?category=nonexistent_item_xyz")
    is_empty_search_ok = res_empty_search.status_code == 200 and "No reports match your filters" in res_empty_search.text
    record_test("GET /search (Empty state handling)", is_empty_search_ok, "Graceful empty state rendered")

    # Test 2.5: Report Filing Form Route
    res_form = client.get("/report")
    is_form_ok = res_form.status_code == 200 and "File a Lost or Found Report" in res_form.text
    record_test("GET /report -> 200 OK with submission form", is_form_ok, f"Status: {res_form.status_code}")

    # Test 2.6: POST /report (Found Item with Verification Question & Secret Note)
    with open("uploads/ChatGPT Image Oct 12, 2026, 08_19_35 PM.png", "rb") as f:
        found_data = {
            "type": "Found",
            "category": "electronics",
            "location": "TT",
            "event_time": "2026-10-16T15:30",
            "description": "Found Apple AirPods Pro 2 in white case with green silicone cover",
            "finder_verification_question": "What is the initials or engraving on the lid?",
            "finder_secret_note": "Engraved with initials 'S.M.' and tiny scratched star"
        }
        res_post_found = client.post("/report", data=found_data, files={"photo": ("airpods.png", f, "image/png")})
    is_post_found_ok = res_post_found.status_code == 200 and "saved with AI embeddings" in res_post_found.text or "successfully filed" in res_post_found.text
    match_found_id = re.search(r"ID:\s*#?(\d+)", res_post_found.text)
    found_id = int(match_found_id.group(1)) if match_found_id else None
    record_test("POST /report (Found item creation with proof note)", is_post_found_ok and found_id is not None, f"Created Report ID: {found_id}")

    # Test 2.7: POST /report (Lost Item with high similarity triggering >= 80% match alert)
    with open("uploads/ChatGPT Image Oct 12, 2026, 08_19_35 PM.png", "rb") as f:
        lost_data = {
            "type": "Lost",
            "category": "electronics",
            "location": "TT",
            "event_time": "2026-10-16T15:00", # Before found time
            "description": "Lost Apple AirPods Pro 2 in white case with green silicone cover"
        }
        res_post_lost = client.post("/report", data=lost_data, files={"photo": ("airpods_lost.png", f, "image/png")})
    is_post_lost_ok = res_post_lost.status_code == 200
    has_high_match_banner = "Instant AI Match Found" in res_post_lost.text or "Match Found" in res_post_lost.text
    record_test("POST /report (AI Multi-Modal High-Match >=80% Trigger)", is_post_lost_ok and has_high_match_banner, 
                f"Status: {res_post_lost.status_code}, High match detected: {has_high_match_banner}")

    # Test 2.8: Report Detail View
    res_detail = client.get(f"/report/{found_id}")
    is_detail_ok = res_detail.status_code == 200 and "AI Multi-Modal Candidate Matches" in res_detail.text
    record_test(f"GET /report/{found_id} -> 200 OK with candidate rankings", is_detail_ok, f"Status: {res_detail.status_code}")

    # Test 2.9: Non-existent report detail view (404 Not Found)
    res_bad_detail = client.get("/report/999999")
    is_bad_detail_ok = res_bad_detail.status_code == 404
    record_test("GET /report/999999 -> 404 Not Found", is_bad_detail_ok, f"Status: {res_bad_detail.status_code}")

    # Test 2.10: Claim Modal Route
    res_claim_modal = client.get(f"/report/{found_id}/claim-modal")
    is_claim_modal_ok = res_claim_modal.status_code == 200 and "What is the initials or engraving on the lid?" in res_claim_modal.text
    record_test(f"GET /report/{found_id}/claim-modal -> 200 OK with verification prompt", is_claim_modal_ok, "Proof question displayed accurately")

    # Test 2.11: Non-existent claim modal (404)
    res_bad_claim_modal = client.get("/report/999999/claim-modal")
    is_bad_modal_404 = res_bad_claim_modal.status_code == 404
    record_test("GET /report/999999/claim-modal -> 404 Not Found", is_bad_modal_404, f"Status: {res_bad_claim_modal.status_code}")

    # Test 2.12: Submit Claim Proof (POST /report/{id}/claim)
    claim_payload = {
        "claimant_name": "Sarah Connor",
        "claimant_email": "s.connor@campus.edu",
        "claimant_reg_no": "22BCE8841",
        "proof_answer": "Initials 'S.M.' on top lid with a tiny engraved star"
    }
    res_claim_sub = client.post(f"/report/{found_id}/claim", data=claim_payload)
    is_claim_sub_ok = res_claim_sub.status_code == 200 and "Claim Submitted Successfully" in res_claim_sub.text
    new_claim = db.query(models.Claim).filter(models.Claim.report_id == found_id).order_by(models.Claim.id.desc()).first()
    record_test("POST /report/{id}/claim -> 200 OK (Claim recorded as PENDING)", 
                is_claim_sub_ok and new_claim is not None and new_claim.status == models.ClaimStatus.PENDING,
                f"Claim ID: {getattr(new_claim, 'id', None)}, Status: {getattr(new_claim, 'status', None)}")

    # Test 2.13: Claims Review Dashboard (GET /claims)
    res_claims_dash = client.get("/claims")
    is_claims_dash_ok = res_claims_dash.status_code == 200 and "Claim Review" in res_claims_dash.text
    record_test("GET /claims -> 200 OK with side-by-side comparison", is_claims_dash_ok, f"Status: {res_claims_dash.status_code}")

    # Test 2.14: Claims Status Filter (GET /claims?status=Pending)
    res_claims_pending = client.get("/claims?status=Pending")
    is_claims_filter_ok = res_claims_pending.status_code == 200 and "Pending Review" in res_claims_pending.text
    record_test("GET /claims?status=Pending -> 200 OK with filter", is_claims_filter_ok, "Filtered list rendered")

    # Test 2.15: Approve Claim (POST /claims/{id}/approve)
    res_approve = client.post(f"/claims/{new_claim.id}/approve")
    db.refresh(new_claim)
    target_rep = db.query(models.Report).filter(models.Report.id == found_id).first()
    is_approved_ok = (res_approve.status_code == 200 and 
                      new_claim.status == models.ClaimStatus.APPROVED and 
                      target_rep.status == models.ReportStatus.CLAIM_PENDING)
    record_test(f"POST /claims/{new_claim.id}/approve -> Report transitions to CLAIM_PENDING", 
                is_approved_ok, f"Claim: {new_claim.status.value}, Report: {target_rep.status.value}")

    # Test 2.16: Handover Modal (GET /claims/{id}/handover-modal)
    res_handover_modal = client.get(f"/claims/{new_claim.id}/handover-modal")
    is_handover_modal_ok = res_handover_modal.status_code == 200 and "22BCE8841" in res_handover_modal.text
    record_test("GET /claims/{id}/handover-modal -> 200 OK (Reg No prefilled)", is_handover_modal_ok, "Student Reg No prefilled")

    # Test 2.17: Complete Physical Handover (POST /claims/{id}/handover)
    handover_payload = {
        "claimant_reg_no": "22BCE8841",
        "claimant_name": "Sarah Connor",
        "verified_by": "Security Officer V. Sharma",
        "notes": "Verified student ID card. Case unlocked in presence of officer."
    }
    res_handover = client.post(f"/claims/{new_claim.id}/handover", data=handover_payload)
    db.refresh(new_claim)
    db.refresh(target_rep)
    h_log = db.query(models.HandoverLog).filter(models.HandoverLog.claim_id == new_claim.id).first()
    is_handover_ok = (res_handover.status_code == 200 and 
                      new_claim.status == models.ClaimStatus.RESOLVED and 
                      target_rep.status == models.ReportStatus.RESOLVED and 
                      h_log is not None and h_log.claimant_reg_no == "22BCE8841")
    record_test("POST /claims/{id}/handover -> Report transitions to RESOLVED & HandoverLog created", 
                is_handover_ok, f"Report status: {target_rep.status.value}, Log ID: {getattr(h_log, 'id', None)}")

    # Test 2.18: Reject Claim Route (POST /claims/{id}/reject)
    dummy_claim = models.Claim(
        report_id=dummy_report.id,
        claimant_name="Bad Actor",
        claimant_email="bad@actor.com",
        claimant_reg_no="00XYZ0000",
        proof_answer="Wrong guess",
        status=models.ClaimStatus.PENDING
    )
    db.add(dummy_claim)
    db.commit()
    db.refresh(dummy_claim)
    res_reject = client.post(f"/claims/{dummy_claim.id}/reject")
    db.refresh(dummy_claim)
    is_reject_ok = res_reject.status_code == 200 and dummy_claim.status == models.ClaimStatus.REJECTED
    record_test(f"POST /claims/{dummy_claim.id}/reject -> Claim transitions to REJECTED", is_reject_ok, f"Claim status: {dummy_claim.status.value}")

    # Test 2.19: Audit Trail Dashboard (GET /audit-logs)
    res_audit = client.get("/audit-logs")
    is_audit_ok = res_audit.status_code == 200 and "22BCE8841" in res_audit.text and "Officer V. Sharma" in res_audit.text
    record_test("GET /audit-logs -> 200 OK with verified student audit log", is_audit_ok, "Audit record displayed in table")

    # Test 2.20: Real-Time SSE Endpoint Handshake
    async def run_sse_test():
        mock_req = MagicMock()
        mock_req.is_disconnected = AsyncMock(side_effect=[False, True])
        sse_res = await alerts_sse(mock_req)
        assert sse_res.media_type == "text/event-stream"
        assert sse_res.headers.get("Cache-Control") == "no-cache"
        gen = sse_res.body_iterator
        first_event = await gen.__anext__()
        assert "connected" in first_event
    asyncio.run(run_sse_test())
    record_test("GET /alerts/sse -> Streams valid text/event-stream handshake", True, "Connected handshake received")


    # --------------------------------------------------------------------------
    # PART 3: UI & UX VALIDATION (DOM, HTMX, RESPONSIVENESS, ACCESSIBILITY)
    # --------------------------------------------------------------------------
    print("\n🎨 [PART 3] UI / UX INTEGRITY & ACCESSIBILITY TESTS")
    print("-" * 75)

    # Test 3.1: HTML5 Viewport Meta Tag
    dom_feed = parse_dom(res_feed.text)
    record_test("Mobile Viewport Meta Tag present (<meta name='viewport'>)", dom_feed.has_viewport, "Responsive scaling enabled")

    # Test 3.2: Mobile Navigation Drawer & Hamburger Menu
    has_mobile_menu = "mobile-menu" in dom_feed.ids
    has_hamburger_btn = any("md:hidden" in c for c in dom_feed.classes)
    record_test("Mobile Responsive Hamburger & Drawer Menu", has_mobile_menu and has_hamburger_btn, 
                f"#mobile-menu exists: {has_mobile_menu}, Hamburger button: {has_hamburger_btn}")

    # Test 3.3: Mobile Filter Accordion in Feed View
    has_filter_fields = "filter-fields" in dom_feed.ids
    record_test("Mobile Filter Accordion Toggle (#filter-fields)", has_filter_fields, "Collapsible on mobile screens")

    # Test 3.4: HTMX Target DOM Resolution (All hx-targets exist in HTML)
    dom_pages = {
        "feed.html": res_feed.text,
        "report_form.html": res_form.text,
        "claims.html": res_claims_dash.text,
        "report_detail.html": res_detail.text
    }
    
    unresolved_targets = []
    for page_name, html in dom_pages.items():
        dom = parse_dom(html)
        for target in dom.hx_targets:
            if target.startswith("#"):
                target_id = target[1:]
                if target_id not in dom.ids:
                    unresolved_targets.append((page_name, target))
    record_test("HTMX Target Resolution (All hx-targets match DOM IDs)", len(unresolved_targets) == 0, 
                f"Unresolved: {unresolved_targets if unresolved_targets else 'None (100% matched)'}")

    # Test 3.5: HTMX Indicator Resolution (All spinners exist in DOM)
    unresolved_indicators = []
    for page_name, html in dom_pages.items():
        dom = parse_dom(html)
        for indicator in dom.hx_indicators:
            if indicator.startswith("#"):
                ind_id = indicator[1:]
                if ind_id not in dom.ids:
                    unresolved_indicators.append((page_name, indicator))
    record_test("HTMX Indicator Resolution (All hx-indicators match DOM elements)", len(unresolved_indicators) == 0,
                f"Unresolved: {unresolved_indicators if unresolved_indicators else 'None (100% matched)'}")

    # Test 3.6: Form Input Validation Attributes (Required, Type, Accept)
    dom_form = parse_dom(res_form.text)
    input_names = {inp['name']: inp for inp in dom_form.inputs if inp['name']}
    
    type_required = input_names.get('type', {}).get('required', False)
    category_required = input_names.get('category', {}).get('required', False)
    location_required = input_names.get('location', {}).get('required', False)
    time_is_datetime = input_names.get('event_time', {}).get('type') == 'datetime-local'
    photo_accepts_images = input_names.get('photo', {}).get('accept') == 'image/*'
    
    form_validations_ok = type_required and category_required and location_required and time_is_datetime and photo_accepts_images
    record_test("Form Input Validation Attributes (required, datetime-local, image/*)", form_validations_ok,
                f"Required fields OK: {type_required and category_required and location_required}, Photo accept: {photo_accepts_images}")

    # Test 3.7: Double-Submit Prevention via hx-disabled-elt
    has_disabled_elt = any("button" in elt for elt in dom_form.hx_disabled_elts)
    record_test("Double-Submit Prevention (hx-disabled-elt on Submit)", has_disabled_elt, "Submit button disabled during ML inference")

    # Test 3.8: Vector Generation Loading Spinner Card
    has_upload_spinner = "upload-spinner" in dom_form.ids
    record_test("Vector Generation Loading Indicator (#upload-spinner)", has_upload_spinner, "Animated spinner card informs user of CLIP/SBERT generation")

    # Test 3.9: Modal Viewport Overflow Protection for Small Screens
    dom_claim_modal = parse_dom(res_claim_modal.text)
    dom_handover_modal = parse_dom(res_handover_modal.text)
    modal_classes = dom_claim_modal.classes.union(dom_handover_modal.classes)
    has_max_h = any("max-h-" in c for c in modal_classes)
    has_overflow_y = "overflow-y-auto" in modal_classes
    record_test("Modal Small-Screen Overflow Protection (max-h-[92vh] overflow-y-auto)", has_max_h and has_overflow_y, 
                f"max-h present: {has_max_h}, overflow-y-auto present: {has_overflow_y}")

    # Test 3.10: Accessibility (Image alt attributes)
    images_without_alt = []
    for page_name, html in dom_pages.items():
        dom = parse_dom(html)
        for img in dom.images:
            if not img['has_alt'] or not img['alt'].strip():
                images_without_alt.append((page_name, img['src']))
    record_test("Image Accessibility (All <img> tags have descriptive alt text)", len(images_without_alt) == 0,
                f"Missing alt: {images_without_alt if images_without_alt else 'None (100% compliant)'}")

    # Test 3.11: Broken Links Spider (Every <a href> returns 200 or 303)
    all_links = set()
    for page_name, html in dom_pages.items():
        dom = parse_dom(html)
        for link in dom.links:
            # Only test internal relative routes
            if link.startswith("/") and not link.startswith("//"):
                all_links.add(link)

    broken_links = []
    for link in all_links:
        res = client.get(link, follow_redirects=True)
        if res.status_code not in (200, 303):
            broken_links.append((link, res.status_code))
    record_test(f"Navigation & Link Spider (Tested {len(all_links)} internal routes)", len(broken_links) == 0,
                f"Broken links: {broken_links if broken_links else '0 broken links (All routes 200 OK)'}")

    # Test 3.12: Graceful HTMX Error Listeners in Base Template
    base_html = res_feed.text
    has_response_err = "htmx:responseError" in base_html
    has_send_err = "htmx:sendError" in base_html
    has_timeout_err = "htmx:timeout" in base_html
    has_error_toast = "showErrorToast" in base_html
    error_handling_ok = has_response_err and has_send_err and has_timeout_err and has_error_toast
    record_test("Graceful HTMX Error & Timeout Handling Listeners", error_handling_ok, 
                f"Listeners: responseError={has_response_err}, sendError={has_send_err}, timeout={has_timeout_err}")

    # Test 3.13: Real-Time SSE Listener Container in Base Layout
    dom_base = parse_dom(res_feed.text)
    has_sse_connect = any("/alerts/sse" in str(f) for f in dom_base.forms) or "/alerts/sse" in base_html
    has_sse_swap = "match-alert" in dom_base.sse_swaps
    has_toast_container = "toast-container" in dom_base.ids
    record_test("Base Template Real-Time SSE Container & Toast Target", has_sse_connect and has_sse_swap and has_toast_container,
                f"SSE connect: {has_sse_connect}, sse-swap: {has_sse_swap}, #toast-container: {has_toast_container}")

    # Test 3.14: Human-Readable Location Building Names in UI
    has_building_name = "Silver Jubilee Tower (SJT)" in res_feed.text or "TT" in res_feed.text or "Library" in res_feed.text
    record_test("Human-Readable Campus Location Badges (Not raw integer IDs)", has_building_name, "Building names rendered in UI badges")

    db.close()

    # --------------------------------------------------------------------------
    # SUMMARY & SCORECARD
    # --------------------------------------------------------------------------
    total_tests = passed_tests + failed_tests
    pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print("\n" + "=" * 75)
    print(f"📊 TEST SUITE SUMMARY: {passed_tests}/{total_tests} TESTS PASSED ({pass_rate:.1f}%)")
    print("=" * 75)
    
    if failed_tests == 0:
        print("🎉 ALL BACKEND AND UI/UX TESTS PASSED PERFECTLY! SYSTEM IS 100% OPERATIONAL.")
    else:
        print(f"⚠️ {failed_tests} TESTS FAILED. Review details above.")

    return failed_tests == 0

if __name__ == "__main__":
    success = run_full_suite()
    sys.exit(0 if success else 1)
