import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, Text, JSON
from sqlalchemy.orm import relationship
from .database import Base

class ReportStatus(enum.Enum):
    OPEN = "Open"
    MATCHED = "Matched"
    CLAIM_PENDING = "Claim pending"
    RETURNED = "Returned"
    RESOLVED = "Resolved"
    CLOSED = "Closed"

class ReportType(enum.Enum):
    LOST = "Lost"
    FOUND = "Found"

class ClaimStatus(enum.Enum):
    PENDING = "Pending"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    RESOLVED = "Resolved"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True) # For campus email login
    reports = relationship("Report", back_populates="owner")

class Location(Base):
    __tablename__ = "locations"
    id = Column(Integer, primary_key=True, index=True)
    building_name = Column(String, unique=True) # e.g., SJT, TT
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    type = Column(Enum(ReportType))
    category = Column(String, index=True)
    description = Column(Text)
    photo_path = Column(String, nullable=True)
    event_time = Column(DateTime)
    status = Column(Enum(ReportStatus), default=ReportStatus.OPEN)
    
    # Used for claim vetting proof questions
    finder_verification_question = Column(String, nullable=True)
    finder_secret_note = Column(Text, nullable=True) 
    
    owner_id = Column(Integer, ForeignKey("users.id"))
    location_id = Column(Integer, ForeignKey("locations.id"))
    
    owner = relationship("User", back_populates="reports")
    embedding = relationship("Embedding", back_populates="report", uselist=False)
    claims = relationship("Claim", back_populates="report", cascade="all, delete-orphan")
    handover_log = relationship("HandoverLog", back_populates="report", uselist=False)

class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    
    # Store vectors computed at upload
    clip_image_vector = Column(JSON, nullable=True) 
    sbert_text_vector = Column(JSON, nullable=True) 
    
    report = relationship("Report", back_populates="embedding")

class Claim(Base):
    __tablename__ = "claims"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)
    claimant_name = Column(String, nullable=False)
    claimant_email = Column(String, nullable=False)
    claimant_reg_no = Column(String, nullable=False) # e.g., "21BCE10234"
    proof_answer = Column(Text, nullable=False)      # Claimant's submitted answer
    status = Column(Enum(ClaimStatus), default=ClaimStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_notes = Column(Text, nullable=True)

    report = relationship("Report", back_populates="claims")
    handover = relationship("HandoverLog", back_populates="claim", uselist=False)

class HandoverLog(Base):
    __tablename__ = "handover_logs"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=True)
    claimant_reg_no = Column(String, nullable=False) # Student Reg No audited
    claimant_name = Column(String, nullable=True)
    verified_by = Column(String, nullable=False)     # Staff/Security/Admin
    timestamp = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    report = relationship("Report", back_populates="handover_log")
    claim = relationship("Claim", back_populates="handover")