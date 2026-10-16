import enum
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, Text, JSON
from sqlalchemy.orm import relationship
from .database import Base

class ReportStatus(enum.Enum):
    OPEN = "Open"
    MATCHED = "Matched"
    CLAIM_PENDING = "Claim pending"
    RETURNED = "Returned"
    CLOSED = "Closed"

class ReportType(enum.Enum):
    LOST = "Lost"
    FOUND = "Found"

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
    finder_secret_note = Column(Text, nullable=True) 
    
    owner_id = Column(Integer, ForeignKey("users.id"))
    location_id = Column(Integer, ForeignKey("locations.id"))
    
    owner = relationship("User", back_populates="reports")
    embedding = relationship("Embedding", back_populates="report", uselist=False)

class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    
    # Store vectors computed at upload
    clip_image_vector = Column(JSON, nullable=True) 
    sbert_text_vector = Column(JSON, nullable=True) 
    
    report = relationship("Report", back_populates="embedding")