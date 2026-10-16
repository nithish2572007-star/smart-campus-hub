import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

# Read database URL from environment or default to local SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./campus_hub.db")

# Normalize heroku/legacy postgres:// syntax if provided
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL.startswith("sqlite"):
    # Ensure directory exists for custom file paths like /app/data/campus_hub.db
    db_file_path = DATABASE_URL.replace("sqlite:///", "")
    if "/" in db_file_path:
        os.makedirs(os.path.dirname(db_file_path), exist_ok=True)
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # Production PostgreSQL connection pooling configuration
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20"))
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependency that provides a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initializes tables, applies schema migrations, and seeds baseline locations/users."""
    from app import models
    Base.metadata.create_all(bind=engine)
    
    # Universal schema inspection and migration (works for SQLite and PostgreSQL)
    inspector = inspect(engine)
    if "reports" in inspector.get_table_names():
        columns = [c["name"] for c in inspector.get_columns("reports")]
        if "finder_verification_question" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE reports ADD COLUMN finder_verification_question TEXT;"))

    # Seed campus locations and default user if missing
    db = SessionLocal()
    try:
        if db.query(models.Location).count() == 0:
            campus_locations = [
                models.Location(building_name="SJT", latitude=12.9716, longitude=79.1589),
                models.Location(building_name="TT", latitude=12.9702, longitude=79.1598),
                models.Location(building_name="Library", latitude=12.9692, longitude=79.1560),
                models.Location(building_name="MB", latitude=12.9698, longitude=79.1558),
                models.Location(building_name="Hostel_Blocks", latitude=12.9730, longitude=79.1620),
            ]
            db.add_all(campus_locations)
            db.commit()

        if db.query(models.User).count() == 0:
            default_user = models.User(email="campus.user@university.edu")
            db.add(default_user)
            db.commit()
    finally:
        db.close()