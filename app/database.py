from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

SQLALCHEMY_DATABASE_URL = "sqlite:///./campus_hub.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initializes tables, applies column migrations, and seeds baseline locations/users."""
    from app import models
    Base.metadata.create_all(bind=engine)
    
    # Safe SQLite column migration for new Phase 3 fields
    with engine.connect() as conn:
        cursor = conn.connection.cursor()
        cursor.execute("PRAGMA table_info(reports);")
        columns = [row[1] for row in cursor.fetchall()]
        if "finder_verification_question" not in columns:
            cursor.execute("ALTER TABLE reports ADD COLUMN finder_verification_question TEXT;")
            conn.connection.commit()

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