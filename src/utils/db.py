"""
SQLAlchemy models and CRUD helpers for ECLIPSE candidate catalog.
Uses SQLite by default; any SQLAlchemy-compatible DB works.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime,
    Boolean, Text, JSON, Index, event
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

DB_URL = os.getenv("ECLIPSE_DB_URL", "sqlite:///eclipse.db")


class Base(DeclarativeBase):
    pass


class Candidate(Base):
    """
    A Threshold Crossing Event (TCE) candidate that has been classified
    by ECLIPSE-PRIME. Stores every field the frontend needs.
    """
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tic_id = Column(Integer, nullable=False, index=True)
    sector = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time_s = Column(Float)

    # ── Classification ─────────────────────────────────────────────────────────
    predicted_class = Column(String(16))       # TRANSIT, EB, BLEND, OTHER
    prob_transit = Column(Float)
    prob_eb = Column(Float)
    prob_blend = Column(Float)
    prob_other = Column(Float)
    confidence = Column(Float)                  # max probability
    conformal_class_set = Column(JSON)          # list of classes in conformal set
    in_conformal_90 = Column(Boolean)
    in_conformal_95 = Column(Boolean)

    # ── Transit parameters ─────────────────────────────────────────────────────
    period = Column(Float)                      # days
    period_err = Column(Float)
    duration = Column(Float)                    # hours
    duration_err = Column(Float)
    depth = Column(Float)                       # fractional
    depth_err = Column(Float)
    depth_ppm = Column(Float)                   # parts per million
    n_transits = Column(Integer)
    odd_even_mismatch = Column(Float)

    # ── SNR ────────────────────────────────────────────────────────────────────
    snr_tls = Column(Float)                     # TLS Signal Detection Efficiency
    snr_transit = Column(Float)                 # photometric SNR

    # ── Conformal period intervals ─────────────────────────────────────────────
    period_lower = Column(Float)
    period_upper = Column(Float)

    # ── Derived planet parameters ──────────────────────────────────────────────
    rp_rearth = Column(Float)                   # planet radius in Earth radii
    t_eq_kelvin = Column(Float)                 # equilibrium temperature (K)

    # ── Stellar parameters (raw values from TIC) ───────────────────────────────
    host_name = Column(String(64))
    teff = Column(Float)                        # K
    log_g = Column(Float)
    stellar_mass = Column(Float)                # M_sun
    stellar_radius = Column(Float)              # R_sun
    tmag = Column(Float)
    contamination = Column(Float)
    distance_pc = Column(Float)                 # parsecs
    luminosity_lsun = Column(Float)             # L_sun (derived)
    ra = Column(Float)
    dec = Column(Float)

    # ── Habitability outputs ───────────────────────────────────────────────────
    esi_score = Column(Float)                   # Earth Similarity Index 0–1
    hz_class = Column(String(16))               # INNER / CONSERVATIVE / OUTER / NONE
    priority_score = Column(Float)              # combined priority 0–1
    tier = Column(Integer)                      # 1, 2, or 3
    rv_amplitude_ms = Column(Float)             # RV semi-amplitude K (m/s)
    in_confirmed_catalog = Column(Boolean, default=False)

    # ── XAI outputs ────────────────────────────────────────────────────────────
    shap_values_json = Column(Text)             # JSON: [{name, value, shap_value}, ...]
    attention_map_b64 = Column(Text)            # base64 PNG attention heatmap
    attention_path = Column(String(256))
    shap_path = Column(String(256))
    pdf_report_path = Column(String(256))

    # ── Phase fold arrays for frontend charts ──────────────────────────────────
    phase_fold_global_json = Column(Text)       # JSON array [2001 floats]
    phase_fold_local_json = Column(Text)        # JSON array [201 floats]
    batman_model_json = Column(Text)            # JSON array [201 floats] batman fit

    # ── Metadata ───────────────────────────────────────────────────────────────
    disposition = Column(String(32))            # final human-reviewed disposition
    notes = Column(Text)

    __table_args__ = (
        Index("ix_sector_class", "sector", "predicted_class"),
        Index("ix_tic_sector", "tic_id", "sector", unique=True),
    )


class Sector(Base):
    """Tracks sector-level processing status."""
    __tablename__ = "sectors"

    id = Column(Integer, primary_key=True)
    sector_number = Column(Integer, unique=True, nullable=False)
    total_tic_ids = Column(Integer, default=0)
    processed = Column(Integer, default=0)
    transit_count = Column(Integer, default=0)
    eb_count = Column(Integer, default=0)
    blend_count = Column(Integer, default=0)
    other_count = Column(Integer, default=0)
    status = Column(String(32), default="pending")  # pending, running, done, error
    started_at = Column(DateTime)
    completed_at = Column(DateTime)


class TrainingRun(Base):
    """Log of each training run for reproducibility."""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), unique=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    config_snapshot = Column(JSON)
    best_val_f1 = Column(Float)
    best_epoch = Column(Integer)
    checkpoint_path = Column(String(256))
    notes = Column(Text)


# ── Engine & Session factory ─────────────────────────────────────────────────

def get_engine(db_url: str = DB_URL):
    """Create SQLAlchemy engine. SQLite gets WAL mode for concurrent access."""
    if db_url.startswith("sqlite"):
        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )

        @event.listens_for(engine, "connect")
        def set_wal_mode(dbapi_conn, _):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    else:
        engine = create_engine(db_url)
    return engine


def init_db(engine=None) -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)


def get_session(engine=None) -> Session:
    engine = engine or get_engine()
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal()


# ── CRUD helpers ─────────────────────────────────────────────────────────────

def upsert_candidate(session: Session, candidate_data: dict) -> Candidate:
    """Insert or update a candidate record by (tic_id, sector)."""
    existing = session.query(Candidate).filter_by(
        tic_id=candidate_data["tic_id"],
        sector=candidate_data["sector"]
    ).first()
    if existing:
        for k, v in candidate_data.items():
            if hasattr(existing, k):
                setattr(existing, k, v)
        candidate = existing
    else:
        # Only pass keys that exist as columns
        valid_cols = {c.key for c in Candidate.__table__.columns}
        filtered = {k: v for k, v in candidate_data.items() if k in valid_cols}
        candidate = Candidate(**filtered)
        session.add(candidate)
    session.commit()
    return candidate


def get_candidates(
    session: Session,
    sector: Optional[int] = None,
    predicted_class: Optional[str] = None,
    min_snr: float = 0.0,
    min_transit_prob: float = 0.0,
    min_score: float = 0.0,
    limit: int = 100,
    offset: int = 0
) -> List[Candidate]:
    """Paginated, filtered candidate query ordered by confidence desc."""
    q = session.query(Candidate)
    if sector is not None:
        q = q.filter(Candidate.sector == sector)
    if predicted_class is not None:
        q = q.filter(Candidate.predicted_class == predicted_class)
    if min_snr > 0:
        q = q.filter(Candidate.snr_tls >= min_snr)
    if min_transit_prob > 0:
        q = q.filter(Candidate.prob_transit >= min_transit_prob)
    if min_score > 0:
        q = q.filter(Candidate.confidence >= min_score)
    q = q.order_by(Candidate.confidence.desc())
    return q.offset(offset).limit(limit).all()


def get_candidate_by_tic(session: Session, tic_id: int, sector: int = 1) -> Optional[Candidate]:
    """Fetch a single candidate by TIC ID and sector."""
    return session.query(Candidate).filter_by(tic_id=tic_id, sector=sector).first()
