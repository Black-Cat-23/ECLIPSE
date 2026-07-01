"""Extended DB utilities for ECLIPSE API."""
from src.utils.db import get_engine, get_session, init_db, Candidate


def upsert_candidate(session, data: dict):
    """Insert or update a candidate record."""
    tic_id = data["tic_id"]
    sector = data["sector"]
    obj = session.query(Candidate).filter_by(tic_id=tic_id, sector=sector).first()
    if obj is None:
        obj = Candidate(**data)
        session.add(obj)
    else:
        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
    session.commit()
    return obj


def get_candidates(
    session, sector=None, predicted_class=None,
    min_snr=0.0, min_transit_prob=0.0, limit=100, offset=0
):
    """Query candidates with optional filters."""
    q = session.query(Candidate)
    if sector is not None:
        q = q.filter(Candidate.sector == sector)
    if predicted_class:
        q = q.filter(Candidate.predicted_class == predicted_class)
    if min_snr > 0:
        q = q.filter(Candidate.snr_tls >= min_snr)
    if min_transit_prob > 0:
        q = q.filter(Candidate.prob_transit >= min_transit_prob)
    return q.offset(offset).limit(limit).all()
