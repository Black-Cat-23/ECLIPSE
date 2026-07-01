"""
GET /api/candidate/{tic_id}   — Full detail for a single candidate.
GET /api/candidates           — Paginated, filterable candidate list.

These endpoints are what the frontend uses to:
  1. Load the candidate table on the dashboard
  2. Load the detailed view when a user clicks a candidate
"""
from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException
from api.schemas import CandidateOut, CandidateListResponse, PredictResponse
from src.utils.db import get_engine, get_session, get_candidates, get_candidate_by_tic
from src.utils.config import DEFAULT_CONFIG

router = APIRouter()


@router.get("/candidates", response_model=CandidateListResponse)
def list_candidates(
    sector: Optional[int] = Query(None, description="Filter by TESS sector"),
    predicted_class: Optional[str] = Query(None, description="TRANSIT | EB | BLEND | OTHER"),
    min_snr: float = Query(0.0, ge=0.0),
    min_transit_prob: float = Query(0.0, ge=0.0, le=1.0),
    min_score: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence score"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """
    Return a ranked, filtered list of ECLIPSE candidates.
    Ordered by confidence score descending.
    """
    engine = get_engine(DEFAULT_CONFIG.api.db_url)
    session = get_session(engine)
    try:
        from src.utils.db import Candidate
        q = session.query(Candidate)
        if sector is not None:
            q = q.filter(Candidate.sector == sector)
        if predicted_class:
            q = q.filter(Candidate.predicted_class == predicted_class.upper())
        if min_snr > 0:
            q = q.filter(Candidate.snr_tls >= min_snr)
        if min_transit_prob > 0:
            q = q.filter(Candidate.prob_transit >= min_transit_prob)
        if min_score > 0:
            q = q.filter(Candidate.confidence >= min_score)

        total = q.count()
        rows = q.order_by(Candidate.confidence.desc()).offset(offset).limit(limit).all()

        return CandidateListResponse(
            total=total,
            candidates=[CandidateOut.model_validate(r) for r in rows]
        )
    finally:
        session.close()


@router.get("/candidate/{tic_id}", response_model=PredictResponse)
def get_candidate_detail(
    tic_id: int,
    sector: int = Query(1, ge=1, le=99, description="TESS sector"),
):
    """
    Return the full pipeline result for a single candidate.
    If the candidate isn't in the DB yet, trigger a live pipeline run.
    """
    engine = get_engine(DEFAULT_CONFIG.api.db_url)
    session = get_session(engine)
    try:
        row = get_candidate_by_tic(session, tic_id=tic_id, sector=sector)
    finally:
        session.close()

    if row is not None:
        # Reconstruct full response from DB
        from api.routes.predict import _db_row_to_response
        # return _db_row_to_response(row)
        pass

    # Not in DB — run the pipeline live and return result
    from api.routes.predict import _get_pipeline, _result_dict_to_response, _save_result_to_db
    pipe = _get_pipeline(sector)
    result = pipe.run(tic_id=tic_id)

    if result.get("error") and not result.get("period"):
        raise HTTPException(
            status_code=404,
            detail=f"TIC {tic_id} not found in sector {sector} or pipeline failed: {result.get('error')}"
        )

    # Save for next time
    _save_result_to_db(result)

    return _result_dict_to_response(result)
