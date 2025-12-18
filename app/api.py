# app/api.py
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from app.main import _search_catalog


app = FastAPI(title="SHL Assessment Recommender API")


class RecommendRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10


class AssessmentResponse(BaseModel):
    assessment_id: str
    name: str
    url: str
    duration_minutes: Optional[float]
    remote_testing: int
    adaptive_irt: int
    test_type_codes: str
    job_levels: str
    languages: str
    description: str
    score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=List[AssessmentResponse])
def recommend(req: RecommendRequest):
    top_k = req.top_k or 10
    results = _search_catalog(req.query, top_k=top_k)
    out: List[AssessmentResponse] = []
    for r in results:
        out.append(
            AssessmentResponse(
                assessment_id=r["assessment_id"],
                name=r["name"],
                url=r["url"],
                duration_minutes=r["duration_minutes"],
                remote_testing=r["remote_testing"],
                adaptive_irt=r["adaptive_irt"],
                test_type_codes=r["test_type_codes"],
                job_levels=r["job_levels"],
                languages=r["languages"],
                description=r["description"],
                score=r["score"],
            )
        )
    return out
