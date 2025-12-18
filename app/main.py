# app/main.py
from __future__ import annotations

from typing import List, Dict, Any

from .recommender import recommend_for_query
from .catalog import AssessmentItem


def _item_to_dict(rank: int, it: AssessmentItem, score: float) -> Dict[str, Any]:
    return {
        "rank": rank,
        "assessment_id": it.assessment_id,
        "name": it.name,
        "url": it.url,
        "duration_minutes": it.duration_minutes,
        "remote_testing": it.remote_testing,
        "adaptive_irt": it.adaptive_irt,
        "test_type_codes": it.test_type_codes,
        "job_levels": it.job_levels,
        "languages": it.languages,
        "description": it.description,
        "score": score,
    }


def _search_catalog(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    ranked = recommend_for_query(query, top_k=top_k)
    return [
        _item_to_dict(rank=i, it=item, score=score)
        for i, (item, score) in enumerate(ranked, start=1)
    ]
