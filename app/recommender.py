# app/recommender.py
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.linalg import norm

from .catalog import AssessmentItem, get_all_items_with_embeddings
from .embeddings import encode_texts


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (norm(b, axis=1, keepdims=True) + 1e-8)
    return a_n @ b_n.T


def recommend_for_query(query: str, top_k: int = 10) -> List[Tuple[AssessmentItem, float]]:
    """Return top_k (item, score) for a free-text query."""
    items, catalog_emb = get_all_items_with_embeddings()

    q_emb = encode_texts([query])
    sims = _cosine_sim(q_emb, catalog_emb)[0]

    idx = np.argsort(-sims)[:top_k]
    results: List[Tuple[AssessmentItem, float]] = [
        (items[i], float(sims[i])) for i in idx
    ]
    return results
