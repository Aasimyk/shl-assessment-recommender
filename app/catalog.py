# app/catalog.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .embeddings import encode_texts, load_embeddings, save_embeddings  # [web:49]


CATALOG_PATH = Path("data/catalog_preprocessed.csv")


@dataclass
class AssessmentItem:
    assessment_id: str
    name: str
    url: str
    remote_testing: int
    adaptive_irt: int
    duration_minutes: Optional[float]
    test_type_codes: str
    job_levels: str
    languages: str
    description: str
    text_for_embedding: str


def load_catalog() -> List[AssessmentItem]:
    items: List[AssessmentItem] = []
    with CATALOG_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur_raw = row.get("duration_minutes") or ""
            try:
                dur_val: Optional[float] = float(dur_raw)
            except ValueError:
                dur_val = None

            items.append(
                AssessmentItem(
                    assessment_id=row["assessment_id"],
                    name=row["name"],
                    url=row["url"],
                    remote_testing=int(row.get("remote_testing", 0) or 0),
                    adaptive_irt=int(row.get("adaptive_irt", 0) or 0),
                    duration_minutes=dur_val,
                    test_type_codes=row.get("test_type_codes", ""),
                    job_levels=row.get("job_levels", ""),
                    languages=row.get("languages", ""),
                    description=row.get("description", ""),
                    text_for_embedding=row.get("text_for_embedding", ""),
                )
            )
    return items


def build_catalog_index() -> None:
    """Compute and persist embeddings for all catalog items."""
    items = load_catalog()
    texts = [it.text_for_embedding or it.description or it.name for it in items]
    emb = encode_texts(texts)
    save_embeddings(emb)


def get_all_items_with_embeddings():
    """Return (items, embeddings) aligned by index."""
    items = load_catalog()
    emb = load_embeddings()
    if len(items) != emb.shape[0]:
        raise ValueError(
            f"Items ({len(items)}) and embeddings ({emb.shape[0]}) lengths differ."
        )
    return items, emb
