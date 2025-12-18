# app/embeddings.py
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer  # [web:46]


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # [web:43]
EMBEDDINGS_PATH = Path("data/catalog_embeddings.npy")


def get_model() -> SentenceTransformer:
    """Load (and cache) the sentence-transformers model."""
    global _MODEL
    try:
        return _MODEL
    except NameError:
        _MODEL = SentenceTransformer(MODEL_NAME)
        return _MODEL


def encode_texts(texts: List[str]) -> np.ndarray:
    """Encode a list of texts into embeddings."""
    model = get_model()
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return emb.astype("float32")


def save_embeddings(embeddings: np.ndarray) -> None:
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)


def load_embeddings() -> np.ndarray:
    return np.load(EMBEDDINGS_PATH)
