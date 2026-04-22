"""
embeddings.py
─────────────
Shared embedding utility using sentence-transformers.
Provides consistent, cached embeddings across all modules.
"""

from pathlib import Path
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import yaml

# ─── Load Config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

_MODEL_NAME = CONFIG["embeddings"]["model"]
_BATCH_SIZE = CONFIG["embeddings"]["batch_size"]
_DEVICE = CONFIG["embeddings"]["device"]

# ─── Singleton model instance ─────────────────────────────────────────────────
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load and cache the sentence transformer model."""
    global _model
    if _model is None:
        print(f"⚙️   Loading embedding model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME, device=_DEVICE)
    return _model


def embed_texts(
    texts: list[str],
    show_progress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Embed a list of texts and return an (N, D) numpy array.

    Args:
        texts:          List of complaint strings.
        show_progress:  Show tqdm progress bar.
        normalize:      L2-normalize embeddings (recommended for cosine similarity).

    Returns:
        Numpy array of shape (N, embedding_dim).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=_BATCH_SIZE,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return embeddings


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of embeddings (L2 normalized)."""
    # If already normalized, dot product == cosine similarity
    return a @ b.T


def compute_cluster_centroids(
    embeddings: np.ndarray, labels: list[str] | np.ndarray
) -> dict[str, np.ndarray]:
    """
    Compute the mean embedding (centroid) for each label class.

    Returns:
        Dict mapping topic label → centroid embedding (normalized).
    """
    unique_labels = list(set(labels))
    centroids: dict[str, np.ndarray] = {}
    for label in unique_labels:
        mask = np.array(labels) == label
        cluster_embeddings = embeddings[mask]
        centroid = cluster_embeddings.mean(axis=0)
        # Normalize centroid
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        centroids[label] = centroid
    return centroids
