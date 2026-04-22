"""
new_topic_detector.py
─────────────────────
Detects novel/emerging complaint topics in a stream of incoming complaints.

Algorithm:
1. Embed all incoming complaints using sentence-transformers
2. For each complaint, compute cosine distance to the nearest known topic centroid
3. Flag complaints that are "far" from all known clusters (potential new topics)
4. Run HDBSCAN on the flagged set to find clusters (new topic candidates)
5. Report the new clusters for human review + auto-naming by Gemma 4

Usage (PyCharm):  Imported by daily_pipeline.py and run_new_topic_demo.py
"""

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

_DET = CONFIG["detector"]
DISTANCE_THRESHOLD = _DET["distance_threshold"]
MIN_CLUSTER_SIZE = _DET["min_cluster_size"]
MIN_SAMPLES = _DET["min_samples"]
UMAP_N_NEIGHBORS = _DET["umap_n_neighbors"]
UMAP_N_COMPONENTS_CLUSTER = _DET["umap_n_components_cluster"]


class NewTopicDetector:
    """
    Detects emerging topics in incoming complaint batches.

    Attributes:
        known_centroids: Dict[topic_name -> centroid_embedding] loaded from training.
    """

    def __init__(self, known_centroids: Optional[dict[str, np.ndarray]] = None):
        self.known_centroids = known_centroids or {}

    def load_centroids_from_data(
        self, texts: list[str], labels: list[str]
    ) -> None:
        """Compute centroids from a labeled dataset (called after training)."""
        from src.utils.embeddings import embed_texts, compute_cluster_centroids
        print("⚙️   Computing known topic centroids from labeled data...")
        embeddings = embed_texts(texts, show_progress=True)
        self.known_centroids = compute_cluster_centroids(embeddings, labels)
        print(f"✅  Computed {len(self.known_centroids)} topic centroids.")

    def save_centroids(self, path: Path) -> None:
        """Save centroids as numpy arrays to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for topic, centroid in self.known_centroids.items():
            np.save(path / f"{topic}.npy", centroid)
        print(f"💾  Saved {len(self.known_centroids)} centroids to: {path}")

    def load_centroids(self, path: Path) -> None:
        """Load pre-computed centroids from disk."""
        path = Path(path)
        self.known_centroids = {}
        for npy_file in path.glob("*.npy"):
            topic = npy_file.stem
            self.known_centroids[topic] = np.load(str(npy_file))
        print(f"✅  Loaded {len(self.known_centroids)} centroids from: {path}")

    def _distance_to_nearest_centroid(self, embedding: np.ndarray) -> tuple[str, float]:
        """Return the nearest known topic name and cosine distance."""
        if not self.known_centroids:
            return ("none", 1.0)

        best_topic = ""
        best_sim = -1.0
        for topic, centroid in self.known_centroids.items():
            sim = float(np.dot(embedding, centroid))  # L2 normalized → dot = cosine sim
            if sim > best_sim:
                best_sim = sim
                best_topic = topic

        distance = 1.0 - best_sim  # Convert similarity to distance
        return best_topic, distance

    def flag_unknown_complaints(
        self,
        texts: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Classify each complaint as known or potentially unknown.

        Args:
            texts:      List of complaint strings.
            embeddings: Optional pre-computed embeddings (avoids re-embedding).

        Returns:
            {
                "known": [...],          # List of (text, topic, distance) for known
                "unknown": [...],        # List of (text, nearest_topic, distance) for flagged
                "embeddings": np.ndarray
            }
        """
        from src.utils.embeddings import embed_texts
        if embeddings is None:
            print("⚙️   Embedding incoming complaints...")
            embeddings = embed_texts(texts, show_progress=True)

        known_list = []
        unknown_list = []
        unknown_embeddings = []

        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            nearest_topic, distance = self._distance_to_nearest_centroid(emb)
            record = {
                "text": text,
                "nearest_known_topic": nearest_topic,
                "distance_to_centroid": round(distance, 4),
            }
            if distance > DISTANCE_THRESHOLD:
                unknown_list.append(record)
                unknown_embeddings.append(emb)
            else:
                known_list.append(record)

        print(f"\n📊  Detection Results:")
        print(f"   Total complaints   : {len(texts)}")
        print(f"   → Known topics     : {len(known_list)} ({len(known_list)/len(texts):.1%})")
        print(f"   → Flagged UNKNOWN  : {len(unknown_list)} ({len(unknown_list)/len(texts):.1%})")
        print(f"   Distance threshold : {DISTANCE_THRESHOLD}")

        return {
            "known": known_list,
            "unknown": unknown_list,
            "unknown_embeddings": np.array(unknown_embeddings) if unknown_embeddings else np.empty((0, embeddings.shape[1])),
            "all_embeddings": embeddings,
        }

    def cluster_unknown_complaints(
        self, unknown_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Run HDBSCAN on flagged unknown complaints to find new topic clusters.

        Args:
            unknown_embeddings: Embeddings of flagged complaints.

        Returns:
            Cluster labels (-1 = noise).
        """
        if len(unknown_embeddings) < MIN_CLUSTER_SIZE:
            print(f"⚠️   Only {len(unknown_embeddings)} unknown complaints — not enough for clustering.")
            return np.array([-1] * len(unknown_embeddings))

        try:
            from umap import UMAP
            import hdbscan

            print(f"\n⚙️   Reducing to {UMAP_N_COMPONENTS_CLUSTER}D for clustering...")
            reducer = UMAP(
                n_neighbors=min(UMAP_N_NEIGHBORS, len(unknown_embeddings) - 1),
                min_dist=0.0,
                n_components=min(UMAP_N_COMPONENTS_CLUSTER, len(unknown_embeddings) - 1),
                random_state=42,
                metric="cosine",
            )
            reduced = reducer.fit_transform(unknown_embeddings)

            print(f"⚙️   Running HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE})...")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=MIN_CLUSTER_SIZE,
                min_samples=MIN_SAMPLES,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            cluster_labels = clusterer.fit_predict(reduced)

            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = int((cluster_labels == -1).sum())
            print(f"✅  HDBSCAN found {n_clusters} new topic cluster(s) | {n_noise} noise points")

            return cluster_labels

        except ImportError as e:
            print(f"⚠️   Clustering dependencies not installed: {e}")
            return np.array([-1] * len(unknown_embeddings))

    def detect(self, texts: list[str]) -> dict:
        """
        Full detection pipeline: flag unknowns → cluster → return results.

        Returns:
            {
                "flagged_records":    List of dicts for unknown complaints,
                "cluster_labels":     HDBSCAN labels for unknown complaints,
                "n_new_clusters":     Number of new topic clusters found,
                "known_records":      List of dicts for known complaints,
                "all_embeddings":     All embeddings (for visualization),
            }
        """
        from src.utils.embeddings import embed_texts

        print("\n🔍  Running New Topic Detection Pipeline...")
        print("=" * 55)

        embeddings = embed_texts(texts, show_progress=True)
        results = self.flag_unknown_complaints(texts, embeddings=embeddings)

        cluster_labels = np.array([-1] * len(results["unknown"]))
        n_new_clusters = 0

        if len(results["unknown"]) >= MIN_CLUSTER_SIZE:
            cluster_labels = self.cluster_unknown_complaints(results["unknown_embeddings"])
            n_new_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        return {
            "flagged_records": results["unknown"],
            "cluster_labels": cluster_labels,
            "n_new_clusters": n_new_clusters,
            "known_records": results["known"],
            "all_embeddings": embeddings,
            "unknown_embeddings": results["unknown_embeddings"],
        }
