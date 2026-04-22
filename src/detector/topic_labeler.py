"""
topic_labeler.py
────────────────
Auto-name newly discovered topic clusters using Google Gemma 4 (local via Ollama).

When the NewTopicDetector finds new clusters, this module takes representative
complaints from each cluster and asks Gemma 4 to suggest a topic name and description.

Falls back to keyword extraction if Ollama is not available.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def get_cluster_samples(
    texts: list[str],
    cluster_labels: np.ndarray,
    cluster_id: int,
    n_samples: int = 8,
) -> list[str]:
    """Get representative complaint samples from a specific cluster."""
    indices = np.where(cluster_labels == cluster_id)[0]
    selected = indices[:n_samples] if len(indices) <= n_samples else \
        np.random.choice(indices, size=n_samples, replace=False)
    return [texts[i] for i in selected]


def label_cluster_with_gemma(samples: list[str]) -> dict:
    """Use Gemma 4 to name the cluster. Falls back to keyword method if unavailable."""
    from src.utils.llm_client import is_ollama_available, name_topic_cluster

    if is_ollama_available():
        print("   🤖  Using Gemma 4 (Ollama) for topic naming...")
        try:
            result = name_topic_cluster(samples)
            result["method"] = "gemma4_ollama"
            return result
        except Exception as e:
            print(f"   ⚠️  Gemma 4 failed: {e}. Falling back to keyword extraction.")

    # ── Fallback: Simple keyword extraction ───────────────────────────────────
    print("   📝  Using keyword extraction fallback...")
    return label_cluster_with_keywords(samples)


def label_cluster_with_keywords(samples: list[str]) -> dict:
    """
    Simple fallback: extract most common meaningful words from cluster samples
    to suggest a topic name without any LLM.
    """
    import re
    from collections import Counter

    STOPWORDS = {
        "i", "my", "the", "a", "an", "is", "are", "was", "were", "has", "have",
        "had", "been", "be", "to", "of", "in", "on", "at", "for", "with", "and",
        "or", "but", "not", "it", "its", "this", "that", "these", "those", "from",
        "by", "as", "so", "if", "do", "did", "does", "can", "could", "will",
        "would", "should", "may", "might", "must", "shall", "your", "our",
        "their", "me", "him", "her", "us", "them", "what", "when", "where",
        "who", "which", "how", "why", "am", "no", "yes", "into", "through",
        "about", "up", "down", "out", "there", "here", "then", "than", "also",
        "even", "just", "very", "too", "only", "more", "most", "some", "any",
        "all", "each", "every", "after", "before", "since", "still", "get",
        "got", "keep", "try", "tried", "please", "dear", "bank", "account",
        "app", "card", "banking", "complaint", "issue", "problem", "works",
        "work", "working", "help", "need", "want", "use", "using", "show",
        "shows", "says", "said", "make", "made", "see", "cannot", "ca",
        "nt", "im", "ive", "dont", "doesnt"
    }

    all_words = []
    for sample in samples:
        words = re.findall(r'\b[a-z]{3,}\b', sample.lower())
        all_words.extend([w for w in words if w not in STOPWORDS])

    top_words = [word for word, _ in Counter(all_words).most_common(4)]
    topic_name = "_".join(top_words[:3]) if top_words else "unknown_cluster"
    description = f"Complaints mentioning: {', '.join(top_words)}"

    return {
        "topic_name": topic_name,
        "description": description,
        "method": "keyword_fallback",
    }


def label_all_clusters(
    flagged_texts: list[str],
    cluster_labels: np.ndarray,
) -> list[dict]:
    """
    Label all discovered clusters.

    Args:
        flagged_texts:   Texts of flagged (unknown) complaints.
        cluster_labels:  HDBSCAN cluster assignments for each flagged complaint.

    Returns:
        List of cluster info dicts:
        {
            "cluster_id": int,
            "topic_name": str,
            "description": str,
            "method": str,
            "sample_count": int,
            "sample_complaints": list[str]
        }
    """
    unique_clusters = [c for c in sorted(set(cluster_labels)) if c != -1]

    if not unique_clusters:
        print("   ℹ️   No new clusters found (all flagged as noise).")
        return []

    print(f"\n🏷️   Labeling {len(unique_clusters)} new topic cluster(s) with Gemma 4...")
    results = []

    for cluster_id in unique_clusters:
        samples = get_cluster_samples(flagged_texts, cluster_labels, cluster_id)
        print(f"\n   Cluster {cluster_id} ({len(np.where(cluster_labels == cluster_id)[0])} complaints):")
        for s in samples[:2]:
            print(f"     → {s[:80]}...")

        label_info = label_cluster_with_gemma(samples)

        results.append({
            "cluster_id": int(cluster_id),
            "topic_name": label_info["topic_name"],
            "description": label_info["description"],
            "method": label_info.get("method", "unknown"),
            "sample_count": int((cluster_labels == cluster_id).sum()),
            "sample_complaints": samples[:3],
        })
        print(f"   ✅  Suggested name: '{label_info['topic_name']}'")
        print(f"       Description  : {label_info['description']}")

    return results
