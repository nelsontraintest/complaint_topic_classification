"""
predict.py
──────────
Load the trained SetFit model and predict the topic of new complaints.

Usage (PyCharm):  Run this file directly for an interactive demo.
    python src/classifier/predict.py

Or import predict_topic() in your own code:
    from src.classifier.predict import predict_topic, predict_batch
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent

# Global variables to hold the model in memory
_model = None
_le = None
_embedder = None


def load_model():
    """Loads the stable classifier and embedder into memory."""
    global _model, _le, _embedder

    model_dir = ROOT / "models" / "stable_classifier"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found at {model_dir}. Please run training first.")

    # Load the Logistic Regression head
    _model = joblib.load(model_dir / "classifier_head.joblib")
    # Load the label encoder (Topic names)
    _le = joblib.load(model_dir / "label_encoder.joblib")
    # Load the Sentence Transformer (The Brain)
    _embedder = SentenceTransformer(str(model_dir / "embedding_model"))

    return _model, _le, _embedder



def predict_topic(complaint: str) -> dict:
    """
    Predict the topic of a single complaint.

    Returns:
        {
            "complaint": str,
            "predicted_topic": str,
            "confidence": float,
            "is_unknown": bool  (True if confidence < threshold)
        }
    """
    model = load_model()
    proba = model.predict_proba([complaint])[0]

    label_names = list(_label_map.get("label2id", {}).keys()) if _label_map else list(range(len(proba)))
    id2label = _label_map.get("id2label", {str(i): str(i) for i in range(len(proba))})

    best_idx = int(np.argmax(proba))
    best_conf = float(proba[best_idx])
    predicted_topic = id2label.get(str(best_idx), str(best_idx))

    return {
        "complaint": complaint,
        "predicted_topic": predicted_topic,
        "confidence": round(best_conf, 4),
        "is_unknown": best_conf < CONFIDENCE_THRESHOLD,
        "all_scores": {
            id2label.get(str(i), str(i)): round(float(p), 4)
            for i, p in enumerate(proba)
        },
    }


def predict_batch(texts, threshold=0.4, show_progress=True):
    """
    Predicts topics for a list of texts.
    If the confidence is below the threshold, it marks it as 'unknown'.
    """
    global _model, _le, _embedder
    if _model is None:
        load_model()

    # 1. Convert texts to embeddings
    if show_progress:
        X = []
        for t in tqdm(texts, desc="Embedding texts"):
            X.append(_embedder.encode(t))
        X = np.array(X)
    else:
        X = _embedder.encode(texts)

    # 2. Get probability distributions
    # predict_proba returns the probability for each class
    probs = _model.predict_proba(X)

    results = []
    for i, prob_dist in enumerate(probs):
        max_prob = np.max(prob_dist)
        pred_idx = np.argmax(prob_dist)
        topic_name = _le.inverse_transform([pred_idx])[0]

        # 3. New Topic Detection Logic:
        # If the highest probability is too low, we flag it as unknown
        if max_prob < threshold:
            results.append({
                "complaint": texts[i],
                "predicted_topic": None,
                "confidence": float(max_prob),
                "is_unknown": True
            })
        else:
            results.append({
                "complaint": texts[i],
                "predicted_topic": topic_name,
                "confidence": float(max_prob),
                "is_unknown": False
            })

    return results


# ─── Interactive Demo ──────────────────────────────────────────────────────────
DEMO_COMPLAINTS = [
    "I can't log into the mobile banking app. It keeps crashing on startup.",
    "My credit card was declined at the restaurant even though I have sufficient credit.",
    "The ETC card wasn't recognized at the highway toll booth today.",
    "I tried to pay with QR code at the supermarket but the payment failed and money was deducted.",
    "My face recognition login no longer works on the banking app after the update.",
    "ATM swallowed my debit card and didn't give me the cash I requested.",
]


def main():
    print("=" * 65)
    print("  Complaint Classifier — Prediction Demo")
    print("=" * 65)
    load_model()

    from rich.console import Console
    from rich.table import Table
    console = Console()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Complaint (truncated)", style="white", width=45)
    table.add_column("Predicted Topic", style="green", width=25)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Status", justify="center", width=10)

    for complaint in DEMO_COMPLAINTS:
        result = predict_topic(complaint)
        status = "⚠️ UNKNOWN" if result["is_unknown"] else "✅ Known"
        conf_color = "red" if result["is_unknown"] else "green"
        table.add_row(
            complaint[:45] + "..." if len(complaint) > 45 else complaint,
            result["predicted_topic"],
            f"[{conf_color}]{result['confidence']:.2%}[/{conf_color}]",
            status,
        )

    console.print(table)
    print(f"\n⚙️   Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}  (below = flagged as unknown)")


if __name__ == "__main__":
    main()
