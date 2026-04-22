"""
predict.py
──────────
Load the trained SetFit model and predict the topic of new complaints.

Usage (PyCharm):  Run this file directly for an interactive demo.
    python src/classifier/predict.py

Or import predict_topic() in your own code:
    from src.classifier.predict import predict_topic, predict_batch
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

MODEL_SAVE_PATH = ROOT / CONFIG["classifier"]["model_save_path"]
CONFIDENCE_THRESHOLD = CONFIG["classifier"]["confidence_threshold"]

# ─── Cached model ─────────────────────────────────────────────────────────────
_model = None
_label_map: dict = {}


def load_model():
    """Load and cache the trained SetFit model."""
    global _model, _label_map
    if _model is not None:
        return _model

    if not MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(
            f"No model found at {MODEL_SAVE_PATH}.\n"
            "Please run src/classifier/train.py first."
        )

    from setfit import SetFitModel
    print(f"⚙️   Loading model from: {MODEL_SAVE_PATH}")
    _model = SetFitModel.from_pretrained(str(MODEL_SAVE_PATH))

    label_map_path = MODEL_SAVE_PATH / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            _label_map = json.load(f)

    print("✅  Model loaded.")
    return _model


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


def predict_batch(complaints: list[str], show_progress: bool = True) -> list[dict]:
    """
    Predict topics for a batch of complaints.

    Returns:
        List of prediction dicts (same format as predict_topic).
    """
    model = load_model()
    id2label = _label_map.get("id2label", {})

    from tqdm import tqdm
    results = []
    iterator = tqdm(complaints, desc="Predicting") if show_progress else complaints

    for complaint in iterator:
        result = predict_topic(complaint)
        results.append(result)

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
