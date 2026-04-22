"""
evaluate.py
───────────
Evaluate the trained SetFit classifier on the test set.
Generates a full classification report and saves a confusion matrix PNG.

Usage (PyCharm):  Run this file directly:
    python src/classifier/evaluate.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

DATA_PATH = ROOT / CONFIG["data"]["labeled_dataset"]
MODEL_SAVE_PATH = ROOT / CONFIG["classifier"]["model_save_path"]


def load_data() -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["complaint_text"])
            labels.append(row["topic"])
    return texts, labels


def evaluate():
    print("=" * 65)
    print("  Complaint Classifier — Evaluation")
    print("=" * 65)

    if not MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(
            f"No model found at {MODEL_SAVE_PATH}.\n"
            "Run src/classifier/train.py first."
        )

    # ── Load Data ──────────────────────────────────────────────────────────────
    print(f"\n📂  Loading data: {DATA_PATH}")
    texts, labels = load_data()
    le = LabelEncoder()
    int_labels = le.fit_transform(labels).tolist()

    _, X_test, _, y_test_int = train_test_split(
        texts, int_labels, test_size=0.15, random_state=42, stratify=int_labels
    )
    y_test = [le.classes_[i] for i in y_test_int]
    print(f"   Test samples: {len(X_test):,}")

    # ── Load Model ─────────────────────────────────────────────────────────────
    from setfit import SetFitModel
    print(f"\n⚙️   Loading model from: {MODEL_SAVE_PATH}")
    model = SetFitModel.from_pretrained(str(MODEL_SAVE_PATH))

    label_map_path = MODEL_SAVE_PATH / "label_map.json"
    id2label = {}
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_map = json.load(f)
        id2label = label_map.get("id2label", {})

    # ── Predict ────────────────────────────────────────────────────────────────
    print("\n🔮  Running predictions on test set...")
    from tqdm import tqdm
    preds_int = []
    for text in tqdm(X_test, desc="Predicting"):
        pred = model.predict([text])
        preds_int.append(int(pred[0]))

    y_pred = [id2label.get(str(p), le.classes_[p] if p < len(le.classes_) else str(p)) for p in preds_int]

    # ── Metrics ────────────────────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 65)
    print("  Summary Metrics")
    print("=" * 65)
    print(f"  Accuracy         : {acc:.4f} ({acc:.1%})")
    print(f"  F1 (macro)       : {f1_macro:.4f}")
    print(f"  F1 (weighted)    : {f1_weighted:.4f}")

    print("\n" + "=" * 65)
    print("  Per-Class Classification Report")
    print("=" * 65)
    print(classification_report(y_test, y_pred, digits=3))

    # ── Save Confusion Matrix ──────────────────────────────────────────────────
    print("🖼️   Saving confusion matrix...")
    from src.utils.visualization import plot_confusion_matrix
    cm_path = ROOT / "data" / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, output_path=cm_path)

    # ── Save metrics JSON ──────────────────────────────────────────────────────
    metrics = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "n_test_samples": len(X_test),
        "n_topics": len(set(y_test)),
    }
    metrics_path = ROOT / "data" / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📁  Metrics saved: {metrics_path}")

    print("\n✅  Evaluation complete!")
    return metrics


if __name__ == "__main__":
    evaluate()
