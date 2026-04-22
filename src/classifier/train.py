"""
train.py
─────────
Fine-tune a SetFit classifier on the labeled complaint dataset.

SetFit (Sentence Fine-tuning) uses contrastive learning on sentence pairs
to adapt a sentence transformer to your classification task with very few examples.
It is state-of-the-art for few-shot text classification.

Usage (PyCharm):  Run this file directly, or via terminal:
    python src/classifier/train.py

Output:
    models/setfit_classifier/    (saved SetFit model)
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

DATA_PATH = ROOT / CONFIG["data"]["labeled_dataset"]
MODEL_SAVE_PATH = ROOT / CONFIG["classifier"]["model_save_path"]
NUM_EPOCHS = CONFIG["classifier"]["num_epochs"]
NUM_ITERATIONS = CONFIG["classifier"]["num_iterations"]


def load_data() -> tuple[list[str], list[str]]:
    """Load labeled complaints from CSV."""
    texts, labels = [], []
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["complaint_text"])
            labels.append(row["topic"])
    return texts, labels


def build_hf_dataset(texts: list[str], labels: list[str]) -> tuple[Dataset, Dataset, LabelEncoder, dict]:
    """Convert to HuggingFace Dataset format required by SetFit."""
    le = LabelEncoder()
    int_labels = le.fit_transform(labels).tolist()
    label_names = le.classes_.tolist()
    label2id = {name: i for i, name in enumerate(label_names)}

    X_train, X_test, y_train, y_test = train_test_split(
        texts, int_labels, test_size=0.15, random_state=42, stratify=int_labels
    )

    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    test_ds = Dataset.from_dict({"text": X_test, "label": y_test})

    return train_ds, test_ds, le, label2id


def train():
    print("=" * 60)
    print("  Complaint Classifier — SetFit Training")
    print("=" * 60)

    # ── Load Data ──────────────────────────────────────────────────────────────
    print(f"\n📂  Loading data from: {DATA_PATH}")
    texts, labels = load_data()
    unique_labels = sorted(set(labels))
    print(f"   Records : {len(texts):,}")
    print(f"   Topics  : {len(unique_labels)}")

    # ── Prepare Dataset ────────────────────────────────────────────────────────
    train_ds, test_ds, le, label2id = build_hf_dataset(texts, labels)
    print(f"   Train   : {len(train_ds):,} samples")
    print(f"   Test    : {len(test_ds):,} samples")

    # ── Initialize SetFit Model ────────────────────────────────────────────────
    embedding_model = CONFIG["embeddings"]["model"]
    print(f"\n🤖  Initializing SetFit with: {embedding_model}")
    model = SetFitModel.from_pretrained(
        f"sentence-transformers/{embedding_model}",
        labels=le.classes_.tolist(),
    )

    # ── Training Arguments ─────────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir=str(MODEL_SAVE_PATH),
        num_epochs=NUM_EPOCHS,
        num_iterations=NUM_ITERATIONS,
        batch_size=32,
        seed=42,
        sampling_strategy="oversampling",
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        metric="accuracy",
    )

    print("\n🚀  Starting SetFit training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n⏱️   Training completed in {elapsed:.1f}s")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\n📊  Evaluating on test set...")
    metrics = trainer.evaluate()
    print(f"   Test Accuracy: {metrics.get('accuracy', metrics):.4f}")

    # ── Save Model ─────────────────────────────────────────────────────────────
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(MODEL_SAVE_PATH))

    # Save label encoder mapping for inference
    label_map = {"label2id": label2id, "id2label": {str(v): k for k, v in label2id.items()}}
    with open(MODEL_SAVE_PATH / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n💾  Model saved to: {MODEL_SAVE_PATH}")
    print("\n✅  Training complete!")

    # ── Detailed Report ────────────────────────────────────────────────────────
    from sklearn.metrics import classification_report
    test_texts = test_ds["text"]
    test_labels_true = [le.classes_[i] for i in test_ds["label"]]
    preds_int = model.predict(test_texts)
    preds_labels = [le.classes_[int(p)] for p in preds_int]

    print("\n" + "=" * 60)
    print("  Classification Report (Test Set)")
    print("=" * 60)
    print(classification_report(test_labels_true, preds_labels, digits=3))

    return model, le


if __name__ == "__main__":
    train()
