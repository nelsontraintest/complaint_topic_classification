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

import pandas as pd
import yaml
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

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
    # 1. Load Data
    data_path = ROOT / CONFIG["data"]["labeled_dataset"]
    df = pd.read_csv(data_path)

    print(f"📂 Loading {len(df)} records for training...")

    # 2. Encode Labels (Topics to Numbers)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['topic'])

    # 3. Split Data
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])

    # 4. Load Embedding Model
    print("🤖 Encoding text into vectors (this may take a minute)...")
    model_name = "all-MiniLM-L6-v2"
    embedder = SentenceTransformer(model_name)

    X_train = embedder.encode(train_df['complaint_text'].tolist(), show_progress_bar=True)
    y_train = train_df['label'].values

    # 5. Train Classifier (Logistic Regression is excellent for embeddings)
    print("🧠 Training classifier head...")
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    clf.fit(X_train, y_train)

    # 6. Save everything to the 'models' folder
    model_dir = ROOT / "models" / "stable_classifier"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, model_dir / "classifier_head.joblib")
    joblib.dump(le, model_dir / "label_encoder.joblib")
    embedder.save(str(model_dir / "embedding_model"))

    # Save test data temporarily for the evaluation script
    test_df.to_csv(ROOT / "data" / "test_split.csv", index=False)

    print(f"✅ Model saved to {model_dir}")
    return clf, le


if __name__ == "__main__":
    train()
