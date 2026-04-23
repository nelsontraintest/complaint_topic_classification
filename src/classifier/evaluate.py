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
import joblib
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent

def evaluate():
    model_dir = ROOT / "models" / "stable_classifier"
    test_path = ROOT / "data" / "test_split.csv"

    # 1. Load Model and Data
    clf = joblib.load(model_dir / "classifier_head.joblib")
    le = joblib.load(model_dir / "label_encoder.joblib")
    embedder = SentenceTransformer(str(model_dir / "embedding_model"))
    df_test = pd.read_csv(test_path)

    # 2. Generate Predictions
    X_test = embedder.encode(df_test['complaint_text'].tolist())
    y_test = le.transform(df_test['topic'])
    y_pred = clf.predict(X_test)

    # 3. Compute Metrics
    report = classification_report(le.inverse_transform(y_test),
                                   le.inverse_transform(y_pred),
                                   output_dict=True)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
        "n_test_samples": len(y_test),
        "n_topics": len(le.classes_)
    }

    # 4. Save Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(ROOT / "data" / "confusion_matrix.png")

    # 5. Save Metrics JSON
    with open(ROOT / "data" / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    evaluate()
