"""
daily_pipeline.py
─────────────────
End-to-end daily complaint ingestion and classification pipeline.

Workflow:
1. Load new complaints from a CSV file
2. Classify each complaint using the trained SetFit model
3. Flag low-confidence predictions as potentially unknown topics
4. Save results (classified + flagged) to output CSVs
5. If enough unknown complaints have accumulated, run new topic detection

Usage (PyCharm):  Run this file directly:
    python src/pipeline/daily_pipeline.py --input data/complaints_with_new_topics.csv

Or import and call run_pipeline() in your code.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

FLAGGED_OUTPUT = ROOT / CONFIG["data"]["flagged_output"]
CONFIDENCE_THRESHOLD = CONFIG["classifier"]["confidence_threshold"]
MIN_CLUSTER_SIZE = CONFIG["detector"]["min_cluster_size"]


def load_complaints_csv(path: Path) -> list[dict]:
    """Load complaints from a CSV file. Expects a 'complaint_text' column."""
    records = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def classify_batch(texts: list[str]) -> list[dict]:
    """Classify a batch using the trained model."""
    from src.classifier.predict import predict_batch, load_model
    load_model()
    return predict_batch(texts, show_progress=True)


def save_results(records: list[dict], predictions: list[dict], output_path: Path) -> None:
    """Save classified complaints to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys()) + ["predicted_topic", "confidence", "is_unknown"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record, pred in zip(records, predictions):
            row = dict(record)
            row["predicted_topic"] = pred["predicted_topic"]
            row["confidence"] = pred["confidence"]
            row["is_unknown"] = pred["is_unknown"]
            writer.writerow(row)


def run_pipeline(
    input_path: Path,
    run_detection: bool = True,
    visualize: bool = True,
) -> dict:
    """
    Run the full daily pipeline on a CSV of incoming complaints.

    Args:
        input_path:     Path to CSV with 'complaint_text' column.
        run_detection:  Whether to run new topic detection on flagged complaints.
        visualize:      Whether to generate UMAP visualization.

    Returns:
        Summary dict with counts and results.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n" + "=" * 65)
    print(f"  Daily Complaint Pipeline  [{timestamp}]")
    print("=" * 65)

    # ── Load Complaints ────────────────────────────────────────────────────────
    print(f"\n📂  Loading complaints: {input_path}")
    records = load_complaints_csv(input_path)
    texts = [r["complaint_text"] for r in records]
    print(f"   Loaded {len(texts):,} complaints")

    # ── Classify ───────────────────────────────────────────────────────────────
    print("\n🔮  Classifying complaints...")
    predictions = classify_batch(texts)

    known_preds = [p for p in predictions if not p["is_unknown"]]
    unknown_preds = [p for p in predictions if p["is_unknown"]]
    unknown_texts = [p["complaint"] for p in unknown_preds]

    print(f"\n📊  Classification Summary:")
    print(f"   → Classified (known topics) : {len(known_preds):,} ({len(known_preds)/len(predictions):.1%})")
    print(f"   → Flagged as unknown        : {len(unknown_preds):,} ({len(unknown_preds)/len(predictions):.1%})")
    print(f"   → Confidence threshold used : {CONFIDENCE_THRESHOLD:.0%}")

    # ── Save Classified Results ────────────────────────────────────────────────
    output_path = ROOT / "data" / f"classified_{timestamp}.csv"
    save_results(records, predictions, output_path)
    print(f"\n💾  Classified results saved: {output_path}")

    # ── Save Flagged Complaints ────────────────────────────────────────────────
    if unknown_preds:
        flagged_records = [records[i] for i, p in enumerate(predictions) if p["is_unknown"]]
        flagged_path = ROOT / "data" / f"flagged_{timestamp}.csv"
        save_results(flagged_records, unknown_preds, flagged_path)
        print(f"⚠️   Flagged complaints saved : {flagged_path}")

    # ── New Topic Detection ────────────────────────────────────────────────────
    detection_results = None
    if run_detection and len(unknown_texts) >= MIN_CLUSTER_SIZE:
        print(f"\n🔍  Running new topic detection on {len(unknown_texts)} flagged complaints...")

        # Load trained centroids
        from src.detector.new_topic_detector import NewTopicDetector
        from src.utils.embeddings import embed_texts

        detector = NewTopicDetector()
        centroids_path = ROOT / "models" / "centroids"

        if centroids_path.exists():
            detector.load_centroids(centroids_path)
        else:
            # Compute centroids from labeled data on the fly
            print("⚙️   Centroids not found — computing from labeled dataset...")
            labeled_path = ROOT / CONFIG["data"]["labeled_dataset"]
            labeled_records = load_complaints_csv(labeled_path)
            labeled_texts = [r["complaint_text"] for r in labeled_records]
            labeled_labels = [r["topic"] for r in labeled_records]
            detector.load_centroids_from_data(labeled_texts, labeled_labels)
            detector.save_centroids(centroids_path)

        detection_results = detector.detect(unknown_texts)

        if detection_results["n_new_clusters"] > 0:
            print(f"\n🆕  Found {detection_results['n_new_clusters']} potential new topic(s)!")

            # Auto-label clusters with Gemma 4
            from src.detector.topic_labeler import label_all_clusters
            new_topic_labels = label_all_clusters(
                unknown_texts,
                detection_results["cluster_labels"],
            )

            # Save new topic suggestions
            suggestions_path = ROOT / "data" / f"new_topics_{timestamp}.json"
            with open(suggestions_path, "w") as f:
                json.dump(new_topic_labels, f, indent=2)
            print(f"\n💾  New topic suggestions saved: {suggestions_path}")

        else:
            print("ℹ️   No new distinct topic clusters detected.")

    # ── Visualization ──────────────────────────────────────────────────────────
    if visualize and len(texts) > 10:
        try:
            print("\n📊  Generating topic space visualization...")
            from src.utils.embeddings import embed_texts
            from src.utils.visualization import reduce_to_2d, plot_topic_space

            emb = embed_texts(texts, show_progress=False)
            emb_2d = reduce_to_2d(emb)
            pred_labels = [p["predicted_topic"] if not p["is_unknown"] else "⚠️ unknown" for p in predictions]

            has_true_topic = "topic" in records[0]
            is_new = [r.get("is_new_topic", "False") == "True" for r in records] if has_true_topic else None

            plot_topic_space(
                emb_2d,
                labels=pred_labels,
                texts=texts,
                title=f"Complaint Topic Space — {timestamp}",
                is_new_topic=is_new,
            )
        except Exception as e:
            print(f"⚠️   Visualization failed: {e}")

    # ── Summary ────────────────────────────────────────────────────────────────
    summary = {
        "timestamp": timestamp,
        "total_complaints": len(texts),
        "classified_known": len(known_preds),
        "flagged_unknown": len(unknown_preds),
        "n_new_clusters": detection_results["n_new_clusters"] if detection_results else 0,
    }

    print("\n" + "=" * 65)
    print("  Pipeline Complete ✅")
    print("=" * 65)
    for k, v in summary.items():
        print(f"  {k:<25}: {v}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Daily Complaint Classification Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT / CONFIG["data"]["new_topic_dataset"]),
        help="Path to input CSV with complaint_text column",
    )
    parser.add_argument("--no-detection", action="store_true", help="Skip new topic detection")
    parser.add_argument("--no-viz", action="store_true", help="Skip UMAP visualization")
    args = parser.parse_args()

    run_pipeline(
        input_path=Path(args.input),
        run_detection=not args.no_detection,
        visualize=not args.no_viz,
    )


if __name__ == "__main__":
    main()
