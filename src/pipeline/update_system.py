"""
update_system.py
────────────────
Re-trains the classifier to incorporate newly confirmed topics.

After the daily pipeline detects new topics and a human confirms/names them,
this script:
1. Combines the original labeled dataset with labeled new-topic complaints
2. Re-trains the SetFit classifier on the expanded topic set
3. Recomputes and saves new centroid vectors
4. Updates config.yaml with the expanded known_topics list

Usage (PyCharm):  Run this file:
    python src/pipeline/update_system.py \
        --new-data data/flagged_20240101_120000.csv \
        --new-labels '{"biometric_auth_failure": [...cluster ids...]}'

Or use the programmatic API: update_system(new_labeled_records)
"""

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def backup_model(model_path: Path) -> Path:
    """Backup the existing model before overwriting."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = model_path.parent / f"{model_path.name}_backup_{timestamp}"
    if model_path.exists():
        shutil.copytree(str(model_path), str(backup_path))
        print(f"📦  Model backed up to: {backup_path}")
    return backup_path


def merge_datasets(
    original_path: Path,
    new_records: list[dict],
    output_path: Path,
) -> list[dict]:
    """Merge original labeled data with newly labeled complaint records."""
    # Load original
    original_records = []
    with open(original_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            original_records.append({"complaint_text": row["complaint_text"], "topic": row["topic"]})

    # Combine
    all_records = original_records + new_records
    print(f"   Original records : {len(original_records):,}")
    print(f"   New records      : {len(new_records):,}")
    print(f"   Total after merge: {len(all_records):,}")

    # Save merged
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["complaint_text", "topic"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"💾  Merged dataset saved: {output_path}")
    return all_records


def update_known_topics_in_config(new_topics: list[str]) -> None:
    """Add newly confirmed topics to config.yaml's known_topics list."""
    config_path = ROOT / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    existing = set(config.get("known_topics", []))
    added = []
    for topic in new_topics:
        if topic not in existing:
            existing.add(topic)
            added.append(topic)

    config["known_topics"] = sorted(existing)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    if added:
        print(f"✅  Added {len(added)} new topic(s) to config.yaml: {added}")
    else:
        print("ℹ️   No new topics to add to config.yaml.")


def update_system(new_labeled_records: list[dict]) -> dict:
    """
    Full system update: merge data → retrain → recompute centroids → update config.

    Args:
        new_labeled_records: List of {'complaint_text': str, 'topic': str} dicts
                             for newly confirmed complaint topics.

    Returns:
        Summary dict with update details.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n" + "=" * 65)
    print(f"  System Update Pipeline  [{timestamp}]")
    print("=" * 65)

    model_path = ROOT / CONFIG["classifier"]["model_save_path"]
    original_data_path = ROOT / CONFIG["data"]["labeled_dataset"]
    updated_data_path = ROOT / CONFIG["data"]["updated_dataset"]

    new_topics_list = sorted(set(r["topic"] for r in new_labeled_records))
    print(f"\n📋  New topics to incorporate: {new_topics_list}")

    # ── Backup Current Model ───────────────────────────────────────────────────
    backup_model(model_path)

    # ── Merge Datasets ─────────────────────────────────────────────────────────
    print("\n📂  Merging datasets...")
    all_records = merge_datasets(original_data_path, new_labeled_records, updated_data_path)

    # ── Update Config ──────────────────────────────────────────────────────────
    print("\n⚙️   Updating config.yaml...")
    update_known_topics_in_config(new_topics_list)

    # ── Reload Config & Retrain ────────────────────────────────────────────────
    with open(ROOT / "config.yaml") as f:
        updated_config = yaml.safe_load(f)

    print("\n🚀  Re-training classifier on expanded topic set...")

    # Temporarily patch DATA_PATH in train.py to use updated dataset
    import sys
    sys.path.insert(0, str(ROOT))

    # Override the data path used by train.py by swapping the labeled dataset
    original_labeled = original_data_path.read_text()
    shutil.copy(str(updated_data_path), str(original_data_path))

    try:
        from importlib import import_module, reload
        import src.classifier.train as train_module
        reload(train_module)
        train_module.train()
    finally:
        # Restore original labeled data file
        original_data_path.write_text(original_labeled)

    # ── Recompute Centroids ────────────────────────────────────────────────────
    print("\n⚙️   Recomputing topic centroids...")
    from src.detector.new_topic_detector import NewTopicDetector

    detector = NewTopicDetector()
    texts = [r["complaint_text"] for r in all_records]
    labels = [r["topic"] for r in all_records]
    detector.load_centroids_from_data(texts, labels)

    centroids_path = ROOT / "models" / "centroids"
    detector.save_centroids(centroids_path)

    summary = {
        "timestamp": timestamp,
        "new_topics_added": new_topics_list,
        "total_records_after_update": len(all_records),
        "total_topics_after_update": len(set(r["topic"] for r in all_records)),
    }

    print("\n" + "=" * 65)
    print("  ✅  System Update Complete!")
    print("=" * 65)
    for k, v in summary.items():
        print(f"  {k:<35}: {v}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Update classifier with newly confirmed topics")
    parser.add_argument(
        "--new-data",
        type=str,
        required=True,
        help="Path to CSV of newly labeled complaints (complaint_text, topic columns)",
    )
    args = parser.parse_args()

    new_data_path = Path(args.new_data)
    if not new_data_path.exists():
        raise FileNotFoundError(f"New data file not found: {new_data_path}")

    new_records = []
    with open(new_data_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            new_records.append({"complaint_text": row["complaint_text"], "topic": row["topic"]})

    print(f"📂  Loaded {len(new_records)} new labeled records from: {new_data_path}")
    update_system(new_records)


if __name__ == "__main__":
    main()
