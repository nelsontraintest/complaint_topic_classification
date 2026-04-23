"""
run_training.py
───────────────
Full training workflow — run this script directly in PyCharm.

What this does:
1. Generates the labeled dataset (if not yet done)
2. Fine-tunes the SetFit classifier
3. Prints full classification report
4. Saves confusion matrix as PNG
5. Saves eval metrics to JSON

Usage (PyCharm):  Right-click → Run 'run_training'
"""


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from rich.console import Console
console = Console()

with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

LABELED_PATH = ROOT / CONFIG["data"]["labeled_dataset"]


def main():
    console.print("\n[bold green]═══ Complaint Classifier — Training Workflow ═══[/bold green]\n")

    # ── Step 1: Generate Data (if needed) ────────────────────────────────────
    if not LABELED_PATH.exists():
        console.print("[yellow]📊  Labeled dataset not found. Generating now...[/yellow]")
        from src.data_generation.generate_dataset import main as gen_main
        gen_main()
    else:
        import pandas as pd
        df = pd.read_csv(LABELED_PATH)
        console.print(f"[green]✅  Labeled dataset found: {len(df):,} records[/green]")

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 1/3: Training SetFit classifier...[/bold cyan]")
    from src.classifier.train import train
    clf, le = train()

    # ── Step 3: Evaluate ──────────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 2/3: Evaluating classifier...[/bold cyan]")
    from src.classifier.evaluate import evaluate
    metrics = evaluate()

    # ── Step 4: Compute & Save Centroids ──────────────────────────────────────
    console.print("\n[bold cyan]Step 3/3: Computing topic centroids for new topic detection...[/bold cyan]")
    from src.detector.new_topic_detector import NewTopicDetector
    import csv

    texts, labels = [], []
    with open(LABELED_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["complaint_text"])
            labels.append(row["topic"])

    detector = NewTopicDetector()
    detector.load_centroids_from_data(texts, labels)
    centroids_path = ROOT / "models" / "centroids"
    detector.save_centroids(centroids_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    from rich.table import Table, box
    console.print("\n")
    console.rule("[bold green]Training Complete ✅[/bold green]")

    table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
    table.add_column("Metric", style="white", width=25)
    table.add_column("Value", justify="right", width=15)
    table.add_row("Accuracy", f"{metrics['accuracy']:.1%}")
    table.add_row("F1 (macro)", f"{metrics['f1_macro']:.4f}")
    table.add_row("F1 (weighted)", f"{metrics['f1_weighted']:.4f}")
    table.add_row("Test Samples", f"{metrics['n_test_samples']:,}")
    table.add_row("Topics", str(metrics["n_topics"]))
    console.print(table)

    console.print(f"\n📁  Model saved   : [cyan]models/setfit_classifier/[/cyan]")
    console.print(f"📁  Centroids     : [cyan]models/centroids/[/cyan]")
    console.print(f"📁  Confusion mat : [cyan]data/confusion_matrix.png[/cyan]")
    console.print(f"📁  Metrics JSON  : [cyan]data/eval_metrics.json[/cyan]")


if __name__ == "__main__":
    main()
