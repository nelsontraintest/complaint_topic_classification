"""
run_new_topic_demo.py
─────────────────────
New Topic Detection Demo — run this directly in PyCharm.

What this does:
1. Loads the mixed dataset (old + new topics)
2. Uses the trained classifier to predict known topics
3. Flags low-confidence predictions as potentially unknown
4. Runs HDBSCAN to cluster the unknown complaints
5. Uses Gemma 4 (local) to auto-name each new cluster
6. Generates an interactive UMAP visualization (opens in browser)
7. Prints a full summary report

Usage (PyCharm):  Right-click → Run 'run_new_topic_demo'

Prerequisites:
  1. Run scripts/run_training.py first to train the classifier
  2. Run src/data_generation/generate_new_topic_data.py to generate mixed data
  3. For Gemma 4 topic naming: ollama pull gemma3:4b
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

NEW_TOPIC_PATH = ROOT / CONFIG["data"]["new_topic_dataset"]
LABELED_PATH = ROOT / CONFIG["data"]["labeled_dataset"]


def main():
    console.print("\n[bold green]═══ New Topic Detection Demo ═══[/bold green]\n")

    # ── Check Prerequisites ────────────────────────────────────────────────────
    model_path = ROOT / CONFIG["classifier"]["model_save_path"]
    if not model_path.exists():
        console.print("[red]❌  No trained model found![/red]")
        console.print("[yellow]   Run: python scripts/run_training.py[/yellow]")
        return

    if not NEW_TOPIC_PATH.exists():
        console.print("[yellow]⚠️  Mixed dataset not found. Generating now...[/yellow]")
        from src.data_generation.generate_new_topic_data import main as gen_main
        gen_main()

    # ── Load Dataset ───────────────────────────────────────────────────────────
    df = pd.read_csv(NEW_TOPIC_PATH)
    texts = df["complaint_text"].tolist()
    true_topics = df["topic"].tolist() if "topic" in df.columns else None
    is_new = df["is_new_topic"].tolist() if "is_new_topic" in df.columns else None

    console.print(f"📂  Loaded [green]{len(texts):,}[/green] complaints from mixed dataset")
    if true_topics:
        n_new = sum(1 for t in df.get("is_new_topic", []) if str(t) == "True")
        console.print(f"   → Known topic complaints : {len(texts) - n_new}")
        console.print(f"   → New topic complaints   : {n_new}")

    # ── Classify with Current Model ────────────────────────────────────────────
    console.print("\n[bold cyan]Step 1/4: Classifying with trained model...[/bold cyan]")
    from src.classifier.predict import predict_batch, load_model
    load_model()
    predictions = predict_batch(texts, show_progress=True)

    known_preds = [p for p in predictions if not p["is_unknown"]]
    unknown_preds = [p for p in predictions if p["is_unknown"]]
    unknown_texts = [p["complaint"] for p in unknown_preds]
    unknown_indices = [i for i, p in enumerate(predictions) if p["is_unknown"]]

    console.print(f"\n📊  Classification Results:")
    console.print(f"   → Classified as known : [green]{len(known_preds)}[/green] ({len(known_preds)/len(predictions):.1%})")
    console.print(f"   → Flagged as unknown  : [yellow]{len(unknown_preds)}[/yellow] ({len(unknown_preds)/len(predictions):.1%})")

    # ── New Topic Detection ────────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 2/4: Running new topic detection...[/bold cyan]")

    from src.detector.new_topic_detector import NewTopicDetector
    detector = NewTopicDetector()

    centroids_path = ROOT / "models" / "centroids"
    if centroids_path.exists():
        detector.load_centroids(centroids_path)
    else:
        console.print("[yellow]  Computing centroids from labeled data...[/yellow]")
        import csv as csv_mod
        ltexts, llabels = [], []
        with open(LABELED_PATH, encoding="utf-8") as f:
            for row in csv_mod.DictReader(f):
                ltexts.append(row["complaint_text"])
                llabels.append(row["topic"])
        detector.load_centroids_from_data(ltexts, llabels)
        detector.save_centroids(centroids_path)

    detection = detector.detect(unknown_texts)
    cluster_labels = detection["cluster_labels"]
    n_new = detection["n_new_clusters"]

    # ── Auto-Label New Clusters with Gemma 4 ──────────────────────────────────
    console.print(f"\n[bold cyan]Step 3/4: Auto-labeling {n_new} new cluster(s) with Gemma 4...[/bold cyan]")
    from src.detector.topic_labeler import label_all_clusters
    new_topic_info = label_all_clusters(unknown_texts, cluster_labels)

    # ── Visualization ──────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 4/4: Generating UMAP visualization...[/bold cyan]")
    from src.utils.embeddings import embed_texts
    from src.utils.visualization import reduce_to_2d, plot_topic_space

    all_emb = embed_texts(texts, show_progress=False)
    emb_2d = reduce_to_2d(all_emb)
    pred_labels = [
        p["predicted_topic"] if not p["is_unknown"] else "⚠️ unknown"
        for p in predictions
    ]
    is_new_bool = [str(v) == "True" for v in is_new] if is_new is not None else None

    plot_topic_space(
        emb_2d,
        labels=pred_labels,
        texts=texts,
        title="Complaint Space — Known + New Topics (UMAP)",
        is_new_topic=is_new_bool,
        output_path=ROOT / "data" / "umap_new_topic_demo.html",
    )

    # ── Results Summary ────────────────────────────────────────────────────────
    console.rule("\n[bold green]New Topic Detection Results[/bold green]")

    if new_topic_info:
        table = Table(show_header=True, header_style="bold yellow", box=box.ROUNDED)
        table.add_column("Cluster", justify="center", width=8)
        table.add_column("Suggested Topic Name", style="yellow", width=30)
        table.add_column("Count", justify="right", width=8)
        table.add_column("Method", width=18)
        table.add_column("Description", style="dim", width=45)

        for info in new_topic_info:
            table.add_row(
                str(info["cluster_id"]),
                info["topic_name"],
                str(info["sample_count"]),
                info["method"],
                info["description"][:43] + ".." if len(info["description"]) > 43 else info["description"],
            )
        console.print(table)
    else:
        console.print("[yellow]  No distinct new topic clusters found.[/yellow]")

    # ── Accuracy on new topic detection ───────────────────────────────────────
    if is_new is not None:
        true_new_indices = set(i for i, v in enumerate(is_new) if str(v) == "True")
        detected_unknown = set(unknown_indices)
        correctly_flagged = len(true_new_indices & detected_unknown)
        missed = len(true_new_indices - detected_unknown)
        false_alarms = len(detected_unknown - true_new_indices)

        console.print(f"\n[bold]Detection Quality:[/bold]")
        console.print(f"  New complaints correctly flagged : [green]{correctly_flagged}[/green] / {len(true_new_indices)}")
        console.print(f"  New complaints missed            : [red]{missed}[/red]")
        console.print(f"  False positives (known → flagged): [yellow]{false_alarms}[/yellow]")
        recall = correctly_flagged / max(len(true_new_indices), 1)
        console.print(f"  Recall on new topics             : [bold]{recall:.1%}[/bold]")

    console.print("\n[bold green]✅  Demo complete! Open data/umap_new_topic_demo.html in your browser.[/bold green]")


if __name__ == "__main__":
    main()
