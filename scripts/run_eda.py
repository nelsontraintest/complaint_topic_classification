"""
run_eda.py
──────────
Exploratory Data Analysis — run this script directly in PyCharm.

What this does:
1. Prints dataset statistics and class distribution table
2. Shows 3 sample complaints per topic
3. Analyzes text length distribution
4. Saves summary stats to CSV
5. Generates interactive Plotly bar chart of class distribution

Usage (PyCharm):  Right-click → Run 'run_eda'
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
import plotly.express as px
from rich.console import Console
from rich.table import Table
from rich import box

with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

console = Console()


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def print_overview(df: pd.DataFrame, name: str) -> None:
    console.rule(f"[bold cyan]Dataset: {name}[/bold cyan]")
    console.print(f"  Shape       : [green]{df.shape[0]:,} rows × {df.shape[1]} columns[/green]")
    console.print(f"  Topics      : [green]{df['topic'].nunique()}[/green]")
    console.print(f"  Avg length  : [green]{df['complaint_text'].str.len().mean():.0f} characters[/green]")
    console.print(f"  Min / Max   : [green]{df['complaint_text'].str.len().min()} / {df['complaint_text'].str.len().max()} chars[/green]")


def print_distribution(df: pd.DataFrame) -> None:
    console.rule("[bold cyan]Topic Distribution[/bold cyan]")
    counts = df["topic"].value_counts()

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Topic", style="white", width=35)
    table.add_column("Count", justify="right", width=8)
    table.add_column("Pct", justify="right", width=8)
    table.add_column("Bar", width=30)

    total = len(df)
    for topic, count in counts.items():
        pct = count / total
        bar = "█" * int(pct * 100)
        is_new = topic in CONFIG.get("new_topics", [])
        style = "yellow" if is_new else "green"
        tag = " 🆕" if is_new else ""
        table.add_row(
            f"[{style}]{topic}{tag}[/{style}]",
            str(count),
            f"{pct:.1%}",
            f"[{style}]{bar}[/{style}]",
        )

    console.print(table)


def print_samples(df: pd.DataFrame, n_per_topic: int = 2) -> None:
    console.rule("[bold cyan]Sample Complaints Per Topic[/bold cyan]")
    for topic in sorted(df["topic"].unique()):
        samples = df[df["topic"] == topic]["complaint_text"].sample(
            min(n_per_topic, len(df[df["topic"] == topic])), random_state=42
        ).tolist()
        console.print(f"\n[bold yellow]{topic}[/bold yellow]")
        for i, s in enumerate(samples, 1):
            console.print(f"  [dim]{i}.[/dim] {s[:120]}{'...' if len(s) > 120 else ''}")


def plot_distribution(df: pd.DataFrame, title: str, output_path: Path) -> None:
    counts = df["topic"].value_counts().reset_index()
    counts.columns = ["topic", "count"]
    counts["type"] = counts["topic"].apply(
        lambda t: "New Topic" if t in CONFIG.get("new_topics", []) else "Known Topic"
    )

    fig = px.bar(
        counts,
        x="count",
        y="topic",
        color="type",
        orientation="h",
        title=title,
        labels={"count": "Number of Complaints", "topic": "Topic"},
        template="plotly_dark",
        color_discrete_map={"Known Topic": "#4ade80", "New Topic": "#facc15"},
        height=600,
    )
    fig.update_layout(
        font=dict(family="Inter, Arial", size=12),
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        yaxis={"categoryorder": "total ascending"},
    )
    fig.write_html(str(output_path))
    print(f"📊  Chart saved: {output_path}")
    import webbrowser
    webbrowser.open(str(output_path))


def main():
    console.print("\n[bold green]═══ Complaint Classification — EDA ═══[/bold green]\n")

    labeled_path = ROOT / CONFIG["data"]["labeled_dataset"]
    new_topic_path = ROOT / CONFIG["data"]["new_topic_dataset"]

    # ── Dataset 1: Labeled ─────────────────────────────────────────────────────
    if labeled_path.exists():
        df1 = load_dataset(labeled_path)
        print_overview(df1, labeled_path.name)
        print_distribution(df1)
        print_samples(df1, n_per_topic=2)
        plot_distribution(df1, "Known Topics — Complaint Distribution", ROOT / "data" / "eda_known_topics.html")
    else:
        console.print(f"[red]⚠️  File not found: {labeled_path}[/red]")
        console.print("[yellow]Run: python src/data_generation/generate_dataset.py[/yellow]")

    # ── Dataset 2: Mixed with New Topics ──────────────────────────────────────
    if new_topic_path.exists():
        df2 = load_dataset(new_topic_path)
        console.print("\n")
        print_overview(df2, new_topic_path.name)
        print_distribution(df2)
        plot_distribution(df2, "Mixed Topics — Known + New Topics", ROOT / "data" / "eda_mixed_topics.html")
    else:
        console.print(f"\n[yellow]⚠️  New-topic dataset not found. Run generate_new_topic_data.py[/yellow]")

    console.print("\n[bold green]✅  EDA complete![/bold green]")


if __name__ == "__main__":
    main()
