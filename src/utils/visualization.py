"""
visualization.py
────────────────
Interactive UMAP visualizations for the complaint topic space.

Outputs HTML files viewable in any browser (opens automatically).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

_UMAP_CFG = CONFIG["detector"]


def reduce_to_2d(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Reduce high-dimensional embeddings to 2D using UMAP."""
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("Install umap-learn: pip install umap-learn")

    print("⚙️   Running UMAP dimensionality reduction to 2D...")
    reducer = UMAP(
        n_neighbors=_UMAP_CFG["umap_n_neighbors"],
        min_dist=_UMAP_CFG["umap_min_dist"],
        n_components=2,
        random_state=random_state,
        metric="cosine",
    )
    return reducer.fit_transform(embeddings)


def plot_topic_space(
    embeddings_2d: np.ndarray,
    labels: list[str],
    texts: Optional[list[str]] = None,
    title: str = "Complaint Topic Space",
    output_path: Optional[Path] = None,
    highlight_unknowns: bool = False,
    is_new_topic: Optional[list[bool]] = None,
) -> None:
    """
    Create an interactive Plotly scatter plot of the complaint embedding space.

    Args:
        embeddings_2d:    (N, 2) array from UMAP.
        labels:           Topic label for each point.
        texts:            Optional complaint texts for hover tooltips.
        title:            Plot title.
        output_path:      Save path for HTML file (auto-opens if None).
        highlight_unknowns: Color-code unknown/flagged complaints differently.
        is_new_topic:     Boolean mask — True = new/emerging topic.
    """
    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "topic": labels,
    })

    if texts:
        # Truncate for tooltip readability
        df["complaint"] = [t[:120] + "..." if len(t) > 120 else t for t in texts]
    else:
        df["complaint"] = ""

    if is_new_topic is not None:
        df["type"] = ["🆕 New Topic" if n else "Known Topic" for n in is_new_topic]
        symbol_map = {"Known Topic": "circle", "🆕 New Topic": "star"}
        color_col = "type"
    else:
        color_col = "topic"
        symbol_map = None

    hover_data = {"complaint": True, "x": False, "y": False}
    if is_new_topic is not None:
        hover_data["type"] = True

    fig = px.scatter(
        df,
        x="x", y="y",
        color=color_col,
        symbol="type" if is_new_topic is not None else None,
        symbol_map=symbol_map,
        hover_data=hover_data,
        title=title,
        template="plotly_dark",
        width=1100,
        height=750,
        opacity=0.80,
    )

    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
    fig.update_layout(
        title_font_size=20,
        legend_title_text="Topic",
        font=dict(family="Inter, Arial, sans-serif", size=13),
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
    )

    if output_path is None:
        output_path = ROOT / "data" / "topic_space_visualization.html"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"📊  Interactive visualization saved: {output_path}")

    # Auto-open in browser
    import webbrowser
    webbrowser.open(str(output_path))


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    output_path: Optional[Path] = None,
) -> None:
    """Plot and save a confusion matrix as PNG using seaborn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Classifier Confusion Matrix (Normalized)", fontsize=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path is None:
        output_path = ROOT / "data" / "confusion_matrix.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🖼️   Confusion matrix saved: {output_path}")
