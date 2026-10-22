"""Static report figures (PNG) for the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_gains(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], out: Path, mark: float = 0.5
) -> None:
    """Cumulative gains: share of churners reached vs. share of users contacted."""
    colors = [BLUE, ORANGE]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(
        [0, 100], [0, 100], color=INK_MUTED, linewidth=1, linestyle="--", label="Random selection"
    )
    offsets = [(-34, 8), (8, -16)]  # first series labelled above-left, second below-right
    for (name, (x, y)), color, offset in zip(curves.items(), colors, offsets, strict=False):
        ax.plot(x * 100, y * 100, color=color, linewidth=2, label=name)
        at = y[np.searchsorted(x, mark)] * 100
        ax.scatter([mark * 100], [at], s=50, color=color, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.annotate(
            f"{at:.0f}%",
            (mark * 100, at),
            textcoords="offset points",
            xytext=offset,
            fontsize=9,
            color=INK,
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    _style(
        ax,
        "Churners reached by contacting the riskiest users first (test period)",
        "Users contacted, ranked by predicted risk (%)",
        "Churners reached (%)",
    )
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_importance(importance: pd.DataFrame, out: Path, top_n: int = 12) -> None:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.barh(top["feature"], top["importance"], color=BLUE, height=0.6)
    for y, v in enumerate(top["importance"]):
        ax.text(v, y, f"  {v:.3f}", va="center", fontsize=9, color=INK)
    _style(
        ax,
        "What predicts churn (permutation importance, test period)",
        "Drop in ROC-AUC when the feature is shuffled",
        "",
    )
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_churn_by_recency(df: pd.DataFrame, out: Path) -> None:
    """Churn rate by days since last activity: the 'silence' signal, before any model."""
    bins = [0, 1, 3, 7, 14, 21, 28]
    labels = ["1", "2–3", "4–7", "8–14", "15–21", "22–28"]
    bucket = pd.cut(df["days_since_last_active"], bins=bins, labels=labels)
    rates = df.groupby(bucket, observed=False)["churned"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.bar(rates.index.astype(str), rates["mean"] * 100, color=BLUE, width=0.6)
    for i, rate in enumerate(rates["mean"]):
        ax.text(i, rate * 100 + 1.5, f"{rate:.0%}", ha="center", fontsize=9, color=INK)
    _style(
        ax,
        "Churn rate by days since last activity (regular users, all snapshots)",
        "Days since the user was last active, at the snapshot date",
        "Churn rate (%)",
    )
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_churn_by_engagement(df: pd.DataFrame, out: Path, active_col: str, cutoff: int) -> None:
    """Churn rate by active days in the window, across everyone seen in the data."""
    bins = [0, 1, 2, 3, 5, 10, 28]
    labels = ["1", "2", "3", "4–5", "6–10", "11–28"]
    bucket = pd.cut(df[active_col], bins=bins, labels=labels)
    rates = df.groupby(bucket, observed=False)["churned"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    colors = [INK_MUTED if int(lbl.split("–")[0]) < cutoff else BLUE for lbl in labels]
    ax.bar(labels, rates["mean"] * 100, color=colors, width=0.6)
    for i, (rate, n) in enumerate(zip(rates["mean"], rates["size"], strict=True)):
        ax.text(i, rate * 100 + 1.5, f"{rate:.0%}", ha="center", fontsize=9, color=INK)
        ax.text(i, 2, f"n={n:,}", ha="center", fontsize=8, color=SURFACE)
    _style(
        ax,
        "Churn rate by how often the user was active (all users, all snapshots)",
        f"Active days in the 28 days before the snapshot (blue = modelled, {cutoff}+ days)",
        "Churn rate (%)",
    )
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
