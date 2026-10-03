"""Plot generation from saved experiment CSV files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CLASSICAL = ["FCFS", "SSTF", "SCAN", "C-SCAN"]
ORDER = ["uniform", "localized", "directional", "bursty"]
MARKERS = ["o", "s", "^", "D", "x", "+"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_classical_by_workload(summary: pd.DataFrame, figures_dir: Path) -> None:
    classical = summary[summary["method"].isin(CLASSICAL)]
    pivot = classical.pivot(index="workload_type", columns="method", values="mean_total_movement").loc[ORDER]

    ax = pivot.plot(kind="bar", figsize=(7.2, 4.2), color=["#4c4c4c", "#8a8a8a", "#c0c0c0", "#ffffff"], edgecolor="black")
    ax.set_title("Classical Scheduler Movement by Workload Type")
    ax.set_xlabel("Workload type")
    ax.set_ylabel("Mean total head movement (cylinders)")
    ax.legend(title="Scheduler", ncols=2)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "classical_scheduler_comparison.png")
    plt.close()


def plot_selector_comparison(summary: pd.DataFrame, figures_dir: Path) -> None:
    methods = ["FCFS", "SSTF", "SCAN", "C-SCAN", "Learned selector", "Oracle best"]
    selected = summary[summary["method"].isin(methods)]
    pivot = selected.pivot(index="workload_type", columns="method", values="mean_total_movement").loc[ORDER, methods]

    ax = pivot.plot(kind="bar", figsize=(8.0, 4.4), color=["#525252", "#737373", "#969696", "#bdbdbd", "#f0f0f0", "#ffffff"], edgecolor="black")
    ax.set_title("Learned Selector Compared with Fixed Schedulers")
    ax.set_xlabel("Workload type")
    ax.set_ylabel("Mean total head movement (cylinders)")
    ax.legend(title="Method", ncols=2)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "learned_selector_comparison.png")
    plt.close()


def plot_workload_shift(shift: pd.DataFrame, figures_dir: Path) -> None:
    columns = ["FCFS_movement", "SSTF_movement", "SCAN_movement", "C-SCAN_movement", "learned_selector_movement", "oracle_best_movement"]
    labels = ["FCFS", "SSTF", "SCAN", "C-SCAN", "Learned selector", "Oracle best"]
    colors = ["#1a1a1a", "#4d4d4d", "#737373", "#969696", "#000000", "#bdbdbd"]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), "-"]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for column, label, marker, color, linestyle in zip(columns, labels, MARKERS, colors, linestyles):
        ax.plot(
            shift["window_id"],
            shift[column],
            label=label,
            marker=marker,
            color=color,
            linestyle=linestyle,
            linewidth=1.25,
            markersize=4,
        )

    boundary = int(shift["window_id"].max() // 2)
    ax.axvline(boundary + 0.5, color="black", linestyle="--", linewidth=1)
    ax.text(boundary / 2, ax.get_ylim()[1] * 0.95, "Directional phase", ha="center", va="top")
    ax.text(boundary + 1 + boundary / 2, ax.get_ylim()[1] * 0.95, "Uniform phase", ha="center", va="top")
    ax.set_title("Scheduler Behaviour Under a Workload Shift")
    ax.set_xlabel("Window")
    ax.set_ylabel("Total head movement (cylinders)")
    ax.legend(ncols=2)
    plt.tight_layout()
    plt.savefig(figures_dir / "workload_shift_behavior.png")
    plt.close()


def plot_confusion_matrix(confusion: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(confusion.values, cmap="Greys")
    ax.set_title("Decision Tree Confusion Matrix")
    ax.set_xlabel("Predicted scheduler")
    ax.set_ylabel("Actual best scheduler")
    ax.set_xticks(range(len(confusion.columns)), confusion.columns)
    ax.set_yticks(range(len(confusion.index)), confusion.index)

    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            value = int(confusion.iloc[row, col])
            color = "white" if value > confusion.values.max() / 2 else "black"
            ax.text(col, row, value, ha="center", va="center", color=color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(figures_dir / "decision_tree_confusion_matrix.png")
    plt.close()


def generate_figures(results_dir: Path, figures_dir: Path) -> None:
    """Generate all publication figures from saved CSV files."""

    _style()
    summary = pd.read_csv(results_dir / "summary_results.csv")
    shift = pd.read_csv(results_dir / "workload_shift_results.csv")
    confusion = pd.read_csv(results_dir / "confusion_matrix.csv", index_col=0)

    plot_classical_by_workload(summary, figures_dir)
    plot_selector_comparison(summary, figures_dir)
    plot_workload_shift(shift, figures_dir)
    plot_confusion_matrix(confusion, figures_dir)
