"""Shared matplotlib styling so every chart reads as one system."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import COLORS, FIGURES


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": COLORS["surface"],
        "axes.facecolor": COLORS["surface"],
        "axes.edgecolor": COLORS["axis"],
        "axes.labelcolor": COLORS["ink2"],
        "axes.titlecolor": COLORS["ink"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "xtick.labelcolor": COLORS["ink2"],
        "ytick.labelcolor": COLORS["ink2"],
        "legend.frameon": False,
        "legend.labelcolor": COLORS["ink2"],
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "lines.linewidth": 2,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def titled(ax, title: str, takeaway: str = "") -> None:
    """Bold title plus a grey takeaway line, so every chart states its point."""
    ax.set_title(title, pad=24 if takeaway else 8)
    if takeaway:
        ax.text(0, 1.02, takeaway, transform=ax.transAxes, fontsize=9.5,
                color=COLORS["ink2"], va="bottom", ha="left")


def save(fig, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def fig_titled(fig, title: str, takeaway: str = "") -> None:
    """Figure-level title + takeaway for multi-panel charts (placed clear of panel titles)."""
    fig.suptitle(title, x=0.01, y=1.075, ha="left", va="bottom", fontsize=13, fontweight="bold", color="#0b0b0b")
    if takeaway:
        fig.text(0.01, 1.02, takeaway, fontsize=9.5, color="#52514e", ha="left", va="bottom")
