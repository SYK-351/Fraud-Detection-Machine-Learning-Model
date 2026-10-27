"""Shared plotting helpers with consistent 300 dpi exports."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_figure(fig, name: str, directory: str | Path = "figures") -> None:
    """Save a finished figure as high-resolution PNG and vector PDF."""
    target=Path(directory); target.mkdir(parents=True,exist_ok=True); fig.tight_layout()
    fig.savefig(target/f"{name}.png",dpi=300,bbox_inches="tight")
    fig.savefig(target/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)
