"""
Plot training metrics saved by train.py.

Reads a ``metrics.pkl`` file produced by ``train.py`` (a list of per-step
dicts with keys ``step``, ``loss``, ``reinforce_loss``, ``ce_loss``,
``mean_reward``, and ``entropy``) and saves a multi-panel figure.

Usage
-----
    # Default: reads output/metrics.pkl, saves output/metrics.png
    python plot_metrics.py

    # Custom paths
    python plot_metrics.py --metrics_file run1/metrics.pkl --output run1/metrics.png

    # Show interactively instead of saving
    python plot_metrics.py --show

    # Apply a smoothing window (exponential moving average)
    python plot_metrics.py --smooth 50
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List


def load_history(path: str) -> List[dict]:
    with open(path, "rb") as fh:
        history = pickle.load(fh)
    if not history:
        raise ValueError(f"Metrics file is empty: {path}")
    return history


def ema_smooth(values: list, alpha: float) -> list:
    """Exponential moving average smoothing."""
    smoothed = []
    s = values[0]
    for v in values:
        s = alpha * s + (1.0 - alpha) * v
        smoothed.append(s)
    return smoothed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot EBP training metrics from a metrics.pkl file."
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        default="output/metrics.pkl",
        help="Path to the metrics pickle file (default: output/metrics.pkl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to save the figure (e.g. output/metrics.png). "
            "Defaults to <metrics_file parent>/metrics.png. "
            "Ignored when --show is set."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively instead of saving to disk.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        metavar="W",
        help=(
            "Apply an exponential moving average with this window size "
            "(0 = no smoothing, default). Raw values are plotted in the "
            "background for reference."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Figure DPI for saved images (default: 120).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "matplotlib is required to plot metrics. "
            "Install it with: pip install matplotlib"
        )

    history = load_history(args.metrics_file)

    # All metrics to plot: (key, y-axis label, subplot title)
    panels = [
        ("loss",           "Loss",        "Total Loss"),
        ("reinforce_loss", "Loss",        "REINFORCE Loss"),
        ("ce_loss",        "Loss",        "Cross-Entropy Loss"),
        ("mean_reward",    "Reward",      "Mean Feature-Matching Reward"),
        ("entropy",        "Entropy/NLL", "Entropy Proxy (per-token NLL)"),
    ]

    # Determine EMA alpha from the requested window size.
    # alpha ≈ 1 - 2/(W+1)  so that the effective window ≈ W steps.
    if args.smooth > 1:
        alpha = 1.0 - 2.0 / (args.smooth + 1)
        alpha = max(0.0, min(alpha, 0.9999))
    else:
        alpha = 0.0  # no smoothing

    ncols = 3
    nrows = -(-len(panels) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten()

    for ax, (key, ylabel, title) in zip(axes_flat, panels):
        missing_steps = [m["step"] for m in history if key not in m]
        if missing_steps:
            import warnings
            warnings.warn(
                f"Metric '{key}' is missing from {len(missing_steps)} record(s) "
                f"(steps {missing_steps[:5]}{'...' if len(missing_steps) > 5 else ''}). "
                "Those steps will be skipped.",
                stacklevel=2,
            )
        valid = [(m["step"], m[key]) for m in history if key in m]
        if not valid:
            ax.set_title(f"{title}\n(no data)", fontsize=11)
            ax.set_visible(True)
            continue
        plot_steps, values = zip(*valid)

        if alpha > 0.0:
            # Plot raw values as a faint background
            ax.plot(plot_steps, values, color="tab:blue", alpha=0.2, linewidth=0.8)
            smoothed = ema_smooth(list(values), alpha)
            ax.plot(
                plot_steps, smoothed,
                color="tab:blue", linewidth=1.5,
                label=f"EMA (α={alpha:.3f})",
            )
            ax.legend(fontsize=8)
        else:
            ax.plot(plot_steps, values, color="tab:blue", linewidth=1.0)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)

    # Hide any unused subplot panels
    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    fig.suptitle("EBP Training Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if args.show:
        plt.show()
    else:
        out_path = args.output
        if out_path is None:
            out_path = str(Path(args.metrics_file).with_name("metrics.png"))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
