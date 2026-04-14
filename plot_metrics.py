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

    # All metrics to plot: (key, y-axis label, subplot title, include_val)
    panels = [
        ("loss",           "Loss",        "Total Loss",                 True),
        ("reinforce_loss", "Loss",        "REINFORCE Loss",             True),
        ("ce_loss",        "Loss",        "Cross-Entropy Loss",         True),
        ("mean_reward",    "Reward",      "Mean Feature-Matching Reward", True),
        ("entropy",        "Entropy/NLL", "Entropy Proxy (per-token NLL)", True),
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
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    for ax, (key, ylabel, title, include_val) in zip(axes_flat, panels):
        # 1. Plot train data
        valid_train = [(m["step"], m[key]) for m in history if key in m]
        if not valid_train:
            ax.set_title(f"{title}\n(no data)", fontsize=11)
            ax.set_visible(True)
            continue
        train_steps, train_values = zip(*valid_train)

        color_train = "tab:blue"
        color_val = "tab:orange"

        if alpha > 0.0:
            # Plot raw train values as a faint background
            ax.plot(train_steps, train_values, color=color_train, alpha=0.15, linewidth=0.6)
            smoothed = ema_smooth(list(train_values), alpha)
            ax.plot(
                train_steps, smoothed,
                color=color_train, linewidth=1.5,
                label="Train (EMA)",
            )
        else:
            ax.plot(train_steps, train_values, color=color_train, linewidth=1.0, label="Train")

        # 2. Plot validation data (if present and requested)
        val_key = f"val_{key}"
        valid_val = [(m["step"], m[val_key]) for m in history if val_key in m]
        if include_val and valid_val:
            val_steps, val_values = zip(*valid_val)
            # Validation is usually sparse, so we use markers + lines
            ax.plot(
                val_steps, val_values,
                color=color_val, marker="o", markersize=4,
                linewidth=1.2, label="Val",
            )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)
        
        # Only show legend if we have two types of data or if smoothing is on
        if (include_val and valid_val) or alpha > 0.0:
            ax.legend(fontsize=8)

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
