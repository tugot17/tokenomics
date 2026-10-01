"""Plot a scoring sweep: decisions/s, input tokens/s and burst latency."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter


def plot_score_benchmark(results_dir, output_image):
    path = Path(results_dir)
    metadata = json.loads((path / "metadata.json").read_text())
    rows = json.loads((path / "summary.json").read_text())
    lengths = sorted({r["query_tokens"] for r in rows})
    items = sorted({r["items"] for r in rows})
    metrics = [("decisions_per_second", "Decisions / second"),
               ("input_tokens_per_second", "Processed input tokens / second"),
               ("burst_latency_ms", "Batch completion time (ms)")]
    fig, axes = plt.subplots(3, len(lengths), figsize=(5 * len(lengths), 9), squeeze=False)
    for col, length in enumerate(lengths):
        for item in items:
            points = sorted((r for r in rows if r["query_tokens"] == length and r["items"] == item),
                            key=lambda r: r["concurrency"])
            x = [r["concurrency"] for r in points]
            for row, (metric, label) in enumerate(metrics):
                ax = axes[row, col]
                mean = np.array([r[metric]["mean"] for r in points])
                std = np.array([r[metric]["std"] for r in points])
                line, = ax.plot(x, mean, "o-", label=f"{item} items/request")
                ax.fill_between(x, np.maximum(0, mean - std), mean + std, color=line.get_color(), alpha=.15)
                ax.set_xscale("log", base=2)
                ax.set_xticks(x, labels=x)
                ax.set_ylabel(label)
                ax.set_xlabel("Concurrent requests")
                ax.yaxis.set_major_formatter(EngFormatter(sep=""))
                ax.grid(alpha=.2)
        for ax in axes[:, col]:
            ax.set_ylim(bottom=0)
        axes[0, col].set_title(f"{length:,} shared-query tokens")
    args = metadata["arguments"]
    fig.suptitle(f"{args['model']} · {args['mode'].upper()}")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.5, .965), ncol=min(len(items), 4), frameon=False)
    fig.text(.5, .025, f"Mean ± sample SD, {args['num_runs']} rounds · end-to-end bursts\n"
             "Input tokens use server-reported usage; MIS counts the shared query once per request.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .93))
    fig.savefig(output_image, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("output_image")
    args = parser.parse_args()
    plot_score_benchmark(args.results_dir, args.output_image)


if __name__ == "__main__":
    main()
