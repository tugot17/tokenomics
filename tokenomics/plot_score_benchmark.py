"""Plot a scoring sweep: decisions/s, input tokens/s and burst latency."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter


def plot_score_benchmark(results_dir, output_image):
    path = Path(results_dir)
    metadata = json.loads((path / "metadata.json").read_text())
    rows = [json.loads(file.read_text()) for file in path.glob("*.json")
            if file.stem.isdigit()]
    if not rows:
        raise ValueError("No measured scoring results found")
    rows.sort(key=lambda row: row["batch_size"])
    x = [row["batch_size"] for row in rows]
    metrics = [("decisions_per_second", "Completed candidate sets / second"),
               ("input_tokens_per_second", "Input tokens / second"),
               ("latency_ms_mean", "Mean successful request latency (ms)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (metric, label) in zip(axes, metrics):
        y = [row[metric] if row[metric] is not None else float("nan") for row in rows]
        ax.plot(x, y, "o-")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, labels=x)
        ax.set_xlabel("Concurrent candidate sets")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(EngFormatter(sep=""))
        ax.grid(alpha=.2)
    failed = sum(row["failed_candidate_sets"] for row in rows)
    fig.suptitle(f"{metadata['model']} · {metadata['formulation']} · {failed} failed sets")
    fig.tight_layout()
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
