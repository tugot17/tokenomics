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
    folders = sorted(path.glob("state-*"), key=lambda p: p.name) if isinstance(metadata.get("state_tokens"), list) else [path]
    rows = [json.loads(file.read_text()) for folder in folders for file in folder.glob("*.json") if file.stem.isdigit()]
    if not rows:
        raise ValueError("No measured scoring results found")
    # Accept results produced before request metrics were unified.
    for row in rows:
        if "requests_per_second" not in row:
            row["requests_per_second"] = row["decisions_per_second"]
        if "failed_requests" not in row:
            row["failed_requests"] = row["failed_candidate_sets"]
    rows.sort(key=lambda row: row["batch_size"])
    x = sorted({row["batch_size"] for row in rows})
    lengths = sorted({row.get("state_tokens") for row in rows}, key=lambda n: str(n))
    questions = metadata.get("workload") == "questions"
    metrics = [("requests_per_second", "Completed candidate sets / second"),
               ("input_tokens_per_second", "Input tokens / second"),
               ("latency_ms_mean", "Mean successful request latency (ms)")]
    if questions:
        metrics[0] = ("questions_per_second", "Answered questions / second")
        metrics.insert(0, ("requests_per_second", "Completed requests / second"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for ax, (metric, label) in zip(axes, metrics):
        for length in lengths:
            points = [row for row in rows if row.get("state_tokens") == length]
            y = [row[metric] if row[metric] is not None else float("nan") for row in points]
            ax.plot([row["batch_size"] for row in points], y, "o-", label=f"{length} state tokens" if length else None)
        if lengths != [None]:
            ax.legend(frameon=False)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, labels=x)
        ax.set_xlabel("Concurrent HTTP requests")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(EngFormatter(sep=""))
        ax.grid(alpha=.2)
    failed = sum(row["failed_requests"] for row in rows)
    label = "independent questions" if questions else metadata["formulation"]
    fig.suptitle(f"{metadata['model']} · {label} · {failed} failed requests")
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
