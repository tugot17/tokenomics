"""Plot scoring throughput and request latency across concurrency."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter


def load_score_results(results_dir):
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
    lengths = sorted({row.get("state_tokens") for row in rows},
                     key=lambda n: (not isinstance(n, int), n if isinstance(n, int) else str(n)))
    return metadata, [(length, [row for row in rows if row.get("state_tokens") == length]) for length in lengths]


def plot_score_benchmark(results_dir, output_image):
    metadata, groups = load_score_results(results_dir)
    rows = [row for _, points in groups for row in points]
    x = sorted({row["batch_size"] for row in rows})
    questions = metadata.get("workload") == "questions"
    metrics = [("requests_per_second", "Requests/s", 1),
               ("input_tokens_per_second", "Input tokens/s", 1),
               ("latency_ms_mean", "Latency (s)", .001)]
    if questions:
        metrics.insert(1, ("questions_per_second", "Questions/s", 1))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for ax, (metric, label, scale) in zip(axes, metrics):
        for length, points in groups:
            y = [row[metric] * scale if row[metric] is not None else float("nan") for row in points]
            name = f"{length} tokens" if isinstance(length, int) else length
            ax.plot([row["batch_size"] for row in points], y, "o-", label=name)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, labels=x)
        ax.set_xlabel("Concurrent HTTP requests")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        if metric != "latency_ms_mean":
            ax.yaxis.set_major_formatter(EngFormatter(sep=""))
        ax.grid(alpha=.2)
    failed = sum(row["failed_requests"] for row in rows)
    label = "independent questions" if questions else metadata["formulation"]
    title = f"{metadata['model']} · {label}"
    if failed:
        title += f" · {failed} failed requests"
    fig.suptitle(title)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False)
    fig.tight_layout(rect=(0, .12 if handles else 0, 1, .94))
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
