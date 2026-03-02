#!/usr/bin/env python3
"""
Plot benchmark results on a 6-panel dashboard.

Supports both single-file and multi-file comparison modes:
- Single file: Plot one benchmark result
- Multi files: Compare multiple benchmark results on the same dashboard

Auto-detects mode based on command-line arguments.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, NamedTuple
from matplotlib.gridspec import GridSpec
from pathlib import Path


# Constants
FIGURE_SIZE = (16, 12)
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
MARKER_STYLES = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
PREFILL_COLORS = ['#ff7f0e', '#ffb366', '#ff9933', '#ffcc99']  # Orange shades
DECODE_COLORS = ['#2ca02c', '#5cb85c', '#7fcc7f', '#a3d9a5']   # Green shades
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Bar chart constants
BAR_WIDTH_FACTOR = 0.8
BAR_LETTER_FONTSIZE = 8
BAR_ANNOTATION_FONTSIZE = 7


class BenchmarkData(NamedTuple):
    """Container for benchmark data."""
    label: str
    metrics: Dict
    data: Dict
    model: str
    scenario: str


def generate_label_from_metadata(metadata: Dict, json_file: str) -> str:
    """Generate a display label from benchmark metadata (LoRA strategy or description)."""
    lora_config = metadata.get("lora_config")
    if lora_config:
        strategy = lora_config.get("strategy", "")
        lora_names = lora_config.get("lora_names", [])

        strategy_labels = {
            "single": "Single LoRA",
            "zipf": f"Zipf {len(lora_names)} LoRAs",
            "uniform": f"Uniform {len(lora_names)} LoRAs",
        }

        if strategy in strategy_labels:
            return strategy_labels[strategy]
        elif strategy == "all-unique":
            return f"{len(lora_names)} Unique LoRAs"
        elif strategy == "mixed":
            ratio = lora_config.get("base_model_ratio", 0)
            return f"Mixed {int((1-ratio)*100)}% LoRA"
        else:
            return strategy

    # Use description or filename
    desc = metadata.get("description", "")
    if "baseline" in desc.lower() or "no lora" in desc.lower():
        return "Baseline (No LoRA)"

    return desc or Path(json_file).stem


def load_benchmark_data(json_file: str) -> Tuple[str, Dict, str, str]:
    """Load benchmark JSON and return (label, data, model, scenario)."""
    with open(json_file, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    model = metadata.get("model", "Unknown")
    scenario = metadata.get("scenario", "")
    label = generate_label_from_metadata(metadata, json_file)

    return label, data, model, scenario


def extract_metrics(data: Dict) -> Dict:
    """Extract all metrics from a single benchmark result.

    Auto-detects the sweep axis: concurrency_levels (sustained mode) or
    batch_sizes (burst mode). Falls back to inferring from result keys.
    """
    metadata = data.get("metadata", {})

    # Auto-detect sweep axis
    if "concurrency_levels" in metadata:
        sweep_values = sorted(map(int, metadata["concurrency_levels"]))
        sweep_label = "Concurrency"
    elif "batch_sizes" in metadata:
        sweep_values = sorted(map(int, metadata["batch_sizes"]))
        sweep_label = "Batch Size"
    else:
        # Fallback: infer from result keys
        sweep_values = sorted(int(k) for k in data.get("results", {}).keys() if k.isdigit())
        sweep_label = "Batch Size"

    def _get(d, key, stat):
        return d.get(key, {}).get(stat, 0)

    metrics = {
        'sweep_values': sweep_values,
        'sweep_label': sweep_label,
        # Keep batch_sizes as alias for backward compat with internal references
        'batch_sizes': sweep_values,
        'ttft_mean': [], 'ttft_std': [],
        'output_throughput_mean': [], 'output_throughput_std': [],
        'e2e_tps_mean': [], 'e2e_tps_std': [],
        'steady_state_median_mean': [], 'steady_state_median_std': [],
        'decode_time_mean': [], 'decode_time_std': [],
    }

    for sv in sweep_values:
        entry = data["results"][str(sv)]
        prefill = entry.get("prefill_metrics", {})
        decode = entry.get("decode_metrics", {})
        batch_m = entry.get("batch_metrics", {})
        phased = entry.get("phased_metrics", {})
        ss = phased.get("steady_state_tps", {})

        metrics['ttft_mean'].append(_get(prefill, "ttft", "mean"))
        metrics['ttft_std'].append(_get(prefill, "ttft", "std"))
        metrics['output_throughput_mean'].append(_get(decode, "output_throughput", "mean"))
        metrics['output_throughput_std'].append(_get(decode, "output_throughput", "std"))
        metrics['decode_time_mean'].append(_get(decode, "decode_time", "mean"))
        metrics['decode_time_std'].append(_get(decode, "decode_time", "std"))
        # Support both new (e2e_tps) and old (wall_clock_tps) JSON keys
        e2e = batch_m.get("e2e_tps") or batch_m.get("wall_clock_tps") or {}
        metrics['e2e_tps_mean'].append(e2e.get("mean", 0))
        metrics['e2e_tps_std'].append(e2e.get("std", 0))
        metrics['steady_state_median_mean'].append(ss.get("median", 0) or 0)
        metrics['steady_state_median_std'].append(ss.get("median_std", 0) or 0)

    return metrics


def add_annotations(ax, x_positions, y_values, benchmark_idx, format_string='{:.2f}s', y_offset_base=15, y_offset_step=12):
    """Add value annotations to plot points with vertical spreading per benchmark."""
    y_offset = y_offset_base + (benchmark_idx * y_offset_step)
    for x, y_val in zip(x_positions, y_values):
        if y_val > 0:
            ax.annotate(format_string.format(y_val),
                       xy=(x, y_val), xytext=(0, y_offset),
                       textcoords='offset points', ha='center',
                       fontsize=9, fontweight='bold')


def plot_line_with_errorband(ax, x_positions, y_mean, y_std, color, marker, label):
    """Plot a line with shaded error band."""
    ax.plot(x_positions, y_mean,
           color=color, linewidth=2, marker=marker,
           markersize=8, label=label)

    # Error band with clipping at 0
    y_lower = [max(0, m - s) for m, s in zip(y_mean, y_std)]
    y_upper = [m + s for m, s in zip(y_mean, y_std)]
    ax.fill_between(x_positions, y_lower, y_upper, color=color, alpha=0.15)


def configure_matplotlib_style() -> None:
    """Configure matplotlib style settings."""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False


def setup_line_plot(ax, x_positions, batch_sizes, benchmarks, metric_mean, metric_std,
                    title, ylabel, xlabel, format_string):
    """Setup a line plot with multiple benchmarks."""
    for idx, benchmark in enumerate(benchmarks):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        marker = MARKER_STYLES[idx % len(MARKER_STYLES)]
        display_label = benchmark.label

        plot_line_with_errorband(ax, x_positions,
                                 benchmark.metrics[metric_mean],
                                 benchmark.metrics[metric_std],
                                 color, marker, display_label)
        add_annotations(ax, x_positions, benchmark.metrics[metric_mean], idx,
                       format_string=format_string)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(batch_sizes)


def setup_bar_chart(ax, x_positions, batch_sizes, benchmarks, xlabel='Batch Size'):
    """Setup a stacked bar chart for latency breakdown."""
    num_benchmarks = len(benchmarks)
    bar_width = BAR_WIDTH_FACTOR / num_benchmarks

    for idx, benchmark in enumerate(benchmarks):
        bar_positions = [x + (idx - num_benchmarks/2 + 0.5) * bar_width for x in x_positions]

        # Use different shades for each configuration
        prefill_color = PREFILL_COLORS[idx % len(PREFILL_COLORS)]
        decode_color = DECODE_COLORS[idx % len(DECODE_COLORS)]

        # Get letter for this configuration
        letter = LETTERS[idx] if idx < len(LETTERS) else str(idx)
        display_label = benchmark.label

        # Stacked bars: TTFT on bottom, decode_time on top
        ax.bar(bar_positions, benchmark.metrics['ttft_mean'], bar_width,
               label=f'({letter}) {display_label}',
               color=prefill_color, alpha=0.8,
               edgecolor='black', linewidth=0.5)

        ax.bar(bar_positions, benchmark.metrics['decode_time_mean'], bar_width,
               bottom=benchmark.metrics['ttft_mean'],
               color=decode_color, alpha=0.8,
               edgecolor='black', linewidth=0.5)

        # Add letter labels on each bar
        for bar_pos, ttft, decode_time in zip(bar_positions,
                                              benchmark.metrics['ttft_mean'],
                                              benchmark.metrics['decode_time_mean']):
            if ttft > 0:
                mid_height = ttft + decode_time / 2
                ax.text(bar_pos, mid_height, letter,
                        ha='center', va='center',
                        fontsize=BAR_LETTER_FONTSIZE, fontweight='bold', color='white',
                        bbox=dict(boxstyle='circle,pad=0.1', facecolor='black', alpha=0.7))

        # Add total time annotations on top of each bar
        total_times = [t + d for t, d in zip(benchmark.metrics['ttft_mean'],
                                             benchmark.metrics['decode_time_mean'])]
        for bar_pos, total in zip(bar_positions, total_times):
            if total > 0:
                ax.annotate(f'{total:.2f}s',
                            xy=(bar_pos, total), xytext=(0, 5),
                            textcoords='offset points', ha='center',
                            fontsize=BAR_ANNOTATION_FONTSIZE, fontweight='bold')

    ax.set_title('Latency Breakdown: Prefill vs Decode',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(batch_sizes)


def plot_phased_metrics_panel(ax, benchmarks: List['BenchmarkData']) -> None:
    """Plot time-series of output tok/s per bucket with steady-state median line."""
    plotted = False
    for idx, benchmark in enumerate(benchmarks):
        # phased_metrics is stored per batch_size in the results
        # We pick the largest batch size since it's most interesting
        batch_sizes = benchmark.metrics['batch_sizes']
        largest_batch = str(max(batch_sizes))
        entry = benchmark.data["results"].get(largest_batch, {})
        pm = entry.get("phased_metrics", {})
        ts = pm.get("time_series", {})

        tokens_per_bucket = ts.get("output_tokens_per_bucket", [])
        bucket_size = benchmark.data.get("metadata", {}).get("bucket_size_seconds", 1.0)

        if not tokens_per_bucket:
            continue

        plotted = True
        n_buckets = len(tokens_per_bucket)
        time_axis = [i * bucket_size for i in range(n_buckets)]
        tps_per_bucket = [t / bucket_size for t in tokens_per_bucket]

        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        display_label = benchmark.label

        # Step plot for bucket data (tokens are counts per bucket, not continuous)
        ax.step(time_axis, tps_per_bucket, where='post', color=color,
                linewidth=1.8, alpha=0.85,
                label=f'{display_label} (batch={largest_batch})')
        ax.fill_between(time_axis, tps_per_bucket, step='post',
                        color=color, alpha=0.1)

        # Median steady-state line
        steady_state_median = pm.get("steady_state_tps", {}).get("median")
        if steady_state_median is not None:
            ax.axhline(y=steady_state_median, color=color, linestyle='--', linewidth=1.5,
                        alpha=0.8, label=f'Steady-state median: {steady_state_median:.0f} tok/s')

    if plotted:
        ax.set_ylim(bottom=0)
        ax.set_title('Time-Series: Output Tokens/s per Bucket (Largest Batch Size)',
                      fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Output Tokens/s', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    return plotted


def plot_multiple_benchmarks(json_files: List[str], output_image: str) -> None:
    """Create multi-panel dashboard comparing benchmark results."""
    configure_matplotlib_style()

    # Load all benchmarks
    benchmarks = []
    for json_file in json_files:
        label, data, model, scenario = load_benchmark_data(json_file)
        metrics = extract_metrics(data)
        benchmarks.append(BenchmarkData(label, metrics, data, model, scenario))

    # Build title from model/scenario info
    unique_models = set(b.model for b in benchmarks)
    unique_scenarios = set(b.scenario for b in benchmarks)
    title_model = benchmarks[0].model if len(unique_models) == 1 else f"Multiple Models ({len(unique_models)})"
    title_scenario = benchmarks[0].scenario if len(unique_scenarios) == 1 and benchmarks[0].scenario else (
        "" if len(unique_scenarios) == 1 else "Multiple Scenarios"
    )

    # Detect execution mode from first benchmark
    first_metadata = benchmarks[0].data.get("metadata", {})
    execution_mode = first_metadata.get("execution_mode", "burst")
    mode_label = "Sustained" if execution_mode == "sustained" else "Burst"

    # Check if any benchmark has phased_metrics
    has_phased = any(
        entry.get("phased_metrics", {}).get("time_series", {}).get("output_tokens_per_bucket")
        for b in benchmarks
        for entry in b.data.get("results", {}).values()
    )

    # Get sweep values and label from first benchmark
    sweep_values = benchmarks[0].metrics['sweep_values']
    sweep_label = benchmarks[0].metrics['sweep_label']
    x_positions = range(len(sweep_values))

    fig = plt.figure(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] + 5))
    gs = GridSpec(3, 2, hspace=0.35, wspace=0.3, height_ratios=[1, 1, 1])

    # ===== Plot 1: TTFT (Prefill Phase) =====
    ax1 = fig.add_subplot(gs[0, 0])
    setup_line_plot(ax1, x_positions, sweep_values, benchmarks,
                    'ttft_mean', 'ttft_std',
                    'Prefill Phase: Time to First Token',
                    'TTFT (seconds)', sweep_label, '{:.2f}s')

    # ===== Plot 2: Decode Throughput Per Request =====
    ax2 = fig.add_subplot(gs[0, 1])
    setup_line_plot(ax2, x_positions, sweep_values, benchmarks,
                    'output_throughput_mean', 'output_throughput_std',
                    'Decode Phase: Output Throughput per Request',
                    'Output Tokens/second (per request)', sweep_label, '{:.1f} tok/s')

    # ===== Plot 3: End-to-End Throughput =====
    ax3 = fig.add_subplot(gs[1, 0])
    setup_line_plot(ax3, x_positions, sweep_values, benchmarks,
                    'e2e_tps_mean', 'e2e_tps_std',
                    'Output Combined Throughput',
                    'Output Tokens/second', sweep_label, '{:.1f} tok/s')

    # ===== Plot 4: Latency Breakdown (Stacked Bar) =====
    ax4 = fig.add_subplot(gs[1, 1])
    setup_bar_chart(ax4, x_positions, sweep_values, benchmarks, xlabel=sweep_label)

    # ===== Plot 5: Steady-State Decode Throughput =====
    ax5 = fig.add_subplot(gs[2, 0])
    setup_line_plot(ax5, x_positions, sweep_values, benchmarks,
                    'steady_state_median_mean', 'steady_state_median_std',
                    'Steady-State Output Throughput',
                    'Output Tokens/second', sweep_label, '{:.1f} tok/s')

    # ===== Plot 6: Phased Metrics Time-Series (if available) =====
    if has_phased:
        ax6 = fig.add_subplot(gs[2, 1])
        plot_phased_metrics_panel(ax6, benchmarks)

    # Main title
    title_parts = [f"Benchmark ({len(benchmarks)} configs, {mode_label})"]
    if title_model != "Unknown":
        title_parts.append(f"Model: {title_model}")
    if title_scenario:
        title_parts.append(f"Scenario: {title_scenario}")

    main_title = " | ".join(title_parts)
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)

    # Save with high quality
    plt.savefig(output_image,
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"📊 Multi-benchmark comparison plot saved to: {output_image}")
    print(f"📈 Compared {len(benchmarks)} configurations across {len(sweep_values)} {sweep_label.lower()} levels")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Single file:  python plot_completion_benchmark.py <json_file> <output_image>")
        print("  Multi files:  python plot_completion_benchmark.py <output_image> <json_file1> [json_file2] ...")
        print()
        print("Examples:")
        print("  Single: python plot_completion_benchmark.py benchmark.json output.png")
        print("  Multi:  python plot_completion_benchmark.py comparison.png \\")
        print("              lora_benchmark_results/00_baseline_no_lora.json \\")
        print("              lora_benchmark_results/01_single_lora.json \\")
        print("              lora_benchmark_results/03_all_unique_8_loras.json")
        sys.exit(1)

    # Auto-detect mode based on first argument
    # If first arg is a JSON file -> single mode (backward compatible)
    # Otherwise -> multi mode
    first_arg = sys.argv[1]

    if first_arg.endswith('.json') and Path(first_arg).exists():
        # Single file mode (backward compatible)
        json_files = [sys.argv[1]]
        output_image = sys.argv[2]
    else:
        # Multi-file mode
        output_image = sys.argv[1]
        json_files = sys.argv[2:]

    for f in json_files:
        if not Path(f).exists():
            print(f"Error: File not found: {f}")
            sys.exit(1)

    print(f"📊 Plotting {len(json_files)} benchmark(s)")
    plot_multiple_benchmarks(json_files, output_image)
