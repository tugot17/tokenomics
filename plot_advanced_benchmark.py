#!/usr/bin/env python3
"""
Plot benchmark results on a 4-panel dashboard.

Supports both single-file and multi-file comparison modes:
- Single file: Plot one benchmark result
- Multi files: Compare multiple benchmark results on the same dashboard

Auto-detects mode based on command-line arguments.
"""

import json
import sys
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
    """
    Generate a meaningful label from benchmark metadata.

    Parameters
    ----------
    metadata : Dict
        Benchmark metadata dictionary
    json_file : str
        Path to JSON file (fallback for label)

    Returns
    -------
    str
        Generated label
    """
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
    """
    Load benchmark data and extract label, model, and scenario.

    Returns
    -------
    Tuple[str, Dict, str, str]
        (label, data, model, scenario)
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    model = metadata.get("model", "Unknown")
    scenario = metadata.get("scenario", "")
    label = generate_label_from_metadata(metadata, json_file)

    return label, data, model, scenario


def _extract_stat(metrics_dict: Dict, metric_name: str, stat: str) -> float:
    """Helper to safely extract mean/std from nested metrics."""
    return metrics_dict.get(metric_name, {}).get(stat, 0)


def extract_metrics(data: Dict) -> Dict:
    """Extract all metrics from a single benchmark result."""
    batch_sizes = sorted(map(int, data["metadata"]["batch_sizes"]))

    metrics = {
        'batch_sizes': batch_sizes,
        'ttft_mean': [], 'ttft_std': [],
        'input_throughput_mean': [], 'input_throughput_std': [],
        'output_throughput_mean': [], 'output_throughput_std': [],
        'combined_throughput_mean': [], 'combined_throughput_std': [],
        'decode_time_mean': [], 'decode_time_std': [],
        'tpot_mean': [], 'tpot_std': []
    }

    for batch in batch_sizes:
        entry = data["results"][str(batch)]
        prefill = entry.get("prefill_metrics", {})
        decode = entry.get("decode_metrics", {})
        batch_metrics = entry.get("batch_metrics", {})

        # Prefill metrics
        metrics['ttft_mean'].append(_extract_stat(prefill, "ttft", "mean"))
        metrics['ttft_std'].append(_extract_stat(prefill, "ttft", "std"))
        metrics['input_throughput_mean'].append(_extract_stat(prefill, "input_throughput", "mean"))
        metrics['input_throughput_std'].append(_extract_stat(prefill, "input_throughput", "std"))

        # Decode metrics
        metrics['output_throughput_mean'].append(_extract_stat(decode, "output_throughput", "mean"))
        metrics['output_throughput_std'].append(_extract_stat(decode, "output_throughput", "std"))
        metrics['decode_time_mean'].append(_extract_stat(decode, "decode_time", "mean"))
        metrics['decode_time_std'].append(_extract_stat(decode, "decode_time", "std"))
        metrics['tpot_mean'].append(_extract_stat(decode, "tpot", "mean"))
        metrics['tpot_std'].append(_extract_stat(decode, "tpot", "std"))

        # Batch metrics
        metrics['combined_throughput_mean'].append(_extract_stat(batch_metrics, "combined_throughput", "mean"))
        metrics['combined_throughput_std'].append(_extract_stat(batch_metrics, "combined_throughput", "std"))

    return metrics


def add_annotations(ax, x_positions, y_values, benchmark_idx, format_string='{:.2f}s', y_offset_base=15, y_offset_step=12):
    """
    Add value annotations to plot points with vertical spreading.

    Parameters
    ----------
    ax : matplotlib axes
        The axes to annotate
    x_positions : list
        X coordinates for annotations
    y_values : list
        Y coordinates for annotations
    benchmark_idx : int
        Index of benchmark (for vertical spreading)
    format_string : str
        Format string for annotation values (default: '{:.2f}s')
    y_offset_base : int
        Base vertical offset in points (default: 15)
    y_offset_step : int
        Vertical offset increment per benchmark (default: 12)
    """
    y_offset = y_offset_base + (benchmark_idx * y_offset_step)
    for x, y_val in zip(x_positions, y_values):
        if y_val > 0:
            ax.annotate(format_string.format(y_val),
                       xy=(x, y_val), xytext=(0, y_offset),
                       textcoords='offset points', ha='center',
                       fontsize=9, fontweight='bold')


def plot_line_with_errorband(ax, x_positions, y_mean, y_std, color, marker, label):
    """
    Plot a line with error band (shaded region).

    Parameters
    ----------
    ax : matplotlib axes
        The axes to plot on
    x_positions : list
        X coordinates
    y_mean : list
        Mean values
    y_std : list
        Standard deviation values
    color : str
        Line color
    marker : str
        Marker style
    label : str
        Legend label
    """
    ax.plot(x_positions, y_mean,
           color=color, linewidth=2, marker=marker,
           markersize=8, label=label)

    # Error band with clipping at 0
    y_lower = [max(0, m - s) for m, s in zip(y_mean, y_std)]
    y_upper = [m + s for m, s in zip(y_mean, y_std)]
    ax.fill_between(x_positions, y_lower, y_upper, color=color, alpha=0.15)


def get_display_label(benchmark: 'BenchmarkData', single_model: bool) -> str:
    """
    Get display label for a benchmark, optionally including model name.

    Parameters
    ----------
    benchmark : BenchmarkData
        The benchmark data
    single_model : bool
        Whether all benchmarks use the same model

    Returns
    -------
    str
        Display label for the benchmark
    """
    if single_model:
        return benchmark.label
    return f"{benchmark.model}: {benchmark.label}"


def validate_files(json_files: List[str]) -> None:
    """
    Validate that all JSON files exist.

    Parameters
    ----------
    json_files : List[str]
        List of JSON file paths to validate

    Raises
    ------
    SystemExit
        If any file doesn't exist
    """
    for json_file in json_files:
        if not Path(json_file).exists():
            print(f"❌ Error: File not found: {json_file}")
            sys.exit(1)


def configure_matplotlib_style() -> None:
    """Configure matplotlib style settings for professional plots."""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False


def setup_line_plot(ax, x_positions, batch_sizes, benchmarks, metric_mean, metric_std,
                    single_model, title, ylabel, xlabel, format_string):
    """
    Setup a line plot with multiple benchmarks.

    Parameters
    ----------
    ax : matplotlib axes
        The axes to plot on
    x_positions : range
        X coordinate positions
    batch_sizes : list
        Batch size labels for x-axis
    benchmarks : list of BenchmarkData
        Benchmark data to plot
    metric_mean : str
        Key for mean values in metrics dict
    metric_std : str
        Key for std values in metrics dict
    single_model : bool
        Whether all benchmarks use the same model
    title : str
        Plot title
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    format_string : str
        Format string for annotations
    """
    for idx, benchmark in enumerate(benchmarks):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        marker = MARKER_STYLES[idx % len(MARKER_STYLES)]
        display_label = get_display_label(benchmark, single_model)

        plot_line_with_errorband(ax, x_positions,
                                 benchmark.metrics[metric_mean],
                                 benchmark.metrics[metric_std],
                                 color, marker, display_label)
        add_annotations(ax, x_positions, benchmark.metrics[metric_mean], idx,
                       format_string=format_string)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(batch_sizes)


def setup_bar_chart(ax, x_positions, batch_sizes, benchmarks, single_model):
    """
    Setup a stacked bar chart for latency breakdown.

    Parameters
    ----------
    ax : matplotlib axes
        The axes to plot on
    x_positions : range
        X coordinate positions
    batch_sizes : list
        Batch size labels for x-axis
    benchmarks : list of BenchmarkData
        Benchmark data to plot
    single_model : bool
        Whether all benchmarks use the same model
    """
    num_benchmarks = len(benchmarks)
    bar_width = BAR_WIDTH_FACTOR / num_benchmarks

    for idx, benchmark in enumerate(benchmarks):
        bar_positions = [x + (idx - num_benchmarks/2 + 0.5) * bar_width for x in x_positions]

        # Use different shades for each configuration
        prefill_color = PREFILL_COLORS[idx % len(PREFILL_COLORS)]
        decode_color = DECODE_COLORS[idx % len(DECODE_COLORS)]

        # Get letter for this configuration
        letter = LETTERS[idx] if idx < len(LETTERS) else str(idx)
        display_label = get_display_label(benchmark, single_model)

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
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_xlabel('Batch Size', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(batch_sizes)


def plot_multiple_benchmarks(json_files: List[str], output_image: str) -> None:
    """
    Create 4-panel dashboard comparing multiple benchmark results.

    Parameters
    ----------
    json_files : List[str]
        List of JSON files to compare
    output_image : str
        Path for saving the output figure
    """
    configure_matplotlib_style()

    # Load all benchmarks
    benchmarks = []
    for json_file in json_files:
        label, data, model, scenario = load_benchmark_data(json_file)
        metrics = extract_metrics(data)
        benchmarks.append(BenchmarkData(label, metrics, data, model, scenario))

    # Determine if we have multiple models or scenarios
    all_models = [b.model for b in benchmarks]
    all_scenarios = [b.scenario for b in benchmarks]
    unique_models = set(all_models)
    unique_scenarios = set(all_scenarios)

    # Check if all benchmarks have the same model/scenario
    single_model = len(unique_models) == 1
    single_scenario = len(unique_scenarios) == 1

    # Decide what to show in the title
    if single_model:
        title_model = all_models[0]
    else:
        title_model = f"Multiple Models ({len(unique_models)})"

    if single_scenario and all_scenarios[0]:
        title_scenario = all_scenarios[0]
    else:
        title_scenario = "" if single_scenario else f"Multiple Scenarios"

    # Create figure with 2x2 dashboard layout
    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # Get x positions from first benchmark
    batch_sizes = benchmarks[0].metrics['batch_sizes']
    x_positions = range(len(batch_sizes))

    # ===== Plot 1: TTFT (Prefill Phase) =====
    ax1 = fig.add_subplot(gs[0, 0])
    setup_line_plot(ax1, x_positions, batch_sizes, benchmarks,
                    'ttft_mean', 'ttft_std', single_model,
                    'Prefill Phase: Time to First Token',
                    'TTFT (seconds)', 'Batch Size', '{:.2f}s')

    # ===== Plot 2: Decode Throughput Per Request =====
    ax2 = fig.add_subplot(gs[0, 1])
    setup_line_plot(ax2, x_positions, batch_sizes, benchmarks,
                    'output_throughput_mean', 'output_throughput_std', single_model,
                    'Decode Phase: Throughput per Request',
                    'Tokens/second (per request)', 'Batch Size', '{:.1f} tok/s')

    # ===== Plot 3: Combined System Throughput (Aggregate) =====
    ax3 = fig.add_subplot(gs[1, 0])
    setup_line_plot(ax3, x_positions, batch_sizes, benchmarks,
                    'combined_throughput_mean', 'combined_throughput_std', single_model,
                    'Decode Phase: Aggregate Batch Throughput',
                    'Total Tokens/second', 'Batch Size', '{:.1f} tok/s')

    # ===== Plot 4: Latency Breakdown (Stacked Bar) =====
    ax4 = fig.add_subplot(gs[1, 1])
    setup_bar_chart(ax4, x_positions, batch_sizes, benchmarks, single_model)

    # Main title
    title_parts = [f"Benchmark Comparison ({len(benchmarks)} configurations)"]
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
    print(f"📈 Compared {len(benchmarks)} configurations across {len(batch_sizes)} batch sizes")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Single file:  python plot_advanced_benchmark.py <json_file> <output_image>")
        print("  Multi files:  python plot_advanced_benchmark.py <output_image> <json_file1> [json_file2] ...")
        print()
        print("Examples:")
        print("  Single: python plot_advanced_benchmark.py benchmark.json output.png")
        print("  Multi:  python plot_advanced_benchmark.py comparison.png \\")
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
        json_file = sys.argv[1]
        output_image = sys.argv[2]
        validate_files([json_file])

        print(f"📊 Single-file mode: plotting {json_file}")
        plot_multiple_benchmarks([json_file], output_image)
    else:
        # Multi-file mode
        output_image = sys.argv[1]
        json_files = sys.argv[2:]
        validate_files(json_files)

        print(f"📊 Multi-file mode: comparing {len(json_files)} benchmarks")
        plot_multiple_benchmarks(json_files, output_image)
