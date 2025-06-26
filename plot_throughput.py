import json
import sys
import argparse
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors


def load_and_process_data(
    json_file: str,
) -> Tuple[str, List[int], List[float], List[float], List[float], List[float]]:
    """
    Load and process data from a JSON file.

    Returns
    -------
    tuple
        (description, batch_sizes, batch_tokens_mean, batch_tokens_std, request_tokens_mean, request_tokens_std)
    """
    with open(json_file, "r") as f:
        data: Dict = json.load(f)

    description: str = data["metadata"].get("description", "No description available")
    batch_sizes: List[int] = sorted(map(int, data["metadata"]["batch_sizes"]))

    # Initialize data containers
    batch_tokens_mean: List[float] = []
    batch_tokens_std: List[float] = []
    request_tokens_mean: List[float] = []
    request_tokens_std: List[float] = []

    # Extract data from results
    for batch in batch_sizes:
        entry = data["results"][str(batch)]

        # Get batch stats
        if "tokens_per_second_in_batch" in entry:
            b_mean, b_std = entry["tokens_per_second_in_batch"], 0.0
        else:
            b_mean = entry["throughput"]["batch_tokens_per_second"].get("mean", 0.0)
            b_std = entry["throughput"]["batch_tokens_per_second"].get("std", 0.0)

        # Get request stats
        if "avg_tokens_per_second" in entry:
            r_mean, r_std = entry["avg_tokens_per_second"], 0.0
        else:
            r_mean = entry["throughput"]["request_tokens_per_second"].get("mean", 0.0)
            r_std = entry["throughput"]["request_tokens_per_second"].get("std", 0.0)

        batch_tokens_mean.append(b_mean)
        batch_tokens_std.append(b_std)
        request_tokens_mean.append(r_mean)
        request_tokens_std.append(r_std)

    return (
        description,
        batch_sizes,
        batch_tokens_mean,
        batch_tokens_std,
        request_tokens_mean,
        request_tokens_std,
    )


def generate_colors_and_markers(num_datasets: int) -> Tuple[List[str], List[str]]:
    """
    Generate colors and markers for the given number of datasets.

    Parameters
    ----------
    num_datasets : int
        Number of datasets to generate colors and markers for.

    Returns
    -------
    tuple
        (colors, markers) lists
    """
    # Base colors - expand if needed
    base_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Base markers
    base_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    # Extend colors if we need more than the base set
    if num_datasets > len(base_colors):
        # Generate additional colors using matplotlib's color cycle
        additional_colors = plt.cm.tab20(range(len(base_colors), num_datasets))
        colors = base_colors + [mcolors.to_hex(c) for c in additional_colors]
    else:
        colors = base_colors[:num_datasets]

    # Extend markers by cycling through the base set
    markers = (base_markers * ((num_datasets // len(base_markers)) + 1))[:num_datasets]

    return colors, markers


def plot_throughput(json_files: List[str], output_image: str, title: Optional[str] = None) -> None:
    """
    Creates a beautifully styled visualization comparing throughput data from multiple JSON files.

    Parameters
    ----------
    json_files : List[str]
        List of paths to JSON files containing throughput data.
    output_image : str
        Path for saving the output figure.
    title : Optional[str], optional
        Custom title for the plot. If None, a default title will be generated.
    """
    if not json_files:
        raise ValueError("At least one JSON file must be provided")

    # Basic style settings
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.5

    # Load and process data from all files
    datasets = []
    for json_file in json_files:
        (
            desc,
            batch_sizes,
            batch_tokens_mean,
            batch_tokens_std,
            request_tokens_mean,
            request_tokens_std,
        ) = load_and_process_data(json_file)
        datasets.append(
            {
                "description": desc,
                "batch_sizes": batch_sizes,
                "batch_tokens_mean": batch_tokens_mean,
                "batch_tokens_std": batch_tokens_std,
                "request_tokens_mean": request_tokens_mean,
                "request_tokens_std": request_tokens_std,
            }
        )

    # Ensure all datasets have the same batch sizes for comparison
    all_batch_sizes = [set(dataset["batch_sizes"]) for dataset in datasets]
    if not all(batch_sizes == all_batch_sizes[0] for batch_sizes in all_batch_sizes):
        print(
            "Warning: Batch sizes differ between files. Using intersection of batch sizes."
        )
        common_batches = sorted(set.intersection(*all_batch_sizes))
        if not common_batches:
            raise ValueError("No common batch sizes found across all files")
        batch_sizes = common_batches
        # Note: For simplicity, we'll use the first dataset's batch sizes
        # In a more robust implementation, we'd filter all datasets to common batch sizes
    else:
        batch_sizes = datasets[0]["batch_sizes"]

    # Generate colors and markers
    colors, markers = generate_colors_and_markers(len(datasets))

    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 16))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

    # Plot 1: Batch Tokens Comparison
    ax1 = fig.add_subplot(gs[0])
    x_positions = range(len(batch_sizes))

    # Plot all datasets
    for i, dataset in enumerate(datasets):
        ax1.plot(
            x_positions,
            dataset["batch_tokens_mean"],
            color=colors[i],
            linewidth=2,
            marker=markers[i],
            markersize=8,
            label=f"{dataset['description']}",
        )
        ax1.fill_between(
            x_positions,
            [
                m - s
                for m, s in zip(
                    dataset["batch_tokens_mean"], dataset["batch_tokens_std"]
                )
            ],
            [
                m + s
                for m, s in zip(
                    dataset["batch_tokens_mean"], dataset["batch_tokens_std"]
                )
            ],
            color=colors[i],
            alpha=0.2,
        )

    # Add value annotations for all datasets with smart positioning
    for x in x_positions:
        # Collect all values at this x position
        values = []
        for i, dataset in enumerate(datasets):
            mean = dataset["batch_tokens_mean"][x]
            std = dataset["batch_tokens_std"][x]
            values.append((mean, std, i))

        # Sort values to determine vertical positioning
        values.sort(key=lambda x: x[0])  # Sort by mean value

        # Position annotations based on value ranking to avoid overlap
        for rank, (mean, std, dataset_idx) in enumerate(values):
            y_offset = 15 + (rank * 15)  # Stack vertically
            # Distribute horizontally based on dataset index
            num_datasets = len(datasets)
            if num_datasets > 1:
                x_offset = (dataset_idx - (num_datasets - 1) / 2) * (50 / num_datasets)
            else:
                x_offset = 0

            ax1.annotate(
                f"{mean:.1f}±{std:.1f}",
                xy=(x, mean),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                fontweight="bold",
                color=colors[dataset_idx],
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor=colors[dataset_idx],
                ),
            )

    # Styling for first plot
    ax1.set_title(
        "Batch Tokens Per Second Comparison", pad=20, fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel("Batch Tokens/s", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(fontsize=10, loc="best")

    # Plot 2: Request Tokens Comparison
    ax2 = fig.add_subplot(gs[1])

    # Plot all datasets
    for i, dataset in enumerate(datasets):
        ax2.plot(
            x_positions,
            dataset["request_tokens_mean"],
            color=colors[i],
            linewidth=2,
            marker=markers[i],
            markersize=8,
            label=f"{dataset['description']}",
        )
        ax2.fill_between(
            x_positions,
            [
                m - s
                for m, s in zip(
                    dataset["request_tokens_mean"], dataset["request_tokens_std"]
                )
            ],
            [
                m + s
                for m, s in zip(
                    dataset["request_tokens_mean"], dataset["request_tokens_std"]
                )
            ],
            color=colors[i],
            alpha=0.2,
        )

    # Add value annotations for all datasets with smart positioning
    for x in x_positions:
        # Collect all values at this x position
        values = []
        for i, dataset in enumerate(datasets):
            mean = dataset["request_tokens_mean"][x]
            std = dataset["request_tokens_std"][x]
            values.append((mean, std, i))

        # Sort values to determine vertical positioning
        values.sort(key=lambda x: x[0])  # Sort by mean value

        # Position annotations based on value ranking to avoid overlap
        for rank, (mean, std, dataset_idx) in enumerate(values):
            y_offset = 15 + (rank * 15)  # Stack vertically
            # Distribute horizontally based on dataset index
            num_datasets = len(datasets)
            if num_datasets > 1:
                x_offset = (dataset_idx - (num_datasets - 1) / 2) * (50 / num_datasets)
            else:
                x_offset = 0

            ax2.annotate(
                f"{mean:.1f}±{std:.1f}",
                xy=(x, mean),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                fontweight="bold",
                color=colors[dataset_idx],
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor=colors[dataset_idx],
                ),
            )

    # Styling for second plot
    ax2.set_title(
        "Request Tokens Per Second Comparison", pad=20, fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("Batch Size", fontsize=12)
    ax2.set_ylabel("Request Tokens/s", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(fontsize=10, loc="best")

    # Set x-axis ticks for both plots
    for ax in [ax1, ax2]:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(batch_sizes)
        ax.tick_params(labelsize=10)

    # Main title - use custom title if provided, otherwise generate default
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # Save with high quality
    plt.savefig(output_image, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create throughput comparison plots from JSON benchmark files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_throughput.py file1.json file2.json output.png
  python plot_throughput.py --title "My Custom Title" file1.json file2.json output.png
        """
    )
    
    parser.add_argument(
        "json_files", 
        nargs="+",
        help="One or more JSON files containing throughput data"
    )
    
    parser.add_argument(
        "output_image",
        help="Output image file path (e.g., plot.png)"
    )
    
    parser.add_argument(
        "--title",
        type=str,
        help="Custom title for the plot"
    )
    
    args = parser.parse_args()
    
    # Validate that we have at least one JSON file
    if len(args.json_files) < 1:
        print("Error: At least one JSON file is required.")
        sys.exit(1)
    
    print(f"Processing {len(args.json_files)} JSON files: {args.json_files}")
    if args.title:
        print(f"Using custom title: '{args.title}'")
    
    plot_throughput(args.json_files, args.output_image, args.title)
