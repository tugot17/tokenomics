import json
import sys
import matplotlib.pyplot as plt
from typing import List, Dict
from matplotlib.gridspec import GridSpec

def plot_throughput(json_file: str, output_image: str) -> None:
    """
    Creates a beautifully styled visualization of throughput data.
    
    Parameters
    ----------
    json_file : str
        Path to the JSON file containing throughput data.
    output_image : str
        Path for saving the output figure.
    """
    # Basic style settings
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5
    
    # Load and process data
    with open(json_file, "r") as f:
        data: Dict = json.load(f)

    description_subtitle: str = data["metadata"].get("description", "No description available")
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

    # Create figure with custom layout
    fig = plt.figure(figsize=(12, 14))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Style parameters
    main_color = '#1f77b4'  # Blue
    secondary_color = '#2ca02c'  # Green
    
    # Plot 1: Batch Tokens
    ax1 = fig.add_subplot(gs[0])
    x_positions = range(len(batch_sizes))
    
    # Main line and confidence interval
    ax1.plot(x_positions, batch_tokens_mean, 
            color=main_color, linewidth=2, marker='o', 
            markersize=8, label='Batch Tokens Per Second')
    ax1.fill_between(x_positions,
                    [m - s for m, s in zip(batch_tokens_mean, batch_tokens_std)],
                    [m + s for m, s in zip(batch_tokens_mean, batch_tokens_std)],
                    color=main_color, alpha=0.2)

    # Add value annotations
    for x, mean, std in zip(x_positions, batch_tokens_mean, batch_tokens_std):
        ax1.annotate(f'{mean:.1f} ± {std:.1f}',
                    xy=(x, mean), xytext=(0, 15),
                    textcoords='offset points', ha='center',
                    fontsize=10, fontweight='bold')

    # Styling for first plot
    ax1.set_title('Batch Tokens Per Second vs. Batch Size', 
                 pad=20, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Batch Tokens/s', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Request Tokens
    ax2 = fig.add_subplot(gs[1])
    
    # Main line and confidence interval
    ax2.plot(x_positions, request_tokens_mean, 
            color=secondary_color, linewidth=2, marker='o',
            markersize=8, label='Request Tokens Per Second')
    ax2.fill_between(x_positions,
                    [m - s for m, s in zip(request_tokens_mean, request_tokens_std)],
                    [m + s for m, s in zip(request_tokens_mean, request_tokens_std)],
                    color=secondary_color, alpha=0.2)

    # Add value annotations
    for x, mean, std in zip(x_positions, request_tokens_mean, request_tokens_std):
        ax2.annotate(f'{mean:.1f} ± {std:.1f}',
                    xy=(x, mean), xytext=(0, 15),
                    textcoords='offset points', ha='center',
                    fontsize=10, fontweight='bold')

    # Styling for second plot
    ax2.set_title('Request Tokens Per Second vs. Batch Size',
                 pad=20, fontsize=14, fontweight='bold')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Request Tokens/s', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Set x-axis ticks for both plots
    for ax in [ax1, ax2]:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(batch_sizes)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=10)

    # Main title
    fig.suptitle(description_subtitle, 
                fontsize=16, fontweight='bold', 
                y=0.98)

    # Save with high quality
    plt.savefig(output_image, 
                dpi=300, bbox_inches='tight', 
                facecolor='white')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <json_file> <output_image>")
        sys.exit(1)

    json_file = sys.argv[1]
    output_image = sys.argv[2]
    plot_throughput(json_file, output_image)