import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
from matplotlib.gridspec import GridSpec
import seaborn as sns

def parse_config_key(config_key: str) -> Tuple[int, str]:
    """Parse configuration key to extract batch size and sequence length."""
    # Format: "batch_X_seq_Y" where X is batch size, Y is sequence length
    parts = config_key.split('_')
    batch_size = int(parts[1])
    seq_length = parts[3]  # Could be int or string
    
    # Try to convert to int, keep as string if not possible
    try:
        seq_length = int(seq_length)
    except ValueError:
        pass
    
    return batch_size, seq_length

def plot_embedding_benchmark(json_file: str, output_image: str) -> None:
    """
    Creates a comprehensive visualization of embedding benchmark data.
    
    Parameters
    ----------
    json_file : str
        Path to the JSON file containing embedding benchmark data.
    output_image : str
        Path for saving the output figure.
    """
    # Set up plotting style to match LLM throughput plots
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5
    
    # Load data
    with open(json_file, "r") as f:
        data: Dict = json.load(f)
    
    metadata = data["metadata"]
    results = data["results"]
    
    # Parse configurations
    configs = {}
    batch_sizes = set()
    seq_lengths = set()
    
    for config_key, config_data in results.items():
        batch_size, seq_length = parse_config_key(config_key)
        configs[config_key] = {
            'batch_size': batch_size,
            'seq_length': seq_length,
            'data': config_data
        }
        batch_sizes.add(batch_size)
        seq_lengths.add(seq_length)
    
    batch_sizes = sorted(list(batch_sizes))
    seq_lengths = sorted(list(seq_lengths), key=lambda x: (isinstance(x, str), x))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # Style parameters to match LLM throughput plots
    main_color = '#1f77b4'  # Blue
    secondary_color = '#2ca02c'  # Green
    colors = [main_color, secondary_color, '#ff7f0e', '#d62728']  # Blue, Green, Orange, Red
    color_map = {seq_len: colors[i % len(colors)] for i, seq_len in enumerate(seq_lengths)}
    
    # Plot 1: Batch Embeddings Per Second
    ax1 = fig.add_subplot(gs[0, 0])
    for seq_length in seq_lengths:
        batch_eps_means = []
        batch_eps_stds = []
        x_positions = []
        
        for batch_size in batch_sizes:
            config_key = f"batch_{batch_size}_seq_{seq_length}"
            if config_key in configs:
                config_data = configs[config_key]['data']
                mean_val = config_data['throughput']['batch_embeddings_per_second']['mean']
                std_val = config_data['throughput']['batch_embeddings_per_second']['std']
                batch_eps_means.append(mean_val)
                batch_eps_stds.append(std_val)
                x_positions.append(batch_size)
        
        if batch_eps_means:
            # Main line
            ax1.plot(x_positions, batch_eps_means, 
                    color=color_map[seq_length], linewidth=2, marker='o', 
                    markersize=8, label=f'{seq_length} words')
            # Confidence interval
            ax1.fill_between(x_positions,
                            [m - s for m, s in zip(batch_eps_means, batch_eps_stds)],
                            [m + s for m, s in zip(batch_eps_means, batch_eps_stds)],
                            color=color_map[seq_length], alpha=0.2)
            
            # Add value annotations
            for x, mean, std in zip(x_positions, batch_eps_means, batch_eps_stds):
                ax1.annotate(f'{mean:.1f} ± {std:.1f}',
                            xy=(x, mean), xytext=(0, 15),
                            textcoords='offset points', ha='center',
                            fontsize=10, fontweight='bold')
    
    ax1.set_title('Batch Embeddings Per Second', 
                 pad=20, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Embeddings/sec', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.tick_params(labelsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(batch_sizes)
    ax1.set_xticklabels(batch_sizes)
    
    # Plot 2: Batch Tokens Per Second
    ax2 = fig.add_subplot(gs[0, 1])
    for seq_length in seq_lengths:
        batch_tps_means = []
        batch_tps_stds = []
        x_positions = []
        
        for batch_size in batch_sizes:
            config_key = f"batch_{batch_size}_seq_{seq_length}"
            if config_key in configs:
                config_data = configs[config_key]['data']
                mean_val = config_data['throughput']['batch_tokens_per_second']['mean']
                std_val = config_data['throughput']['batch_tokens_per_second']['std']
                batch_tps_means.append(mean_val)
                batch_tps_stds.append(std_val)
                x_positions.append(batch_size)
        
        if batch_tps_means:
            # Main line
            ax2.plot(x_positions, batch_tps_means, 
                    color=color_map[seq_length], linewidth=2, marker='s', 
                    markersize=8, label=f'{seq_length} words')
            # Confidence interval
            ax2.fill_between(x_positions,
                            [m - s for m, s in zip(batch_tps_means, batch_tps_stds)],
                            [m + s for m, s in zip(batch_tps_means, batch_tps_stds)],
                            color=color_map[seq_length], alpha=0.2)
            
            # Add value annotations
            for x, mean, std in zip(x_positions, batch_tps_means, batch_tps_stds):
                ax2.annotate(f'{mean:.0f} ± {std:.0f}',
                            xy=(x, mean), xytext=(0, 15),
                            textcoords='offset points', ha='center',
                            fontsize=10, fontweight='bold')
    
    ax2.set_title('Batch Tokens Per Second',
                 pad=20, fontsize=14, fontweight='bold')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Tokens/sec', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.tick_params(labelsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(batch_sizes)
    ax2.set_xticklabels(batch_sizes)
    
    # Plot 3: Request Processing Time
    ax3 = fig.add_subplot(gs[1, 0])
    for seq_length in seq_lengths:
        batch_times_means = []
        batch_times_stds = []
        x_positions = []
        
        for batch_size in batch_sizes:
            config_key = f"batch_{batch_size}_seq_{seq_length}"
            if config_key in configs:
                config_data = configs[config_key]['data']
                mean_val = config_data['timings']['batch_total_seconds']['mean']
                std_val = config_data['timings']['batch_total_seconds']['std']
                batch_times_means.append(mean_val * 1000)  # Convert to ms
                batch_times_stds.append(std_val * 1000)
                x_positions.append(batch_size)
        
        if batch_times_means:
            # Main line
            ax3.plot(x_positions, batch_times_means, 
                    color=color_map[seq_length], linewidth=2, marker='^', 
                    markersize=8, label=f'{seq_length} words')
            # Confidence interval
            ax3.fill_between(x_positions,
                            [m - s for m, s in zip(batch_times_means, batch_times_stds)],
                            [m + s for m, s in zip(batch_times_means, batch_times_stds)],
                            color=color_map[seq_length], alpha=0.2)
            
            # Add value annotations
            for x, mean, std in zip(x_positions, batch_times_means, batch_times_stds):
                ax3.annotate(f'{mean:.1f} ± {std:.1f}',
                            xy=(x, mean), xytext=(0, 15),
                            textcoords='offset points', ha='center',
                            fontsize=10, fontweight='bold')
    
    ax3.set_title('Batch Processing Time',
                 pad=20, fontsize=14, fontweight='bold')
    ax3.set_xlabel('Batch Size', fontsize=12)
    ax3.set_ylabel('Time (ms)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.tick_params(labelsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_xscale('log', base=2)
    ax3.set_xticks(batch_sizes)
    ax3.set_xticklabels(batch_sizes)
    
    # Plot 4: Heatmap of Embeddings Per Second
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Prepare data for heatmap
    heatmap_data = np.zeros((len(seq_lengths), len(batch_sizes)))
    for i, seq_length in enumerate(seq_lengths):
        for j, batch_size in enumerate(batch_sizes):
            config_key = f"batch_{batch_size}_seq_{seq_length}"
            if config_key in configs:
                config_data = configs[config_key]['data']
                heatmap_data[i, j] = config_data['throughput']['batch_embeddings_per_second']['mean']
            else:
                heatmap_data[i, j] = np.nan
    
    # Create heatmap
    im = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Embeddings/sec', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax4.set_xticks(range(len(batch_sizes)))
    ax4.set_xticklabels(batch_sizes)
    ax4.set_yticks(range(len(seq_lengths)))
    ax4.set_yticklabels(seq_lengths)
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Sequence Length')
    ax4.set_title('Performance Heatmap\n(Embeddings/sec)', fontsize=14, fontweight='bold')
    
    # Add text annotations to heatmap
    for i in range(len(seq_lengths)):
        for j in range(len(batch_sizes)):
            if not np.isnan(heatmap_data[i, j]):
                text = f'{heatmap_data[i, j]:.1f}'
                ax4.text(j, i, text, ha='center', va='center', 
                        color='white' if heatmap_data[i, j] > np.nanmax(heatmap_data) * 0.6 else 'black',
                        fontweight='bold')
    
    # Main title
    title = f"Embedding Benchmark Results - {metadata['model']}"
    if metadata.get('description'):
        title += f"\n{metadata['description']}"
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Add metadata info
    info_text = (f"Runs: {metadata['num_runs']} | "
                f"Concurrent: {metadata.get('concurrent_requests', True)} | "
                f"API: {metadata['api_base']}")
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic')
    
    # Save figure
    plt.savefig(output_image, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Embedding benchmark plot saved to: {output_image}")



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_embedding_benchmark.py <json_file> <output_image>")
        print("Example: python plot_embedding_benchmark.py embedding_benchmark_results.json plot.png")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_image = sys.argv[2]
    
    # Create main plot
    plot_embedding_benchmark(json_file, output_image) 