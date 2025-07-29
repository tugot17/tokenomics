import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from matplotlib.gridspec import GridSpec

def plot_advanced_benchmark(json_file: str, output_image: str) -> None:
    """
    Creates a comprehensive visualization of enhanced benchmark data with focus on
    prefill vs decode performance for long context analysis.
    
    Parameters
    ----------
    json_file : str
        Path to the JSON file containing enhanced benchmark data.
    output_image : str
        Path for saving the output figure.
    """
    # Enhanced style settings for professional look
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Load and process data
    with open(json_file, "r") as f:
        data: Dict = json.load(f)

    metadata = data["metadata"]
    description = metadata.get("description", "Enhanced Benchmark Results")
    model = metadata.get("model", "Unknown Model")
    scenario = metadata.get("scenario", "")
    batch_sizes = sorted(map(int, metadata["batch_sizes"]))
    
    # Initialize data containers
    ttft_mean, ttft_std = [], []
    input_throughput_mean, input_throughput_std = [], []
    output_throughput_mean, output_throughput_std = [], []
    combined_throughput_mean, combined_throughput_std = [], []
    decode_time_mean, decode_time_std = [], []
    tpot_mean, tpot_std = [], []
    success_rates = []
    
    # Extract data from results
    for batch in batch_sizes:
        entry = data["results"][str(batch)]
        
        # Prefill metrics
        prefill = entry.get("prefill_metrics", {})
        ttft_mean.append(prefill.get("ttft", {}).get("mean", 0))
        ttft_std.append(prefill.get("ttft", {}).get("std", 0))
        input_throughput_mean.append(prefill.get("input_throughput", {}).get("mean", 0))
        input_throughput_std.append(prefill.get("input_throughput", {}).get("std", 0))
        
        # Decode metrics  
        decode = entry.get("decode_metrics", {})
        output_throughput_mean.append(decode.get("output_throughput", {}).get("mean", 0))
        output_throughput_std.append(decode.get("output_throughput", {}).get("std", 0))
        decode_time_mean.append(decode.get("decode_time", {}).get("mean", 0))
        decode_time_std.append(decode.get("decode_time", {}).get("std", 0))
        tpot_mean.append(decode.get("tpot", {}).get("mean", 0))
        tpot_std.append(decode.get("tpot", {}).get("std", 0))
        
        # Batch metrics
        batch_metrics = entry.get("batch_metrics", {})
        combined_throughput_mean.append(batch_metrics.get("combined_throughput", {}).get("mean", 0))
        combined_throughput_std.append(batch_metrics.get("combined_throughput", {}).get("std", 0))
        
        # Reliability
        reliability = entry.get("reliability", {})
        success_rates.append(reliability.get("success_rate", 100.0))

    # Create figure with 2x2 dashboard layout
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # Color scheme
    prefill_color = '#ff7f0e'  # Orange for prefill
    decode_color = '#2ca02c'   # Green for decode  
    combined_color = '#1f77b4' # Blue for combined
    reliability_color = '#d62728' # Red for reliability

    x_positions = range(len(batch_sizes))
    
    # Plot 1: TTFT (Prefill Phase) vs Batch Size
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x_positions, ttft_mean, 
            color=prefill_color, linewidth=2, marker='o', 
            markersize=8, label='Time to First Token')
    ax1.fill_between(x_positions,
                    [m - s for m, s in zip(ttft_mean, ttft_std)],
                    [m + s for m, s in zip(ttft_mean, ttft_std)],
                    color=prefill_color, alpha=0.2)
    
    # Add value annotations
    for x, mean, std in zip(x_positions, ttft_mean, ttft_std):
        if mean > 0:  # Only annotate if we have data
            ax1.annotate(f'{mean:.2f}s',
                        xy=(x, mean), xytext=(0, 15),
                        textcoords='offset points', ha='center',
                        fontsize=9, fontweight='bold')
    
    ax1.set_title('Prefill Phase: Time to First Token', 
                 fontsize=14, fontweight='bold', color=prefill_color)
    ax1.set_ylabel('TTFT (seconds)', fontsize=11)
    ax1.set_xlabel('Batch Size', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(batch_sizes)
    
    # Plot 2: Decode Performance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x_positions, output_throughput_mean, 
            color=decode_color, linewidth=2, marker='s', 
            markersize=8, label='Per-Request Decode Speed')
    ax2.fill_between(x_positions,
                    [m - s for m, s in zip(output_throughput_mean, output_throughput_std)],
                    [m + s for m, s in zip(output_throughput_mean, output_throughput_std)],
                    color=decode_color, alpha=0.2)
    
    # Add value annotations  
    for x, mean, std in zip(x_positions, output_throughput_mean, output_throughput_std):
        if mean > 0:
            ax2.annotate(f'{mean:.1f} tok/s',
                        xy=(x, mean), xytext=(0, 15),
                        textcoords='offset points', ha='center',
                        fontsize=9, fontweight='bold')
    
    ax2.set_title('Decode Phase: Throughput per Request', 
                 fontsize=14, fontweight='bold', color=decode_color)
    ax2.set_ylabel('Tokens/second (per request)', fontsize=11)
    ax2.set_xlabel('Batch Size', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(batch_sizes)
    
    # Plot 3: Combined System Throughput
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x_positions, combined_throughput_mean, 
            color=combined_color, linewidth=2, marker='^', 
            markersize=8, label='Combined System Throughput')
    ax3.fill_between(x_positions,
                    [m - s for m, s in zip(combined_throughput_mean, combined_throughput_std)],
                    [m + s for m, s in zip(combined_throughput_mean, combined_throughput_std)],
                    color=combined_color, alpha=0.2)
    
    # Add value annotations
    for x, mean, std in zip(x_positions, combined_throughput_mean, combined_throughput_std):
        if mean > 0:
            ax3.annotate(f'{mean:.1f} tok/s',
                        xy=(x, mean), xytext=(0, 15),
                        textcoords='offset points', ha='center',
                        fontsize=9, fontweight='bold')
    
    ax3.set_title('Decode Phase: Batch Throughput', 
                 fontsize=14, fontweight='bold', color=combined_color)
    ax3.set_ylabel('Total Tokens/second', fontsize=11)
    ax3.set_xlabel('Batch Size', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(batch_sizes)
    
    # Plot 4: Latency Breakdown (Stacked Bar)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Create stacked bars showing TTFT + Decode Time
    width = 0.6
    ttft_bars = ax4.bar(x_positions, ttft_mean, width, 
                       label='Prefill (TTFT)', color=prefill_color, alpha=0.8)
    decode_bars = ax4.bar(x_positions, decode_time_mean, width, 
                         bottom=ttft_mean, label='Decode Time', 
                         color=decode_color, alpha=0.8)
    
    # Add total time annotations
    total_times = [t + d for t, d in zip(ttft_mean, decode_time_mean)]
    for x, total in zip(x_positions, total_times):
        if total > 0:
            ax4.annotate(f'{total:.2f}s total',
                        xy=(x, total), xytext=(0, 8),
                        textcoords='offset points', ha='center',
                        fontsize=9, fontweight='bold')
    
    ax4.set_title('Latency Breakdown: Prefill vs Decode', 
                 fontsize=14, fontweight='bold')
    ax4.set_ylabel('Time (seconds)', fontsize=11)
    ax4.set_xlabel('Batch Size', fontsize=11)
    ax4.legend(fontsize=10, loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(x_positions)
    ax4.set_xticklabels(batch_sizes)
    
    # Main title with model and scenario info
    title_parts = [description]
    if model != "Unknown Model":
        title_parts.append(f"Model: {model}")
    if scenario:
        title_parts.append(f"Scenario: {scenario}")
    
    main_title = " | ".join(title_parts)
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.95)
    
    # Add summary statistics box
    if ttft_mean and output_throughput_mean:
        avg_ttft = np.mean([x for x in ttft_mean if x > 0])
        avg_decode_speed = np.mean([x for x in output_throughput_mean if x > 0])
        max_combined = max([x for x in combined_throughput_mean if x > 0], default=0)
        avg_success = np.mean(success_rates)
        
        summary_text = f"Avg TTFT: {avg_ttft:.2f}s | Avg Decode: {avg_decode_speed:.1f} tok/s | Max System: {max_combined:.1f} tok/s | Success: {avg_success:.1f}%"
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    # Save with high quality
    plt.savefig(output_image, 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📊 Enhanced benchmark plot saved to: {output_image}")
    print(f"📈 Analyzed {len(batch_sizes)} batch sizes with focus on prefill/decode separation")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_advanced_benchmark.py <json_file> <output_image>")
        print("Example: python plot_advanced_benchmark.py test_30k.json benchmark_results.png")
        sys.exit(1)

    json_file = sys.argv[1]
    output_image = sys.argv[2]
    plot_advanced_benchmark(json_file, output_image)