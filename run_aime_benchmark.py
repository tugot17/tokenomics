from time import perf_counter_ns
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as np
from typing import List, Dict, Any
import torch
import statistics
import vllm
import json
import os
from datetime import datetime

def create_sample_conversations(dataset_name: str, num_samples: int, seed: int = 42) -> List[List[Dict[str, str]]]:
    """Create conversation samples from the dataset."""
    ds = load_dataset(dataset_name)
    sampled_dataset = ds['train'].shuffle(seed=seed).select(range(num_samples))
    
    return [
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": question["Question"]}
        ] for question in sampled_dataset
    ]

def clear_cuda_cache():
    """Clear CUDA cache and ensure garbage collection."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def run_single_benchmark(
    llm: LLM,
    conversations: List[List[Dict[str, str]]],
    sampling_params: SamplingParams
) -> Dict[str, float]:
    """Run a single benchmark iteration and return metrics."""
    start = perf_counter_ns()
    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False)
    end = perf_counter_ns()
    
    total_time_seconds = (end - start) / 1e9
    lengths = [
        {
            "input_length": len(output.prompt_token_ids),
            "output_length": len(output.outputs[0].token_ids)
        }
        for output in outputs
    ]
    
    total_output_tokens = sum(length["output_length"] for length in lengths)
    tokens_per_second = total_output_tokens / total_time_seconds
    tokens_per_second_per_request = tokens_per_second / len(conversations)
        
    return {
        "total_output_tokens": total_output_tokens,
        "total_time_seconds": total_time_seconds,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_per_request": tokens_per_second_per_request
    }

def create_experiment_metadata(
    model_path: str,
    dataset_name: str,
    batch_sizes: List[int],
    num_runs: int,
    tensor_parallel_size: int,
    max_tokens: int,
    temperature: float,
    seed: int,
    num_warmup_steps: int
) -> Dict[str, Any]:
    """Create metadata dictionary for the experiment."""
    return {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "dataset_name": dataset_name,
        "batch_sizes": batch_sizes,
        "num_runs": num_runs,
        "tensor_parallel_size": tensor_parallel_size,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "num_warmup_steps": num_warmup_steps,
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_info": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None
    }

def save_results(results: Dict[str, Any], filename: str):
    """Save results to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_previous_results(filename: str) -> Dict[str, Any]:
    """Load previous results if they exist."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def print_batch_results(batch_size: int, metrics: Dict[str, Dict[str, float]]):
    """Print results for a single batch size."""
    print(f"\nResults for Batch Size: {batch_size}")
    print("-" * 80)
    
    for metric_name, stats in metrics.items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Std:  {stats['std']:.2f}")
        print(f"  Min:  {stats['min']:.2f}")
        print(f"  Max:  {stats['max']:.2f}")
    
    print("=" * 80)

def run_benchmarks(
    model_path: str,
    dataset_name: str,
    batch_sizes: List[int],
    results_file: str = "benchmark_results.json",
    num_runs: int = 5,
    tensor_parallel_size: int = 8,
    max_tokens: int = 2000,
    temperature: float = 0.5,
    seed: int = 42,
    num_warmup_steps: int = 5
) -> Dict[str, Any]:
    """
    Run benchmarks for different batch sizes and multiple iterations.
    Saves results incrementally and can resume from previous runs.
    """
    # Load previous results if they exist
    previous_results = load_previous_results(results_file)
    
    # Initialize or load results dictionary
    if previous_results is None:
        results = {
            "metadata": create_experiment_metadata(
                model_path, dataset_name, batch_sizes, num_runs,
                tensor_parallel_size, max_tokens, temperature, seed, num_warmup_steps
            ),
            "results": {}
        }
    else:
        results = previous_results
        print(f"Resuming from previous run. Already completed batch sizes: {list(results['results'].keys())}")
    
    # Initialize model
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
    )
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    
    # Warmup runs
    print("\nPerforming warmup runs...")
    for step in range(num_warmup_steps):
        run_single_benchmark(
            llm,
            create_sample_conversations(dataset_name, num_samples=batch_sizes[-1]),
            SamplingParams(temperature=temperature, max_tokens=100)
        )
    clear_cuda_cache()
    
    # Main benchmark runs
    for batch_size in tqdm(batch_sizes):
        # Skip if this batch size was already processed
        if str(batch_size) in results["results"]:
            print(f"\nSkipping batch size {batch_size} (already processed)")
            continue
            
        print(f"\nRunning benchmarks for batch size {batch_size}")
        batch_metrics = []
        conversations = create_sample_conversations(dataset_name, batch_size, seed)
        
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            metrics = run_single_benchmark(llm, conversations, sampling_params)
            batch_metrics.append(metrics)
            clear_cuda_cache()
        
        # Calculate statistics for this batch size
        batch_results = {
            metric: {
                "mean": statistics.mean(m[metric] for m in batch_metrics),
                "std": statistics.stdev(m[metric] for m in batch_metrics) if num_runs > 1 else 0,
                "min": min(m[metric] for m in batch_metrics),
                "max": max(m[metric] for m in batch_metrics)
            }
            for metric in batch_metrics[0].keys()
        }
        
        # Update results and save
        results["results"][str(batch_size)] = batch_results
        save_results(results, results_file)
        
        # Print results for this batch size
        print_batch_results(batch_size, batch_results)
    
    return results

def print_benchmark_results(results: Dict[str, Any]) -> None:
    """Print formatted benchmark results."""
    print("\nExperiment Metadata:")
    print("-" * 80)
    for key, value in results["metadata"].items():
        print(f"{key}: {value}")
    
    print("\nBenchmark Results:")
    print("=" * 80)
    for batch_size, metrics in results["results"].items():
        print_batch_results(int(batch_size), metrics)

# Example usage
if __name__ == "__main__":
    MODEL_PATH = "/nfs/checkpoint-tuning/deepseek/DeepSeek-R1-Distill-Llama-70B"
    DATASET_NAME = "gneubig/aime-1983-2024"
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    
    results = run_benchmarks(
        model_path=MODEL_PATH,
        dataset_name=DATASET_NAME,
        batch_sizes=BATCH_SIZES,
        num_runs=3,
        tensor_parallel_size=8,
        max_tokens=5000,
        temperature=0.5
    )
    
    print_benchmark_results(results)