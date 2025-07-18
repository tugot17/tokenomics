"""
Enhanced OAI server benchmark with scenario-based sampling.

This script extends the original benchmark with GenAI-Bench inspired
sampling capabilities while maintaining the exact same output format.
"""

import argparse
import asyncio
import aiohttp
import time
import statistics
import json
import multiprocessing
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sampling import Scenario, TextSampler, BatchSampler, DatasetConfig, DatasetLoader


async def single_request(session: aiohttp.ClientSession, api_base: str, model: str,
                        request_data: Dict[str, Any], temperature: float, max_tokens: int,
                        semaphore: asyncio.Semaphore, start_time: float) -> Dict:
    """Make a single API request with timing."""
    async with semaphore:
        request_start = time.time()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer dummy-key"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": request_data["prompt"]}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            async with session.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                request_end = time.time()
                
                if response.status == 200:
                    data = await response.json()
                    usage = data.get("usage", {})
                    
                    return {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "start_time": request_start,
                        "end_time": request_end,
                        "relative_start": request_start - start_time,
                        "relative_end": request_end - start_time,
                        "success": True
                    }
                else:
                    error_text = await response.text()
                    return {
                        "error": f"API error {response.status}: {error_text}",
                        "success": False
                    }
                    
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }


async def run_batch_async(user_requests: List[Dict], api_base: str, model: str,
                         temperature: float, max_tokens: int, max_concurrent: int) -> Dict:
    """Run batch of requests asynchronously."""
    
    start_time = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for request_data in user_requests:
            task = single_request(session, api_base, model, request_data, temperature, max_tokens, semaphore, start_time)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    batch_total_seconds = end_time - start_time
    
    # Process results
    successful_results = [r for r in results if r.get("success", False)]
    
    if not successful_results:
        return {
            "tokens": {"input_per_request": [], "output_per_request": []},
            "timings": {"batch_total_seconds": batch_total_seconds, "fastest_seconds": 0, "slowest_seconds": 0, "spread_seconds": 0},
            "throughput": {"batch_tokens_per_second": 0, "request_tokens_per_second": []}
        }
    
    # Extract metrics
    input_tokens = [r["input_tokens"] for r in successful_results]
    output_tokens = [r["output_tokens"] for r in successful_results]
    
    # Calculate timing metrics (relative to batch start)
    relative_ends = [r["relative_end"] for r in successful_results]
    fastest_seconds = min(relative_ends)
    slowest_seconds = max(relative_ends)
    spread_seconds = slowest_seconds - fastest_seconds
    
    # Calculate throughput metrics
    total_output_tokens = sum(output_tokens)
    batch_tokens_per_second = total_output_tokens / batch_total_seconds if batch_total_seconds > 0 else 0
    
    # Calculate per-request tokens per second
    request_tokens_per_second = []
    for r in successful_results:
        request_duration = r["end_time"] - r["start_time"]
        if request_duration > 0:
            tps = r["output_tokens"] / request_duration
            request_tokens_per_second.append(tps)
    
    return {
        "tokens": {
            "input_per_request": input_tokens,
            "output_per_request": output_tokens
        },
        "timings": {
            "batch_total_seconds": batch_total_seconds,
            "fastest_seconds": fastest_seconds,
            "slowest_seconds": slowest_seconds,
            "spread_seconds": spread_seconds
        },
        "throughput": {
            "batch_tokens_per_second": batch_tokens_per_second,
            "request_tokens_per_second": request_tokens_per_second
        }
    }


def worker_process(worker_id: int, user_requests: List[Dict], api_base: str, model: str,
                   temperature: float, max_tokens: int, max_concurrent: int, result_queue: multiprocessing.Queue):
    """Worker process for multiprocessing."""
    try:
        result = asyncio.run(run_batch_async(user_requests, api_base, model, temperature, max_tokens, max_concurrent))
        result_queue.put(result)
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")
        result_queue.put(None)


def aggregate_metrics(worker_results: List[Dict]) -> Dict:
    """Aggregate metrics from multiple workers (same format as original)."""
    
    # Combine all token lists
    all_input_tokens = []
    all_output_tokens = []
    all_tps = []
    
    # Track timing info for overall batch timing
    all_fastest = []
    all_slowest = []
    
    total_output_tokens = 0
    latest_end = 0
    
    for metrics in worker_results:
        all_input_tokens.extend(metrics["tokens"]["input_per_request"])
        all_output_tokens.extend(metrics["tokens"]["output_per_request"])
        all_tps.extend(metrics["throughput"]["request_tokens_per_second"])
        
        all_fastest.append(metrics["timings"]["fastest_seconds"])
        all_slowest.append(metrics["timings"]["slowest_seconds"])
        
        total_output_tokens += sum(metrics["tokens"]["output_per_request"])
        
        # For overall batch timing, we need to consider that workers ran in parallel
        # So we take the maximum batch time across all workers
        latest_end = max(latest_end, metrics["timings"]["batch_total_seconds"])
    
    # Aggregate metrics
    aggregated = {
        "tokens": {
            "input_per_request": all_input_tokens,
            "output_per_request": all_output_tokens
        },
        "timings": {
            "batch_total_seconds": latest_end,  # Max time across all workers
            "fastest_seconds": min(all_fastest) if all_fastest else 0,
            "slowest_seconds": max(all_slowest) if all_slowest else 0,
            "spread_seconds": max(all_slowest) - min(all_fastest) if all_slowest and all_fastest else 0
        },
        "throughput": {
            "batch_tokens_per_second": total_output_tokens / latest_end if latest_end > 0 else 0,
            "request_tokens_per_second": all_tps
        }
    }
    
    return aggregated


def run_multiprocess_benchmark(user_requests: List[Dict], api_base: str, model: str,
                             temperature: float, max_tokens: int, n_cores: int, max_asyncio_connections: int = 512) -> Dict:
    """Run benchmark using multiple processes (same format as original)."""
    
    total_requests = len(user_requests)

    # Set the number of cores s.t. the asyncio semaphore has asyncio_connections connections
    cores_for_asyncio_connections = total_requests // max_asyncio_connections + 1
    
    # Handle case where we have fewer requests than cores
    actual_cores = min(n_cores, cores_for_asyncio_connections)
    
    # Split requests across processes
    requests_per_process = total_requests // actual_cores
    remainder = total_requests % actual_cores
    
    # Create request chunks for each process
    request_chunks = []
    start_idx = 0
    
    for i in range(actual_cores):
        # Add one extra request to first 'remainder' processes
        chunk_size = requests_per_process + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        
        request_chunks.append(user_requests[start_idx:end_idx])
        start_idx = end_idx
    
    # Calculate max concurrent connections per process
    max_concurrent_per_process = max(1, requests_per_process + 1)
    
    result_queue = multiprocessing.Queue(maxsize=actual_cores * 2)
    
    # Create and start processes
    processes = []
    for i, chunk in enumerate(request_chunks):
        p = multiprocessing.Process(
            target=worker_process,
            args=(i, chunk, api_base, model, temperature, max_tokens, max_concurrent_per_process, result_queue)
        )
        processes.append(p)
        p.start()
    
    worker_results = []
    for _ in range(actual_cores):
        try:
            result = result_queue.get(timeout=1800)  # 30 minute timeout per process
            if result is not None:
                worker_results.append(result)
        except:
            print(f"Warning: Failed to get result from a worker process")
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    if not worker_results:
        raise RuntimeError("No worker processes completed successfully")
    
    return aggregate_metrics(worker_results)


def calculate_stats(runs_data: List[Dict]) -> Dict:
    """Calculate statistics across multiple runs (same format as original)."""
    
    def safe_mean(values):
        return statistics.mean(values) if values else 0
    
    def safe_std(values):
        return statistics.stdev(values) if len(values) > 1 else 0
    
    # For list-based metrics, flatten all values from all runs
    all_input_tokens = []
    all_output_tokens = []
    all_request_tps = []
    
    # For single-value metrics, collect one value per run
    batch_total_seconds = []
    fastest_seconds = []
    slowest_seconds = []
    spread_seconds = []
    batch_tokens_per_second = []
    
    for run_data in runs_data:
        all_input_tokens.extend(run_data["tokens"]["input_per_request"])
        all_output_tokens.extend(run_data["tokens"]["output_per_request"])
        all_request_tps.extend(run_data["throughput"]["request_tokens_per_second"])
        
        batch_total_seconds.append(run_data["timings"]["batch_total_seconds"])
        fastest_seconds.append(run_data["timings"]["fastest_seconds"])
        slowest_seconds.append(run_data["timings"]["slowest_seconds"])
        spread_seconds.append(run_data["timings"]["spread_seconds"])
        batch_tokens_per_second.append(run_data["throughput"]["batch_tokens_per_second"])
    
    return {
        "tokens": {
            "input_per_request": {
                "mean": safe_mean(all_input_tokens),
                "std": safe_std(all_input_tokens)
            },
            "output_per_request": {
                "mean": safe_mean(all_output_tokens),
                "std": safe_std(all_output_tokens)
            }
        },
        "timings": {
            "batch_total_seconds": {
                "mean": safe_mean(batch_total_seconds),
                "std": safe_std(batch_total_seconds)
            },
            "fastest_seconds": {
                "mean": safe_mean(fastest_seconds),
                "std": safe_std(fastest_seconds)
            },
            "slowest_seconds": {
                "mean": safe_mean(slowest_seconds),
                "std": safe_std(slowest_seconds)
            },
            "spread_seconds": {
                "mean": safe_mean(spread_seconds),
                "std": safe_std(spread_seconds)
            }
        },
        "throughput": {
            "batch_tokens_per_second": {
                "mean": safe_mean(batch_tokens_per_second),
                "std": safe_std(batch_tokens_per_second)
            },
            "request_tokens_per_second": {
                "mean": safe_mean(all_request_tps),
                "std": safe_std(all_request_tps)
            }
        }
    }


def main():
    """Main function to run the enhanced benchmark."""
    parser = argparse.ArgumentParser(description="Enhanced OAI server benchmark with scenario-based sampling")
    
    # API configuration
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="API base URL")
    parser.add_argument("--api-key", default="dummy-key", help="API key")
    parser.add_argument("--model", required=True, help="Model name")
    
    # Scenario configuration
    parser.add_argument("--scenario", required=True, help="Scenario string (e.g., 'N(480,240)/(300,150)', 'D(100,100)')")
    parser.add_argument("--dataset-config", required=True, help="Path to dataset configuration JSON")
    parser.add_argument("--tokenizer", help="Tokenizer name (defaults to model name)")
    
    # Benchmark parameters (same as original)
    parser.add_argument("--batch-sizes", default="1,2,4,8", help="Comma-separated batch sizes")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs per batch size")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature parameter")
    parser.add_argument("--description", default="Enhanced benchmark", help="Description of the benchmark")
    
    # Output configuration
    parser.add_argument("--results-file", default="enhanced_benchmark_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    
    # Initialize scenario and dataset
    scenario = Scenario.from_string(args.scenario)
    dataset_config = DatasetConfig.from_file(args.dataset_config)
    dataset_loader = DatasetLoader(dataset_config)
    
    # Initialize tokenizer (default to model name if not specified)
    tokenizer_name = args.tokenizer if args.tokenizer else args.model
    
    # Pre-generate all requests to avoid tokenizer in worker processes
    print("🔄 Pre-generating requests for all batch sizes...")
    all_requests = {}
    text_sampler = TextSampler(tokenizer_name, dataset_loader)
    batch_sampler = BatchSampler(text_sampler)
    
    # Generate all requests upfront
    for batch_size in batch_sizes:
        all_requests[batch_size] = []
        for run_idx in range(args.num_runs):
            sampled_requests = batch_sampler.sample_batch(scenario, batch_size)
            user_requests = []
            for request in sampled_requests:
                user_requests.append({
                    "prompt": request.prompt,
                    "max_tokens": args.max_tokens,
                    "target_input_tokens": request.target_input_tokens,
                    "target_output_tokens": request.target_output_tokens
                })
            all_requests[batch_size].append(user_requests)
    
    print(f"📊 Initialized benchmark with scenario: {args.scenario}")
    print(f"📚 Loaded dataset: {len(dataset_loader)} samples")
    print(f"🚀 Starting enhanced benchmark: {args.description}")
    
    # Create results structure (same format as original)
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "scenario": args.scenario,
            "dataset_config": dataset_config.config,
            "api_base": args.api_base,
            "batch_sizes": batch_sizes,
            "num_runs": args.num_runs,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "description": args.description
        },
        "results": {}
    }
    
    # Run benchmark for each batch size
    for batch_size in batch_sizes:
        print(f"\n🔄 Testing batch size: {batch_size}")
        
        runs_data = []
        for run_idx in range(args.num_runs):
            print(f"  Run {run_idx + 1}/{args.num_runs}")
            
            # Use pre-generated requests (avoids tokenizer in worker process)
            user_requests = all_requests[batch_size][run_idx]
            
            # Run the batch (using multiprocessing like original)
            run_data = run_multiprocess_benchmark(
                user_requests, args.api_base, args.model, args.temperature, 
                args.max_tokens, multiprocessing.cpu_count()
            )
            
            runs_data.append(run_data)
            
            # Print progress (same format as original)
            if run_data["tokens"]["output_per_request"]:
                total_output_tokens = sum(run_data["tokens"]["output_per_request"])
                batch_time = run_data["timings"]["batch_total_seconds"]
                throughput = run_data["throughput"]["batch_tokens_per_second"]
                
                print(f"    ✅ Batch time: {batch_time:.2f}s, Output tokens: {total_output_tokens}, Throughput: {throughput:.1f} tokens/s")
            else:
                print(f"    ❌ Run failed")
        
        # Calculate statistics for this batch size
        results["results"][str(batch_size)] = calculate_stats(runs_data)
    
    # Save results (same format as original)
    with open(args.results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Benchmark completed! Results saved to {args.results_file}")


if __name__ == "__main__":
    main()