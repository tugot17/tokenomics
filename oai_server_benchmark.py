import argparse
import asyncio
import aiohttp
import time
import statistics
import json
import multiprocessing
import os
from datetime import datetime
from datasets import load_dataset
from typing import List, Dict, Any, Tuple

class DatasetHandler:
    @staticmethod
    def aime_handler(num_samples, seed):
        ds = load_dataset("gneubig/aime-1983-2024")
        dataset_size = len(ds["train"])
        if num_samples <= dataset_size:
            sampled_dataset = ds["train"].shuffle(seed=seed).select(range(num_samples))
        else:
            print(f"Requested {num_samples} samples, dataset has {dataset_size} samples")
            print(f"Will repeat dataset {(num_samples + dataset_size - 1) // dataset_size} times")
            
            shuffled_ds = ds["train"].shuffle(seed=seed)
            
            full_cycles = num_samples // dataset_size
            remainder = num_samples % dataset_size
            
            indices = []
            for cycle in range(full_cycles):
                indices.extend(range(dataset_size))
            if remainder > 0:
                indices.extend(range(remainder))
            
            sampled_dataset = shuffled_ds.select(indices)
        
        conversations = []
        for item in sampled_dataset:
            messages = [
                {"role": "system", "content": "You are an assistant that helps students solve challenging problems."},
                {"role": "user", "content": item["Question"]}
            ]
            conversations.append(messages)
        
        return conversations

DATASET_HANDLERS = {
    "aime": DatasetHandler.aime_handler
}

def create_sample_conversations(dataset_key: str, num_samples: int, seed: int = 42):
    handler = DATASET_HANDLERS.get(dataset_key)
    if not handler:
        raise ValueError(f"Unknown dataset key: {dataset_key}")
    return handler(num_samples, seed)

class AsyncBenchmark:
    def __init__(self, api_base: str, model: str, max_concurrent: int):
        self.api_base = api_base
        self.model = model
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    def create_session(self):
        """Create aiohttp session with optimized connection settings"""
        # Connection settings optimized for high-concurrency benchmarking
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,  
            limit_per_host=self.max_concurrent,
            ttl_dns_cache=300,  # DNS cache TTL - longer is better for benchmarking
            use_dns_cache=True,
            keepalive_timeout=30,  # Keep connections alive for reuse
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=300,  # Total request timeout
            connect=30,  # Connection establishment timeout
            sock_read=60,  # Individual socket read timeout
            sock_connect=10  # Socket connection timeout
        )

        # Use large read buffer (10MB) like sglang for handling large responses
        read_bufsize = 10 * 1024 * 1024  # 10 MB buffer
        
        return aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            read_bufsize=read_bufsize,
            headers={"Authorization": "Bearer sk-dummy"}
        )
    
    async def call_completion(self, session: aiohttp.ClientSession, messages: List[Dict], 
                            temperature: float, max_tokens: int, request_id: int = 0) -> Tuple[int, int, float, float, float]:
        """Make async API call with semaphore-based concurrency control"""
        async with self.semaphore:
            try:
                start_time = time.perf_counter()
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                async with session.post(f"{self.api_base}/chat/completions", json=payload) as response:
                    if response.status != 200:
                        print(f"Request {request_id} failed with status {response.status}")
                        return 0, 0, 0, 0, 0
                    
                    data = await response.json()
                    end_time = time.perf_counter()
                    
                    elapsed = end_time - start_time
                    prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                    completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
                    tokens_per_second = completion_tokens / elapsed if elapsed > 0 else 0
                    
                    return prompt_tokens, completion_tokens, tokens_per_second, start_time, end_time
                    
            except Exception as e:
                print(f"Error in request {request_id}: {e}")
                return 0, 0, 0, 0, 0

    async def run_benchmark(self, conversations: List[List[Dict]], temperature: float, max_tokens: int):
        """Run benchmark with async concurrency"""
        batch_start_time = time.perf_counter()
        
        async with self.create_session() as session:
            # Create tasks for all requests
            tasks = [
                self.call_completion(session, conv, temperature, max_tokens, i)
                for i, conv in enumerate(conversations)
            ]
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        batch_end_time = time.perf_counter()
        
        # Filter out exceptions and failed requests
        valid_results = [r for r in results if isinstance(r, tuple) and r[0] > 0]
        
        if not valid_results:
            print("No valid results obtained!")
            return None
            
        relative_end_times = [end_time - batch_start_time for _, _, _, _, end_time in valid_results]
        
        prompt_tokens_list = [r[0] for r in valid_results]
        completion_tokens_list = [r[1] for r in valid_results]
        tps_list = [r[2] for r in valid_results]
        
        metrics = {
            "tokens": {
                "input_per_request": prompt_tokens_list,
                "output_per_request": completion_tokens_list
            },
            "timings": {
                "batch_total_seconds": batch_end_time - batch_start_time,
                "fastest_seconds": min(relative_end_times),
                "slowest_seconds": max(relative_end_times),
                "spread_seconds": max(relative_end_times) - min(relative_end_times)
            },
            "throughput": {
                "batch_tokens_per_second": sum(completion_tokens_list) / (batch_end_time - batch_start_time) if batch_end_time > batch_start_time else 0,
                "request_tokens_per_second": tps_list
            }
        }

        return metrics

def worker_process(worker_id: int, conversations: List[List[Dict]], api_base: str, model: str, 
                  temperature: float, max_tokens: int, max_concurrent: int, result_queue):
    """Worker process that runs async benchmark for a subset of conversations"""
    
    async def run_worker():
        # print(f"Worker {worker_id} starting with {len(conversations)} conversations")
        
        benchmark = AsyncBenchmark(api_base, model, max_concurrent)
        metrics = await benchmark.run_benchmark(conversations, temperature, max_tokens)
        
        if metrics:
            # Add worker ID to metrics for debugging
            metrics["worker_id"] = worker_id
            result_queue.put(metrics)
            # print(f"Worker {worker_id} completed successfully")
        else:
            # print(f"Worker {worker_id} failed")
            result_queue.put(None)
    
    # Run the async function in this process
    asyncio.run(run_worker())

def aggregate_metrics(worker_results: List[Dict]) -> Dict:
    """Aggregate metrics from multiple worker processes"""
    if not worker_results:
        return None
    
    # Combine all token lists
    all_input_tokens = []
    all_output_tokens = []
    all_tps = []
    
    # Track timing info for overall batch timing
    earliest_start = float('inf')
    latest_end = 0
    all_fastest = []
    all_slowest = []
    
    total_output_tokens = 0
    
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
            "fastest_seconds": min(all_fastest),
            "slowest_seconds": max(all_slowest),
            "spread_seconds": max(all_slowest) - min(all_fastest)
        },
        "throughput": {
            "batch_tokens_per_second": total_output_tokens / latest_end if latest_end > 0 else 0,
            "request_tokens_per_second": all_tps
        }
    }
    
    return aggregated

def run_multiprocess_benchmark(conversations: List[List[Dict]], api_base: str, model: str,
                             temperature: float, max_tokens: int, n_cores: int, max_asyncio_connections: int = 512) -> Dict:
    """Run benchmark using multiple processes"""
    
    total_conversations = len(conversations)

    # set the number of cores s.t. the asyncio semaphore has asyncio_connections connections
    cores_for_asyncio_connections = total_conversations // max_asyncio_connections + 1
    
    # Handle case where we have fewer conversations than cores
    actual_cores = min(n_cores, cores_for_asyncio_connections)
    
    # Split conversations across processes
    conversations_per_process = total_conversations // actual_cores
    remainder = total_conversations % actual_cores
    
    # Create conversation chunks for each process
    conversation_chunks = []
    start_idx = 0
    
    for i in range(actual_cores):
        # Add one extra conversation to first 'remainder' processes
        chunk_size = conversations_per_process + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        
        conversation_chunks.append(conversations[start_idx:end_idx])
        start_idx = end_idx
    
    # Calculate max concurrent connections per process
    # Use a reasonable limit per process to avoid overwhelming the system
    max_concurrent_per_process = max(1, conversations_per_process + 1)

    
    result_queue = multiprocessing.Queue(maxsize=actual_cores * 2)
    
    # Create and start processes
    processes = []
    for i, chunk in enumerate(conversation_chunks):
        p = multiprocessing.Process(
            target=worker_process,
            args=(i, chunk, api_base, model, temperature, max_tokens, max_concurrent_per_process, result_queue)
        )
        processes.append(p)
        p.start()
    
    worker_results = []
    for _ in range(actual_cores):
        try:
            result = result_queue.get(timeout=600)  # 10 minute timeout per process
            if result is not None:
                worker_results.append(result)
        except:
            print(f"Warning: Failed to get result from a worker process")
    
    # Now wait for all processes to complete
    for p in processes:
        p.join(timeout=600)  # 10 minute timeout per process
        if p.is_alive():
            print(f"Warning: Process {p.pid} didn't finish in time, terminating")
            p.terminate()
            p.join()
    
    # Aggregate results
    if worker_results:
        return aggregate_metrics(worker_results)
    else:
        return None

def calculate_stats(run_metrics):
    def compute_stats(metrics_list, key_path):
        if isinstance(metrics_list[0][key_path[0]][key_path[1]], list):
            # For token lists and tps lists
            values = [item for m in metrics_list for item in m[key_path[0]][key_path[1]]]
        else:
            # For single values
            values = [get_nested_value(m, key_path) for m in metrics_list]
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def get_nested_value(d, path):
        for key in path:
            d = d[key]
        return d
    
    stats = {}
    for category in ["tokens", "timings", "throughput"]:
        stats[category] = {}
        for metric in run_metrics[0][category]:
            stats[category][metric] = compute_stats(run_metrics, [category, metric])
    
    return stats

def save_results(results: dict, filename: str):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

async def main():
    parser = argparse.ArgumentParser(
        description="Multiprocess Async Benchmark for vLLM Server using OpenAI Chat Completion API"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model tag to use (e.g., 'distill-llama-8b').")
    parser.add_argument("--dataset_key", type=str, default="aime",
                        help="Dataset key (e.g., 'aime', 'conversation')")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1",
                        help="Base URL of the vLLM server API.")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8",
                        help="Comma-separated batch sizes (e.g., '1,2,4,8').")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs per batch size.")
    parser.add_argument("--warmup_runs", type=int, default=3,
                        help="Number of warmup runs before each batch size.")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum tokens to generate per request.")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature for generation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset sampling.")
    parser.add_argument("--n_cores", type=int, default=int(os.cpu_count()/2),
                        help="Number of processes to use for parallel execution.")
    parser.add_argument("--description", type=str, default="",
                        help="Optional description or notes for the experiment.")
    parser.add_argument("--results_file", type=str, default="async_benchmark_results.json",
                        help="Path to JSON file for saving results.")
    args = parser.parse_args()

    batch_sizes = [int(bs.strip()) for bs in args.batch_sizes.split(",") if bs.strip()]
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "dataset_key": args.dataset_key,
            "api_base": args.api_base,
            "batch_sizes": batch_sizes,
            "num_runs": args.num_runs,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "warmup_runs": args.warmup_runs,
            "n_cores": args.n_cores,
            "description": args.description
        },
        "results": {}
    }
    
    for batch_size in batch_sizes:
        print(f"\n=== Benchmarking Batch Size: {batch_size} ===")
        
        print(f"Performing {args.warmup_runs} warmup runs...", end="", flush=True)
        for _ in range(args.warmup_runs):
            conversations = create_sample_conversations(args.dataset_key, num_samples=1, seed=args.seed)
            # Simple warmup with single process
            benchmark = AsyncBenchmark(args.api_base, args.model, max_concurrent=1)
            async with benchmark.create_session() as session:
                await benchmark.call_completion(
                    session,
                    conversations[0], 
                    args.temperature, 
                    max_tokens=30
                )
        print(" done")
        
        run_metrics = []

        for run in range(1, args.num_runs + 1):
            print(f" Run {run}/{args.num_runs} ... ", end="", flush=True)
            conversations = create_sample_conversations(args.dataset_key, num_samples=batch_size, seed=args.seed)
            
            metrics = run_multiprocess_benchmark(
                conversations, 
                args.api_base, 
                args.model,
                args.temperature, 
                args.max_tokens, 
                args.n_cores
            )
            
            if metrics:
                run_metrics.append(metrics)
                
                # Calculate mean for this run's output stats
                mean_output = statistics.mean(metrics['tokens']['output_per_request'])
                mean_tps = statistics.mean(metrics['throughput']['request_tokens_per_second'])
                
                print(f" Run {run}/{args.num_runs}:")
                print(f"  Output tokens/req: {mean_output:.2f}")
                print(f"  Batch time: {metrics['timings']['batch_total_seconds']:.2f}s")
                print(f"  Avg Request TPS: {mean_tps:.2f}")
                print(f"  Batch TPS: {metrics['throughput']['batch_tokens_per_second']:.2f}")
            else:
                print(" FAILED")

        if run_metrics:
            stats = calculate_stats(run_metrics)
            results["results"][str(batch_size)] = stats

            print("\nSummary:")
            for category, metrics in stats.items():
                print(f"\n{category.title()}:")
                for metric, values in metrics.items():
                    print(f"  {metric}: {values['mean']:.2f} ± {values['std']:.2f}")

    save_results(results, args.results_file)
    print(f"\nBenchmark results saved to {args.results_file}")

if __name__ == "__main__":
    # Enable multiprocessing support
    multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())