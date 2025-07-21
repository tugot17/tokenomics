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

# Environment setup for clean execution (GenAI-Bench pattern)
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from sampling import Scenario, TextSampler, BatchSampler, DatasetConfig, DatasetLoader


async def single_request(session: aiohttp.ClientSession, api_base: str, model: str,
                        request_data: Dict[str, Any], temperature: float, max_tokens: int,
                        semaphore: asyncio.Semaphore, start_time: float, tokenizer=None) -> Dict:
    """Make a single API request with TTFT timing (streaming enabled)."""
    async with semaphore:
        request_start = time.time()
        time_at_first_token = None
        generated_text = ""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer dummy-key"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": request_data["prompt"]}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,  # Enable streaming for TTFT measurement
            "stream_options": {"include_usage": True}
        }
        
        try:
            async with session.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                
                if response.status == 200:
                    # Process streaming response to capture TTFT and exact usage
                    api_usage = None
                    
                    async for line in response.content:
                        line_text = line.decode('utf-8').strip()
                        
                        if line_text.startswith('data: '):
                            data_text = line_text[6:]  # Remove 'data: ' prefix
                            
                            if data_text == '[DONE]':
                                break
                            
                            try:
                                chunk_data = json.loads(data_text)
                                choices = chunk_data.get('choices', [])
                                
                                # Extract usage info (GenAI-Bench pattern)
                                if 'usage' in chunk_data:
                                    api_usage = chunk_data['usage']
                                
                                # Process content from choices
                                if choices:
                                    delta = choices[0].get('delta', {})
                                    content = delta.get('content', '')
                                    
                                    if content and time_at_first_token is None:
                                        time_at_first_token = time.time()
                                    
                                    if content:
                                        generated_text += content
                                        
                            except json.JSONDecodeError:
                                continue  # Skip invalid JSON chunks
                    
                    request_end = time.time()
                    
                    # Calculate TTFT-based metrics
                    ttft = time_at_first_token - request_start if time_at_first_token else 0
                    e2e_latency = request_end - request_start
                    output_latency = e2e_latency - ttft if ttft > 0 else e2e_latency
                    
                    # Get exact token counts from API usage (GenAI-Bench approach)
                    if api_usage:
                        input_tokens = api_usage.get("prompt_tokens", 0)
                        output_tokens = api_usage.get("completion_tokens", 0)
                    else:
                        # Fallback: use target for input, tokenize output with actual tokenizer
                        input_tokens = request_data.get("target_input_tokens", 0)
                        if tokenizer and generated_text:
                            try:
                                output_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
                            except Exception as e:
                                output_tokens = int(len(generated_text.split()) * 1.3)
                                print(f"⚠️  Tokenizer failed ({e}), using word estimation")
                        else:
                            output_tokens = int(len(generated_text.split()) * 1.3) if generated_text else 0
                        print(f"⚠️  No API usage info - using tokenizer fallback for output tokens")
                    
                    # Calculate throughput metrics like GenAI-Bench
                    input_throughput = input_tokens / ttft if ttft > 0 else 0
                    output_throughput = (output_tokens - 1) / output_latency if output_latency > 0 and output_tokens > 1 else 0
                    tpot = output_latency / (output_tokens - 1) if output_tokens > 1 else 0
                    
                    return {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "start_time": request_start,
                        "end_time": request_end,
                        "time_at_first_token": time_at_first_token,
                        "relative_start": request_start - start_time,
                        "relative_end": request_end - start_time,
                        # GenAI-Bench style metrics
                        "ttft": ttft,
                        "e2e_latency": e2e_latency,
                        "output_latency": output_latency,
                        "tpot": tpot,
                        "input_throughput": input_throughput,  # tokens/sec for prefill
                        "output_throughput": output_throughput,  # tokens/sec for decode
                        "generated_text": generated_text,
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
                         temperature: float, max_tokens: int, max_concurrent: int, tokenizer=None) -> Dict:
    """Run batch of requests asynchronously."""
    
    start_time = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for request_data in user_requests:
            task = single_request(session, api_base, model, request_data, temperature, max_tokens, semaphore, start_time, tokenizer)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    batch_total_seconds = end_time - start_time
    
    # Process results and track failures
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    
    total_requests = len(user_requests)
    success_count = len(successful_results)
    failure_count = len(failed_results)
    
    print(f"    📊 Request Results: {success_count}/{total_requests} successful, {failure_count} failed")
    
    if failure_count > 0:
        # Log some failure reasons
        error_samples = [r.get("error", "Unknown error") for r in failed_results[:3]]  # Show first 3 errors
        print(f"    ❌ Sample errors: {error_samples}")
    
    if not successful_results:
        return {
            "tokens": {"input_per_request": [], "output_per_request": []},
            "timings": {"batch_total_seconds": batch_total_seconds, "fastest_seconds": 0, "slowest_seconds": 0, "spread_seconds": 0},
            "prefill_metrics": {"ttft_per_request": [], "input_throughput_per_request": []},
            "decode_metrics": {"output_throughput_per_request": [], "tpot_per_request": []},
            "batch_metrics": {"total_output_tokens": 0, "batch_duration": batch_total_seconds, "combined_throughput": 0, "total_requests": 0},
            "failures": {"total_requests": total_requests, "successful": success_count, "failed": failure_count}
        }
    
    # Extract original metrics
    input_tokens = [r["input_tokens"] for r in successful_results]
    output_tokens = [r["output_tokens"] for r in successful_results]
    
    # Extract TTFT-based metrics (GenAI-Bench style)
    ttft_values = [r.get("ttft", 0) for r in successful_results if r.get("ttft", 0) > 0]
    input_throughput_values = [r.get("input_throughput", 0) for r in successful_results if r.get("input_throughput", 0) > 0]
    output_throughput_values = [r.get("output_throughput", 0) for r in successful_results if r.get("output_throughput", 0) > 0]
    tpot_values = [r.get("tpot", 0) for r in successful_results if r.get("tpot", 0) > 0]
    
    # Calculate timing metrics (relative to batch start)
    relative_ends = [r["relative_end"] for r in successful_results]
    fastest_seconds = min(relative_ends)
    slowest_seconds = max(relative_ends)
    spread_seconds = slowest_seconds - fastest_seconds
    
    
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
        # Prefill phase metrics (input processing)
        "prefill_metrics": {
            "ttft_per_request": ttft_values,  # Time to first token
            "input_throughput_per_request": input_throughput_values,  # Prefill speed
        },
        # Decode phase metrics (output generation)
        "decode_metrics": {
            "output_throughput_per_request": output_throughput_values,  # Decode speed per request
            "tpot_per_request": tpot_values  # Time per output token
        },
        # Batch-level system metrics (combined performance)
        "batch_metrics": {
            "total_output_tokens": sum(output_tokens),
            "batch_duration": batch_total_seconds,
            "combined_throughput": sum(output_throughput_values),  # Sum of all individual decode speeds
            "total_requests": len(successful_results)
        },
        # Request success/failure tracking
        "failures": {
            "total_requests": total_requests,
            "successful": success_count,
            "failed": failure_count
        }
    }


def worker_process(worker_id: int, user_requests: List[Dict], api_base: str, model: str,
                   temperature: float, max_tokens: int, max_concurrent: int, result_queue: multiprocessing.Queue, tokenizer_name: str = None):
    """Worker process for multiprocessing."""
    try:
        # Load tokenizer in worker process if needed
        tokenizer = None
        if tokenizer_name:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        result = asyncio.run(run_batch_async(user_requests, api_base, model, temperature, max_tokens, max_concurrent, tokenizer))
        result_queue.put(result)
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")
        result_queue.put(None)


def aggregate_metrics(worker_results: List[Dict]) -> Dict:
    """Aggregate metrics from multiple workers (enhanced with TTFT)."""
    
    # Combine all token lists
    all_input_tokens = []
    all_output_tokens = []
    
    # Combine TTFT metrics
    all_ttft = []
    all_input_throughput = []
    all_output_throughput = []
    all_tpot = []
    
    # Track timing info for overall batch timing
    all_fastest = []
    all_slowest = []
    
    # Track failure counts
    total_requests = 0
    total_successful = 0
    total_failed = 0
    
    # Track batch-level metrics
    all_combined_throughput = []
    all_batch_duration = []
    all_total_output_tokens = []
    all_total_requests = []
    
    latest_end = 0
    
    for metrics in worker_results:
        all_input_tokens.extend(metrics["tokens"]["input_per_request"])
        all_output_tokens.extend(metrics["tokens"]["output_per_request"])
        
        # Aggregate prefill metrics
        all_ttft.extend(metrics.get("prefill_metrics", {}).get("ttft_per_request", []))
        all_input_throughput.extend(metrics.get("prefill_metrics", {}).get("input_throughput_per_request", []))
        
        # Aggregate decode metrics
        all_output_throughput.extend(metrics.get("decode_metrics", {}).get("output_throughput_per_request", []))
        all_tpot.extend(metrics.get("decode_metrics", {}).get("tpot_per_request", []))
        
        # Aggregate batch-level metrics
        batch_metrics = metrics.get("batch_metrics", {})
        all_combined_throughput.append(batch_metrics.get("combined_throughput", 0))
        all_batch_duration.append(batch_metrics.get("batch_duration", 0))
        all_total_output_tokens.append(batch_metrics.get("total_output_tokens", 0))
        all_total_requests.append(batch_metrics.get("total_requests", 0))
        
        # Aggregate failure counts
        failures = metrics.get("failures", {})
        total_requests += failures.get("total_requests", 0)
        total_successful += failures.get("successful", 0)
        total_failed += failures.get("failed", 0)
        
        all_fastest.append(metrics["timings"]["fastest_seconds"])
        all_slowest.append(metrics["timings"]["slowest_seconds"])
        
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
        # Aggregated prefill metrics
        "prefill_metrics": {
            "ttft_per_request": all_ttft,
            "input_throughput_per_request": all_input_throughput
        },
        # Aggregated decode metrics  
        "decode_metrics": {
            "output_throughput_per_request": all_output_throughput,
            "tpot_per_request": all_tpot
        },
        # Aggregated batch-level metrics
        "batch_metrics": {
            "combined_throughput_per_batch": all_combined_throughput,  # Sum of individual throughputs per batch
            "batch_duration_per_batch": all_batch_duration,
            "total_output_tokens_per_batch": all_total_output_tokens,
            "total_requests_per_batch": all_total_requests
        },
        # Aggregated failure tracking
        "failures": {
            "total_requests": total_requests,
            "successful": total_successful,
            "failed": total_failed
        }
    }
    
    return aggregated


def run_multiprocess_benchmark(user_requests: List[Dict], api_base: str, model: str,
                             temperature: float, max_tokens: int, n_cores: int, tokenizer_name: str = None, max_asyncio_connections: int = 512) -> Dict:
    """Run benchmark using multiple processes (same format as original)."""
    
    # Disable HuggingFace Transformers' tokenizer parallelism in Rust
    # to prevent conflicts with Python multiprocessing (GenAI-Bench pattern)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
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
            args=(i, chunk, api_base, model, temperature, max_tokens, max_concurrent_per_process, result_queue, tokenizer_name)
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
    """Calculate statistics across multiple runs (enhanced with TTFT metrics)."""
    
    def safe_mean(values):
        return statistics.mean(values) if values else 0
    
    def safe_std(values):
        return statistics.stdev(values) if len(values) > 1 else 0
    
    # For list-based metrics, flatten all values from all runs
    all_input_tokens = []
    all_output_tokens = []
    
    # Individual request metrics (flattened across runs)
    all_ttft = []
    all_input_throughput = []
    all_output_throughput = []
    all_tpot = []
    
    # Batch-level metrics (one value per run)
    all_combined_throughput = []
    all_batch_duration = []
    
    # For single-value metrics, collect one value per run
    batch_total_seconds = []
    fastest_seconds = []
    slowest_seconds = []
    spread_seconds = []
    
    # Failure metrics (aggregated across runs)
    total_requests = 0
    total_successful = 0
    total_failed = 0
    
    for run_data in runs_data:
        all_input_tokens.extend(run_data["tokens"]["input_per_request"])
        all_output_tokens.extend(run_data["tokens"]["output_per_request"])
        
        # Prefill metrics
        prefill_metrics = run_data.get("prefill_metrics", {})
        all_ttft.extend(prefill_metrics.get("ttft_per_request", []))
        all_input_throughput.extend(prefill_metrics.get("input_throughput_per_request", []))
        
        # Decode metrics
        decode_metrics = run_data.get("decode_metrics", {})
        all_output_throughput.extend(decode_metrics.get("output_throughput_per_request", []))
        all_tpot.extend(decode_metrics.get("tpot_per_request", []))
        
        # Batch-level metrics (one value per run)
        batch_metrics = run_data.get("batch_metrics", {})
        all_combined_throughput.append(batch_metrics.get("combined_throughput", 0))
        all_batch_duration.append(batch_metrics.get("batch_duration", 0))
        
        batch_total_seconds.append(run_data["timings"]["batch_total_seconds"])
        fastest_seconds.append(run_data["timings"]["fastest_seconds"])
        slowest_seconds.append(run_data["timings"]["slowest_seconds"])
        spread_seconds.append(run_data["timings"]["spread_seconds"])
        
        # Aggregate failure metrics
        failures = run_data.get("failures", {})
        total_requests += failures.get("total_requests", 0)
        total_successful += failures.get("successful", 0)
        total_failed += failures.get("failed", 0)
    
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
        # Prefill phase statistics
        "prefill_metrics": {
            "ttft": {
                "mean": safe_mean(all_ttft),
                "std": safe_std(all_ttft)
            },
            "input_throughput": {
                "mean": safe_mean(all_input_throughput),
                "std": safe_std(all_input_throughput)
            }
        },
        # Decode phase statistics
        "decode_metrics": {
            "output_throughput": {
                "mean": safe_mean(all_output_throughput),
                "std": safe_std(all_output_throughput)
            },
            "tpot": {
                "mean": safe_mean(all_tpot),
                "std": safe_std(all_tpot)
            }
        },
        # Batch-level combined statistics
        "batch_metrics": {
            "combined_throughput": {
                "mean": safe_mean(all_combined_throughput),
                "std": safe_std(all_combined_throughput)
            },
            "batch_duration": {
                "mean": safe_mean(all_batch_duration),
                "std": safe_std(all_batch_duration)
            }
        },
        # Failure/Success metrics
        "reliability": {
            "total_requests": total_requests,
            "successful_requests": total_successful, 
            "failed_requests": total_failed,
            "success_rate": (total_successful / total_requests * 100) if total_requests > 0 else 0,
            "failure_rate": (total_failed / total_requests * 100) if total_requests > 0 else 0
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
    
    # Create samplers for on-demand generation
    text_sampler = TextSampler(tokenizer_name, dataset_loader)
    batch_sampler = BatchSampler(text_sampler)
    
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
            
            # Generate batch of requests using scenario (on-demand)
            sampled_requests = batch_sampler.sample_batch(scenario, batch_size)
            user_requests = []
            for request in sampled_requests:
                # Use scenario's output tokens unless overridden by --max-tokens
                max_tokens = args.max_tokens if args.max_tokens is not None else request.target_output_tokens
                
                # Ensure max_tokens is reasonable (at least 1, at most 4096)
                max_tokens = max(1, min(4096, int(max_tokens)))
                
                user_requests.append({
                    "prompt": request.prompt,
                    "max_tokens": max_tokens,
                    "target_input_tokens": request.target_input_tokens,
                    "target_output_tokens": request.target_output_tokens
                })
            
            # Run the batch (using multiprocessing like original)
            run_data = run_multiprocess_benchmark(
                user_requests, args.api_base, args.model, args.temperature, 
                args.max_tokens, multiprocessing.cpu_count(), tokenizer_name
            )
            
            runs_data.append(run_data)
            
            # Print progress with proper TTFT metrics and failure tracking
            failures = run_data.get("failures", {})
            total_req = failures.get("total_requests", 0)
            successful = failures.get("successful", 0)
            failed = failures.get("failed", 0)
            
            if run_data["tokens"]["output_per_request"]:
                total_input_tokens = sum(run_data["tokens"]["input_per_request"])
                total_output_tokens = sum(run_data["tokens"]["output_per_request"])
                batch_time = run_data["timings"]["batch_total_seconds"]
                
                # Extract prefill and decode metrics for reporting
                prefill_metrics = run_data.get("prefill_metrics", {})
                decode_metrics = run_data.get("decode_metrics", {})
                
                ttft_values = prefill_metrics.get("ttft_per_request", [])
                input_throughput_values = prefill_metrics.get("input_throughput_per_request", [])
                output_throughput_values = decode_metrics.get("output_throughput_per_request", [])
                
                # Calculate averages per request
                avg_ttft = statistics.mean(ttft_values) if ttft_values else 0
                avg_input_throughput = statistics.mean(input_throughput_values) if input_throughput_values else 0
                avg_output_throughput = statistics.mean(output_throughput_values) if output_throughput_values else 0
                
                # Extract batch-level combined throughput
                batch_metrics = run_data.get("batch_metrics", {})
                combined_throughput = batch_metrics.get("combined_throughput", 0)
                
                success_rate = (successful / total_req * 100) if total_req > 0 else 0
                print(f"    ✅ Batch: {batch_time:.2f}s | Success: {successful}/{total_req} ({success_rate:.1f}%)")
                if failed > 0:
                    print(f"       ❌ {failed} requests failed")
                print(f"       Tokens - Input: {total_input_tokens:,} | Output: {total_output_tokens:,}")
                print(f"       Combined Throughput: {combined_throughput:.1f} tok/s (sum of all decode speeds)")
                
                if avg_ttft > 0:
                    print(f"       Per-Request Avg - TTFT: {avg_ttft:.3f}s | Prefill: {avg_input_throughput:.1f} tok/s | Decode: {avg_output_throughput:.1f} tok/s")
            else:
                print(f"    ❌ All {total_req} requests failed")
        
        # Calculate statistics for this batch size
        results["results"][str(batch_size)] = calculate_stats(runs_data)
    
    # Save results (same format as original)
    with open(args.results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Benchmark completed! Results saved to {args.results_file}")


if __name__ == "__main__":
    main()