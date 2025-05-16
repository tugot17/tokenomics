import argparse
import concurrent.futures
import time
import statistics
import json
from datetime import datetime
from datasets import load_dataset
from openai import OpenAI

class DatasetHandler:
    @staticmethod
    def aime_handler(num_samples, seed):
        ds = load_dataset("gneubig/aime-1983-2024")
        sampled_dataset = ds["train"].shuffle(seed=seed).select(range(num_samples))
        
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

def create_sample_conversations(client, model: str, dataset_key: str, num_samples: int, seed: int = 42):
    handler = DATASET_HANDLERS.get(dataset_key)
    if not handler:
        raise ValueError(f"Unknown dataset key: {dataset_key}")
    return handler(num_samples, seed)

def call_server_completion(client, model: str, messages, temperature: float, max_tokens: int):
    try:
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        tokens_per_second = completion_tokens / elapsed if elapsed > 0 else 0
        return prompt_tokens, completion_tokens, tokens_per_second, start_time, end_time
    except Exception as e:
        print(f"Error during API call: {e}")
        return 0, 0, 0, 0, 0

def run_benchmark(client, model: str, conversations, temperature: float, max_tokens: int):
    batch_start_time = time.perf_counter()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(conversations)) as executor:
        futures = [
            executor.submit(call_server_completion, client, model, conv, temperature, max_tokens)
            for conv in conversations
        ]
        for future in concurrent.futures.as_completed(futures):
            prompt_tokens, completion_tokens, tps, start_time, end_time = future.result()
            results.append((prompt_tokens, completion_tokens, tps, start_time, end_time))
    
    batch_end_time = time.perf_counter()
    
    relative_end_times = [end_time - batch_start_time for _, _, _, _, end_time in results]
    
    prompt_tokens_list = [r[0] for r in results]
    completion_tokens_list = [r[1] for r in results]
    tps_list = [r[2] for r in results]
    
    metrics = {
        "tokens": {
            "input_per_request": prompt_tokens_list,  # Store full list
            "output_per_request": completion_tokens_list  # Store full list
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

def main():
    parser = argparse.ArgumentParser(
        description="Simplified Benchmark for vLLM Server using OpenAI Chat Completion API"
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
    parser.add_argument("--description", type=str, default="",
                        help="Optional description or notes for the experiment.")
    parser.add_argument("--results_file", type=str, default="server_benchmark_results.json",
                        help="Path to JSON file for saving results.")
    args = parser.parse_args()

    batch_sizes = [int(bs.strip()) for bs in args.batch_sizes.split(",") if bs.strip()]
    client = OpenAI(api_key="sk-dummy", base_url=args.api_base)

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
            "description": args.description
        },
        "results": {}
    }

    print(f"Starting benchmark for model: {args.model}")
    for batch_size in batch_sizes:
        print(f"\n=== Benchmarking Batch Size: {batch_size} ===")
        
        print(f"Performing {args.warmup_runs} warmup runs...", end="", flush=True)
        for _ in range(args.warmup_runs):
            conversations = create_sample_conversations(
                client, args.model, args.dataset_key, num_samples=1, seed=args.seed)
            call_server_completion(client, args.model, conversations[0], args.temperature, max_tokens=30)
        print(" done")
        
        run_metrics = []

        for run in range(1, args.num_runs + 1):
            print(f" Run {run}/{args.num_runs} ... ", end="", flush=True)
            conversations = create_sample_conversations(
                client, args.model, args.dataset_key, num_samples=batch_size, seed=args.seed)
            metrics = run_benchmark(client, args.model, conversations, args.temperature, args.max_tokens)
            run_metrics.append(metrics)
            
            # Calculate mean for this run's output stats
            mean_output = statistics.mean(metrics['tokens']['output_per_request'])
            mean_tps = statistics.mean(metrics['throughput']['request_tokens_per_second'])
            
            print(f" Run {run}/{args.num_runs}:")
            print(f"  Output tokens/req: {mean_output:.2f}")
            print(f"  Batch time: {metrics['timings']['batch_total_seconds']:.2f}s")
            print(f"  Avg Request TPS: {mean_tps:.2f}")

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
    main()
