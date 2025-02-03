#!/usr/bin/env python3
import argparse
import concurrent.futures
import time
import statistics
import json
from datetime import datetime

from datasets import load_dataset
import openai
from openai import OpenAI

def create_sample_conversations(dataset_name: str, num_samples: int, seed: int = 42):
    """
    Load a dataset and create a list of conversation samples.
    Each sample is a list of messages (in OpenAI Chat Completion format).
    Assumes that each item in the dataset has a key "Question".
    """
    ds = load_dataset(dataset_name)
    # Shuffle and select the first num_samples examples from the train split.
    sampled_dataset = ds["train"].shuffle(seed=seed).select(range(num_samples))
    conversations = []
    for item in sampled_dataset:
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": item["Question"]}
        ]
        conversations.append(conversation)
    return conversations

def call_server_completion(client, model: str, messages, temperature: float, max_tokens: int):
    """
    Call the vLLM server (using the OpenAI ChatCompletion API) for a single conversation.
    Returns the number of completion tokens generated or 0 on error.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Try to get the token count from the response.
        if hasattr(response, 'usage') and response.usage:
            return response.usage.completion_tokens
        elif isinstance(response, dict) and "usage" in response:
            return response["usage"].get("completion_tokens", 0)
        else:
            return 0
    except Exception as e:
        print(f"Error during API call: {e}")
        return 0

def run_benchmark(client, model: str, conversations, temperature: float, max_tokens: int):
    """
    Run a benchmark for one batch of conversations concurrently.
    Returns a dictionary with metrics for the run:
      - total_output_tokens: total tokens generated across requests
      - elapsed_time: total time taken in seconds
      - tokens_per_second: overall tokens per second
      - tokens_per_request_per_second: tokens per second divided by the number of requests
    """
    start_time = time.perf_counter()
    tokens_list = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(call_server_completion, client, model, conv, temperature, max_tokens)
            for conv in conversations
        ]
        for future in concurrent.futures.as_completed(futures):
            tokens_list.append(future.result())
    end_time = time.perf_counter()

    total_tokens = sum(tokens_list)
    elapsed = end_time - start_time
    tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0
    tokens_per_request_per_second = tokens_per_second / len(conversations) if conversations else 0

    return {
        "total_output_tokens": total_tokens,
        "elapsed_time": elapsed,
        "tokens_per_second": tokens_per_second,
        "tokens_per_request_per_second": tokens_per_request_per_second
    }

def save_results(results: dict, filename: str):
    """Save results to a JSON file."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="Simplified Benchmark for vLLM Server using OpenAI Chat Completion API"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model tag to use (e.g., 'distill-llama-8b').")
    parser.add_argument("--dataset", type=str, default="gneubig/aime-1983-2024",
                        help="Hugging Face dataset name (assumes items have 'Question').")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1",
                        help="Base URL of the vLLM server API.")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8",
                        help="Comma-separated batch sizes (e.g., '1,2,4,8').")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs per batch size.")
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

    # Convert batch_sizes string into a list of integers.
    batch_sizes = [int(bs.strip()) for bs in args.batch_sizes.split(",") if bs.strip()]

    # Configure the OpenAI API client to use the vLLM server.
    client = OpenAI(api_key="sk-dummy", base_url=args.api_base)

    # Initialize a results dictionary with metadata.
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "dataset": args.dataset,
            "api_base": args.api_base,
            "batch_sizes": batch_sizes,
            "num_runs": args.num_runs,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "description": args.description
        },
        "results": {}
    }

    print(f"Starting benchmark for model: {args.model}")
    for batch_size in batch_sizes:
        print(f"\n=== Benchmarking Batch Size: {batch_size} ===")
        run_metrics = []  # Collect metrics from each run

        for run in range(1, args.num_runs + 1):
            print(f" Run {run}/{args.num_runs} ... ", end="", flush=True)
            # Generate conversation samples.
            conversations = create_sample_conversations(args.dataset, num_samples=batch_size, seed=args.seed)
            metrics = run_benchmark(client, args.model, conversations, args.temperature, args.max_tokens)
            run_metrics.append(metrics)
            print(f"tokens: {metrics['total_output_tokens']}, time: {metrics['elapsed_time']:.2f}s, "
                  f"TPS: {metrics['tokens_per_second']:.2f}, "
                  f"TPS/Request: {metrics['tokens_per_request_per_second']:.2f}")

        # Compute average values for each metric across the runs.
        average_metrics = {
            key: statistics.mean([m[key] for m in run_metrics])
            for key in run_metrics[0]
        }

        # Save only the aggregate average metrics for this batch size.
        results["results"][str(batch_size)] = average_metrics

        print(" Summary:")
        print(f"  Average tokens: {average_metrics['total_output_tokens']:.2f}")
        print(f"  Average time:   {average_metrics['elapsed_time']:.2f}s")
        print(f"  Average TPS:    {average_metrics['tokens_per_second']:.2f}")
        print(f"  Average TPS/Request: {average_metrics['tokens_per_request_per_second']:.2f}")

    # Save final results to JSON.
    save_results(results, args.results_file)
    print(f"\nBenchmark results saved to {args.results_file}")

if __name__ == "__main__":
    main()

# ./server_benchmark.py --model distill-llama-8b --api_base http://localhost:8000/v1 --batch_sizes 1,2,4,8 --num_runs 3 --max_tokens 100 --temperature 0.5 --description "LLama 8B TP8 A100s" --results_file my_server_benchmark.json
