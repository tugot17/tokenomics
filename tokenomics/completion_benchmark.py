"""
OAI server benchmark with scenario-based sampling.
"""

import argparse
import asyncio
import aiohttp
import time
import json
import os
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from .sampling import Scenario, TextSampler, DatasetConfig, DatasetLoader, use_seed, derive_seed
from .io import round_floats, atomic_write_json


def _safe_mean(values):
    """Return the mean of values, or 0 if empty."""
    return float(np.mean(values)) if values else 0


def _safe_std(values):
    """Return the population std of values, or 0 if fewer than 2."""
    return float(np.std(values)) if len(values) > 1 else 0


class LoRAConfig:
    """Configuration for LoRA distribution in benchmark."""

    VALID_STRATEGIES = {"single", "uniform", "zipf", "mixed", "all-unique"}

    def __init__(self, strategy: str, lora_names: List[str],
                 base_model_ratio: float = 0.0, zipf_alpha: float = 1.0):
        # Validation
        if not lora_names:
            raise ValueError("lora_names cannot be empty")
        if not 0.0 <= base_model_ratio <= 1.0:
            raise ValueError(f"base_model_ratio must be between 0 and 1, got {base_model_ratio}")
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Valid strategies: {self.VALID_STRATEGIES}")
        if zipf_alpha <= 0:
            raise ValueError(f"zipf_alpha must be positive, got {zipf_alpha}")

        self.strategy = strategy
        self.lora_names = lora_names
        self.base_model_ratio = base_model_ratio
        self.zipf_alpha = zipf_alpha


    def _apply_base_model_ratio(self, lora_choices: List[Optional[str]]) -> List[Optional[str]]:
        """Apply base_model_ratio to replace some LoRA assignments with None (base model)."""
        if self.base_model_ratio == 0.0:
            return lora_choices

        return [None if random.random() < self.base_model_ratio else lora
                for lora in lora_choices]

    def _get_lora_assignments(self, batch_size: int) -> List[str]:
        """Get raw LoRA assignments based on strategy (before applying base_model_ratio)."""
        if self.strategy == "single":
            # All requests use the same LoRA
            return [self.lora_names[0]] * batch_size

        elif self.strategy in ("uniform", "all-unique"):
            # Round-robin across all LoRAs (cycles if batch > num loras)
            return [self.lora_names[i % len(self.lora_names)] for i in range(batch_size)]

        elif self.strategy == "zipf":
            # Zipf distribution (power law) - some LoRAs more popular
            zipf_samples = np.random.zipf(self.zipf_alpha, batch_size)
            return [self.lora_names[(sample - 1) % len(self.lora_names)] for sample in zipf_samples]

        elif self.strategy == "mixed":
            return [random.choice(self.lora_names) for _ in range(batch_size)]

    def assign_lora(self, batch_size: int) -> List[Optional[str]]:
        """Assign LoRA names to a batch of requests based on strategy."""
        raw_assignments = self._get_lora_assignments(batch_size)
        return self._apply_base_model_ratio(raw_assignments)


async def _iter_sse_data(response):
    """Yield SSE data payloads, robust to chunk boundaries.

    Properly handles cases where a single SSE event spans multiple TCP chunks,
    or multiple events arrive in a single chunk.
    """
    buffer = ""
    event_data_lines = []

    async for chunk in response.content.iter_any():
        if not chunk:
            continue
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            raw_line, buffer = buffer.split("\n", 1)
            line = raw_line.rstrip("\r")

            if line == "":
                # Empty line = end of SSE event
                if event_data_lines:
                    yield "\n".join(event_data_lines)
                    event_data_lines = []
                continue

            if line.startswith(":"):
                # SSE comment, skip
                continue
            if line.startswith("data:"):
                event_data_lines.append(line[5:].lstrip())

    # Flush remaining data
    if event_data_lines:
        yield "\n".join(event_data_lines)


def _extract_text_from_delta(delta: Dict[str, Any]) -> str:
    """Extract streamed text from common API delta layouts.

    Handles OpenAI, Anthropic, and other common response formats.
    """
    pieces = []
    for key in ("reasoning_content", "content", "text", "reasoning", "output_text"):
        value = delta.get(key)
        if isinstance(value, str):
            pieces.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    pieces.append(item)
                elif isinstance(item, dict):
                    for sub_key in ("reasoning_content", "content", "text", "output_text"):
                        sub_val = item.get(sub_key)
                        if isinstance(sub_val, str):
                            pieces.append(sub_val)
    return "".join(pieces)


async def single_request(session: aiohttp.ClientSession, api_base: str, model: str,
                        request_data: Dict[str, Any], temperature: float,
                        timeout: int, api_key: str = "dummy-key") -> Dict:
    """Make a single streaming API request, returning per-request metrics."""
    request_start = time.perf_counter()
    time_at_first_token = None
    chunk_timestamps = []

    lora_name = request_data.get("lora_name")
    model_str = f"{model}:{lora_name}" if lora_name else model

    payload = {
        "model": model_str,
        "messages": [{"role": "user", "content": request_data["prompt"]}],
        "max_tokens": request_data["max_tokens"],
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True}
    }

    try:
        async with session.post(
            f"{api_base}/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:

            if response.status != 200:
                return {"error": f"API error {response.status}: {await response.text()}", "success": False}

            api_usage = None

            async for data_text in _iter_sse_data(response):
                if data_text == "[DONE]":
                    break

                try:
                    chunk_data = json.loads(data_text)
                except json.JSONDecodeError:
                    continue

                if "usage" in chunk_data:
                    api_usage = chunk_data["usage"]

                choices = chunk_data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    text = _extract_text_from_delta(delta)

                    if text:
                        if time_at_first_token is None:
                            time_at_first_token = time.perf_counter()
                        chunk_timestamps.append(time.perf_counter())

            request_end = time.perf_counter()

            ttft = time_at_first_token - request_start if time_at_first_token else 0
            e2e_latency = request_end - request_start
            output_latency = e2e_latency - ttft if ttft > 0 else e2e_latency

            if api_usage:
                input_tokens = api_usage.get("prompt_tokens", 0)
                output_tokens = api_usage.get("reasoning_tokens", 0) + api_usage.get("completion_tokens", 0)
            else:
                input_tokens = request_data.get("target_input_tokens", 0)
                output_tokens = len(chunk_timestamps)  # 1 chunk ≈ 1 token
                print(f"⚠️  No API usage info — using chunk count as token estimate")

            input_throughput = input_tokens / ttft if ttft > 0 else 0
            output_throughput = (output_tokens - 1) / output_latency if output_latency > 0 and output_tokens > 1 else 0
            tpot = output_latency / (output_tokens - 1) if output_tokens > 1 and output_latency > 0 else 0

            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "start_time": request_start,
                "end_time": request_end,
                "ttft": ttft,
                "output_latency": output_latency,
                "input_throughput": input_throughput,
                "output_throughput": output_throughput,
                "tpot": tpot,
                "chunk_timestamps": chunk_timestamps,
                "success": True
            }

    except Exception as e:
        return {"error": str(e), "success": False}


async def run_batch_async(user_requests: List[Dict], api_base: str, model: str,
                         temperature: float, timeout: int,
                         max_concurrency: Optional[int] = None,
                         api_key: str = "dummy-key") -> Dict:
    """Run batch of requests asynchronously.

    Args:
        max_concurrency: If set, limits the number of concurrent in-flight
            requests using a semaphore (sustained-load mode). When None, all
            requests are fired at once (burst mode).
    """
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    start_time = time.perf_counter()

    async def send_request(session, req):
        if semaphore is None:
            return await single_request(session, api_base, model, req, temperature, timeout, api_key)

        async with semaphore:
            return await single_request(session, api_base, model, req, temperature, timeout, api_key)

    # Match sglang's read_bufsize (10MB) to avoid event loop bottleneck at high concurrency.
    # Default 64KB buffer can cause excessive read syscalls under pressure.
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(
        connector=connector,
        read_bufsize=10 * 1024**2,
    ) as session:
        tasks = [send_request(session, req) for req in user_requests]
        results = await asyncio.gather(*tasks)

    batch_total_seconds = time.perf_counter() - start_time

    successful_results = [r for r in results if r.get("success", False)]
    total_requests = len(user_requests)
    success_count = len(successful_results)
    failure_count = total_requests - success_count

    return {
        "tokens": {
            "input_per_request": [r["input_tokens"] for r in successful_results],
            "output_per_request": [r["output_tokens"] for r in successful_results],
        },
        "batch_total_seconds": batch_total_seconds,
        "prefill_metrics": {
            "ttft_per_request": [r["ttft"] for r in successful_results if r["ttft"] > 0],
            "input_throughput_per_request": [r["input_throughput"] for r in successful_results if r["input_throughput"] > 0],
        },
        "decode_metrics": {
            "output_throughput_per_request": [r["output_throughput"] for r in successful_results if r["output_throughput"] > 0],
            "tpot_per_request": [r["tpot"] for r in successful_results if r.get("tpot", 0) > 0],
            "decode_time_per_request": [r["output_latency"] for r in successful_results if r["output_latency"] > 0],
        },
        "failures": {
            "total_requests": total_requests,
            "successful": success_count,
            "failed": failure_count,
        },
        "request_details": [
            {
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "ttft": r["ttft"],
                "output_tokens": r["output_tokens"],
                "output_latency": r["output_latency"],
                "chunk_timestamps": r.get("chunk_timestamps", []),
            }
            for r in successful_results
        ],
    }


def run_benchmark_batch(user_requests: List[Dict], api_base: str, model: str,
                        temperature: float, timeout: int,
                        max_concurrency: Optional[int] = None,
                        api_key: str = "dummy-key") -> Dict:
    """Run a batch of requests and return results."""
    result = asyncio.run(run_batch_async(user_requests, api_base, model, temperature, timeout,
                                         max_concurrency=max_concurrency, api_key=api_key))

    failures = result["failures"]
    print(f"    📊 {failures['successful']}/{failures['total_requests']} successful"
          + (f", {failures['failed']} failed" if failures['failed'] > 0 else ""))

    return result


def compute_phased_metrics(
    request_details: List[Dict],
    bucket_size: float = 0.05,
    steady_state_threshold: float = 0.8,
    target_concurrency: Optional[int] = None,
) -> Dict:
    """Compute time-bucketed metrics with steady-state filtering.

    steady_state_threshold: fraction of reference concurrency required
    for a bucket to be considered steady-state (default 0.8 = 80%).
    target_concurrency: if set (sustained mode), use this as the reference
    instead of observed peak. Avoids bucket-boundary inflation of peak.
    """
    if not request_details:
        return {}

    t0 = min(r["start_time"] for r in request_details)
    t_end = max(r["end_time"] for r in request_details)
    n_buckets = int((t_end - t0) / bucket_size) + 1

    output_tokens_per_bucket = [0] * n_buckets
    active_requests_per_bucket = [0] * n_buckets

    # Sweep-line for active request counting: O(n log n) instead of O(n * buckets)
    events = []  # (bucket_index, +1 or -1)
    for r in request_details:
        decode_start = r["start_time"] + r.get("ttft", 0)
        req_start_b = int((decode_start - t0) / bucket_size)
        req_end_b = int((r["end_time"] - t0) / bucket_size)
        events.append((req_start_b, 1))
        events.append((min(req_end_b + 1, n_buckets), -1))

        # Bin output tokens by chunk timestamp
        for ts in r.get("chunk_timestamps", []):
            b = int((ts - t0) / bucket_size)
            if 0 <= b < n_buckets:
                output_tokens_per_bucket[b] += 1

    events.sort()
    active = 0
    ei = 0
    for b in range(n_buckets):
        while ei < len(events) and events[ei][0] <= b:
            active += events[ei][1]
            ei += 1
        active_requests_per_bucket[b] = active

    peak_active = max(active_requests_per_bucket) if active_requests_per_bucket else 0
    # Use target concurrency as reference when available (sustained mode),
    # otherwise fall back to observed peak (burst mode).
    reference = target_concurrency if target_concurrency else peak_active
    threshold = reference * steady_state_threshold

    steady_state_tps = [
        output_tokens_per_bucket[i] / bucket_size
        for i in range(n_buckets)
        if active_requests_per_bucket[i] >= threshold
        and output_tokens_per_bucket[i] > 0
    ]

    return {
        "steady_state_tps": {
            "median": float(np.median(steady_state_tps)) if steady_state_tps else None,
        },
        "peak_active_requests": peak_active,
        "time_series": {
            "output_tokens_per_bucket": output_tokens_per_bucket,
            "active_requests_per_bucket": active_requests_per_bucket,
        },
    }


def _aggregate_phased_metrics(per_run_metrics: List[Dict]) -> Dict:
    """Aggregate phased metrics computed per-run into a single summary.

    Takes the mean/std of per-run steady-state medians and end-to-end TPS.
    Uses the time_series from the last run for plotting (representative).
    """
    if not per_run_metrics:
        return {}

    steady_state_medians = [m["steady_state_tps"]["median"] for m in per_run_metrics
                            if m.get("steady_state_tps", {}).get("median") is not None]
    peak_actives = [m["peak_active_requests"] for m in per_run_metrics
                    if m.get("peak_active_requests", 0) > 0]

    # Use the last run's time_series for plotting (representative of the batch size)
    last_ts = per_run_metrics[-1].get("time_series", {})

    return {
        "steady_state_tps": {
            "median": _safe_mean(steady_state_medians) or None,
            "median_std": _safe_std(steady_state_medians),
        },
        "peak_active_requests": max(peak_actives) if peak_actives else 0,
        "time_series": last_ts,
    }


def calculate_stats(runs_data: List[Dict], steady_state_threshold: float = 0.8,
                    target_concurrency: Optional[int] = None) -> Dict:
    """Calculate statistics across multiple runs (enhanced with TTFT metrics)."""
    # For list-based metrics, flatten all values from all runs
    all_input_tokens = []
    all_output_tokens = []
    
    # Individual request metrics (flattened across runs)
    all_ttft = []
    all_input_throughput = []
    all_output_throughput = []
    all_tpot = []
    all_decode_time = []

    # Batch-level metrics (one value per run)
    all_e2e_tps = []
    batch_total_seconds = []
    
    # Failure metrics (aggregated across runs)
    total_requests = 0
    total_successful = 0
    total_failed = 0

    # Per-run phased metrics
    all_run_phased_metrics = []

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
        all_decode_time.extend(decode_metrics.get("decode_time_per_request", []))
        
        batch_total_seconds.append(run_data["batch_total_seconds"])

        # End-to-end TPS: total_output_tokens / wall_time
        run_wall_time = run_data["batch_total_seconds"]
        run_total_output = sum(run_data["tokens"]["output_per_request"])
        all_e2e_tps.append(run_total_output / run_wall_time if run_wall_time > 0 else 0)
        
        # Aggregate failure metrics
        failures = run_data.get("failures", {})
        total_requests += failures.get("total_requests", 0)
        total_successful += failures.get("successful", 0)
        total_failed += failures.get("failed", 0)

        # Compute phased metrics per run (not pooled — runs are sequential)
        run_request_details = run_data.get("request_details", [])
        if run_request_details:
            run_phased = compute_phased_metrics(
                run_request_details,
                steady_state_threshold=steady_state_threshold,
                target_concurrency=target_concurrency,
            )
            all_run_phased_metrics.append(run_phased)

    # Aggregate phased metrics across runs
    phased_metrics = _aggregate_phased_metrics(all_run_phased_metrics)

    return {
        "tokens": {
            "input_per_request": {"mean": _safe_mean(all_input_tokens), "std": _safe_std(all_input_tokens)},
            "output_per_request": {"mean": _safe_mean(all_output_tokens), "std": _safe_std(all_output_tokens)},
        },
        "prefill_metrics": {
            "ttft": {"mean": _safe_mean(all_ttft), "std": _safe_std(all_ttft)},
            "input_throughput": {"mean": _safe_mean(all_input_throughput), "std": _safe_std(all_input_throughput)},
        },
        "decode_metrics": {
            "output_throughput": {"mean": _safe_mean(all_output_throughput), "std": _safe_std(all_output_throughput)},
            "tpot": {"mean": _safe_mean(all_tpot), "std": _safe_std(all_tpot)},
            "decode_time": {"mean": _safe_mean(all_decode_time), "std": _safe_std(all_decode_time)},
        },
        "batch_metrics": {
            "e2e_tps": {"mean": _safe_mean(all_e2e_tps), "std": _safe_std(all_e2e_tps)},
            "wall_time": {"mean": _safe_mean(batch_total_seconds), "std": _safe_std(batch_total_seconds)},
        },
        "reliability": {
            "total_requests": total_requests,
            "successful": total_successful,
            "failed": total_failed,
        },
        "phased_metrics": phased_metrics,
    }


def warmup_server(api_base: str, model: str, temperature: float, timeout: int,
                  text_sampler, num_warmup_runs: int = 3, api_key: str = "dummy-key"):
    """Perform warmup runs to prepare the server before benchmarking."""
    print(f"Performing {num_warmup_runs} warmup runs...", end="", flush=True)

    warmup_scenario = Scenario.from_string("N(100,50)/(50,25)")

    async def run_warmup():
        async with aiohttp.ClientSession() as session:
            for _ in range(num_warmup_runs):
                req = text_sampler.sample_batch(warmup_scenario, 1)[0]
                warmup_data = {
                    "prompt": req.prompt,
                    "max_tokens": max(1, min(4096, int(req.target_output_tokens))),
                }
                try:
                    await single_request(session, api_base, model, warmup_data, temperature, timeout, api_key)
                except Exception:
                    pass

    asyncio.run(run_warmup())
    print(" done")


def main():
    """Main function to run the enhanced benchmark."""
    parser = argparse.ArgumentParser(description="OAI server benchmark with scenario-based sampling")
    
    # API configuration
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="API base URL")
    parser.add_argument("--api-key", default="dummy-key", help="API key")
    parser.add_argument("--model", required=True, help="Model name")
    
    # Scenario configuration
    parser.add_argument("--scenario", required=True, help="Scenario string (e.g., 'N(480,240)/(300,150)', 'D(100,100)')")
    parser.add_argument("--dataset-config", default=None, help="Path to dataset configuration JSON (defaults to bundled AIME dataset)")
    parser.add_argument("--tokenizer", help="Tokenizer name (defaults to model name)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic sampling")
    parser.add_argument("--timeout", type=int, default=3000, help="Timeout in seconds per request.")
    
    # Benchmark parameters
    parser.add_argument("--batch-sizes", default=None, help="Comma-separated batch sizes (burst mode)")
    parser.add_argument("--max-concurrency", default=None, help="Comma-separated concurrency levels (sustained mode)")
    parser.add_argument("--num-prompts", type=int, default=None,
                        help="Total prompts per sweep point in sustained mode (default: max(64, 8*concurrency))")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs per sweep point")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Number of warmup runs before each sweep point")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature parameter")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Upper bound on max_tokens per request (default: 4096)")
    parser.add_argument("--description", default="Benchmark", help="Description of the benchmark")
    
    # Output configuration
    parser.add_argument("--results-dir", default="completion_results/", help="Output directory for per-sweep-value result files")

    # LoRA configuration (direct parameters)
    parser.add_argument("--lora-strategy", help="LoRA distribution strategy (single, uniform, zipf, mixed, all-unique)")
    parser.add_argument("--lora-names", help="Comma-separated LoRA adapter names")
    parser.add_argument("--base-model-ratio", type=float, default=0.0, help="Fraction of requests using base model without LoRA (0.0-1.0)")
    parser.add_argument("--zipf-alpha", type=float, default=1.0, help="Zipf distribution alpha parameter (default: 1.0)")

    # Phased metrics configuration
    parser.add_argument("--steady-state-threshold", type=float, default=0.8,
                        help="Fraction of peak active requests required for a bucket to count as steady-state (0.0-1.0, default: 0.8)")

    args = parser.parse_args()

    # Determine execution mode and sweep values
    if args.batch_sizes is not None and args.max_concurrency is not None:
        parser.error("--batch-sizes and --max-concurrency are mutually exclusive")

    if args.max_concurrency is not None:
        execution_mode = "sustained"
        sweep_values = [int(x.strip()) for x in args.max_concurrency.split(",")]
    else:
        execution_mode = "burst"
        sweep_values = [int(x.strip()) for x in (args.batch_sizes or "1,2,4,8").split(",")]

    # Create LoRA config from command-line arguments
    lora_config = None
    if args.lora_strategy and args.lora_names:
        lora_names_list = [name.strip() for name in args.lora_names.split(",")]
        lora_config = LoRAConfig(
            strategy=args.lora_strategy,
            lora_names=lora_names_list,
            base_model_ratio=args.base_model_ratio,
            zipf_alpha=args.zipf_alpha
        )
        print(f"🔧 LoRA config: strategy={lora_config.strategy}, {len(lora_config.lora_names)} LoRAs, base_model_ratio={lora_config.base_model_ratio}")

    # Initialize scenario and dataset
    scenario = Scenario.from_string(args.scenario)
    if args.dataset_config is None:
        from tokenomics import EXAMPLES_DIR
        args.dataset_config = str(EXAMPLES_DIR / "dataset_configs" / "aime_simple.json")
    dataset_config = DatasetConfig.from_file(args.dataset_config)
    dataset_loader = DatasetLoader(dataset_config)

    tokenizer_name = args.tokenizer or args.model

    text_sampler = TextSampler(tokenizer_name, dataset_loader)

    print(f"📊 Initialized benchmark with scenario: {args.scenario}")
    print(f"📚 Loaded dataset: {len(dataset_loader)} samples")
    print(f"⚙️  Execution mode: {execution_mode}")
    if lora_config:
        print(f"🎯 LoRA strategy: {lora_config.strategy} with {len(lora_config.lora_names)} adapters")
    print(f"🚀 Starting benchmark: {args.description}")

    # Create results structure
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "scenario": args.scenario,
        "dataset_config": dataset_config.config,
        "api_base": args.api_base,
        "execution_mode": execution_mode,
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "temperature": args.temperature,
        "description": args.description,
        "seed": args.seed,
        "lora_config": {
            "strategy": lora_config.strategy,
            "lora_names": lora_config.lora_names,
            "base_model_ratio": lora_config.base_model_ratio,
        } if lora_config else None,
        "steady_state_threshold": args.steady_state_threshold,
        "bucket_size_seconds": 0.05,
    }

    if execution_mode == "sustained":
        metadata["concurrency_levels"] = sweep_values
        metadata["num_prompts"] = args.num_prompts
    else:
        metadata["batch_sizes"] = sweep_values

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Run benchmark for each sweep value
    for sweep_value in sweep_values:
        if execution_mode == "sustained":
            num_prompts = args.num_prompts or max(64, 8 * sweep_value)
            max_concurrency = sweep_value
            print(f"\n🔄 Testing concurrency: {sweep_value} ({num_prompts} prompts)")
        else:
            num_prompts = sweep_value
            max_concurrency = None
            print(f"\n🔄 Testing batch size: {sweep_value}")

        warmup_seed = derive_seed(args.seed, sweep_value, -1)
        with use_seed(warmup_seed):
            warmup_server(args.api_base, args.model, args.temperature, args.timeout,
                         text_sampler, args.warmup_runs, api_key=args.api_key)

        runs_data = []
        for run_idx in range(args.num_runs):
            print(f"  Run {run_idx + 1}/{args.num_runs}")

            # Generate batch of requests using scenario (on-demand)
            run_seed = derive_seed(args.seed, sweep_value, run_idx)
            with use_seed(run_seed):
                sampled_requests = text_sampler.sample_batch(scenario, num_prompts)

            # Assign LoRAs if config is provided
            lora_assignments = None
            if lora_config:
                lora_assignments = lora_config.assign_lora(num_prompts)

            user_requests = []
            for idx, request in enumerate(sampled_requests):
                request_dict = {
                    "prompt": request.prompt,
                    "max_tokens": max(1, min(args.max_tokens, int(request.target_output_tokens))),
                    "target_input_tokens": request.target_input_tokens,
                    "target_output_tokens": request.target_output_tokens,
                }
                if lora_assignments:
                    request_dict["lora_name"] = lora_assignments[idx]
                user_requests.append(request_dict)

            run_data = run_benchmark_batch(
                user_requests, args.api_base, args.model, args.temperature, args.timeout,
                max_concurrency=max_concurrency, api_key=args.api_key
            )

            runs_data.append(run_data)

            # Print per-run progress
            if run_data["tokens"]["output_per_request"]:
                total_output = sum(run_data["tokens"]["output_per_request"])
                wall_time = run_data["batch_total_seconds"]
                wall_tps = total_output / wall_time if wall_time > 0 else 0

                ttft_vals = run_data["prefill_metrics"]["ttft_per_request"]
                avg_ttft = _safe_mean(ttft_vals)

                print(f"    ✅ {wall_time:.2f}s | {total_output:,} tokens | {wall_tps:.0f} tok/s | TTFT {avg_ttft:.3f}s")
            else:
                print(f"    ❌ All requests failed")

        # Calculate statistics for this sweep value
        batch_stats = calculate_stats(runs_data, steady_state_threshold=args.steady_state_threshold,
                                       target_concurrency=max_concurrency)

        # Print phased metrics summary
        pm = batch_stats.get("phased_metrics", {})
        bm = batch_stats.get("batch_metrics", {})
        steady_state = pm.get("steady_state_tps", {})
        if steady_state.get("median") is not None:
            std_str = f" +/- {steady_state['median_std']:.1f}" if steady_state.get("median_std", 0) > 0 else ""
            wall_tps = bm.get("e2e_tps", {}).get("mean", 0)
            print(f"  Phased Metrics:")
            print(f"    Steady-state TPS (median): {steady_state['median']:.1f}{std_str} tok/s")
            print(f"    End-to-end TPS:            {wall_tps:.1f} tok/s")
            print(f"    Peak active requests:      {pm['peak_active_requests']}")

        # Write per-sweep-value result file atomically
        result_entry = {
            "metadata": metadata,
            "sweep_value": sweep_value,
            "result": batch_stats,
        }
        result_path = os.path.join(args.results_dir, f"{sweep_value}.json")
        atomic_write_json(result_path, round_floats(result_entry))
        print(f"  💾 Saved {result_path}")

    print(f"\n✅ Benchmark completed! Results saved to {args.results_dir}")


if __name__ == "__main__":
    main()