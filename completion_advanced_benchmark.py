"""
Enhanced OAI server benchmark with scenario-based sampling.
"""

import argparse
import asyncio
import aiohttp
import time
import statistics
import json
import multiprocessing
import os
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from sampling import Scenario, TextSampler, BatchSampler, DatasetConfig, DatasetLoader, use_seed, derive_seed


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

        elif self.strategy == "uniform":
            # Uniformly distribute across all LoRAs (round-robin)
            return [self.lora_names[i % len(self.lora_names)] for i in range(batch_size)]

        elif self.strategy == "zipf":
            # Zipf distribution (power law) - some LoRAs more popular
            zipf_samples = np.random.zipf(self.zipf_alpha, batch_size)
            return [self.lora_names[(sample - 1) % len(self.lora_names)] for sample in zipf_samples]

        elif self.strategy == "mixed":
            # Random mix
            return [random.choice(self.lora_names) for _ in range(batch_size)]

        elif self.strategy == "all-unique":
            # Each request gets a unique LoRA (cycles if batch > num loras)
            return [self.lora_names[i % len(self.lora_names)] for i in range(batch_size)]

        else:
            # This should never happen due to validation in __init__
            raise ValueError(f"Unknown LoRA strategy: {self.strategy}")

    def assign_lora(self, batch_size: int) -> List[Optional[str]]:
        """Assign LoRA names to a batch of requests based on strategy."""
        raw_assignments = self._get_lora_assignments(batch_size)
        return self._apply_base_model_ratio(raw_assignments)


async def _iter_sse_data(stream) -> str:
    """Yield concatenated SSE data payloads, robust to chunk boundaries."""
    buffer = ""
    event_data_lines: List[str] = []

    async for chunk in stream.iter_any():
        if not chunk:
            continue
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            raw_line, buffer = buffer.split("\n", 1)
            line = raw_line.rstrip("\r")

            if line == "":
                if event_data_lines:
                    yield "\n".join(event_data_lines)
                    event_data_lines = []
                continue

            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                event_data_lines.append(line[5:].lstrip())

    if event_data_lines:
        yield "\n".join(event_data_lines)


def _extract_text_from_delta(delta: Dict[str, Any]) -> str:
    """Extract streamed text from common delta layouts."""
    pieces: List[str] = []
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


def _summarize_run_results(results: List[Dict[str, Any]], total_requests: int, batch_total_seconds: float) -> Dict:
    """Summarize per-request results for one run window."""
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    success_count = len(successful_results)
    failure_count = len(failed_results)

    if not successful_results:
        return {
            "tokens": {"input_per_request": [], "output_per_request": []},
            "timings": {"batch_total_seconds": batch_total_seconds, "fastest_seconds": 0, "slowest_seconds": 0, "spread_seconds": 0},
            "prefill_metrics": {
                "ttft_per_request": [],
                "input_throughput_per_request": [],
                "ttft_missing_count": 0,
            },
            "decode_metrics": {"output_throughput_per_request": [], "tpot_per_request": [], "decode_time_per_request": []},
            "batch_metrics": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "batch_duration": batch_total_seconds,
                "total_token_throughput": 0,
                "output_token_throughput": 0,
                "combined_decode_throughput_sum": 0,
                "combined_throughput": 0,
                "total_requests": 0,
            },
            "failures": {"total_requests": total_requests, "successful": success_count, "failed": failure_count}
        }

    input_tokens = [r["input_tokens"] for r in successful_results]
    output_tokens = [r["output_tokens"] for r in successful_results]

    ttft_values = [r.get("ttft", 0) for r in successful_results if r.get("ttft", 0) > 0]
    ttft_missing_count = max(0, len(successful_results) - len(ttft_values))
    input_throughput_values = [r.get("input_throughput", 0) for r in successful_results if r.get("input_throughput", 0) > 0]
    output_throughput_values = [r.get("output_throughput", 0) for r in successful_results if r.get("output_throughput", 0) > 0]
    tpot_values = [r.get("tpot", 0) for r in successful_results if r.get("tpot", 0) > 0]
    decode_time_values = [r.get("decode_time", 0) for r in successful_results if r.get("decode_time", 0) > 0]

    lora_names = [r.get("lora_name") for r in successful_results]
    unique_loras = set(lora_names) - {None}
    base_model_count = lora_names.count(None)

    relative_ends = [r["relative_end"] for r in successful_results]
    fastest_seconds = min(relative_ends)
    slowest_seconds = max(relative_ends)
    spread_seconds = slowest_seconds - fastest_seconds

    total_input_tokens = sum(input_tokens)
    total_output_tokens = sum(output_tokens)
    total_tokens = total_input_tokens + total_output_tokens
    total_token_throughput = total_tokens / batch_total_seconds if batch_total_seconds > 0 else 0
    output_token_throughput = total_output_tokens / batch_total_seconds if batch_total_seconds > 0 else 0
    decode_throughput_sum = sum(output_throughput_values)

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
        "prefill_metrics": {
            "ttft_per_request": ttft_values,
            "input_throughput_per_request": input_throughput_values,
            "ttft_missing_count": ttft_missing_count,
        },
        "decode_metrics": {
            "output_throughput_per_request": output_throughput_values,
            "tpot_per_request": tpot_values,
            "decode_time_per_request": decode_time_values
        },
        "batch_metrics": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "batch_duration": batch_total_seconds,
            "total_token_throughput": total_token_throughput,
            "output_token_throughput": output_token_throughput,
            "combined_decode_throughput_sum": decode_throughput_sum,
            "combined_throughput": total_token_throughput,
            "total_requests": len(successful_results)
        },
        "failures": {
            "total_requests": total_requests,
            "successful": success_count,
            "failed": failure_count
        },
        "lora_metrics": {
            "lora_names_per_request": lora_names,
            "unique_loras_count": len(unique_loras),
            "unique_loras": list(unique_loras),
            "base_model_count": base_model_count,
            "base_model_percentage": (base_model_count / len(successful_results) * 100) if successful_results else 0
        }
    }


async def single_request(session: aiohttp.ClientSession, api_base: str, model: str,
                        request_data: Dict[str, Any], temperature: float,
                        timeout: int, start_time: float, tokenizer=None) -> Dict:
    """Make a single API request with TTFT timing (streaming enabled)."""
    request_start = time.time()
    time_at_first_token = None
    first_choice_time = None
    generated_text = ""

    # Extract max_tokens and optional LoRA name from request_data
    max_tokens = request_data["max_tokens"]
    lora_name = request_data.get("lora_name")

    # Build model string with LoRA if specified
    if lora_name:
        model_with_lora = f"{model}:{lora_name}"
    else:
        model_with_lora = model

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer dummy-key"
    }

    payload = {
        "model": model_with_lora,
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
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            
            if response.status == 200:
                # Process streaming response to capture TTFT and exact usage.
                api_usage = None
                
                async for data_text in _iter_sse_data(response.content):
                    if data_text == '[DONE]':
                        break
                    try:
                        chunk_data = json.loads(data_text)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk_data.get('choices', [])
                    if 'usage' in chunk_data:
                        api_usage = chunk_data['usage']

                    if choices:
                        if first_choice_time is None:
                            first_choice_time = time.time()
                        delta = choices[0].get('delta', {}) or {}
                        text = _extract_text_from_delta(delta)

                        # Set TTFT on first meaningful generation delta.
                        has_token_event = bool(text) or bool(delta.get("tool_calls")) or bool(delta.get("function_call"))
                        if has_token_event and time_at_first_token is None:
                            time_at_first_token = time.time()
                        if text:
                            generated_text += text
                
                request_end = time.time()
                
                # Calculate TTFT-based metrics.
                ttft = time_at_first_token - request_start if time_at_first_token else 0
                e2e_latency = request_end - request_start
                output_latency = e2e_latency - ttft if ttft > 0 else e2e_latency
                
                # Get exact token counts from API usage
                if api_usage:
                    input_tokens = api_usage.get("prompt_tokens", 0)
                    output_tokens = api_usage.get("reasoning_tokens", 0) + api_usage.get("completion_tokens", 0)
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

                # Fallback for providers/models that emit no text-bearing deltas before completion.
                if time_at_first_token is None and first_choice_time is not None and output_tokens > 0:
                    time_at_first_token = first_choice_time
                    ttft = time_at_first_token - request_start
                    output_latency = e2e_latency - ttft if ttft > 0 else e2e_latency
                
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
                    "ttft": ttft,
                    "e2e_latency": e2e_latency,
                    "output_latency": output_latency,
                    "tpot": tpot,
                    "input_throughput": input_throughput,  # tokens/sec for prefill
                    "output_throughput": output_throughput,  # tokens/sec for decode
                    "decode_time": output_latency,  # total time spent in decode phase
                    "lora_name": lora_name,  # Track which LoRA was used
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


async def run_batch_async(user_requests: List[Dict], api_base: str, model: str, temperature: float,
                          timeout: int, max_concurrent: int,tokenizer=None) -> Dict:
    """Run batch of requests asynchronously."""
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for request_data in user_requests:
            task = single_request(session, api_base, model, request_data, temperature, timeout, start_time, tokenizer)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    batch_total_seconds = end_time - start_time
    return _summarize_run_results(results, total_requests=len(user_requests), batch_total_seconds=batch_total_seconds)


async def run_refill_async(user_requests: List[Dict], api_base: str, model: str, temperature: float,
                           timeout: int, max_concurrent: int, tokenizer=None) -> Dict:
    """Run requests with a refill loop to maintain steady concurrency."""

    start_time = time.time()
    results = []
    total_requests = len(user_requests)
    queue: asyncio.Queue = asyncio.Queue()
    for req in user_requests:
        queue.put_nowait(req)

    # Keep at most max_concurrent in-flight; refill as workers complete.
    worker_count = min(max_concurrent, total_requests) if total_requests > 0 else 0

    async with aiohttp.ClientSession() as session:
        async def worker_loop():
            while True:
                try:
                    request_data = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    result = await single_request(
                        session, api_base, model, request_data, temperature, timeout, start_time, tokenizer
                    )
                    results.append(result)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker_loop()) for _ in range(worker_count)]
        if workers:
            await asyncio.gather(*workers)

    end_time = time.time()
    batch_total_seconds = end_time - start_time
    return _summarize_run_results(results, total_requests=total_requests, batch_total_seconds=batch_total_seconds)


def _truncate_prompt_to_cap(prompt: str, tokenizer, max_input_tokens: int) -> tuple[str, int]:
    """Truncate prompt to at most max_input_tokens and return updated prompt/token count."""
    if max_input_tokens <= 0:
        try:
            token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        except Exception:
            token_count = int(len(prompt.split()) * 1.3)
        return prompt, token_count

    try:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) <= max_input_tokens:
            return prompt, len(token_ids)
        truncated_ids = token_ids[:max_input_tokens]
        truncated_prompt = tokenizer.decode(
            truncated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return truncated_prompt, len(truncated_ids)
    except Exception:
        words = prompt.split()
        approx_words = max(1, int(max_input_tokens / 1.3))
        truncated_prompt = " ".join(words[:approx_words])
        return truncated_prompt, int(len(truncated_prompt.split()) * 1.3)


def build_request_dict(request, tokenizer, max_input_tokens_cap: int, max_output_tokens_cap: int) -> Dict[str, Any]:
    """Build one request payload with optional hard caps for input/output token lengths."""
    prompt, capped_input_tokens = _truncate_prompt_to_cap(
        request.prompt,
        tokenizer,
        max_input_tokens_cap,
    )

    uncapped_output = int(request.target_output_tokens)
    if max_output_tokens_cap > 0:
        capped_output = min(uncapped_output, max_output_tokens_cap)
    else:
        capped_output = uncapped_output
    capped_output = max(1, min(4096, capped_output))

    return {
        "prompt": prompt,
        "max_tokens": capped_output,
        "target_input_tokens": capped_input_tokens,
        "target_output_tokens": capped_output,
    }


def worker_process(worker_id: int, user_requests: List[Dict], api_base: str, model: str,
                   temperature: float, timeout: int, max_concurrent: int, result_queue: multiprocessing.Queue):
    """Worker process for multiprocessing."""
    try:
        # No tokenizer loading in worker process - rely on API usage info for accurate token counts
        result = asyncio.run(run_batch_async(user_requests, api_base, model, temperature, timeout, max_concurrent, tokenizer=None))
        result_queue.put(result)
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")
        result_queue.put(None)


def aggregate_metrics(worker_results: List[Dict]) -> Dict:
    """Aggregate metrics from multiple workers (enhanced with TTFT)."""
    all_input_tokens: List[int] = []
    all_output_tokens: List[int] = []
    all_ttft: List[float] = []
    all_input_throughput: List[float] = []
    all_output_throughput: List[float] = []
    all_tpot: List[float] = []
    all_decode_time: List[float] = []
    all_fastest: List[float] = []
    all_slowest: List[float] = []

    total_requests = 0
    total_successful = 0
    total_failed = 0
    ttft_missing_count = 0
    latest_end = 0.0

    for metrics in worker_results:
        all_input_tokens.extend(metrics.get("tokens", {}).get("input_per_request", []))
        all_output_tokens.extend(metrics.get("tokens", {}).get("output_per_request", []))
        all_ttft.extend(metrics.get("prefill_metrics", {}).get("ttft_per_request", []))
        all_input_throughput.extend(metrics.get("prefill_metrics", {}).get("input_throughput_per_request", []))
        all_output_throughput.extend(metrics.get("decode_metrics", {}).get("output_throughput_per_request", []))
        all_tpot.extend(metrics.get("decode_metrics", {}).get("tpot_per_request", []))
        all_decode_time.extend(metrics.get("decode_metrics", {}).get("decode_time_per_request", []))
        ttft_missing_count += int(metrics.get("prefill_metrics", {}).get("ttft_missing_count", 0))

        failures = metrics.get("failures", {})
        total_requests += int(failures.get("total_requests", 0))
        total_successful += int(failures.get("successful", 0))
        total_failed += int(failures.get("failed", 0))

        timings = metrics.get("timings", {})
        all_fastest.append(float(timings.get("fastest_seconds", 0)))
        all_slowest.append(float(timings.get("slowest_seconds", 0)))
        latest_end = max(latest_end, float(timings.get("batch_total_seconds", 0)))

    total_input_tokens = sum(all_input_tokens)
    total_output_tokens = sum(all_output_tokens)
    total_tokens = total_input_tokens + total_output_tokens
    total_token_throughput = total_tokens / latest_end if latest_end > 0 else 0
    output_token_throughput = total_output_tokens / latest_end if latest_end > 0 else 0
    decode_throughput_sum = sum(all_output_throughput)

    return {
        "tokens": {
            "input_per_request": all_input_tokens,
            "output_per_request": all_output_tokens
        },
        "timings": {
            "batch_total_seconds": latest_end,
            "fastest_seconds": min(all_fastest) if all_fastest else 0,
            "slowest_seconds": max(all_slowest) if all_slowest else 0,
            "spread_seconds": max(all_slowest) - min(all_fastest) if all_slowest and all_fastest else 0
        },
        "prefill_metrics": {
            "ttft_per_request": all_ttft,
            "input_throughput_per_request": all_input_throughput,
            "ttft_missing_count": ttft_missing_count,
        },
        "decode_metrics": {
            "output_throughput_per_request": all_output_throughput,
            "tpot_per_request": all_tpot,
            "decode_time_per_request": all_decode_time
        },
        "batch_metrics": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "batch_duration": latest_end,
            "total_token_throughput": total_token_throughput,
            "output_token_throughput": output_token_throughput,
            "combined_decode_throughput_sum": decode_throughput_sum,
            "combined_throughput": total_token_throughput,
            "total_requests": total_successful
        },
        "failures": {
            "total_requests": total_requests,
            "successful": total_successful,
            "failed": total_failed
        }
    }


def run_multiprocess_benchmark(user_requests: List[Dict], api_base: str, model: str,
                             temperature: float, timeout: int, n_cores: int) -> Dict:
    """Run benchmark using multiple processes (same format as original)."""
    
    # Disable HuggingFace Transformers' tokenizer parallelism in Rust
    # to prevent conflicts with Python multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    total_requests = len(user_requests)
    
    # Use all available cores, but don't exceed the number of requests
    actual_cores = min(n_cores, total_requests)
    
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
            args=(i, chunk, api_base, model, temperature, timeout, max_concurrent_per_process, result_queue)
        )
        processes.append(p)
        p.start()
    
    # Unified multiprocessing wait budget for queue retrieval and process joins
    mp_wait_timeout = 600 + timeout*4

    worker_results = []
    for _ in range(actual_cores):
        try:
            result = result_queue.get(timeout=mp_wait_timeout)
            if result is not None:
                worker_results.append(result)
        except:
            print(f"Warning: Failed to get result from a worker process")
    
    # Wait for all processes to complete
    for p in processes:
        p.join(timeout=mp_wait_timeout)
        if p.is_alive():
            print(f"Warning: Process {p.pid} didn't finish in time, terminating")
            p.terminate()
            p.join(5)
    
    if not worker_results:
        raise RuntimeError("No worker processes completed successfully")
    
    # Print aggregated results from all workers
    aggregated_results = aggregate_metrics(worker_results)
    
    # Calculate total success/failure across all workers
    total_failures = aggregated_results.get("failures", {})
    total_requests = total_failures.get("total_requests", 0)
    successful = total_failures.get("successful", 0)
    failed = total_failures.get("failed", 0)
    
    print(f"    📊 Request Results: {successful}/{total_requests} successful, {failed} failed")
    
    if failed > 0:
        print(f"    ❌ {failed} requests failed")
    
    return aggregated_results


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
    all_decode_time = []
    
    # Batch-level metrics (one value per run)
    all_total_token_throughput = []
    all_output_token_throughput = []
    all_combined_decode_throughput_sum = []
    all_batch_duration = []
    all_total_tokens_per_run = []
    all_total_output_tokens_per_run = []
    all_total_input_tokens_per_run = []
    all_success_requests_per_run = []
    all_ttft_missing_count = []
    
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
        all_decode_time.extend(decode_metrics.get("decode_time_per_request", []))
        
        # Batch-level metrics (one value per run)
        batch_metrics = run_data.get("batch_metrics", {})
        all_total_token_throughput.append(batch_metrics.get("total_token_throughput", batch_metrics.get("combined_throughput", 0)))
        all_output_token_throughput.append(batch_metrics.get("output_token_throughput", 0))
        all_combined_decode_throughput_sum.append(batch_metrics.get("combined_decode_throughput_sum", 0))
        all_batch_duration.append(batch_metrics.get("batch_duration", 0))
        all_total_tokens_per_run.append(batch_metrics.get("total_tokens", 0))
        all_total_output_tokens_per_run.append(batch_metrics.get("total_output_tokens", 0))
        all_total_input_tokens_per_run.append(batch_metrics.get("total_input_tokens", 0))
        all_success_requests_per_run.append(batch_metrics.get("total_requests", 0))
        all_ttft_missing_count.append(prefill_metrics.get("ttft_missing_count", 0))
        
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
            },
            "ttft_missing_count_per_run": {
                "mean": safe_mean(all_ttft_missing_count),
                "std": safe_std(all_ttft_missing_count)
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
            },
            "decode_time": {
                "mean": safe_mean(all_decode_time),
                "std": safe_std(all_decode_time)
            }
        },
        # Batch-level combined statistics
        "batch_metrics": {
            "combined_throughput": {
                "mean": safe_mean(all_total_token_throughput),
                "std": safe_std(all_total_token_throughput)
            },
            "total_token_throughput": {
                "mean": safe_mean(all_total_token_throughput),
                "std": safe_std(all_total_token_throughput)
            },
            "output_token_throughput": {
                "mean": safe_mean(all_output_token_throughput),
                "std": safe_std(all_output_token_throughput)
            },
            "combined_decode_throughput_sum": {
                "mean": safe_mean(all_combined_decode_throughput_sum),
                "std": safe_std(all_combined_decode_throughput_sum)
            },
            "batch_duration": {
                "mean": safe_mean(all_batch_duration),
                "std": safe_std(all_batch_duration)
            },
            "total_tokens": {
                "mean": safe_mean(all_total_tokens_per_run),
                "std": safe_std(all_total_tokens_per_run)
            },
            "total_input_tokens": {
                "mean": safe_mean(all_total_input_tokens_per_run),
                "std": safe_std(all_total_input_tokens_per_run)
            },
            "total_output_tokens": {
                "mean": safe_mean(all_total_output_tokens_per_run),
                "std": safe_std(all_total_output_tokens_per_run)
            },
            "successful_requests": {
                "mean": safe_mean(all_success_requests_per_run),
                "std": safe_std(all_success_requests_per_run)
            }
        },
        # Failure/Success metrics
        "reliability": {
            "total_requests": total_requests,
            "successful_requests": total_successful, 
            "failed_requests": total_failed,
            "success_rate": (total_successful / total_requests * 100) if total_requests > 0 else 0,
            "failure_rate": (total_failed / total_requests * 100) if total_requests > 0 else 0,
            "ttft_missing_count": int(sum(all_ttft_missing_count)),
            "ttft_coverage_rate": ((total_successful - sum(all_ttft_missing_count)) / total_successful * 100)
                                 if total_successful > 0 else 0
        }
    }


def warmup_server(api_base: str, model: str, temperature: float, timeout: int,
                  text_sampler, batch_sampler, num_warmup_runs: int = 3,
                  max_input_tokens_cap: int = 0, max_output_tokens_cap: int = 0):
    """Perform warmup runs to prepare the server before benchmarking."""
    print(f"Performing {num_warmup_runs} warmup runs...", end="", flush=True)
    
    # Create a simple scenario for warmup (small tokens)
    warmup_scenario = Scenario.from_string("N(100,50)/(50,25)")
    
    async def run_warmup():
        for _ in range(num_warmup_runs):
            # Generate a single warmup request
            sampled_requests = batch_sampler.sample_batch(warmup_scenario, 1)
            warmup_request = sampled_requests[0]
            
            # Use scenario's sampled output tokens (consistent with main benchmark)
            max_tokens = warmup_request.target_output_tokens
            max_tokens = max(1, min(4096, int(max_tokens)))  # Same bounds as main benchmark
            
            warmup_data = build_request_dict(
                warmup_request,
                text_sampler.tokenizer,
                max_input_tokens_cap=max_input_tokens_cap,
                max_output_tokens_cap=max_output_tokens_cap,
            )
            warmup_data["max_tokens"] = min(warmup_data["max_tokens"], max_tokens)
            
            # Use tokenizer from text_sampler (already loaded)
            tokenizer = text_sampler.tokenizer
            
            # Run single warmup request (simple, not multiprocessing)
            async with aiohttp.ClientSession() as session:
                try:
                    await single_request(session, api_base, model, warmup_data, temperature, timeout, time.time(), tokenizer)
                except Exception:
                    pass  # Ignore warmup errors
    
    # Run the async warmup in its own event loop
    asyncio.run(run_warmup())
    print(" done")


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
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic sampling")
    parser.add_argument("--timeout", type=int, default=3000, help="Timeout in seconds per request.")
    
    # Benchmark parameters (same as original)
    parser.add_argument("--batch-sizes", default="1,2,4,8", help="Comma-separated batch sizes")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs per batch size")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Number of warmup runs before each batch size")
    parser.add_argument("--execution-mode", choices=["burst", "refill"], default="burst",
                        help="burst: fixed one-shot batch, refill: maintain max concurrency with prompt refill.")
    parser.add_argument("--num-prompts", type=int, default=0,
                        help="Total prompts per run in refill mode (must be > max_concurrency). 0 => auto.")
    parser.add_argument("--max-concurrency", type=int, default=0,
                        help="Override max in-flight requests. 0 => use current batch_size.")
    parser.add_argument("--max-input-tokens-cap", type=int, default=0,
                        help="Hard cap for input tokens per request after sampling. 0 => no cap.")
    parser.add_argument("--max-output-tokens-cap", type=int, default=0,
                        help="Hard cap for output max_tokens per request. 0 => no cap.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature parameter")
    parser.add_argument("--description", default="Enhanced benchmark", help="Description of the benchmark")
    
    # Output configuration
    parser.add_argument("--results-file", default="completion_advanced_benchmark_results.json", help="Output file for results")

    # LoRA configuration (direct parameters)
    parser.add_argument("--lora-strategy", help="LoRA distribution strategy (single, uniform, zipf, mixed, all-unique)")
    parser.add_argument("--lora-names", help="Comma-separated LoRA adapter names")
    parser.add_argument("--base-model-ratio", type=float, default=0.0, help="Fraction of requests using base model without LoRA (0.0-1.0)")
    parser.add_argument("--zipf-alpha", type=float, default=1.0, help="Zipf distribution alpha parameter (default: 1.0)")

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

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
    dataset_config = DatasetConfig.from_file(args.dataset_config)
    dataset_loader = DatasetLoader(dataset_config)

    # Initialize tokenizer (default to model name if not specified)
    tokenizer_name = args.tokenizer if args.tokenizer else args.model

    # Create samplers for on-demand generation
    text_sampler = TextSampler(tokenizer_name, dataset_loader)
    batch_sampler = BatchSampler(text_sampler)

    print(f"📊 Initialized benchmark with scenario: {args.scenario}")
    print(f"📚 Loaded dataset: {len(dataset_loader)} samples")
    if lora_config:
        print(f"🎯 LoRA strategy: {lora_config.strategy} with {len(lora_config.lora_names)} adapters")
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
            "warmup_runs": args.warmup_runs,
            "execution_mode": args.execution_mode,
            "num_prompts": args.num_prompts,
            "max_concurrency": args.max_concurrency,
            "max_input_tokens_cap": args.max_input_tokens_cap,
            "max_output_tokens_cap": args.max_output_tokens_cap,
            "temperature": args.temperature,
            "description": args.description,
            "seed": args.seed,
            "lora_config": {
                "strategy": lora_config.strategy if lora_config else None,
                "lora_names": lora_config.lora_names if lora_config else [],
                "base_model_ratio": lora_config.base_model_ratio if lora_config else 0.0
            } if lora_config else None
        },
        "results": {}
    }
    
    # Run benchmark for each batch size
    for batch_size in batch_sizes:
        effective_max_concurrency = args.max_concurrency if args.max_concurrency > 0 else batch_size
        prompts_per_run = batch_size
        if args.execution_mode == "refill":
            prompts_per_run = args.num_prompts if args.num_prompts > 0 else max(32, effective_max_concurrency * 4)
            if prompts_per_run <= effective_max_concurrency:
                raise ValueError(
                    f"num_prompts ({prompts_per_run}) must be > max_concurrency ({effective_max_concurrency}) in refill mode"
                )

        print(
            f"\n🔄 Testing batch size: {batch_size} | mode={args.execution_mode} "
            f"| max_concurrency={effective_max_concurrency} | prompts_per_run={prompts_per_run}"
        )
        
        warmup_seed = derive_seed(args.seed, batch_size, -1)
        with use_seed(warmup_seed):
            warmup_server(args.api_base, args.model, args.temperature, args.timeout,
                         text_sampler, batch_sampler, args.warmup_runs,
                         max_input_tokens_cap=args.max_input_tokens_cap,
                         max_output_tokens_cap=args.max_output_tokens_cap)
        
        runs_data = []
        for run_idx in range(args.num_runs):
            print(f"  Run {run_idx + 1}/{args.num_runs}")
            
            # Generate batch of requests using scenario (on-demand)
            run_seed = derive_seed(args.seed, batch_size, run_idx)
            with use_seed(run_seed):
                sampled_requests = batch_sampler.sample_batch(scenario, prompts_per_run)

            # Assign LoRAs if config is provided
            lora_assignments = None
            if lora_config:
                lora_assignments = lora_config.assign_lora(prompts_per_run)

            user_requests = []
            for idx, request in enumerate(sampled_requests):
                request_dict = build_request_dict(
                    request,
                    text_sampler.tokenizer,
                    max_input_tokens_cap=args.max_input_tokens_cap,
                    max_output_tokens_cap=args.max_output_tokens_cap,
                )

                # Add LoRA assignment if available
                if lora_assignments:
                    request_dict["lora_name"] = lora_assignments[idx]

                user_requests.append(request_dict)
            
            if args.execution_mode == "refill":
                run_data = asyncio.run(
                    run_refill_async(
                        user_requests,
                        args.api_base,
                        args.model,
                        args.temperature,
                        args.timeout,
                        effective_max_concurrency,
                        tokenizer=text_sampler.tokenizer,
                    )
                )
                failures = run_data.get("failures", {})
                total_req = failures.get("total_requests", 0)
                successful = failures.get("successful", 0)
                failed = failures.get("failed", 0)
                print(f"    📊 Request Results: {successful}/{total_req} successful, {failed} failed")
            else:
                # Run the batch (legacy one-shot mode with multiprocessing)
                run_data = run_multiprocess_benchmark(
                    user_requests, args.api_base, args.model, args.temperature, args.timeout,
                    multiprocessing.cpu_count()
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
                decode_time_values = decode_metrics.get("decode_time_per_request", [])
                
                # Calculate averages per request
                avg_ttft = statistics.mean(ttft_values) if ttft_values else 0
                avg_input_throughput = statistics.mean(input_throughput_values) if input_throughput_values else 0
                avg_output_throughput = statistics.mean(output_throughput_values) if output_throughput_values else 0
                avg_decode_time = statistics.mean(decode_time_values) if decode_time_values else 0
                
                # Extract batch-level throughput metrics
                batch_metrics = run_data.get("batch_metrics", {})
                total_token_throughput = batch_metrics.get(
                    "total_token_throughput",
                    batch_metrics.get("combined_throughput", 0),
                )
                decode_sum_throughput = batch_metrics.get("combined_decode_throughput_sum", 0)
                
                success_rate = (successful / total_req * 100) if total_req > 0 else 0
                print(f"    ✅ Batch: {batch_time:.2f}s | Success: {successful}/{total_req} ({success_rate:.1f}%)")
                if failed > 0:
                    print(f"       ❌ {failed} requests failed")
                print(f"       Tokens - Input: {total_input_tokens:,} | Output: {total_output_tokens:,}")
                print(f"       Total Throughput: {total_token_throughput:.1f} tok/s (input+output over batch window)")
                print(f"       Decode Sum Throughput (legacy): {decode_sum_throughput:.1f} tok/s")
                
                if avg_ttft > 0:
                    print(f"       Per-Request Avg - TTFT: {avg_ttft:.3f}s | Prefill: {avg_input_throughput:.1f} tok/s | Decode: {avg_output_throughput:.1f} tok/s | Decode Time: {avg_decode_time:.3f}s")
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
