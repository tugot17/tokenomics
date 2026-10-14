"""Benchmark candidate ranking or independent questions through SGLang /v1/score."""
import argparse
import copy
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import aiohttp

from tokenomics.io import atomic_write_json
from tokenomics.scoring import (
    build_request, load_records, parse_questions, parse_response,
    resolve_label_ids, validate_config,
)


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def summarize(bursts, workload="ranking"):
    requests = [r for b in bursts for r in b["requests"]]
    successful = [r for r in requests if r["success"]]
    elapsed = sum(b["wall_seconds"] for b in bursts)
    decisions = [a for r in successful for a in r["answers"]] if workload == "questions" else successful
    latencies = [r["latency_ms"] for r in successful]
    result = {"attempted_requests": len(requests), "completed_requests": len(successful),
              "failed_requests": len(requests) - len(successful),
              "requests_per_second": len(successful) / elapsed,
              "input_tokens_per_second": sum(r["prompt_tokens"] for r in successful) / elapsed,
              "latency_ms_mean": statistics.mean(latencies) if latencies else None,
              "latency_ms_p95": percentile(latencies, .95) if latencies else None,
              "timing": "sum of measured burst completion times; successful-request latency excludes failures"}
    if workload == "questions":
        result.update(completed_questions=len(decisions), questions_per_second=len(decisions) / elapsed,
                      attempted_questions=sum(r["question_count"] for r in requests),
                      decision_unit="one independently answered question")
    else:
        result.update(completed_candidates=sum(r["candidate_count"] for r in successful),
                      decision_unit="one complete candidate set (one HTTP request)")
    return result


def state_scenario(length):
    from tokenomics.sampling.scenarios import Scenario

    if isinstance(length, int):
        expression = f"D({length},1)"
    elif length.startswith("N("):
        expression = length + "/(1,0)"
    elif length.startswith("U("):
        expression = length + "/(1,1)"
    else:
        raise ValueError("State tokens must be a positive integer, N(mean,std), or U(min,max)")
    scenario = Scenario.from_string(expression)
    lower = getattr(scenario, "input_tokens", getattr(scenario, "mean_input", getattr(scenario, "min_input", 0)))
    if lower < 1 or getattr(scenario, "max_input", lower) < lower:
        raise ValueError("State token lengths must be positive; uniform max must be >= min")
    return scenario


def synthetic_records(records, tokenizer, length, count, seed):
    words = "river forest market garden bridge mountain paper orange silver window morning evening quiet bright walk read build carry find follow Yes No A B C".split()
    from tokenomics.sampling.sampler import use_seed

    scenario = state_scenario(length)
    generated = []
    for index in range(count):
        with use_seed(seed + index):
            target, _ = scenario.sample()
        target = int(target)
        rng = random.Random(seed + index)
        for _ in range(16):
            text = " ".join(rng.choices(words, k=target * 2 + 16))
            ids = tokenizer.encode(text, add_special_tokens=False)[:target]
            state = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if len(tokenizer.encode(state, add_special_tokens=False)) == target:
                break
        else:
            raise ValueError(f"Cannot generate an exact {target}-token state with this tokenizer")
        record = copy.deepcopy(records[index % len(records)])
        record["state"] = state
        generated.append(record)
    return generated


async def benchmark(args):
    config = json.loads(Path(args.config).read_text())
    if not isinstance(config, dict):
        raise ValueError("Config must be an object")
    workload = getattr(args, "workload", "ranking")
    if config.get("formulation", args.formulation) != args.formulation:
        raise ValueError("Config formulation conflicts with --formulation")
    if config.get("workload", workload) != workload:
        raise ValueError("Config workload conflicts with --workload")
    config.update(workload=workload, formulation=args.formulation)
    validate_config(config)
    records = load_records(args.dataset, workload)
    lengths = args.state_tokens
    if workload == "ranking":
        groups = [config["labels"]]
    else:
        groups = [question["labels"] for record in records for question in record["questions"]]
    field = "token_id" if workload == "ranking" else "token_ids"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(getattr(args, "tokenizer", None) or args.model)
    for labels in groups:
        resolve_label_ids(labels, tokenizer, multiple=workload == "questions")
        ids = [t for label in labels for t in (label[field] if workload == "questions" else [label[field]])]
        if len(set(ids)) != len(ids):
            raise ValueError("Different labels must not share token IDs")
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=False)
    count = args.num_prompts or max(len(records), max(args.batch_sizes))
    atomic_write_json(str(out / "metadata.json"), {
        "model": args.model, "workload": workload, "formulation": args.formulation,
        "state_tokens": lengths, "seed": args.seed, "requests_per_length": count,
    })
    all_ok = True
    for length in lengths:
        generated = synthetic_records(records, tokenizer, length, count, args.seed)
        folder = out / f"state-{length}"
        folder.mkdir()
        all_ok = await benchmark_setting(args, config, generated, tokenizer, folder, length) and all_ok
    return all_ok


async def benchmark_setting(args, config, records, tokenizer, out, state_tokens):
    workload = config["workload"]
    tokenizer_path = getattr(args, "tokenizer", None) or args.model
    bodies = [build_request(r, config, args.model, tokenizer) for r in records]
    encoded = [encode(body) for body in bodies]
    metadata = {"workload": workload, "formulation": args.formulation, "config": config, "model": args.model, "tokenizer": tokenizer_path, "api_base": args.api_base,
                "dataset_sha256": hashlib.sha256(encode(records)).hexdigest(),
                "request_sha256": [hashlib.sha256(body).hexdigest() for body in encoded],
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name("scoring.py").read_bytes()).hexdigest(),
                "requests": len(records), "batch_sizes": args.batch_sizes,
                "state_tokens": state_tokens,
                "num_runs": args.num_runs, "warmup_runs": args.warmup_runs, "seed": args.seed,
                "http": "pooled connections, no retries, burst execution"}
    atomic_write_json(str(out / "metadata.json"), metadata)
    with (out / "requests.jsonl").open("w") as f:
        for record, body in zip(records, bodies):
            lengths = {}
            if tokenizer is not None:
                lengths["state_tokens"] = len(tokenizer.encode(record["state"], add_special_tokens=False))
                lengths["prompt_tokens_per_item"] = [
                    len(body["query"]) + len(item) if isinstance(body["query"], list)
                    else len(tokenizer.encode(body["query"] + item, add_special_tokens=False))
                    for item in body["items"]
                ]
            f.write(json.dumps({"record": record, "body": body, **lengths}) + "\n")
    if args.dry_run:
        return True
    return await measure(args, config, records, bodies, encoded, out, metadata)


async def measure(args, config, records, bodies, encoded, out, metadata):
    workload = config["workload"]
    root = args.api_base.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout,
                                     connector=aiohttp.TCPConnector(limit=max(args.batch_sizes))) as session:
        # Runtime details are optional metadata, never a client mode constraint.
        try:
            async with session.get(root + "/get_server_info",
                                   timeout=aiohttp.ClientTimeout(total=min(args.timeout, 5))) as response:
                response.raise_for_status()
                metadata["server"] = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            metadata["server_info_error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(str(out / "metadata.json"), metadata)

        count_field = "question_count" if workload == "questions" else "candidate_count"
        record_field = "questions" if workload == "questions" else "candidates"

        async def request(index):
            started = time.perf_counter()
            result = {"record_index": index,
                      count_field: len(records[index][record_field]),
                      "success": False}
            try:
                async with session.post(root + "/v1/score", data=encoded[index]) as response:
                    response.raise_for_status()
                    data = await response.json()
                if workload == "questions":
                    answers, tokens = parse_questions(data, bodies[index], records[index])
                    result.update(answers=answers)
                else:
                    selected, tokens = parse_response(data, bodies[index], config, result["candidate_count"])
                    result.update(selected_index=selected)
                result.update(success=True, prompt_tokens=tokens, scores=data["scores"])
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError, KeyError) as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            result["latency_ms"] = (time.perf_counter() - started) * 1000
            return result

        all_ok = True
        with (out / "bursts.jsonl").open("w") as raw:
            for batch_size in args.batch_sizes:
                for _ in range(args.warmup_runs):
                    for start in range(0, len(records), batch_size):
                        warm = await asyncio.gather(*(request(i) for i in range(start, min(start + batch_size, len(records)))))
                        if not all(r["success"] for r in warm):
                            atomic_write_json(str(out / "warmup_failure.json"), warm)
                            raise RuntimeError("Scoring warmup failed; see warmup_failure.json")
                bursts = []
                for run in range(args.num_runs):
                    order = list(range(len(records)))
                    random.Random(args.seed + run).shuffle(order)
                    for start in range(0, len(order), batch_size):
                        started = time.perf_counter()
                        responses = await asyncio.gather(*(request(i) for i in order[start:start + batch_size]))
                        burst = {"batch_size": batch_size, "run": run, "requests": responses,
                                 "wall_seconds": time.perf_counter() - started}
                        bursts.append(burst)
                        raw.write(json.dumps(burst) + "\n")
                        raw.flush()
                result = summarize(bursts, workload)
                result.update(batch_size=batch_size, workload=workload, formulation=config["formulation"],
                              state_tokens=metadata["state_tokens"])
                atomic_write_json(str(out / f"{batch_size}.json"), result)
                all_ok = all_ok and result["failed_requests"] == 0
                print(json.dumps(result), flush=True)
        return all_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", help="Tokenizer repository or local directory; defaults to --model")
    parser.add_argument("--workload", choices=("ranking", "questions"), default="ranking")
    parser.add_argument("--formulation", choices=("pointwise", "setwise"),
                        help="Required for ranking only; no default")
    parser.add_argument("--config", required=True, help="Scoring prompt and label JSON config")
    parser.add_argument("--dataset", required=True, help="JSONL candidate/question templates; --state-tokens generates shared text")
    parser.add_argument("--api-base", default="http://localhost:30000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--batch-sizes", default="1,2,4,8,16", help="Concurrent HTTP requests per burst")
    parser.add_argument("--state-tokens", default="1024", help="Shared-state tokens (default: 1024): integer, fixed-length list, N(mean,std), or U(min,max)")
    parser.add_argument("--num-prompts", type=int, help="Generated requests per length (default: enough to fill concurrency)")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--results-dir", required=True, help="New output directory; existing results are never overwritten")
    parser.add_argument("--dry-run", action="store_true", help="Write requests/metadata without contacting a server")
    args = parser.parse_args()
    if args.workload == "ranking" and args.formulation is None:
        parser.error("--formulation is required for ranking")
    if args.workload == "questions" and args.formulation is not None:
        parser.error("--formulation applies only to ranking")
    try:
        spec = args.state_tokens.strip()
        args.state_tokens = [spec] if spec.startswith(("N(", "U(")) else [int(x) for x in spec.split(",")]
        if len(set(args.state_tokens)) != len(args.state_tokens):
            raise ValueError("State token lengths must be unique")
        for length in args.state_tokens:
            state_scenario(length)
        args.batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        if not args.batch_sizes or any(x < 1 for x in args.batch_sizes) or len(set(args.batch_sizes)) != len(args.batch_sizes):
            raise ValueError("batch sizes must be unique positive integers")
        if args.num_runs < 1 or args.warmup_runs < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("num-runs/timeout must be positive and warmup-runs nonnegative")
        if args.num_prompts is not None and args.num_prompts < 1:
            raise ValueError("num-prompts must be positive")
        ok = asyncio.run(benchmark(args))
    except (ValueError, RuntimeError, OSError, asyncio.TimeoutError, aiohttp.ClientError) as exc:
        parser.exit(1, f"error: {exc}\n")
    if not ok:
        parser.exit(1, "error: measured scoring requests failed; inspect bursts.jsonl\n")


if __name__ == "__main__":
    main()
