"""Synchronized burst benchmarks for SGLang's next-token scoring API."""

import argparse
import asyncio
import csv
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from . import __version__
from .io import atomic_write_json

METRICS = ("decisions_per_second", "input_tokens_per_second",
           "burst_latency_ms")


def positive_ints(value):
    values = [int(x) for x in value.split(",")]
    if not values or min(values) < 1 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected distinct positive integers, separated by commas")
    return values


def load_workload(path):
    workload = json.loads(Path(path).read_text())
    if not isinstance(workload, dict):
        raise ValueError("workload must be a JSON object")
    for key in ("query_prefix", "query_suffix", "filler_token_ids", "label_token_ids"):
        validate_ids(workload[key], key, allow_empty=key.startswith("query_"))
    if not isinstance(workload["items"], list) or not workload["items"]:
        raise ValueError("items must be a nonempty list of token-ID lists")
    for item in workload["items"]:
        validate_ids(item, "item")
    if len(set(workload["label_token_ids"])) != len(workload["label_token_ids"]):
        raise ValueError("label_token_ids must be unique")
    return workload


def validate_ids(values, name, allow_empty=False):
    if (not isinstance(values, list) or (not values and not allow_empty)
            or any(type(x) is not int or x < 0 for x in values)):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")


def payloads(workload, model, config, run, seed):
    length, items, concurrency = config
    count = length - len(workload["query_prefix"]) - len(workload["query_suffix"])
    # The first C states match across concurrency and item-count sweeps.
    rng = random.Random(f"{seed}:{length}:{run}")
    bodies = []
    for _ in range(concurrency):
        query = (workload["query_prefix"] + rng.choices(workload["filler_token_ids"], k=count)
                 + workload["query_suffix"])
        body = {"model": model, "query": query, "items": workload["items"][:items],
                "label_token_ids": workload["label_token_ids"], "apply_softmax": True}
        bodies.append(json.dumps(body, separators=(",", ":")).encode())
    return bodies


def input_token_count(workload, config, mode):
    length, items, concurrency = config
    suffixes = sum(map(len, workload["items"][:items]))
    if mode == "mis":
        return concurrency * (length + suffixes + items + 1)
    return concurrency * (items * length + suffixes)


def validate_response(response, items, labels, expected_tokens):
    scores = response.get("scores")
    if not isinstance(scores, list) or len(scores) != items:
        raise ValueError("incorrect number of score rows")
    for row in scores:
        if (not isinstance(row, list) or len(row) != labels
                or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in row)
                or abs(sum(row) - 1) > 1e-4):
            raise ValueError("invalid normalized label scores")
    tokens = response.get("usage", {}).get("prompt_tokens")
    if type(tokens) is not int or tokens != expected_tokens:
        raise ValueError(f"prompt token count: expected {expected_tokens}, received {tokens}")


async def burst(session, url, bodies, config, run, workload, mode):
    length, items, concurrency = config
    processed = input_token_count(workload, config, mode)
    gate = asyncio.Event()

    async def request(body):
        digest = hashlib.sha256(body).hexdigest()
        await gate.wait()
        dispatched = time.perf_counter()
        result = {"sha256": digest}
        try:
            async with session.post(url, data=body) as response:
                response.raise_for_status()
                data = await response.json()
                completed = time.perf_counter()
            validate_response(data, items, len(workload["label_token_ids"]), processed // concurrency)
        except Exception as exc:
            completed = time.perf_counter()
            result["error"] = str(exc)
        result.update(latency_ms=1000 * (completed - dispatched),
                      dispatch_delay_ms=1000 * (dispatched - started),
                      completion_ms=1000 * (completed - started))
        return result

    tasks = [asyncio.create_task(request(body)) for body in bodies]
    await asyncio.sleep(0)
    started_unix = time.time()
    started = time.perf_counter()
    gate.set()
    try:
        requests = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    # Stop at the last decoded response, before client validation/aggregation.
    elapsed = max(r["completion_ms"] for r in requests) / 1000
    result = {"query_tokens": length, "items": items, "concurrency": concurrency,
              "run": run, "started_unix": started_unix, "requests": requests,
              "errors": sum("error" in r for r in requests)}
    if not result["errors"]:
        result.update(input_tokens=processed,
                      decisions_per_second=items * concurrency / elapsed,
                      input_tokens_per_second=processed / elapsed,
                      burst_latency_ms=elapsed * 1000)
    return result


def summarize(rows, configs):
    summary = []
    for length, items, concurrency in configs:
        matching = [r for r in rows if (r["query_tokens"], r["items"], r["concurrency"])
                    == (length, items, concurrency)]
        entry = dict(query_tokens=length, items=items, concurrency=concurrency, runs=len(matching))
        for metric in METRICS:
            values = [r[metric] for r in matching]
            entry[metric] = {"mean": statistics.mean(values),
                             "std": statistics.stdev(values) if len(values) > 1 else 0.0}
        summary.append(entry)
    return summary


async def benchmark(args):
    workload = load_workload(args.workload)
    fixed = len(workload["query_prefix"]) + len(workload["query_suffix"])
    if min(args.query_lengths) <= fixed:
        raise ValueError("query lengths must leave at least one filler token")
    if max(args.items) > len(workload["items"]):
        raise ValueError("item count exceeds the workload's question suffixes")
    configs = list(itertools.product(args.query_lengths, args.items, args.batch_sizes))
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=False)
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    base = args.api_base.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    metadata = {"schema_version": 1, "tokenomics_version": __version__,
                "started_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                "workload_sha256": hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest(),
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "connection_policy": "fresh connections; no retries", "status": "running"}
    atomic_write_json(out / "metadata.json", metadata)
    atomic_write_json(out / "workload.json", workload)
    try:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, force_close=True), headers=headers,
            timeout=aiohttp.ClientTimeout(total=args.timeout),
        ) as session:
            async with session.get(root + "/get_server_info") as response:
                response.raise_for_status()
                info = await response.json()
            server = info.get("server_args", info)
            if bool(server.get("enable_mis", False)) != (args.mode == "mis"):
                raise ValueError("--mode does not match the server's enable_mis setting")
            if not server.get("disable_radix_cache", False):
                raise ValueError("cold-prefix benchmark requires server --disable-radix-cache")
            metadata["server"] = info
            atomic_write_json(out / "metadata.json", metadata)
            rows = []
            with (out / "bursts.jsonl").open("w") as raw:
                for run in range(-args.warmup_runs, args.num_runs):
                    order = configs.copy()
                    random.Random(args.seed + run).shuffle(order)
                    for config in order:
                        bodies = payloads(workload, args.model, config, run, args.seed)
                        row = await burst(session, root + "/v1/score", bodies, config, run, workload, args.mode)
                        row["warmup"] = run < 0
                        raw.write(json.dumps(row) + "\n")
                        raw.flush()
                        if row["errors"]:
                            print(f"WARNING: {row['errors']}/{config[2]} requests failed "
                                  f"(run={run}, query={config[0]}, items={config[1]}, "
                                  f"concurrency={config[2]}). Stopping benchmark; "
                                  f"details: {out / 'bursts.jsonl'}", file=sys.stderr, flush=True)
                            raise RuntimeError(f"{row['errors']} requests failed; see {out / 'bursts.jsonl'}")
                        if run >= 0:
                            rows.append(row)
                        print(f"run={run} query={config[0]} items={config[1]} concurrency={config[2]} "
                              f"{row['decisions_per_second']:.1f} decisions/s, "
                              f"{row['burst_latency_ms']:.1f} ms", flush=True)
        summary = summarize(rows, configs)
        atomic_write_json(out / "summary.json", summary)
        fields = ["query_tokens", "items", "concurrency", "runs"]
        fields += [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]
        with (out / "summary.csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in summary:
                writer.writerow({**{k: row[k] for k in fields[:4]},
                                 **{f"{metric}_{stat}": row[metric][stat]
                                    for metric in METRICS for stat in ("mean", "std")}})
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="failed", error=str(exc) or type(exc).__name__)
        raise
    finally:
        atomic_write_json(out / "metadata.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True, help="JSON file containing tokenized prompt parts")
    parser.add_argument("--mode", choices=("sis", "mis"), required=True)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--query-lengths", type=positive_ints, default=[260, 1040, 5200])
    parser.add_argument("--items", type=positive_ints, default=[1])
    parser.add_argument("--batch-sizes", type=positive_ints, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", required=True, help="New directory; existing results are never overwritten")
    args = parser.parse_args()
    if args.num_runs < 1 or args.warmup_runs < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("num-runs and timeout must be positive; warmup-runs must be nonnegative")
    try:
        asyncio.run(benchmark(args))
    except (ValueError, KeyError, OSError, RuntimeError, aiohttp.ClientError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
