"""Benchmark candidate ranking or independent questions through SGLang /v1/score."""
import argparse
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


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be an object")
    workload = config.get("workload", "ranking")
    if workload not in ("ranking", "questions"):
        raise ValueError("workload must be ranking/questions")
    questions = workload == "questions"
    mode = config.get("formulation")
    if questions and mode is not None:
        raise ValueError("Independent questions do not use a ranking formulation")
    if not questions and mode not in ("pointwise", "setwise"):
        raise ValueError("formulation must be pointwise/setwise")
    for key in ("query_template", "item_template"):
        if not isinstance(config.get(key), str):
            raise ValueError(f"{key} must be a string")
    # Explicit fields prevent accidentally exposing siblings in pointwise requests.
    import string
    item_fields = {"question"} if questions else {"candidate"} if mode == "pointwise" else {"options", "labels"}
    allowed = {"query_template": {"state"}, "item_template": item_fields}
    for key, fields in allowed.items():
        for _, field, spec, conversion in string.Formatter().parse(config[key]):
            if field is not None and (field not in fields or spec or conversion):
                raise ValueError(f"Unsupported placeholder in {key}: {field}")
    required = "{question}" if questions else "{candidate}" if mode == "pointwise" else "{options}"
    if "{state}" not in config["query_template"] or required not in config["item_template"]:
        raise ValueError(f"Templates must include {{state}} and {required} placeholders")
    if questions:
        return
    labels = config.get("labels")
    if not isinstance(labels, list) or len(labels) < 2:
        raise ValueError("At least two labels are required")
    ids, texts = [], []
    for label in labels:
        if not isinstance(label, dict) or not isinstance(label.get("text"), str) or not label["text"]:
            raise ValueError("Each label needs nonempty text and a token_id")
        token_id = label.get("token_id")
        if type(token_id) is not int or token_id < 0:
            raise ValueError("Label token IDs must be nonnegative integers")
        ids.append(token_id)
        texts.append(label["text"])
    if len(set(ids)) != len(ids) or len(set(texts)) != len(texts):
        raise ValueError("Label texts and token IDs must be unique")
    positive = config.get("positive_label_index", 0)
    if type(positive) is not int or not 0 <= positive < len(labels):
        raise ValueError("positive_label_index is outside the labels")


def validate_questions(record):
    questions = record.get("questions")
    if not isinstance(record.get("state"), str) or not isinstance(questions, list) or not questions:
        raise ValueError("Each record needs state and at least one question")
    for question in questions:
        if not isinstance(question, dict) or not isinstance(question.get("prompt"), str) or not question["prompt"]:
            raise ValueError("Each question needs a nonempty prompt")
        labels = question.get("labels")
        if not isinstance(labels, list) or len(labels) < 2:
            raise ValueError("Each question needs at least two labels")
        ids, texts = [], []
        for label in labels:
            if not isinstance(label, dict) or not isinstance(label.get("text"), str) or not label["text"]:
                raise ValueError("Each label needs nonempty text")
            aliases = label.get("token_ids")
            if not isinstance(aliases, list) or not aliases or any(type(t) is not int or t < 0 for t in aliases):
                raise ValueError("Each label needs nonempty nonnegative token_ids")
            ids.extend(aliases)
            texts.append(label["text"])
        if len(set(ids)) != len(ids) or len(set(texts)) != len(texts):
            raise ValueError("Label texts and aliases must be unique within each question")
        expected = question.get("expected_index")
        if expected is not None and (type(expected) is not int or not 0 <= expected < len(labels)):
            raise ValueError("Question expected_index must identify a label")


def load_records(path, workload="ranking"):
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("Each record must be an object")
        if workload == "questions":
            validate_questions(record)
            records.append(record)
            continue
        candidates = record.get("candidates")
        if not isinstance(record.get("state"), str) or not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError("Each record needs state and at least two candidates")
        if not all(isinstance(x, str) and x for x in candidates):
            raise ValueError("Candidates must be nonempty strings")
        expected = record.get("expected_index")
        if expected is not None and (type(expected) is not int or not 0 <= expected < len(candidates)):
            raise ValueError("expected_index must identify a candidate")
        records.append(record)
    if not records:
        raise ValueError("Dataset is empty")
    return records


def build_request(record, config, model):
    validate_config(config)
    if config.get("workload") == "questions":
        validate_questions(record)
        return {"model": model,
                "query": config["query_template"].format(state=record["state"]),
                "items": [config["item_template"].format(question=q["prompt"]) for q in record["questions"]],
                "label_token_ids": list(dict.fromkeys(t for q in record["questions"]
                                                     for label in q["labels"] for t in label["token_ids"])),
                "apply_softmax": False}
    labels = config["labels"]
    query = config["query_template"].format(state=record["state"])
    if config["formulation"] == "pointwise":
        items = [config["item_template"].format(candidate=c) for c in record["candidates"]]
    elif config["formulation"] == "setwise":
        if len(record["candidates"]) > len(labels):
            raise ValueError("Setwise needs one configured label token per candidate")
        labels = labels[:len(record["candidates"])]
        options = "\n".join(f"{label['text']}) {candidate}" for label, candidate in zip(labels, record["candidates"]))
        items = [config["item_template"].format(options=options, labels=", ".join(x["text"] for x in labels))]
    return {"model": model, "query": query, "items": items,
            "label_token_ids": [x["token_id"] for x in labels], "apply_softmax": True}


def response_scores(response, body):
    if not isinstance(response, dict):
        raise ValueError("Score response must be an object")
    scores = response.get("scores")
    if not isinstance(scores, list) or len(scores) != len(body["items"]):
        raise ValueError("Unexpected score row count")
    for row in scores:
        if not isinstance(row, list) or len(row) != len(body["label_token_ids"]):
            raise ValueError("Unexpected label score count")
        if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in row):
            raise ValueError("Scores must be finite probabilities")
        if body["apply_softmax"] and abs(sum(row) - 1) > 1e-4:
            raise ValueError("Scores are not normalized over labels")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Response must include usage")
    tokens = usage.get("prompt_tokens")
    if type(tokens) is not int or tokens < 0:
        raise ValueError("Response must report nonnegative usage.prompt_tokens")
    return scores, tokens


def parse_response(response, body, config, candidate_count):
    scores, tokens = response_scores(response, body)
    if config["formulation"] == "pointwise":
        if len(scores) != candidate_count:
            raise ValueError("Missing pointwise candidate scores")
        values = [row[config.get("positive_label_index", 0)] for row in scores]
    elif config["formulation"] == "setwise":
        values = scores[0]
        if len(values) != candidate_count:
            raise ValueError("Missing setwise choice scores")
    return max(range(len(values)), key=values.__getitem__), tokens


def parse_questions(response, body, record):
    scores, tokens = response_scores(response, body)
    positions = {token: i for i, token in enumerate(body["label_token_ids"])}
    answers = []
    for question, row in zip(record["questions"], scores):
        # /score returns vocabulary probabilities, not logits. Pool aliases first,
        # then normalize only over this question's labels (never the union).
        pooled = [max(row[positions[t]] for t in label["token_ids"]) for label in question["labels"]]
        total = sum(pooled)
        if total <= 0:
            raise ValueError("Question labels have zero probability mass")
        probabilities = [value / total for value in pooled]
        selected = max(range(len(pooled)), key=pooled.__getitem__)
        expected = question.get("expected_index")
        answers.append({"selected_index": selected, "probabilities": probabilities,
                        "correct": None if expected is None else selected == expected})
    return answers, tokens


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
    labelled = [r for r in decisions if r["correct"] is not None]
    latencies = [r["latency_ms"] for r in successful]
    result = {"attempted_requests": len(requests), "completed_requests": len(successful),
              "failed_requests": len(requests) - len(successful),
              "requests_per_second": len(successful) / elapsed,
              "decisions_per_second": len(decisions) / elapsed,
              "input_tokens_per_second": sum(r["prompt_tokens"] for r in successful) / elapsed,
              "latency_ms_mean": statistics.mean(latencies) if latencies else None,
              "latency_ms_p95": percentile(latencies, .95) if latencies else None,
              "timing": "sum of measured burst completion times; successful-request latency excludes failures"}
    accuracy = sum(r["correct"] for r in labelled) / len(labelled) if labelled else None
    if workload == "questions":
        result.update(completed_questions=len(decisions), questions_per_second=len(decisions) / elapsed,
                      attempted_questions=sum(r["question_count"] for r in requests),
                      labelled_successful_questions=len(labelled),
                      accuracy_on_successful_labelled_questions=accuracy,
                      decision_unit="one independently answered question")
    else:
        result.update(attempted_candidate_sets=len(requests), completed_candidate_sets=len(successful),
                      failed_candidate_sets=len(requests) - len(successful),
                      completed_candidates=sum(r["candidate_count"] for r in successful),
                      labelled_successful_sets=len(labelled), accuracy_on_successful_labelled_sets=accuracy,
                      decision_unit="one complete candidate set (one HTTP request)")
    return result


async def benchmark(args):
    config = json.loads(Path(args.config).read_text())
    if not isinstance(config, dict):
        raise ValueError("Config must be an object")
    if "formulation" in config and config["formulation"] != args.formulation:
        raise ValueError("Config formulation conflicts with --formulation")
    workload = getattr(args, "workload", "ranking")
    if config.get("workload", workload) != workload:
        raise ValueError("Config workload conflicts with --workload")
    config["workload"] = workload
    config["formulation"] = args.formulation
    validate_config(config)
    records = load_records(args.dataset, workload)
    if args.num_prompts is not None:
        records = records[:args.num_prompts]
    bodies = [build_request(r, config, args.model) for r in records]
    encoded = [encode(body) for body in bodies]
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=False)
    metadata = {"workload": workload, "formulation": args.formulation, "config": config, "model": args.model, "api_base": args.api_base,
                "dataset_sha256": hashlib.sha256(encode(records)).hexdigest(),
                "request_sha256": [hashlib.sha256(body).hexdigest() for body in encoded],
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "requests": len(records), "batch_sizes": args.batch_sizes,
                "num_runs": args.num_runs, "warmup_runs": args.warmup_runs, "seed": args.seed,
                "quality_note": "Different formulations need separate quality evaluation; scores are not assumed equivalent.",
                "http": "fresh connections, no retries, burst execution"}
    if workload == "ranking":
        metadata["candidate_sets"] = len(records)
    atomic_write_json(str(out / "metadata.json"), metadata)
    with (out / "requests.jsonl").open("w") as f:
        for record, body in zip(records, bodies):
            f.write(json.dumps({"record": record, "body": body}) + "\n")
    if args.dry_run:
        return True
    root = args.api_base.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout,
                                     connector=aiohttp.TCPConnector(limit=0, force_close=True)) as session:
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
                      "request_sha256": metadata["request_sha256"][index], "success": False}
            try:
                async with session.post(root + "/v1/score", data=encoded[index]) as response:
                    response.raise_for_status()
                    data = await response.json()
                if workload == "questions":
                    answers, tokens = parse_questions(data, bodies[index], records[index])
                    result.update(answers=answers)
                else:
                    selected, tokens = parse_response(data, bodies[index], config, result["candidate_count"])
                    expected = records[index].get("expected_index")
                    result.update(selected_index=selected, correct=None if expected is None else selected == expected)
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
                result.update(batch_size=batch_size, workload=workload, formulation=config["formulation"])
                atomic_write_json(str(out / f"{batch_size}.json"), result)
                all_ok = all_ok and result["failed_requests"] == 0
                print(json.dumps(result), flush=True)
        return all_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", choices=("ranking", "questions"), default="ranking")
    parser.add_argument("--formulation", choices=("pointwise", "setwise"),
                        help="Required for ranking only; no default")
    parser.add_argument("--config", required=True, help="Scoring prompt and label JSON config")
    parser.add_argument("--dataset", required=True, help="JSONL records: state with candidates (ranking) or questions")
    parser.add_argument("--api-base", default="http://localhost:30000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--batch-sizes", default="1,2,4,8,16", help="Concurrent HTTP requests per burst")
    parser.add_argument("--num-prompts", type=int, help="Maximum dataset records (no automatic repetition)")
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
