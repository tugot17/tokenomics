"""Replay candidate sets through /v1/score or /v1/systemone."""
import argparse
import base64
import io
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


def encode(value, *, sort_keys=True):
    return json.dumps(value, sort_keys=sort_keys, ensure_ascii=False, separators=(",", ":")).encode()


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be an object")
    mode = config.get("formulation")
    if mode not in ("pointwise", "setwise"):
        raise ValueError("formulation must be pointwise/setwise")
    endpoint = config.get("endpoint", "score")
    if endpoint not in ("score", "systemone"):
        raise ValueError("endpoint must be score/systemone")
    if endpoint == "systemone":
        if set(config) - {"endpoint", "formulation", "instructions"}:
            raise ValueError("System One config accepts only endpoint, formulation and instructions")
        instructions = config.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("System One requires nonempty instructions")
        import string
        allowed = {"candidate"} if mode == "pointwise" else set()
        for _, field, spec, conversion in string.Formatter().parse(instructions):
            if field is not None and (field not in allowed or spec or conversion):
                raise ValueError(f"Unsupported instruction placeholder: {field}")
        if mode == "pointwise" and "{candidate}" not in instructions:
            raise ValueError("Pointwise instructions must include {candidate}")
        return
    for key in ("query_template", "item_template"):
        if not isinstance(config.get(key), str):
            raise ValueError(f"{key} must be a string")
    # Explicit fields prevent accidentally exposing siblings in pointwise requests.
    import string
    allowed = {"query_template": {"state"}, "item_template": {"candidate"} if mode == "pointwise" else {"options", "labels"}}
    for key, fields in allowed.items():
        for _, field, spec, conversion in string.Formatter().parse(config[key]):
            if field is not None and (field not in fields or spec or conversion):
                raise ValueError(f"Unsupported placeholder in {key}: {field}")
    required = "{candidate}" if mode == "pointwise" else "{options}"
    if "{state}" not in config["query_template"] or required not in config["item_template"]:
        raise ValueError("Templates must include state and candidate/options placeholders")
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


def load_images(values, directory):
    """Load shared query images before timing; never silently drop an image."""
    if not isinstance(values, list):
        raise ValueError("images must be a list of local paths or base64 image data URIs")
    from PIL import Image
    images = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("Each image must be a nonempty path or data URI")
        if value.startswith("data:"):
            header, separator, payload = value.partition(",")
            if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
                raise ValueError("Images require base64 image data URIs")
            try:
                data = base64.b64decode(payload, validate=True)
            except ValueError as exc:
                raise ValueError("Invalid base64 image") from exc
        else:
            if "://" in value:
                raise ValueError("Use local images or data URIs, not remote URLs")
            data = (directory / value).read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
            mime = Image.MIME.get(image.format)
        if not mime:
            raise ValueError("Unknown image format")
        images.append(f"data:{mime};base64," + base64.b64encode(data).decode("ascii"))
    return images


def load_records(path):
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("Each record must be an object")
        candidates = record.get("candidates")
        if not isinstance(record.get("state"), str) or not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError("Each record needs state and at least two candidates")
        if not all(isinstance(x, str) and x for x in candidates):
            raise ValueError("Candidates must be nonempty strings")
        expected = record.get("expected_index")
        if expected is not None and (type(expected) is not int or not 0 <= expected < len(candidates)):
            raise ValueError("expected_index must identify a candidate")
        if "images" in record:
            record["images"] = load_images(record["images"], Path(path).resolve().parent)
        records.append(record)
    if not records:
        raise ValueError("Dataset is empty")
    return records


def build_request(record, config, model):
    validate_config(config)
    if config.get("endpoint") == "systemone":
        if config["formulation"] == "setwise":
            questions = {"choice": {"type": "choice", "instructions": config["instructions"],
                         "criteria": {str(i): c for i, c in enumerate(record["candidates"])}}}
        else:
            questions = {str(i): {"type": "noul", "instructions": config["instructions"].format(candidate=c)}
                         for i, c in enumerate(record["candidates"])}
        return {"model": model, "state": record["state"], "questions": questions,
                "images": record.get("images", [])}
    if record.get("images"):
        raise ValueError('Images require a config with "endpoint": "systemone"')
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
    body = {"model": model, "query": query, "items": items,
            "label_token_ids": [x["token_id"] for x in labels], "apply_softmax": True}
    return body


def parse_response(response, body, config, candidate_count):
    if not isinstance(response, dict):
        raise ValueError("Score response must be an object")
    if config.get("endpoint") == "systemone":
        answers = response.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(body["questions"]):
            raise ValueError("System One response must answer every question exactly once")
        if config["formulation"] == "setwise":
            answer = answers["choice"]
            if not isinstance(answer, dict):
                raise ValueError("System One answer must be an object")
            probabilities = answer.get("probabilities", {})
            if answer.get("type") != "choice" or not isinstance(probabilities, dict) or set(probabilities) != set(body["questions"]["choice"]["criteria"]):
                raise ValueError("Missing System One choice probabilities")
            scores = [[probabilities[str(i)] for i in range(candidate_count)]]
        else:
            scores = []
            for i in range(candidate_count):
                answer = answers[str(i)]
                if not isinstance(answer, dict):
                    raise ValueError("System One answer must be an object")
                probability = answer.get("noul")
                if answer.get("type") != "noul" or type(probability) not in (int, float):
                    raise ValueError("Missing System One noul probability")
                scores.append([probability, 1 - probability])
        # Reuse the same probability, cardinality and token-accounting checks.
        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("System One response must include usage")
        adapted = {"scores": scores, "usage": {"prompt_tokens": usage.get("input_tokens")}}
        adapted_body = {"items": [None] * len(scores), "label_token_ids": [None] * len(scores[0])}
        return parse_response(adapted, adapted_body, {"formulation": config["formulation"]}, candidate_count)
    scores = response.get("scores")
    if not isinstance(scores, list) or len(scores) != len(body["items"]):
        raise ValueError("Unexpected score row count")
    for row in scores:
        if not isinstance(row, list) or len(row) != len(body["label_token_ids"]):
            raise ValueError("Unexpected label score count")
        if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in row):
            raise ValueError("Scores must be finite probabilities")
        if abs(sum(row) - 1) > 1e-4:
            raise ValueError("Scores are not normalized over labels")
    if config["formulation"] == "pointwise":
        if len(scores) != candidate_count:
            raise ValueError("Missing pointwise candidate scores")
        values = [row[config.get("positive_label_index", 0)] for row in scores]
    elif config["formulation"] == "setwise":
        values = scores[0]
        if len(values) != candidate_count:
            raise ValueError("Missing setwise choice scores")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Response must include usage")
    tokens = usage.get("prompt_tokens")
    if type(tokens) is not int or tokens < 0:
        raise ValueError("Response must report nonnegative usage.prompt_tokens")
    return max(range(len(values)), key=values.__getitem__), tokens


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def summarize(bursts):
    requests = [r for b in bursts for r in b["requests"]]
    successful = [r for r in requests if r["success"]]
    elapsed = sum(b["wall_seconds"] for b in bursts)
    labelled = [r for r in successful if r["correct"] is not None]
    latencies = [r["latency_ms"] for r in successful]
    return {"attempted_candidate_sets": len(requests), "completed_candidate_sets": len(successful),
            "failed_candidate_sets": len(requests) - len(successful),
            "completed_candidates": sum(r["candidate_count"] for r in successful),
            "decisions_per_second": len(successful) / elapsed,
            "decision_unit": "one complete candidate set (one HTTP request)",
            "submitted_images_per_second": sum(r.get("image_count", 0) for r in successful) / elapsed,
            "input_tokens_per_second": sum(r["prompt_tokens"] for r in successful) / elapsed,
            "latency_ms_mean": statistics.mean(latencies) if latencies else None,
            "latency_ms_p95": percentile(latencies, .95) if latencies else None,
            "labelled_successful_sets": len(labelled),
            "accuracy_on_successful_labelled_sets": sum(r["correct"] for r in labelled) / len(labelled) if labelled else None,
            "timing": "sum of measured burst completion times; successful-set request latency excludes failures"}


async def benchmark(args):
    config = json.loads(Path(args.config).read_text())
    if not isinstance(config, dict):
        raise ValueError("Config must be an object")
    if "formulation" in config and config["formulation"] != args.formulation:
        raise ValueError("Config formulation conflicts with --formulation")
    config["formulation"] = args.formulation
    validate_config(config)
    records = load_records(args.dataset)
    if args.num_prompts is not None:
        records = records[:args.num_prompts]
    bodies = [build_request(r, config, args.model) for r in records]
    # System One assigns answer labels in criteria insertion order.
    encoded = [encode(body, sort_keys=config.get("endpoint") != "systemone") for body in bodies]
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=False)
    metadata = {"formulation": args.formulation, "endpoint": config.get("endpoint", "score"), "config": config, "model": args.model, "api_base": args.api_base,
                "dataset_sha256": hashlib.sha256(encode(records)).hexdigest(),
                "request_sha256": [hashlib.sha256(body).hexdigest() for body in encoded],
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "candidate_sets": len(records), "batch_sizes": args.batch_sizes,
                "num_runs": args.num_runs, "warmup_runs": args.warmup_runs, "seed": args.seed,
                "quality_note": "Different formulations need separate quality evaluation; scores are not assumed equivalent.",
                "http": "fresh connections, no retries, burst execution",
                "images_per_record": [len(r.get("images", [])) for r in records],
                "vision_note": "Images are shared query inputs. Loading/encoding is excluded; upload and server processing are timed. Replays reuse images: disable server image/prefix caches to measure uncached vision throughput."}
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

        if any(record.get("images") for record in records):
            # Old Pydantic servers may silently ignore unknown request fields.
            async with session.get(root + "/openapi.json") as response:
                response.raise_for_status()
                schema = await response.json()
            properties = schema.get("components", {}).get("schemas", {}).get("SystemOneRequest", {}).get("properties", {})
            if "images" not in properties:
                raise RuntimeError("Server does not advertise images for SystemOneRequest; use a server with /v1/systemone image support")
            if metadata.get("server", {}).get("enable_mis"):
                raise RuntimeError("Vision scoring currently requires SIS (disable MIS)")

        async def request(index):
            started = time.perf_counter()
            result = {"record_index": index, "candidate_count": len(records[index]["candidates"]),
                      "request_sha256": metadata["request_sha256"][index], "success": False,
                      "image_count": len(records[index].get("images", []))}
            try:
                async with session.post(root + "/v1/" + config.get("endpoint", "score"), data=encoded[index]) as response:
                    response.raise_for_status()
                    data = await response.json()
                selected, tokens = parse_response(data, bodies[index], config, result["candidate_count"])
                expected = records[index].get("expected_index")
                result.update(success=True, selected_index=selected, prompt_tokens=tokens,
                              correct=None if expected is None else selected == expected)
                field = "answers" if config.get("endpoint") == "systemone" else "scores"
                result[field] = data[field]
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
                result = summarize(bursts)
                result.update(batch_size=batch_size, formulation=config["formulation"])
                atomic_write_json(str(out / f"{batch_size}.json"), result)
                all_ok = all_ok and result["failed_candidate_sets"] == 0
                print(json.dumps(result), flush=True)
        return all_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--formulation", required=True, choices=("pointwise", "setwise"),
                        help="Required scoring formulation; no default")
    parser.add_argument("--config", required=True, help="Scoring prompt and label JSON config")
    parser.add_argument("--dataset", required=True, help="JSONL records: state, candidates, optional expected_index and images")
    parser.add_argument("--api-base", default="http://localhost:30000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--batch-sizes", default="1,2,4,8,16", help="Concurrent complete candidate sets per burst")
    parser.add_argument("--num-prompts", type=int, help="Maximum dataset records (no automatic repetition)")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--results-dir", required=True, help="New output directory; existing results are never overwritten")
    parser.add_argument("--dry-run", action="store_true", help="Write requests/metadata without contacting a server")
    args = parser.parse_args()
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
