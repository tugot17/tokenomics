"""Scoring payloads: validate inputs, render prompts, and interpret answers."""

import json
import math
import os
import string
from pathlib import Path


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
    # Explicit templates override the tokenizer's chat formatting.
    chat = not any(key in config for key in ("query_template", "item_template"))
    if chat:
        if questions:
            config.setdefault("question_template", "{state}\n\nQUESTION:\n{question}")
            fields = {"state", "question"}
            key = "question_template"
        else:
            key = "prompt_template"
            fields = {"state", "candidate"} if mode == "pointwise" else {"state", "options", "labels"}
        allowed = {key: fields}
        required = {key: fields - {"labels"}}
    else:
        if "question_template" in config or "prompt_template" in config:
            raise ValueError("Use a full prompt template or query_template/item_template, not both")
        style = "questions" if questions else mode
        item_fields = {"questions": {"question"}, "pointwise": {"candidate"},
                       "setwise": {"options", "labels"}}[style]
        allowed = {"query_template": {"state"}, "item_template": item_fields}
        required = {"query_template": {"state"}, "item_template": item_fields - {"labels"}}
    for key, fields in allowed.items():
        if not isinstance(config.get(key), str):
            raise ValueError(f"{key} must be a string")
        found = set()
        for _, field, spec, conversion in string.Formatter().parse(config[key]):
            if field is not None:
                if field not in fields or spec or conversion:
                    raise ValueError(f"Unsupported placeholder in {key}: {field}")
                found.add(field)
        if not required[key] <= found:
            raise ValueError(f"{key} must include {sorted(required[key])}")
    if questions:
        return
    labels = config.get("labels")
    if not isinstance(labels, list) or len(labels) < 2:
        raise ValueError("At least two labels are required")
    ids, texts = [], []
    for label in labels:
        if not isinstance(label, dict) or not isinstance(label.get("text"), str) or not label["text"]:
            raise ValueError("Each label needs nonempty text")
        token_id = label.get("token_id")
        if "token_id" not in label:
            texts.append(label["text"])
            continue
        if type(token_id) is not int or token_id < 0:
            raise ValueError("Label token IDs must be nonnegative integers")
        ids.append(token_id)
        texts.append(label["text"])
    if len(set(ids)) != len(ids) or len(set(texts)) != len(texts):
        raise ValueError("Label texts and token IDs must be unique")
    positive = config.get("positive_label_index", 0)
    if type(positive) is not int or not 0 <= positive < len(labels):
        raise ValueError("positive_label_index is outside the labels")


def resolve_label_ids(labels, tokenizer, multiple=False):
    field = "token_ids" if multiple else "token_id"
    for label in labels:
        if field in label:
            ids = label[field] if multiple else [label[field]]
            if tokenizer is not None and any(tokenizer.convert_ids_to_tokens(t) is None for t in ids):
                raise ValueError(f"Label {label['text']!r} contains IDs outside the tokenizer vocabulary")
            continue
        spellings = label.get("aliases", [label["text"]])
        ids = []
        for spelling in spellings:
            encoded = tokenizer.encode(spelling, add_special_tokens=False)
            if len(encoded) != 1 or encoded[0] in tokenizer.all_special_ids:
                raise ValueError(f"Label alias {spelling!r} must encode as one non-special token")
            ids.extend(encoded)
        ids = list(dict.fromkeys(ids))
        label[field] = ids if multiple else ids[0]


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
            if "token_ids" in label:
                aliases = label["token_ids"]
                if not isinstance(aliases, list) or not aliases or any(type(t) is not int or t < 0 for t in aliases):
                    raise ValueError("Each label needs nonempty nonnegative token_ids")
                ids.extend(aliases)
            if "aliases" in label:
                aliases = label["aliases"]
                if not isinstance(aliases, list) or not aliases or any(not isinstance(t, str) or not t for t in aliases):
                    raise ValueError("aliases must be a nonempty list of strings")
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
        record.setdefault("state", "")
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


def chat_inputs(contents, tokenizer):
    if tokenizer is None or not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat template; supply explicit query_template and item_template")
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    ) for content in contents]
    if any(not prompt for prompt in prompts):
        raise ValueError("Chat template produced an empty prompt")
    # Preserve every full prompt and leave each item nonempty.
    prefix_len = min(len(os.path.commonprefix(prompts)), min(map(len, prompts)) - 1)
    return prompts[0][:prefix_len], [prompt[prefix_len:] for prompt in prompts]


def build_request(record, config, model, tokenizer=None):
    if config.get("workload") == "questions":
        if "question_template" in config:
            contents = [config["question_template"].format(state=record["state"], question=q["prompt"])
                        for q in record["questions"]]
            query, items = chat_inputs(contents, tokenizer)
        else:
            query = config["query_template"].format(state=record["state"])
            items = [config["item_template"].format(question=q["prompt"]) for q in record["questions"]]
        ids = list(dict.fromkeys(t for q in record["questions"] for label in q["labels"] for t in label["token_ids"]))
        return {"model": model, "query": query, "items": items, "label_token_ids": ids, "apply_softmax": False}
    labels = config["labels"]
    values = {"state": record["state"]}
    if config["formulation"] == "pointwise":
        fields = [dict(values, candidate=c) for c in record["candidates"]]
    else:
        if len(record["candidates"]) > len(labels):
            raise ValueError("Setwise needs one configured label token per candidate")
        labels = labels[:len(record["candidates"])]
        options = "\n".join(f"{label['text']}) {candidate}" for label, candidate in zip(labels, record["candidates"]))
        fields = [dict(values, options=options, labels=", ".join(label["text"] for label in labels))]
    if "prompt_template" in config:
        query, items = chat_inputs([config["prompt_template"].format(**field) for field in fields], tokenizer)
    else:
        query = config["query_template"].format(**values)
        items = [config["item_template"].format(**field) for field in fields]
    return {"model": model, "query": query, "items": items,
            "label_token_ids": [label["token_id"] for label in labels], "apply_softmax": True}


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


