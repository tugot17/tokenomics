# Scoring

`tokenomics score` calls SGLang's `/v1/score` endpoint.
`--formulation pointwise|setwise` is required, with no default.
SIS/MIS is configured on the server.

| Formulation | Request | Selection |
| --- | --- | --- |
| `pointwise` | One item per candidate | Highest positive-label probability |
| `setwise` | All candidates in one item | Highest choice-label probability |

Run both against the same dataset:

```bash
for formulation in pointwise setwise; do
  tokenomics score --model LiquidAI/d1-ns-trio-xheavy \
    --api-base http://localhost:30000/v1 \
    --formulation "$formulation" \
    --config examples/scoring/${formulation}.json \
    --dataset examples/scoring/candidate_sets.jsonl \
    --batch-sizes 1,2 --num-runs 5 --warmup-runs 2 \
    --results-dir results/${formulation}
done
```

Use a new results directory for each run. Add `--dry-run` to save requests
without contacting the server. Authentication uses `--api-key` or `OPENAI_API_KEY`.

## Inputs

Each dataset JSONL row contains:

```json
{"state": "Charged twice.", "candidates": ["Refund.", "Troubleshoot."], "expected_index": 0}
```

`expected_index` is optional and zero-based. Each row needs at least two candidates.
`--num-prompts` limits rows; `--num-runs` repeats them in shuffled order controlled
by `--seed`.

The config supplies prompt templates and label token IDs:

- `query_template`: uses `{state}`.
- Pointwise `item_template`: uses `{candidate}`. `positive_label_index` defaults to 0.
- Setwise `item_template`: uses `{options}` and optionally `{labels}`.
  Supply one distinct single-token label per candidate; smaller sets use the first labels.

Templates are sent literally. Use the model's trained prompt format and verify
label token IDs for its tokenizer. The supplied prompts and three records are
smoke-test examples; D1 quality on the setwise A/B/C task has not been validated.
Pointwise and setwise can produce different predictions.

## Results

`--batch-sizes` sets concurrent candidate sets per burst. The final burst may be
smaller. Use enough dataset rows to reach the requested concurrency.
**One decision is one complete candidate set** in both formulations.

Throughput uses measured burst completion time; latency includes HTTP and response
validation. Warmup, request construction and file writes are excluded. Connections
are fresh, with no retries. Failed requests count toward time but not completed work.

| File | Contents |
| --- | --- |
| `metadata.json` | `formulation`, config, hashes, seed, server settings when available |
| `requests.jsonl` | Source records and rendered requests |
| `bursts.jsonl` | Scores, selections, latency and errors per request |
| `<concurrency>.json` | `formulation`, throughput, latency, completed/failed sets, input-token throughput and accuracy on successful labelled sets |

Measured failures return a nonzero exit code. Warmup failures stop the run and
save `warmup_failure.json`. Report failure counts alongside accuracy.

Plot a completed run with `tokenomics plot-score results/pointwise pointwise.png`.

Run tests with `python -m unittest discover -s tests -v`.
