# Scoring benchmarks

`tokenomics score` benchmarks SGLang's `/v1/score` endpoint with explicit label token IDs and no generation. It sweeps shared-query length, items per request and concurrent HTTP requests. Each item gets one next-token decision: this is pointwise scoring, including when MIS shares the query across items.

## Workload

Supply a JSON file with five token-ID lists (all IDs must belong to the served model's tokenizer):

| Field | Meaning |
|---|---|
| `query_prefix` | Fixed tokens before synthetic state content; may be empty |
| `query_suffix` | Fixed tokens after the state content; may be empty |
| `filler_token_ids` | Nonempty pool sampled uniformly to fill the requested query length |
| `items` | List of nonempty question-suffix token lists, ending at the answer boundary |
| `label_token_ids` | Unique candidate token IDs, in response order |

The query is `query_prefix + sampled filler + query_suffix`. `--query-lengths` counts this entire query, excluding item suffixes. `--items N` takes the first N suffixes. Use the model's actual prompt format, including chat-template tokens, when preparing these parts. Tokenization and prompt construction happen outside the timed benchmark. Labels must each be a single token; the benchmark validates normalized score rows but does not assess answer quality.

For example, a minimal synthetic workload can be prepared with the installed `tokenizers` package:

```python
import json
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("your-model")  # or Tokenizer.from_file("tokenizer.json")
def encode(text):
    return tokenizer.encode(text, add_special_tokens=False).ids

labels = [encode(text) for text in [" yes", " no"]]
assert all(len(ids) == 1 for ids in labels)
workload = {
    "query_prefix": encode("State: "),
    "query_suffix": encode("\n"),
    "filler_token_ids": encode(" science nature technology history music"),
    "items": [encode(f"Question: Does the state mention {word}?\nAnswer:")
              for word in ["science", "music"]],
    "label_token_ids": [ids[0] for ids in labels],
}
with open("workload.json", "w") as file:
    json.dump(workload, file)
```

This example is a performance workload, not a model-specific chat template. Sampling individual token IDs can produce unnatural text and MoE routing that differs from real traffic. For meaningful model comparisons, prepare representative prompt parts and keep the workload file and seed fixed. The first C generated states match across concurrency and item-count settings within each round.

## Server and run

Start SGLang separately with `--disable-radix-cache`. For MIS, also enable `--enable-mis` and use the attention backends required by that model and SGLang version. The harness checks `/get_server_info` for the selected mode and disabled radix cache; it does not change the server configuration. For an SIS/MIS comparison, use matching backend settings and the same workload. MIS support and numerical agreement are model-dependent.

```bash
tokenomics score --model your-model --mode mis \
  --api-base http://localhost:30000/v1 --workload workload.json \
  --query-lengths 260,1040,5200 --items 1,2 \
  --batch-sizes 1,2,4,8,16,32,64,128 \
  --num-runs 5 --warmup-runs 2 --results-dir score_results/

tokenomics plot-score score_results/ score.png
```

Use `--mode sis` against a server without MIS enabled. The mode flag verifies server configuration; it does not toggle it. `OPENAI_API_KEY` supplies optional bearer authentication. `--timeout` is per request (default 300 seconds); `--seed` defaults to 42. The output directory must not exist.

## Measurements

A batch size of C means C concurrent requests, each carrying one shared query and N items: C × N decisions per burst. All shapes receive warmup before measurement; their order is shuffled within each round. Requests are serialized before a common release barrier. Timing includes fresh HTTP connections, scheduling, inference and response decoding. There are no retries. A failed request or invalid response prints a warning with the failed request count and configuration, then stops the run after recording that burst; failed runs have no summary.

- **Decisions/s:** C × N / burst seconds.
- **Processed input tokens/s:** summed server-reported prompt usage / burst seconds. MIS counts the query once per request, all item suffixes, and N+1 separator tokens. SIS counts the query once per item. Exact counts are checked against the submitted token IDs.
- **Burst latency:** common release to the last decoded response, in milliseconds. Per-request latency and dispatch delay are recorded separately.

Rates are calculated per burst, then summarized as mean and sample standard deviation across measured rounds. These synchronized bursts measure end-to-end serving behavior, not sustained capacity or GPU-only execution time. Record host conditions separately when comparing runs.

The output contains `metadata.json` (arguments, version, hashes, server settings and completion status), `workload.json`, `bursts.jsonl` (including warmups, request hashes, timings and failures), and `summary.json` / `summary.csv`. Plotting reads the saved summary without checking completion status. Raw scores and the client API key are not stored.

Run the local HTTP and accounting tests with `python -m unittest discover -s tests -v`; they require no model or GPU.
