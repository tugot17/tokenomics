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

## Vision scoring (SIS)

Add an optional `images` list to each dataset row. Images belong to the shared
query: pointwise sends them for every candidate sequence; setwise sends them with
the single joint-choice sequence. Both formulations retain the same decision unit.

```json
{"state":"<image> What is shown?","candidates":["A cat","A dog"],"images":["images/example.png"],"expected_index":0}
```

Paths are resolved relative to the dataset JSONL file. Base64 `data:image/...`
URIs and multiple images are also supported. Images are decoded/validated and
embedded before timing; missing or malformed images fail instead of becoming
text-only examples. Remote URLs are deliberately unsupported, making replay
independent of remote downloads. Request hashes include the actual image bytes.
Image markers and chat delimiters remain model-specific: provide the exact prompt
format expected by the model's processor. The `<image>` above illustrates LFM2-VL;
it is not a universal template.

Run the same scoring command as above with the vision dataset and `--num-runs 1`.
Use SIS on a multimodal generation server. The client checks that `/openapi.json`
advertises `ScoringRequest.image_data` before sending image requests, because older
servers may silently ignore unknown fields. It refuses servers without that
contract. Text-only benchmarking does not require this schema check.

The SGLang revision used for this PR does not expose images in `/v1/score` yet.
A companion patch is provided at
[`server-patches/sglang-score-images.patch`](server-patches/sglang-score-images.patch),
against SGLang `488869c2d0f3321299488bcf221934df40b06454`:

```bash
# In a checkout of the pinned SGLang revision:
git apply /path/to/tokenomics/examples/scoring/server-patches/sglang-score-images.patch
```

It routes shared images through the existing multimodal generation processor,
using one image list per SIS sequence. It supports last-token label scoring for
both benchmark formulations. MIS, extraction-token readouts, embedding overrides,
and non-generation models are rejected for image requests. No CUDA graph support
for vision is implied by this patch.

Latency includes image upload, server image processing, vision encoding and
language-model scoring; local image loading/encoding is excluded. Existing input
throughput uses the server's `usage.prompt_tokens`, whose image-token accounting
is model/server dependent. `submitted_images_per_second` counts successful
request-level images, **not** crops, patches, or vision-encoder executions.
Pointwise may process each shared image once per candidate; caches may reduce that
work. Metadata records `images_per_record`, and raw requests preserve the images.

Warmup and repeated passes reuse dataset images. Disable prefix and multimodal
embedding caches on the server when measuring uncached vision throughput, and
record that configuration. A one-pass run alone does not prevent warmup cache hits.

Client validation: `python -m unittest discover -s tests -v`.
The companion server routing tests can run without a GPU:

```bash
PYTHONPATH=/path/to/sglang/python python examples/scoring/server-patches/test_score_images.py
```

The companion patch was smoke-tested on one B300 with `LiquidAI/d1-3B-RC` in
BF16 SIS eager mode: a 128×128 image produced 64 image tokens; `/v1/score`
probabilities matched `/generate` label-logprob normalization exactly. Shared-image
batching, malformed-image rejection, and a Tokenomics setwise image request also
passed. See `server-patches/vision-validation.json`. This is functional validation,
not a vision throughput comparison or model-quality evaluation.
