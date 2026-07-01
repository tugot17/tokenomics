# Tokenomics

Benchmarking suite for OpenAI-compatible inference servers. Measures throughput, latency, and steady-state performance.

![Example benchmark](assets/example_visualization.png)

## Install

```bash
pip install tokenomics
```

### From source

```bash
git clone https://github.com/tugot17/tokenomics.git
cd tokenomics
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install -e .
```

## Completion Benchmark

Sends chat completion requests to any OpenAI-compatible server and records per-request and system-wide metrics. Requests are non-streaming by default (max throughput); `--stream` adds TTFT and per-token metrics.

Every run **sweeps concurrency** and writes one result JSON per sweep point. Two execution modes, mutually exclusive:

- `--max-concurrency 1,2,4,…` — **sustained**: holds concurrency constant (realistic production numbers).
- `--batch-sizes 1,2,4,…` — **burst**: fires each batch at once (peak throughput).

```bash
tokenomics completion --model your-model \
  --scenario "D(1024,256)" \
  --max-concurrency 1,2,4,8,16,32,64,128 \
  --results-dir results/
```

### Building requests

Three ways to produce request content, all swept over concurrency the same way:

| Mode | Enable with | Text | Use for |
|------|-------------|------|---------|
| **Synthetic** | `--scenario` | random dataset snippets padded to a token budget | throughput curves at controlled input/output lengths |
| **Dataset replay** | `--replay-dataset` | each dataset row, verbatim | real examples; comparing datasets |
| **Images (VL)** | `--num-images` | fixed filler (or `--scenario`) + synthetic images | vision inference speed |

#### Synthetic (`--scenario`)

A scenario sets input/output token counts; the prompt is built by concatenating random dataset snippets (with replacement) until it reaches the input budget.

| Pattern | Example | Description |
|---------|---------|-------------|
| `D(in,out)` | `D(100,50)` | Fixed token counts |
| `N(mu,sigma)/(mu,sigma)` | `N(100,50)/(50,0)` | Normal distribution |
| `U(min,max)/(min,max)` | `U(50,150)/(20,80)` | Uniform distribution |

The snippet source defaults to a bundled AIME dataset; override with `--dataset-config` (see [Dataset config](#dataset-config)).

#### Dataset replay (`--replay-dataset`)

Sends each dataset row **verbatim as one request** and walks the dataset in order, instead of synthesizing prompts. The prompt set is pinned (same rows, same order across every concurrency level and run), so results line up example by example.

```bash
tokenomics completion --model your-model \
  --replay-dataset --dataset-config examples/dataset_configs/humaneval.json \
  --max-concurrency 1,2,4,8,16,32 --max-tokens 1024 \
  --results-dir results/humaneval/
```

- Sustained mode only; `--scenario` is ignored.
- `--num-prompts N` caps the walk to the first N rows (default: whole dataset).
- List prompt columns (MT-Bench `prompt`, Arena-Hard `turns`) are reduced to the first turn — MT-Bench runs single-turn.

Bundled configs under `examples/dataset_configs/`:

| Config | Dataset | Domain |
|--------|---------|--------|
| `gsm8k.json` | `openai/gsm8k` main (test) | grade-school math |
| `math500.json` | `HuggingFaceH4/MATH-500` (test) | competition math (MATH) |
| `aime25.json` | `math-ai/aime25` (test) | competition math (AIME 2025) |
| `mbpp.json` | `google-research-datasets/mbpp` sanitized (test) | code generation |
| `humaneval.json` | `openai/openai_humaneval` (test) | code completion |
| `lcb.json` | `livecodebench/code_generation_lite` release_v1 (test) | code generation (LiveCodeBench) |
| `mtbench.json` | `HuggingFaceH4/mt_bench_prompts` (train) | multi-turn chat (first turn) |
| `alpaca.json` | `tatsu-lab/alpaca_eval` (eval) | instruction following (AlpacaEval) |
| `arena_hard.json` | `lmarena-ai/arena-hard-auto-v0.1` (train) | hard instruction following (Arena-Hard) |

#### Images (`--num-images`)

Attach images to any run — sent as OpenAI content parts (`image_url` base64 `data:` URIs, accepted by SGLang and vLLM) — turning it into a VL benchmark. Metrics and plotting are unchanged.

```bash
tokenomics completion --model your-vl-model \
  --num-images 5 --image-size 512x512 \
  --max-concurrency 1,2,4,8,16 \
  --results-dir results/vl_512x5/
```

Image runs default to a short workload (the images dominate, the text is padding): a fixed filler of `--input-tokens` (default 32) and `--max-tokens` 32 output.

- `--image-size` is `N` (square) or `WxH` (e.g. `1024x768`, lowercase `x`); `--input-tokens 0` = images only.
- Synthetic images are random-noise PNGs, seeded per request → **unique** (defeats the server's prefix/multimodal caches) yet reproducible. Noise is nearly incompressible (~MBs at 1024×1024), so keep size/count modest or the payload dominates.
- `--input-tokens` and `--scenario` are mutually exclusive; pass `--scenario` to put images on dataset-driven text instead.
- Sweep size/count/length by looping the command (one `--results-dir` each) and overlaying with `plot-completion`.

### Dataset config

A JSON file with a `source` and (usually) a `prompt_column`. File paths are resolved relative to the config file.

```json
{ "source": { "type": "huggingface", "path": "openai/gsm8k",
              "huggingface_kwargs": { "name": "main", "split": "test" } },
  "prompt_column": "question" }
```

`source.type` is `huggingface`, `file` (`.txt`/`.csv`/`.json`), or `aime` (bundled shortcut). See `examples/dataset_configs/` for more.

### Output length & reproducibility

`--ignore-eos` makes every request generate exactly `--max-tokens` (EOS ignored), fixing output length so throughput isn't skewed by content-dependent token counts. Add it when comparing harnesses, servers, or configs; omit it for realistic, content-driven lengths. Supported by SGLang and vLLM — a no-op on servers that ignore the field.

`--max-tokens` defaults to 4096 (32 for image runs) and `--temperature` to 0.7. For fully reproducible runs, use `--temperature 0` with `--ignore-eos`.

### Key Options

| Flag | Description |
|------|-------------|
| `--model` | Model name (required) |
| `--scenario` | Traffic pattern (required unless `--replay-dataset` or `--num-images`) |
| `--api-base` | Server URL (default: `http://localhost:8000/v1`) |
| `--max-concurrency` | Sustained mode sweep points |
| `--batch-sizes` | Burst mode sweep points |
| `--num-prompts` | Prompts per sweep point in sustained mode |
| `--num-runs` | Runs per sweep point (default: 3) |
| `--max-tokens` | Max output tokens (default: 4096; 32 for image runs) |
| `--temperature` | Sampling temperature (default: 0.7) |
| `--ignore-eos` | Generate exactly `--max-tokens`, ignoring EOS (SGLang/vLLM) — fixes output length for clean comparisons |
| `--stream` | Enable SSE streaming for TTFT/per-token metrics |
| `-n` | Completions per request (default: 1) |
| `--dataset-config` | Path to dataset config (default: bundled AIME) |
| `--replay-dataset` | Send each dataset row verbatim (sustained only; ignores `--scenario`) |
| `--num-images` | Attach N synthetic random-noise images per request (0 = text-only) |
| `--image-size` | Synthetic image size: `N` or `WxH` (default: 512) |
| `--input-tokens` | Filler-text length for image runs (default: 32; 0 = images only) |
| `--results-dir` | Output directory (one JSON per sweep value) |
| `--lora-strategy` | LoRA distribution: single, uniform, zipf, mixed, all-unique |
| `--lora-names` | Comma-separated LoRA adapter names |

### Metrics

**Per-request:**
- **TTFT** — time to first token (streaming only)
- **Decode throughput** — output tokens/s per request (streaming only)
- **TPOT** — time per output token (streaming only)
- **Per-request latency** — end-to-end time per request

**System-wide:**
- **End-to-end output throughput** — `total_output_tokens / wall_time`
- **Steady-state output throughput** — median tok/s across time buckets where the batch is >= 80% full (streaming only)

### Plotting

```bash
# Compare multiple benchmarks (overlays each results dir as its own line)
tokenomics plot-completion output.png results_dir1/ results_dir2/
```

**Non-streaming** (default) produces a 2-panel plot:

![Non-streaming example](assets/example_visualization_non_streaming.png)

| Top | Output throughput |
|-----|-------------------|
| **Bottom** | **Per-request latency** |

**Streaming** (`--stream`) produces a 6-panel dashboard:

| | Left | Right |
|---|------|-------|
| **Row 1** | TTFT (with TTFO overlaid when it diverges, i.e. reasoning models) | Decode throughput per request |
| **Row 2** | End-to-end output throughput | Latency breakdown (prefill vs decode) |
| **Row 3** | Steady-state output throughput | Time-series token buckets |

## Embedding Benchmark

Tests concurrent embedding throughput.

```bash
tokenomics embedding \
  --model Qwen/Qwen3-Embedding-4B \
  --sequence_lengths "200" \
  --batch_sizes "1,8,16,32,64,128,256,512" \
  --num_runs 3 \
  --results-dir embedding_results/

tokenomics plot-embedding embedding_results/ embedding_plot.png
```

![Embedding performance](assets/embeddings_speed.png)
