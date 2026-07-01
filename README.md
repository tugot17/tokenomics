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

Sends chat completion requests to any OpenAI-compatible server and records per-request and system-wide metrics.

By default, requests are non-streaming for maximum throughput. Use `--stream` to enable SSE streaming for TTFT and per-token latency metrics.

### Usage

```bash
# Sustained mode — maintains constant concurrency (recommended)
tokenomics completion \
  --scenario "D(1024,256)" \
  --model your-model \
  --max-concurrency 1,2,4,8,16,32,64,128,256,512,1024

# Burst mode — fires all requests at once
tokenomics completion \
  --scenario "D(1024,256)" \
  --model your-model \
  --batch-sizes 1,2,4,8

# Multiple completions per request (e.g. for RL rollouts)
tokenomics completion \
  --scenario "D(1024,256)" \
  --model your-model \
  --max-concurrency 1,2,4,8,16 \
  -n 16

# Streaming mode — enables TTFT and per-token metrics
tokenomics completion \
  --scenario "D(1024,256)" \
  --model your-model \
  --max-concurrency 1,2,4,8 \
  --stream

# Dataset-replay mode — walk a real dataset example by example (no --scenario)
tokenomics completion \
  --replay-dataset \
  --dataset-config examples/dataset_configs/humaneval.json \
  --model your-model \
  --max-concurrency 1,2,4,8,16,32 \
  --max-tokens 1024
```

The two execution modes (`--batch-sizes` and `--max-concurrency`) are mutually exclusive. Burst is good for peak throughput; sustained gives realistic production numbers.

### Traffic Scenarios

| Pattern | Example | Description |
|---------|---------|-------------|
| `D(in,out)` | `D(100,50)` | Fixed token counts |
| `N(mu,sigma)/(mu,sigma)` | `N(100,50)/(50,0)` | Normal distribution |
| `U(min,max)/(min,max)` | `U(50,150)/(20,80)` | Uniform distribution |

### Datasets

The benchmark uses a bundled AIME dataset by default. You can specify a custom dataset with `--dataset-config`.

The benchmark concatenates random text snippets from the dataset until it reaches the input token count specified by the scenario. Snippets are picked with replacement, so even a small dataset can produce long prompts.

#### Dataset config format

A dataset config is a JSON file with a `source` section:

**Local file** (TXT, CSV, or JSON):
```json
{
  "source": { "type": "file", "path": "../data/prompts.txt" },
  "prompt_column": "text"
}
```
File paths are resolved relative to the config file.

**HuggingFace dataset:**
```json
{
  "source": {
    "type": "huggingface",
    "path": "squad",
    "huggingface_kwargs": { "split": "train" }
  },
  "prompt_column": "question"
}
```

**AIME** (built-in shortcut):
```json
{
  "source": { "type": "aime" }
}
```

See `examples/dataset_configs/` for more examples.

### Dataset Replay Mode

`--replay-dataset` sends each dataset row **verbatim as one request** and walks the dataset in order at each concurrency level, instead of synthesizing prompts to the scenario's token budget. Use it to benchmark on real examples and compare datasets. The prompt set is pinned (same rows, same order across every concurrency and run), so results line up example by example.

```bash
tokenomics completion \
  --replay-dataset \
  --dataset-config examples/dataset_configs/humaneval.json \
  --model your-model \
  --max-concurrency 1,2,4,8,16,32 --max-tokens 1024 \
  --results-dir results/humaneval/
```

- Sustained mode only (`--max-concurrency`); `--scenario` is ignored.
- `--num-prompts N` caps the walk to the first N rows (default: whole dataset).
- List prompt columns (MT-Bench `prompt`, Arena-Hard `turns`) are reduced to the first turn's string — MT-Bench runs single-turn.

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

**`--ignore-eos`** makes every request generate exactly `--max-tokens` (EOS ignored), fixing output length so throughput isn't skewed by content-dependent token counts. Add it when comparing harnesses, servers, or configs; omit it for realistic, content-driven lengths. Supported by SGLang and vLLM — servers that don't implement it ignore the field, so it silently has no effect there.

### Vision / multimodal

Attach images to every request with `--num-images` (and `--image-size`), turning any completion run into a VL benchmark. Images are sent as OpenAI content parts (`image_url` base64 `data:` URIs, accepted by SGLang and vLLM), and you sweep concurrency as usual — so latency/throughput vs concurrency via `plot-completion` is unchanged.

```bash
tokenomics completion \
  --model your-vl-model \
  --num-images 5 --image-size 512 \
  --max-concurrency 1,2,4,8,16 \
  --results-dir results/vl_512x5/
```

Image runs default to a short workload (the images dominate, the text is padding): a fixed filler prompt of `--input-tokens` (default 32) and `--max-tokens` 32 output. Override either explicitly, and add `--temperature 0` / `--ignore-eos` if you want fully reproducible generation.

- `--input-tokens` sets the filler length (`0` = images only). `--image-size` is `N` for a square or `WxH` for a rectangle — e.g. `--image-size 512` (512×512) or `--image-size 1024x768` (width × height, lowercase `x`, no spaces).
- Synthetic images are random-noise PNGs built on the fly, seeded per request so they're **unique** (defeating the server's prefix/multimodal caches) yet reproducible. Note noise is nearly incompressible (~MBs at 1024×1024), so keep image size/count sane or the payload dominates.
- Sweep image size/count/text-length by looping the command (one `--results-dir` each) and overlaying with `plot-completion` — nothing is baked in.
- Pass an explicit `--scenario` instead to drive the text from the dataset sampler (e.g. images on top of realistic prompts).

### Key Options

| Flag | Description |
|------|-------------|
| `--scenario` | Traffic pattern (required unless `--replay-dataset`) |
| `--replay-dataset` | Send each dataset row verbatim and walk the dataset at each concurrency level (ignores `--scenario`; sustained mode only) |
| `--model` | Model name (required) |
| `--api-base` | Server URL (default: `http://localhost:8000/v1`) |
| `--batch-sizes` | Burst mode sweep points |
| `--max-concurrency` | Sustained mode sweep points |
| `--num-prompts` | Prompts per sweep point in sustained mode |
| `--num-runs` | Runs per sweep point (default: 3) |
| `--max-tokens` | Max output tokens (default: 4096; 32 for image runs) |
| `--ignore-eos` | Generate exactly `--max-tokens` per request, ignoring EOS (SGLang/vLLM). Fixes output length for clean cross-harness throughput comparison |
| `--num-images` | Attach N synthetic (random-noise) images to each request (0 = text-only, default) |
| `--image-size` | Synthetic image size: `N` or `WxH` (default: 512; used when `--num-images` > 0) |
| `--input-tokens` | Filler-text length for image runs without `--scenario` (default: 32; 0 = images only) |
| `--temperature` | Sampling temperature (default: 0.7) |
| `-n` | Completions per request (default: 1) |
| `--stream` | Enable SSE streaming for TTFT/per-token metrics |
| `--dataset-config` | Path to dataset config (default: bundled AIME) |
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
# Compare multiple benchmarks
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
