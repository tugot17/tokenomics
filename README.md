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

By default the benchmark *synthesizes* prompts — it concatenates random dataset snippets until it hits the scenario's input-token budget. That's fine for raw throughput curves at controlled input/output lengths, but the prompts are stitched-together fragments, so the generated content isn't representative of any real workload. When you want throughput on the *actual* examples in a dataset — and to compare one dataset against another — use replay mode.

`--replay-dataset` sends real examples one at a time:

- Each dataset row is sent **verbatim as one request** — no concatenation, no length padding.
- The benchmark **walks the dataset in order**, generating to natural EOS capped by `--max-tokens`.
- The prompt set is **pinned**: identical rows in the same order across every concurrency level and run, so results are comparable example by example.

```bash
tokenomics completion \
  --replay-dataset \
  --dataset-config examples/dataset_configs/humaneval.json \
  --model your-model \
  --max-concurrency 1,2,4,8,16,32 \
  --max-tokens 1024 \
  --results-dir results/humaneval/
```

Bundled real-dataset configs for replay (under `examples/dataset_configs/`) — common code, math, and QA evaluation sets, each sending its prompt column verbatim:

| Config | Dataset | Domain |
|--------|---------|--------|
| `humaneval.json` | `openai/openai_humaneval` (test) | code completion |
| `mbpp.json` | `google-research-datasets/mbpp` sanitized (test) | code generation |
| `gsm8k.json` | `openai/gsm8k` main (test) | grade-school math |
| `math500.json` | `HuggingFaceH4/MATH-500` (test) | competition math |

Notes:

- `--scenario` is **not used** in this mode (input length comes from the data); only `--max-tokens` bounds the output.
- Requires `--max-concurrency` (sustained mode). Burst mode is rejected — it would walk a different slice of the dataset at each sweep point and break the comparison.
- `--num-prompts N` caps the walk to the first `N` rows; the default is the whole dataset.
- To compare datasets, run once per config and overlay the results with `tokenomics plot-completion ...`. To compare two servers or server configs, run the same matrix against each and overlay — tokenomics stays a pure OpenAI client and measures only what it observes over the wire.

#### Fixed-length comparison (`--ignore-eos`)

Natural-EOS replay (above) is the realistic measurement — report it as the model's throughput on a dataset. But it's the wrong tool when you need to **compare two harnesses, servers, or configs**: the *total* output token count then depends on the generated content, which drifts between runs. On models with **batch-non-invariant decode** (e.g. bf16 MoE on AMD), the same prompt set even produces different total tokens depending on how requests happen to batch — so two tools can report throughputs ~15–20% apart purely from generating different *amounts*, not from measuring differently.

`--ignore-eos` removes that variable: every request generates **exactly `--max-tokens`**, so the workload is identical run-to-run and throughput reflects pure serving speed. Validated against [SpecForge](https://github.com/sgl-project/SpecForge)'s `bench_eagle3.py` on the same SGLang server (LFM2.5-8B-A1B, MI325X): with `--ignore-eos` the two agree within **1–3%** across concurrency 8/16/32; without it they diverged ~17%, entirely from batch-non-invariant token counts.

```bash
tokenomics completion \
  --replay-dataset \
  --dataset-config examples/dataset_configs/humaneval.json \
  --model your-model \
  --max-concurrency 8,16,32 \
  --max-tokens 1024 --ignore-eos --temperature 0 \
  --results-dir results/humaneval/
```

`--ignore-eos` is SGLang-specific (sent as a top-level request field). Use it for apples-to-apples throughput; omit it when you want realistic, content-driven generation lengths.

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
| `--max-tokens` | Max output tokens (default: 4096) |
| `--ignore-eos` | Generate exactly `--max-tokens` per request, ignoring EOS (SGLang). Fixes output length for clean cross-harness throughput comparison |
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
