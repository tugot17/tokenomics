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

### Key Options

| Flag | Description |
|------|-------------|
| `--scenario` | Traffic pattern (required) |
| `--model` | Model name (required) |
| `--api-base` | Server URL (default: `http://localhost:8000/v1`) |
| `--batch-sizes` | Burst mode sweep points |
| `--max-concurrency` | Sustained mode sweep points |
| `--num-prompts` | Prompts per sweep point in sustained mode |
| `--num-runs` | Runs per sweep point (default: 3) |
| `--max-tokens` | Max output tokens (default: 4096) |
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
| **Row 1** | TTFT | Decode throughput per request |
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
