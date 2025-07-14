# Tokenomics 🚀📈📊

The repo enabling measruing of tokens/s performance of OpenAI compatible API. Easily figure out the throughput you get on your setup. 

![Tokens go brrr](assets/tokens.jpg)

## How to run server?

To run the performance tests, you need two things:

- An OpenAI chat-compatible server

- Script calling the API

In principle, you can run any OAI-compatible server, such as [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html?ref=blog.mozilla.ai) or [SGLang](https://docs.sglang.ai/backend/server_arguments.html). 

Both server packages have a lot of parameters, so to facilitate running them, we created two scripts to quickly spin them up with hardcoded values of model paths, TP configuration, etc. You should adapt these to your needs, e.g., use a different path for the model weights, a different TP config, etc.

To run the VLLM server, run:

```bash
./server/vllm_run_server.sh
```

To run the SGLang server run:

```bash
./server/sglang_run_server.sh
```


### How to run performance measuring part?

Running the benchmark is fairly straightforward. You need to specify the model name (tag for vllm/sglang), the dataset on which you intend to test and some other options:

```bash
python oai_server_benchmark.py --model MODEL_NAME --dataset_key DATASET_NAME [OPTIONS]
```

### Key Options
- `--model`: Model tag that you run withing the api (e.g., 'distill-llama-8b')
- `--api_base`: vLLM server URL (default: http://localhost:8000/v1)
- `--batch_sizes`: Comma-separated batch sizes (default: 1,2,4,8)
- `--num_runs`: Number of runs per batch size (default: 3)
- `--dataset_key`: Dataset to use (default: aime)
- `--results_file`: Output JSON file path

E.g. config for VLLM server

```bash
python oai_server_benchmark.py --model distill-llama-8b --dataset_key aime --api_base http://localhost:8000/v1 --batch_sizes 1,2,4,8 --num_runs 3 --max_tokens 100 --temperature 0.5 --description "Deepseek R1 distill 8B TP8 A100s" --results_file my_server_benchmark.json
```

We measure the performance against different batch sizes. Note you need to guarantee that no other tenants are using the API at the same time as you do; otherwise, you will get flawed performance numbers. 

The result is a `.json` file looks something like this:

```json
{
  "metadata": {
    "timestamp": "2025-02-04T21:33:32.902353",
    "model": "distill-llama-8b",
    ...
  },
  "results": {
    ...
    "64": {
      "tokens": {
        "input_per_request": {
          "mean": 112.578125,
          "std": 55.21709922324872
        },
        "output_per_request": {
          "mean": 6211.13125,
          "std": 3030.1517578348125
        }
      },
      "timings": {
        "batch_total_seconds": {
          "mean": 106.69955926425754,
          "std": 5.002876034171101
        },
        "fastest_seconds": {
          "mean": 5.34426054880023,
          "std": 1.6415655967821245
        },
        "slowest_seconds": {
          "mean": 106.69714294858277,
          "std": 5.002909157480276
        },
        "spread_seconds": {
          "mean": 101.35288239978254,
          "std": 4.984463398447948
        }
      },
      "throughput": {
        "batch_tokens_per_second": {
          "mean": 3729.295223546461,
          "std": 122.2431365137911
        },
        "request_tokens_per_second": {
          "mean": 159.1587231541266,
          "std": 8.273409865481359
        }
      }
    }
  }
}
```

Post generation use the following line to plot the throghput The plot can take as many input files as wanted and optionally a title. At least one input file and the output file are required.
```bash
python plot_throughput.py --title "My Custom Title" <PATH-TO-JSON-OUTPUT-0> <PATH-TO-JSON-OUTPUT-1> <PATH-TO-JSON-OUTPUT-2> <PATH-TO-IMAGE-WITH-VISUALIZATION>
```

E.g.

```bash
python plot_throughput.py --title "My Custom Title" my_server_benchmark_model0.json my_server_benchmark_model1.json my_output.png
```

![alt text](assets/example_visualization.png)


## Install requirements


```bash
conda create --name tokenomics python=3.11

conda activate tokenomics

pip install -r requirements.txt 
```

To run a `vllm` server install via 

```bash
pip install vllm
```

To run a sglang server via

```bash
pip install --upgrade pip
pip install sgl-kernel --force-reinstall --no-deps
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
```


## Limitations

In the current version, the benchmark has its limitations. Among the most important ones, we would list:
- Batches are very homogenous—similar prompt length; we don't simulate well the situation where part of the batch is still in the pre-fill phase, while another part is already decoding.
- We don't mix the pre-fill and decoding stages that much. As above. This can potentially severly impact the engine performance and is not reflected in our numbers.
- Some elements of the batch might end before others, e.g., in the batch of 2, one might end after 200 tokens and another go through another 4000. The second element of the batch will be effectively running for the majority of its decoding phase as a batch of 1. This can introduce the false sense of performance.
- The batches are always fixed—we send a batch of 1, 2, ... k, but once they are added to the server, we don't send another request. This is substantially different than the practical server behavior when you are most likely to continuously get the new requests—impacting the available compute/memory bandwidth and resulting in potentially different performance numbers for you.
