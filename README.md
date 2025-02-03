# Tokenomics

The repo enabling measruing of tokens/s performance of OpenAI compatible API. Easily figure out the throughput you get on your setup. 

![Tokens go brrr](tokens.jpg)

## How to run server?

To run the performance tests, you need two things:

- An OpenAI chat-compatible server

- Script calling the API

In principle, you can run any OAI-compatible server, such as [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html?ref=blog.mozilla.ai) or [SGLang](https://docs.sglang.ai/backend/server_arguments.html). 

Both server packages have a lot of parameters, so to facilitate running them, we created two scripts to quickly spin them up with hardcoded values of model paths, TP configuration, etc. You should adapt these to your needs, e.g., use a different path for the model weights, a different TP config, etc.

To run the VLLM server, run:

```bash
./vllm_run_server.sh
```

To run the SGLang server run:

```bash
./sglang_run_server.sh
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
