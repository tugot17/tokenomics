# FastAPI Test Server for OpenAI Chat Completions API

A lightweight FastAPI server that mimics OpenAI's chat completions API for testing and benchmarking.

## Quick Start

```bash
pip install fastapi uvicorn
python test_server.py
```

## Usage

### Basic Usage
```bash
python test_server.py
```

### With Custom Configuration
```bash
python test_server.py --host 0.0.0.0 --port 8001 --prompt-tokens 100 --completion-tokens 150 --timeout 2.0
```

#### Single worker (recommended for async)
```bash
python test_server.py --port 8000 --workers 1 --timeout 5.0
```

#### Multiple workers (for CPU-bound tasks)
```bash
python test_server.py --port 8000 --workers 4 --timeout 5.0
```

### Options
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 8000)
- `--prompt-tokens`: Fixed prompt tokens to return (default: 50)
- `--completion-tokens`: Fixed completion tokens to return (default: 75)
- `--timeout`: Request timeout in seconds (default: 5.0)
- `--workers`: Number of worker processes (default: 1)
- `--reload`: Enable auto-reload for development

## API Endpoints

- `POST /v1/chat/completions` - OpenAI-compatible chat completions
- `GET /v1/models` - List available models
- `GET /health` - Health check with concurrency stats
- `GET /stats` - Detailed server statistics

## Testing

```bash
# Test with benchmark script
python ../oai_server_benchmark.py --model test-model --api_base http://127.0.0.1:8000/v1

# View documentation
curl http://127.0.0.1:8000/docs
``` 