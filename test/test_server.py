#!/usr/bin/env python3
"""
Simplified FastAPI test server that mimics OpenAI chat completions API
for testing the benchmark script with high concurrency support.
"""

import time
import asyncio
import argparse
import threading
from datetime import datetime
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
import uvicorn

# Configuration variables
FIXED_PROMPT_TOKENS = 50
FIXED_COMPLETION_TOKENS = 75
REQUEST_TIMEOUT = 5.0  # seconds

# Statistics tracking (thread-safe)
request_stats = {
    'total_requests': 0,
    'concurrent_requests': 0,
    'max_concurrent': 0,
    'lock': threading.Lock()
}

# Create FastAPI app
app = FastAPI(title="OpenAI Chat Completions Test Server")

def update_stats(increment=True):
    """Update request statistics thread-safely"""
    with request_stats['lock']:
        if increment:
            request_stats['total_requests'] += 1
            request_stats['concurrent_requests'] += 1
            request_stats['max_concurrent'] = max(
                request_stats['max_concurrent'], 
                request_stats['concurrent_requests']
            )
        else:
            request_stats['concurrent_requests'] -= 1

@app.post("/v1/chat/completions")
async def chat_completions(request: Dict[str, Any]):
    """
    Mimics OpenAI's chat completions endpoint with async processing.
    """
    try:
        update_stats(increment=True)
        
        # Extract parameters
        messages = request.get('messages', [])
        model = request.get('model', 'test-model')
        
        # Non-blocking delay
        await asyncio.sleep(REQUEST_TIMEOUT)
        
        # Generate response
        if messages:
            last_message = messages[-1].get('content', 'Hello!')
            response_content = f"Test response to: {last_message[:50]}..."
        else:
            response_content = "Test response with no input message."
        
        # Generate unique request ID
        request_id = f"{int(time.time())}-{id(request)}"
        
        # Create OpenAI-compatible response
        response = {
            "id": f"chatcmpl-test-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": FIXED_PROMPT_TOKENS,
                "completion_tokens": FIXED_COMPLETION_TOKENS,
                "total_tokens": FIXED_PROMPT_TOKENS + FIXED_COMPLETION_TOKENS
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        update_stats(increment=False)

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "test-model",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "test-server"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with concurrency statistics."""
    with request_stats['lock']:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "prompt_tokens": FIXED_PROMPT_TOKENS,
                "completion_tokens": FIXED_COMPLETION_TOKENS,
                "timeout_seconds": REQUEST_TIMEOUT
            },
            "statistics": {
                "total_requests": request_stats['total_requests'],
                "concurrent_requests": request_stats['concurrent_requests'],
                "max_concurrent_reached": request_stats['max_concurrent']
            }
        }

@app.get("/stats")
async def get_stats():
    """Get detailed statistics about server performance."""
    with request_stats['lock']:
        return {
            "total_requests_processed": request_stats['total_requests'],
            "currently_processing": request_stats['concurrent_requests'],
            "max_concurrent_reached": request_stats['max_concurrent'],
            "active_threads": threading.active_count()
        }

def main():
    parser = argparse.ArgumentParser(description='Simplified FastAPI test server for OpenAI chat completions API')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--prompt-tokens', type=int, default=50, help='Fixed prompt tokens to return')
    parser.add_argument('--completion-tokens', type=int, default=75, help='Fixed completion tokens to return')
    parser.add_argument('--timeout', type=float, default=5.0, help='Request timeout in seconds')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    # Update global configuration
    global FIXED_PROMPT_TOKENS, FIXED_COMPLETION_TOKENS, REQUEST_TIMEOUT
    FIXED_PROMPT_TOKENS = args.prompt_tokens
    FIXED_COMPLETION_TOKENS = args.completion_tokens
    REQUEST_TIMEOUT = args.timeout
    
    print(f"Starting FastAPI test server...")
    print(f"Host: {args.host}:{args.port}")
    print(f"Fixed prompt tokens: {FIXED_PROMPT_TOKENS}")
    print(f"Fixed completion tokens: {FIXED_COMPLETION_TOKENS}")
    print(f"Request timeout: {REQUEST_TIMEOUT}s")
    print(f"API Base URL: http://{args.host}:{args.port}/v1")
    print(f"Documentation: http://{args.host}:{args.port}/docs")
    
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload
        )
    except KeyboardInterrupt:
        print("\nShutting down server...")

if __name__ == '__main__':
    main()