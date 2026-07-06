"""core/server.py — OpenAI-compatible REST API for Gemma 4 inference.

Serves on port 8080, compatible with llama-server replacement.
Endpoints:
  POST /v1/chat/completions    — chat completion with streaming support
  POST /v1/completions         — text completion
  GET  /v1/models              — list available models
  GET  /health                 — health check

No external HTTP framework — uses Python's built-in http.server for
zero-dependency deployment.
"""
from __future__ import annotations

import json
import threading
import time
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, List, Dict, Any, Callable

import numpy as np

import tensor_cuda as tc


# ==================================================================
# Chat format utilities
# ==================================================================
CHAT_TEMPLATE = """<start_of_turn>user
{user_message}<end_of_turn>
<start_of_turn>model
"""


def format_chat_messages(messages: List[Dict[str, str]]) -> str:
    """Convert OpenAI chat messages to Gemma 4 prompt format."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            # Gemma 4 doesn't have a system role; prepend to first user
            parts.append(f"System: {content}\n\n")
        elif role == "user":
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
        elif role == "assistant":
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>\n")
    parts.append("<start_of_turn>model\n")
    return "".join(parts)


def tokenize(text: str, tokenizer) -> np.ndarray:
    """Tokenize text to int64 array.  Uses the provided tokenizer."""
    # Tokenizer is expected to be a callable: tokenizer.encode(text) -> list[int]
    if tokenizer is None:
        raise RuntimeError("No tokenizer provided to server")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return np.array([tokens], dtype=np.int64)


def detokenize(tokens: np.ndarray, tokenizer) -> str:
    """Detokenize int64 array to text."""
    if tokenizer is None:
        raise RuntimeError("No tokenizer provided to server")
    return tokenizer.decode(tokens.tolist(), skip_special_tokens=True)


# ==================================================================
# SSE streaming
# ==================================================================
class SSEStream:
    """Server-sent events stream for chunked generation."""

    def __init__(self):
        self.queue = queue.Queue()
        self.done = False

    def put(self, data: dict):
        self.queue.put(json.dumps(data))

    def put_text(self, text: str, finish: bool = False):
        chunk = {
            "choices": [{"delta": {"content": text}, "index": 0}],
            "object": "chat.completion.chunk",
        }
        if finish:
            chunk["choices"][0]["finish_reason"] = "stop"
        self.put(chunk)
        if finish:
            self.done = True
            self.queue.put(None)

    def iter(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            yield f"data: {item}\n\n"
        yield "data: [DONE]\n\n"


# ==================================================================
# Request handler
# ==================================================================
class ChatHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OpenAI-compatible API."""

    # Set by the server factory
    model: Any = None
    tokenizer: Any = None
    model_name: str = "gemma-4-26b-moe"

    def log_message(self, fmt, *args):
        # Suppress default logging; implement custom if needed
        pass

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_sse(self, stream: SSEStream):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        for chunk in stream.iter():
            self.wfile.write(chunk.encode())
            self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            self._handle_models()
        elif self.path == "/health":
            self._handle_health()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            self._send_json({"error": "Empty body"}, 400)
            return

        body = self.rfile.read(content_len).decode()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        if self.path == "/v1/chat/completions":
            self._handle_chat_completions(data)
        elif self.path == "/v1/completions":
            self._handle_completions(data)
        else:
            self._send_json({"error": "Not found"}, 404)

    def _handle_models(self):
        self._send_json({
            "object": "list",
            "data": [{
                "id": self.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mythos",
            }]
        })

    def _handle_health(self):
        health = {
            "status": "ok",
            "model": self.model_name,
            "tensor_cuda": True,
        }
        if self.model is not None and hasattr(self.model, 'kv_manager'):
            kv_mgr = self.model.kv_manager
            if kv_mgr is not None:
                health["kv_cache_mb"] = round(kv_mgr.total_kv_bytes() / (1024 * 1024), 1)
        self._send_json(health)

    def _handle_chat_completions(self, data: dict):
        """Handle /v1/chat/completions request."""
        messages = data.get("messages", [])
        stream = data.get("stream", False)
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        top_k = data.get("top_k", 50)
        stop = data.get("stop", None)

        if self.model is None:
            self._send_json({"error": "Model not loaded"}, 503)
            return

        # Format prompt
        prompt_text = format_chat_messages(messages)
        try:
            prompt_ids = tokenize(prompt_text, self.tokenizer)
        except Exception as e:
            self._send_json({"error": f"Tokenization failed: {e}"}, 500)
            return

        if stream:
            # Streaming response
            sse = SSEStream()
            thread = threading.Thread(
                target=self._generate_stream,
                args=(prompt_ids, max_tokens, temperature, top_p, top_k, sse, stop)
            )
            thread.start()
            self._send_sse(sse)
        else:
            # Non-streaming response
            try:
                result = self._generate_sync(
                    prompt_ids, max_tokens, temperature, top_p, top_k, stop
                )
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": f"Generation failed: {e}"}, 500)

    def _handle_completions(self, data: dict):
        """Handle /v1/completions request."""
        prompt = data.get("prompt", "")
        stream = data.get("stream", False)
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        top_k = data.get("top_k", 50)

        if self.model is None:
            self._send_json({"error": "Model not loaded"}, 503)
            return

        try:
            prompt_ids = tokenize(prompt, self.tokenizer)
        except Exception as e:
            self._send_json({"error": f"Tokenization failed: {e}"}, 500)
            return

        if stream:
            sse = SSEStream()
            thread = threading.Thread(
                target=self._generate_stream,
                args=(prompt_ids, max_tokens, temperature, top_p, top_k, sse, None)
            )
            thread.start()
            self._send_sse(sse)
        else:
            try:
                result = self._generate_sync(
                    prompt_ids, max_tokens, temperature, top_p, top_k, None
                )
                # Adapt to completions format
                result["object"] = "text_completion"
                result["choices"][0]["text"] = result["choices"][0]["message"]["content"]
                del result["choices"][0]["message"]
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": f"Generation failed: {e}"}, 500)

    def _generate_sync(self, prompt_ids: np.ndarray, max_tokens: int,
                       temperature: float, top_p: float, top_k: int,
                       stop_seqs: Optional[List[str]]) -> dict:
        """Synchronous generation, returns full response."""
        model = self.model
        cfg = model.config

        with tc.no_grad():
            gen_ids, _ = model.generate(
                prompt_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_at_eos=True,
            )

        text = detokenize(gen_ids[0], self.tokenizer)

        # Apply stop sequences
        if stop_seqs:
            for stop in stop_seqs:
                idx = text.find(stop)
                if idx >= 0:
                    text = text[:idx]
                    break

        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_ids.shape[1],
                "completion_tokens": gen_ids.shape[1],
                "total_tokens": prompt_ids.shape[1] + gen_ids.shape[1],
            }
        }

    def _generate_stream(self, prompt_ids: np.ndarray, max_tokens: int,
                         temperature: float, top_p: float, top_k: int,
                         sse: SSEStream, stop_seqs: Optional[List[str]]):
        """Streaming generation, pushes chunks to SSE stream."""
        model = self.model
        cfg = model.config
        B = prompt_ids.shape[0]
        generated_tokens = []
        buffer_text = ""

        try:
            with tc.no_grad():
                # Prefill
                logits, caches = model.forward(prompt_ids, caches=None,
                                               last_token_only=True)

                for step in range(max_tokens):
                    # Sample
                    next_token = model._sample_token(logits, temperature, top_p, top_k)
                    tok_id = int(next_token[0, 0])
                    generated_tokens.append(tok_id)

                    # Check EOS
                    if tok_id in cfg.eos_token_ids:
                        break

                    # Decode partial
                    chunk_text = detokenize(np.array([tok_id]), self.tokenizer)
                    buffer_text += chunk_text

                    # Stream word-by-word (simple heuristic)
                    if chunk_text.endswith(" ") or chunk_text.endswith("\n") or step == max_tokens - 1:
                        sse.put_text(buffer_text)
                        buffer_text = ""

                    # Check stop sequences
                    if stop_seqs:
                        full_text = detokenize(np.array(generated_tokens), self.tokenizer)
                        for stop in stop_seqs:
                            if stop in full_text:
                                # Truncate and finish
                                idx = full_text.find(stop)
                                remaining = full_text[idx:]
                                if remaining and remaining != buffer_text:
                                    sse.put_text(remaining[:len(remaining)-len(stop)])
                                sse.put_text("", finish=True)
                                return

                    # Next decode step
                    next_ids = next_token
                    pos = prompt_ids.shape[1] + step + 1
                    logits, caches = model.forward(next_ids, caches=caches,
                                                   position_offset=pos,
                                                   last_token_only=True)

            # Flush remaining buffer
            if buffer_text:
                sse.put_text(buffer_text)
            sse.put_text("", finish=True)

        except Exception as e:
            sse.put_text(f"\n[Error: {e}]")
            sse.put_text("", finish=True)


# ==================================================================
# Server
# ==================================================================
class InferenceServer:
    """OpenAI-compatible inference server for Gemma 4.

    Usage:
        from core.gemma4_runner import Gemma4Runner
        from transformers import AutoTokenizer

        model, info = Gemma4Runner.from_pretrained("/path/to/gguf", qat=True)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-26b-it")

        server = InferenceServer(model, tokenizer, port=8080)
        server.start()   # blocks
    """

    def __init__(self, model, tokenizer, port: int = 8080,
                 model_name: str = "gemma-4-26b-moe"):
        self.model = model
        self.tokenizer = tokenizer
        self.port = port
        self.model_name = model_name
        self._httpd = None
        self._thread = None

    def _make_handler(self):
        """Create a request handler class bound to this server instance."""
        model = self.model
        tokenizer = self.tokenizer
        name = self.model_name

        class BoundHandler(ChatHandler):
            pass

        BoundHandler.model = model
        BoundHandler.tokenizer = tokenizer
        BoundHandler.model_name = name
        return BoundHandler

    def start(self, blocking: bool = True):
        """Start the server. If blocking=True, blocks forever."""
        handler = self._make_handler()
        self._httpd = HTTPServer(("0.0.0.0", self.port), handler)
        print(f"[server] MYTHOS inference server running on port {self.port}")
        print(f"[server] Model: {self.model_name}")
        print(f"[server] Endpoints: /v1/chat/completions, /v1/completions, /v1/models, /health")

        if blocking:
            try:
                self._httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n[server] Shutting down...")
                self._httpd.shutdown()
        else:
            self._thread = threading.Thread(target=self._httpd.serve_forever)
            self._thread.daemon = True
            self._thread.start()

    def stop(self):
        """Stop the server."""
        if self._httpd:
            self._httpd.shutdown()
        if self._thread:
            self._thread.join(timeout=5)


# ==================================================================
# CLI entry point
# ==================================================================
def main():
    """CLI entry point: python -m core.server --model /path/to/gguf --port 8080"""
    import argparse

    parser = argparse.ArgumentParser(description="MYTHOS Gemma 4 Inference Server")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer name or path (default: google/gemma-4-26b-it)")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--compute-dtype", default="bfloat16",
                        choices=["bfloat16", "float16"],
                        help="Compute dtype")
    parser.add_argument("--max-context", type=int, default=131072,
                        help="Maximum context length")
    args = parser.parse_args()

    print(f"[server] Loading model from {args.model}...")

    from core.gemma4_runner import Gemma4Runner

    model, info = Gemma4Runner.from_pretrained(
        args.model, qat=True, compute_dtype=args.compute_dtype
    )
    print(f"[server] Loaded: {info}")

    # Load tokenizer
    tok_name = args.tokenizer or "google/gemma-4-26b-it"
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        print(f"[server] Tokenizer: {tok_name}")
    except ImportError:
        print("[server] Warning: transformers not installed, tokenizer unavailable")
        tokenizer = None

    server = InferenceServer(model, tokenizer, port=args.port)
    server.start(blocking=True)


if __name__ == "__main__":
    main()
