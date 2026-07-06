"""core — TensorCUDA model runner and inference engine.

Gives the tensor_cuda kernel library a complete inference stack:
  - model_loader: safetensors/GGUF -> TensorCUDA weight loading
  - gemma4_runner: Gemma 4 26B MoE forward pass + generation loop
  - kv_manager: multi-resolution KV cache + graft mount
  - server: OpenAI-compatible REST API

No PyTorch. No libtorch. TensorCUDA's own Storage -> NDArray -> Tensor stack only.
"""

from . import mistral7b_tc
from . import qwen35_tc
from . import model_loader
from . import gemma4_runner
from . import kv_manager

__all__ = [
    "mistral7b_tc",
    "qwen35_tc", 
    "model_loader",
    "gemma4_runner",
    "kv_manager",
]
