"""BF16 PyTorch MiniCPM3 reference, full attention and forced fused SDPA.

Prior art: OpenBMB MiniCPM3 (2024), PyTorch SDPA (2023), FlashAttention-2
(Tri Dao 2023, https://arxiv.org/abs/2307.08691), xFormers efficient attention
(PyTorch team 2023, https://pytorch.org/blog/accelerated-pytorch-2/).
Reuses these implementations, not a new attention algorithm. SP3 adds pinned
reference windows/backend receipts. Teacher-forced NLL reuses PROTOCOL-2.
"""
import gc
import importlib.metadata
import os
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
from apa_sp3_common import ART, Red, read, registration, sha, verify_weight_stat
from apa_sp3_model import last512, score_windows


def compact_nll(logits, ids):
    # Prior art: projecting/scoring only required LM positions, standard causal
    # LM evaluation. No attention chunking: all S positions traverse all layers.
    # Reuse the UNCHANGED last512 scorer with 513 local positions; final row is
    # intentionally unused, exactly as in the original full-logit scorer.
    logits = np.asarray(logits)
    if logits.shape[0] != 512 or len(ids) < 513:
        raise Red('compact scorer needs exactly 512 prediction rows')
    padded = np.concatenate([logits, np.zeros_like(logits[:1])], axis=0)
    return last512(padded, np.asarray(ids)[-514:-1])


def stage_snapshot(destination):
    """Byte-identical metadata staging avoids HF symlink-relative-import bugs."""
    source = Path(registration()['model']['snapshot'])
    pins = read(ART/'a4_torch_sources.json')['files']
    for name, digest in pins.items():
        if sha(source/name) != digest:
            raise Red('torch remote source pin changed: '+name)
        shutil.copyfile(source/name, Path(destination)/name)
    # Never duplicate 8GB weights or edit the HF snapshot.
    (Path(destination)/'pytorch_model.bin').symlink_to(source/'pytorch_model.bin')


def reference_class(snapshot):
    import torch
    import transformers
    import transformers.utils.import_utils as imports
    # Prior art: version compatibility adapters, standard software maintenance.
    # These are API metadata repairs, not attention/model arithmetic changes.
    # Verified against byte-identical 2024 remote source on transformers 5.12.
    compatibility = []
    if not hasattr(imports, 'is_torch_fx_available'):
        if transformers.__version__ != '5.12.0':
            raise Red('unvalidated transformers version for legacy FX import')
        imports.is_torch_fx_available = lambda: hasattr(torch, 'fx')
        compatibility.append('legacy is_torch_fx_available = hasattr(torch, fx)')
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module('modeling_minicpm.MiniCPM3ForCausalLM',
                                         snapshot, local_files_only=True)
    if cls._tied_weights_keys == ['lm_head.weight'] and transformers.__version__ == '5.12.0':
        cls._tied_weights_keys = {'lm_head.weight': 'model.embed_tokens.weight'}
        compatibility.append('legacy tied-weight list converted to identical embedding/head mapping')
    # Use the same remote-code registration as AutoModel, so Transformers 5's
    # initialization guard preserves loaded parameters instead of reinitializing
    # this legacy class's in-place normal_ initializer after loading.
    cls.register_for_auto_class('AutoModelForCausalLM')
    return cls, compatibility


class FusedSDPA:
    """Enforce the registered choice on actual tensors, and receipt every layer."""
    def __init__(self, torch):
        self.torch = torch
        self.original = torch.nn.functional.scaled_dot_product_attention
        self.calls = []
        self.expected_S = None

    def __call__(self, q, k, v, attn_mask=None, dropout_p=0., is_causal=False, **kw):
        t = self.torch
        if (q.shape != (1, 40, self.expected_S, 96) or k.shape != q.shape
                or v.shape != (1, 40, self.expected_S, 64) or attn_mask is not None
                or not is_causal or dropout_p != 0. or q.dtype != t.bfloat16):
            raise Red('T attention shape/dtype/mask changed; no attention chunking allowed')
        params = t.backends.cuda.SDPAParams(q, k, v, None, 0., True, False)
        flash = t.backends.cuda.can_use_flash_attention(params, debug=False)
        efficient = t.backends.cuda.can_use_efficient_attention(params, debug=False)
        backend = (t.nn.attention.SDPBackend.FLASH_ATTENTION if flash else
                   t.nn.attention.SDPBackend.EFFICIENT_ATTENTION if efficient else None)
        if backend is None:
            raise Red('T_NO_FUSED_BACKEND: flash and efficient both unsupported')
        row = dict(backend=str(backend), flash_supported=flash, efficient_supported=efficient,
                   q_shape=list(q.shape), v_shape=list(v.shape), sm=list(t.cuda.get_device_capability()))
        self.calls.append(row)
        # PyTorch official sdpa_kernel disables all other backends here, so
        # math fallback is impossible, even if fused launch itself fails.
        with t.nn.attention.sdpa_kernel([backend]):
            return self.original(q, k, v, attn_mask=None, dropout_p=0., is_causal=True, **kw)


def sliced_head_hook(module, args):
    return (args[0][:, -513:-1, :],) + args[1:]


def restore_rotary_buffers(model):
    # Prior art: reconstructing nonpersistent derived buffers after meta-device
    # loading, standard PyTorch lifecycle repair. Use the remote model's OWN
    # _init_rope, unchanged formulas/constants; no new positional encoding.
    # transformers 5.12 leaves this 2024 class's nonpersistent buffers unfilled.
    for layer in model.model.layers:
        layer.self_attn._init_rope()


class TorchModel:
    def __init__(self):
        import torch
        from transformers import AutoConfig
        self.torch = t = torch
        verify_weight_stat()
        if not t.cuda.is_available():
            raise Red('T_GPU_UNAVAILABLE')
        if t.cuda.get_device_capability() != (8, 9):
            raise Red('T expects registered sm_89 device')
        self.stage = tempfile.TemporaryDirectory(prefix='apa_sp3_a4_T_')
        stage_snapshot(self.stage.name)
        cls, self.compatibility = reference_class(self.stage.name)
        config = AutoConfig.from_pretrained(self.stage.name, trust_remote_code=True, local_files_only=True)
        if config.pretraining_tp != 1 or not config.tie_word_embeddings:
            raise Red('T LM-head contract changed')
        self.model, loading = cls.from_pretrained(self.stage.name, config=config, local_files_only=True,
                         trust_remote_code=True, dtype=t.bfloat16, attn_implementation='sdpa',
                         output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise Red('T incomplete checkpoint load: '+str(loading))
        self.model.eval()
        restore_rotary_buffers(self.model)
        self.compatibility.append('rebuild nonpersistent rotary buffers with unchanged remote _init_rope after meta loading')
        self.model.to('cuda')
        if {p.dtype for p in self.model.parameters()} != {t.bfloat16}:
            raise Red('T weights are not all bf16')
        if any(type(l.self_attn).__name__ != 'MiniCPMSdpaAttention' for l in self.model.model.layers):
            raise Red('T remote model did not instantiate SDPA attention')
        self.hook = self.model.lm_head.register_forward_pre_hook(sliced_head_hook)
        self.sdpa = FusedSDPA(t)
        t.nn.functional.scaled_dot_product_attention = self.sdpa

    def peak(self):
        t = self.torch
        return dict(peak_resident_mib=(t.cuda.max_memory_reserved()+self.offset)/(1<<20),
                    peak_allocated_mib=t.cuda.max_memory_allocated()/(1<<20),
                    peak_reserved_mib=t.cuda.max_memory_reserved()/(1<<20),
                    peak_status='ESTIMATE: PyTorch reserved high-water plus pre-call device/context offset; internal driver transients may be missed')

    def forward(self, ids):
        t = self.torch
        S = len(ids)
        self.sdpa.expected_S = S
        before = len(self.sdpa.calls)
        t.cuda.synchronize()
        free, total = t.cuda.mem_get_info()
        self.offset = max(0, total-free-t.cuda.memory_reserved())
        t.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with t.inference_mode():
            output = self.model(input_ids=t.as_tensor(ids[None], device='cuda', dtype=t.long),
                                use_cache=False, output_attentions=False, output_hidden_states=False,
                                return_dict=True)
        t.cuda.synchronize()
        elapsed = time.perf_counter()-start
        if len(self.sdpa.calls)-before != 62 or output.logits.shape[1] != 512:
            raise Red('T missing full-layer SDPA calls or prediction rows')
        result = compact_nll(output.logits[0].float().cpu().numpy(), ids)
        del output
        return dict(result, wall_ms=elapsed*1000, **self.peak())

    def close(self):
        self.torch.nn.functional.scaled_dot_product_attention = self.sdpa.original
        self.hook.remove()
        self.stage.cleanup()


def reference(cell, ids):
    import torch
    model = None
    try:
        model = TorchModel()
        result = score_windows(model, ids, cell['S'])
        return dict(result, arm='T', S=cell['S'], fit=True, weight_bits=16,
                    evidence_class='model perplexity, reference', sdpa_calls=model.sdpa.calls,
                    sdpa_backends=sorted({r['backend'] for r in model.sdpa.calls}),
                    compatibility=model.compatibility, packages={n: importlib.metadata.version(n) for n in ('torch', 'transformers')},
                    source_sha256=sha(ART/'a4_torch_sources.json'),
                    projection='only 512 required LM-head rows; full S attention, no cache, no attention chunks')
    except torch.OutOfMemoryError as e:
        # Fit failure is an explicit non-fit reference result; it supplies no
        # ground-truth number and blocks D. Do not retry/chunk attention.
        if model is not None and hasattr(model, 'offset'):
            peak = model.peak()
        else:
            free,total = torch.cuda.mem_get_info()
            offset = max(0,total-free-torch.cuda.memory_reserved())
            peak = dict(peak_resident_mib=(torch.cuda.max_memory_reserved()+offset)/(1<<20),
                        peak_status='ESTIMATE: reserved high-water plus post-OOM sampled context offset')
        raise Red('T_NON_FIT_OOM: '+str(e), dict(fit=False, S=cell['S'], arm='T',
                  outcome='OOM', ppl=None, phase='load_or_full_prefill',
                  sdpa_calls=model.sdpa.calls if model is not None else [], **peak)) from e
    finally:
        if model is not None:
            model.close()
