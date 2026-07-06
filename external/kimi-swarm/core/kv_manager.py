"""core/kv_manager.py — Multi-resolution KV cache + graft mount.

Manages the KV cache with three precision tiers:
  - FP16 for sliding-window layers (capped at 1024 keys, always)
  - FP16 for global layers recent tokens (configurable window, default 2048)
  - INT4 for global layers long-range tokens (everything beyond the FP16 window)

Also handles graft injection: mounting pre-computed KV artifacts as
positional prefixes in the attention path.

No PyTorch.  TensorCUDA primitives only.
"""
from __future__ import annotations

import gc
import os
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

import tensor_cuda as tc
from core.mistral7b_tc import BlockTC, _cast
from core.gemma4_runner import Gemma4Config, KVRing, _zeros, _grow_cap


# ==================================================================
# Multi-resolution global KV: FP16 recent + INT4 long-range
# ==================================================================
class MultiResGlobalKV:
    """Global-layer KV cache with dual precision:
      - FP16 buffer for the most recent N tokens (quality)
      - INT4-packed buffer for tokens beyond N (compression)

    The FP16 window slides forward as new tokens arrive. When a token
    ages out of the FP16 window, it gets packed to INT4.

    Shape: (B, 1, S, D) where D = global_head_dim (512 for Gemma 4)
    """

    def __init__(self, batch_size: int, head_dim: int,
                 fp16_window: int = 2048, max_context: int = 1_000_000):
        self.B = batch_size
        self.D = head_dim
        self.fp16_window = fp16_window
        self.max_context = max_context

        # FP16 buffer: recent tokens
        self.fp16_cap = _grow_cap(fp16_window + 64)
        self.kb_fp16 = _zeros(batch_size, 1, self.fp16_cap, head_dim)
        self.vb_fp16 = _zeros(batch_size, 1, self.fp16_cap, head_dim)
        self.fp16_count = 0

        # INT4 buffer: long-range tokens
        # Packed: (B, 1, S, D/2) uint8 + scales (B, 1, S, D/32)
        self.int4_cap = 0
        self.kb_int4 = None
        self.vb_int4 = None
        self.scales_k = None
        self.scales_v = None
        self.int4_count = 0

        # Bias mask for invalid rows
        self.bias = None
        self._init_bias()

    def _init_bias(self):
        bias = np.full((self.fp16_cap, 1), -1e4, np.float32)
        if self.fp16_count > 0:
            bias[:self.fp16_count] = 0.0
        self.bias = _cast(tc.tensor(bias))

    def append(self, k1, v1):
        """Append a single token's K/V (B, 1, 1, D)."""
        # Check if FP16 buffer needs to grow
        if self.fp16_count >= self.fp16_cap:
            self._grow_fp16()

        # Write to FP16 buffer
        with tc.no_grad():
            tc.write_rows(self.kb_fp16, k1, self.fp16_count)
            tc.write_rows(self.vb_fp16, v1, self.fp16_count)
            # Unmask this row
            zero_row = _cast(tc.tensor(np.zeros((1, 1), np.float32)))
            tc.write_rows(self.bias, zero_row, self.fp16_count)
        self.fp16_count += 1

        # Check if we need to spill oldest FP16 to INT4
        if self.fp16_count > self.fp16_window:
            self._spill_to_int4(1)

    def _grow_fp16(self):
        """Double FP16 capacity."""
        old_cap = self.fp16_cap
        self.fp16_cap = _grow_cap(old_cap + 1)
        B, _, _, D = self.kb_fp16.shape

        def _grow(buf):
            nb = _zeros(B, 1, self.fp16_cap, D)
            with tc.no_grad():
                tc.write_rows(nb, buf, 0)
            return nb

        self.kb_fp16 = _grow(self.kb_fp16)
        self.vb_fp16 = _grow(self.vb_fp16)

        # Grow bias
        old_bias = self.bias
        bias = np.full((self.fp16_cap, 1), -1e4, np.float32)
        # Copy old valid entries
        bias[:old_cap] = old_bias.numpy().reshape(old_cap, 1)[:old_cap, 0]
        self.bias = _cast(tc.tensor(bias))

    def _spill_to_int4(self, n_tokens: int):
        """Move the oldest n_tokens from FP16 to INT4."""
        if n_tokens <= 0 or self.fp16_count <= self.fp16_window:
            return

        # Tokens to spill: the oldest ones that exceed the window
        spill_count = min(n_tokens, self.fp16_count - self.fp16_window)
        if spill_count <= 0:
            return

        # Slice the oldest tokens from FP16
        k_spill = self.kb_fp16.slice(2, 0, spill_count)
        v_spill = self.vb_fp16.slice(2, 0, spill_count)

        # Pack to INT4
        k_packed, k_scales = tc.kv_int4_pack(k_spill, group=32)
        v_packed, v_scales = tc.kv_int4_pack(v_spill, group=32)

        # Append to INT4 buffer (grow if needed)
        if self.int4_count + spill_count > self.int4_cap:
            self._grow_int4(self.int4_count + spill_count)

        with tc.no_grad():
            tc.write_rows(self.kb_int4, k_packed, self.int4_count)
            tc.write_rows(self.vb_int4, v_packed, self.int4_count)
            tc.write_rows(self.scales_k, k_scales, self.int4_count)
            tc.write_rows(self.scales_v, v_scales, self.int4_count)
        self.int4_count += spill_count

        # Shift FP16 buffer left by spill_count
        # Copy [spill_count:fp16_count] to [0:fp16_count-spill_count]
        remaining = self.fp16_count - spill_count
        if remaining > 0:
            k_rem = self.kb_fp16.slice(2, spill_count, remaining)
            v_rem = self.vb_fp16.slice(2, spill_count, remaining)
            with tc.no_grad():
                tc.write_rows(self.kb_fp16, k_rem, 0)
                tc.write_rows(self.vb_fp16, v_rem, 0)
        self.fp16_count = remaining

        # Update bias
        bias = np.full((self.fp16_cap, 1), -1e4, np.float32)
        if remaining > 0:
            bias[:remaining] = 0.0
        self.bias = _cast(tc.tensor(bias))

        tc.empty_cache()

    def _grow_int4(self, need: int):
        """Grow INT4 buffers to accommodate `need` tokens."""
        B, _, _, D = self.kb_fp16.shape
        new_cap = _grow_cap(need)
        if new_cap <= self.int4_cap:
            return

        D_packed = D // 2
        D_groups = D // 32

        def _grow_packed(buf):
            nb = tc.zeros(B, 1, new_cap, D_packed, dtype="uint8")
            if buf is not None and self.int4_count > 0:
                with tc.no_grad():
                    tc.write_rows(nb, buf.slice(2, 0, self.int4_count), 0)
            return nb

        def _grow_scales(buf):
            nb = tc.zeros(B, 1, new_cap, D_groups, dtype=BlockTC.COMPUTE_DTYPE)
            if buf is not None and self.int4_count > 0:
                with tc.no_grad():
                    tc.write_rows(nb, buf.slice(2, 0, self.int4_count), 0)
            return nb

        self.kb_int4 = _grow_packed(self.kb_int4)
        self.vb_int4 = _grow_packed(self.vb_int4)
        self.scales_k = _grow_scales(self.scales_k)
        self.scales_v = _grow_scales(self.scales_v)
        self.int4_cap = new_cap

    def get_kv(self, need_int4: bool = True):
        """Return the full KV cache for attention.

        Returns concatenated (k, v) where:
          - INT4 portion is unpacked to FP16
          - FP16 portion is used as-is
        """
        parts_k = []
        parts_v = []

        # INT4 portion
        if self.int4_count > 0 and need_int4:
            k_int4 = tc.kv_int4_unpack(
                self.kb_int4, self.scales_k, group=32,
                lo=0, n=self.int4_count, out_dtype=BlockTC.COMPUTE_DTYPE)
            v_int4 = tc.kv_int4_unpack(
                self.vb_int4, self.scales_v, group=32,
                lo=0, n=self.int4_count, out_dtype=BlockTC.COMPUTE_DTYPE)
            parts_k.append(k_int4)
            parts_v.append(v_int4)

        # FP16 portion
        if self.fp16_count > 0:
            parts_k.append(self.kb_fp16.slice(2, 0, self.fp16_count))
            parts_v.append(self.vb_fp16.slice(2, 0, self.fp16_count))

        if not parts_k:
            return None, None

        k = tc.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
        v = tc.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]
        return k, v

    @property
    def total_tokens(self):
        return self.int4_count + self.fp16_count


# ==================================================================
# KV Manager: per-layer cache with multi-resolution for global layers
# ==================================================================
class KVManager:
    """Manages the full per-layer KV cache for Gemma 4 inference.

    For sliding-window layers: standard KVRing (capped at 1024).
    For global layers: MultiResGlobalKV (FP16 recent + INT4 long-range).

    Also handles graft mounting/unmounting.
    """

    def __init__(self, cfg: Gemma4Config, batch_size: int = 1,
                 fp16_window: int = 2048):
        self.cfg = cfg
        self.batch_size = batch_size
        self.fp16_window = fp16_window
        self.caches: List[Any] = []
        self._multi_res: List[Optional[MultiResGlobalKV]] = []

    def init_prefill(self, prefill_caches: list):
        """Initialize from prefill output caches.

        prefill_caches: list of (k, v) tuples per layer from prefill forward.
        """
        self.caches = []
        self._multi_res = []
        for i, (k, v) in enumerate(prefill_caches):
            if Gemma4Config.is_global(i):
                # Global layer: use multi-resolution
                mr = MultiResGlobalKV(self.batch_size, self.cfg.global_head_dim,
                                      self.fp16_window)
                # Write prefill tokens into FP16 buffer
                S = k.shape[2]
                with tc.no_grad():
                    tc.write_rows(mr.kb_fp16, k, 0)
                    tc.write_rows(mr.vb_fp16, v, 0)
                mr.fp16_count = S
                # Update bias
                bias = np.full((mr.fp16_cap, 1), -1e4, np.float32)
                bias[:S] = 0.0
                mr.bias = _cast(tc.tensor(bias))
                self.caches.append(mr)
                self._multi_res.append(mr)
            else:
                # Sliding layer: standard KVRing
                ring = KVRing(k, v, ring_cap=self.cfg.sliding_window)
                self.caches.append(ring)
                self._multi_res.append(None)

    def get_layer_cache(self, layer_idx: int):
        """Get the cache object for a specific layer."""
        return self.caches[layer_idx] if layer_idx < len(self.caches) else None

    def append_decode(self, layer_idx: int, k1, v1):
        """Append a decode-step K/V to the appropriate cache."""
        cache = self.caches[layer_idx]
        if cache is None:
            return
        if isinstance(cache, MultiResGlobalKV):
            cache.append(k1, v1)
        else:
            cache.append(k1, v1, _cast(tc.tensor(np.zeros((1, 1), np.float32))))

    def mount_graft(self, layer_idx: int, graft_k, graft_v,
                    scale: float = 1.0):
        """Mount a graft as a prefix to a layer's KV cache.

        The graft K/V are concatenated in front of the live cache.
        This is called by the graft injection mechanism.
        """
        cache = self.caches[layer_idx]
        if cache is None:
            return

        # For global layers with multi-res, we inject at the attention level
        # (handled by Gemma4AttentionTC.inject_kv). This method is for
        # explicit manipulation.
        if isinstance(cache, MultiResGlobalKV):
            # Prepend graft to the FP16 portion
            # Shift existing FP16 right, write graft at front
            graft_len = graft_k.shape[2]
            old_count = cache.fp16_count

            # Ensure capacity
            while old_count + graft_len > cache.fp16_cap:
                cache._grow_fp16()

            # Shift existing right
            if old_count > 0:
                old_k = cache.kb_fp16.slice(2, 0, old_count)
                old_v = cache.vb_fp16.slice(2, 0, old_count)
                with tc.no_grad():
                    tc.write_rows(cache.kb_fp16, old_k, graft_len)
                    tc.write_rows(cache.vb_fp16, old_v, graft_len)

            # Write graft at front
            with tc.no_grad():
                tc.write_rows(cache.kb_fp16, graft_k, 0)
                tc.write_rows(cache.vb_fp16, graft_v, 0)
            cache.fp16_count = old_count + graft_len

            # Update bias
            bias = np.full((cache.fp16_cap, 1), -1e4, np.float32)
            bias[:cache.fp16_count] = 0.0
            cache.bias = _cast(tc.tensor(bias))

    def unmount_graft(self, layer_idx: int, graft_len: int):
        """Remove a graft prefix from a layer's cache."""
        cache = self.caches[layer_idx]
        if cache is None or not isinstance(cache, MultiResGlobalKV):
            return

        # Shift left by graft_len
        new_count = cache.fp16_count - graft_len
        if new_count > 0:
            rem_k = cache.kb_fp16.slice(2, graft_len, new_count)
            rem_v = cache.vb_fp16.slice(2, graft_len, new_count)
            with tc.no_grad():
                tc.write_rows(cache.kb_fp16, rem_k, 0)
                tc.write_rows(cache.vb_fp16, rem_v, 0)
        cache.fp16_count = max(0, new_count)

        # Update bias
        bias = np.full((cache.fp16_cap, 1), -1e4, np.float32)
        if cache.fp16_count > 0:
            bias[:cache.fp16_count] = 0.0
        cache.bias = _cast(tc.tensor(bias))

    def total_kv_bytes(self) -> int:
        """Report total KV cache memory usage in bytes."""
        total = 0
        for cache in self.caches:
            if cache is None:
                continue
            if isinstance(cache, MultiResGlobalKV):
                # FP16 portion
                total += cache.fp16_count * cache.D * 2 * 2  # k + v, 2 bytes
                # INT4 portion
                if cache.int4_count > 0:
                    D_packed = cache.D // 2
                    D_groups = cache.D // 32
                    total += cache.int4_count * D_packed  # packed bytes
                    total += cache.int4_count * D_groups * 2  # scales (fp16)
            else:
                # KVRing: count valid rows
                n = min(cache.count, cache.cap)
                _, KV, _, D = cache.kb.shape
                total += n * KV * D * 2 * 2
        return total

    def save(self, path: str, position_offset: int = 0):
        """Save all caches to disk as npz."""
        arrs = {"position_offset": np.array([position_offset], np.int64)}
        for i, cache in enumerate(self.caches):
            if isinstance(cache, MultiResGlobalKV):
                # Save both FP16 and INT4 portions
                k_fp16 = cache.kb_fp16.slice(2, 0, cache.fp16_count)
                v_fp16 = cache.vb_fp16.slice(2, 0, cache.fp16_count)
                arrs[f"l{i}_k_fp16"] = k_fp16.float().numpy()
                arrs[f"l{i}_v_fp16"] = v_fp16.float().numpy()
                arrs[f"l{i}_fp16_count"] = np.array([cache.fp16_count], np.int64)
                arrs[f"l{i}_int4_count"] = np.array([cache.int4_count], np.int64)
                if cache.int4_count > 0:
                    arrs[f"l{i}_k_int4"] = cache.kb_int4.slice(
                        2, 0, cache.int4_count).numpy()
                    arrs[f"l{i}_v_int4"] = cache.vb_int4.slice(
                        2, 0, cache.int4_count).numpy()
                    arrs[f"l{i}_sk"] = cache.scales_k.slice(
                        2, 0, cache.int4_count).float().numpy()
                    arrs[f"l{i}_sv"] = cache.scales_v.slice(
                        2, 0, cache.int4_count).float().numpy()
            else:
                k, v = cache.ordered()
                arrs[f"l{i}_k"] = k.float().numpy()
                arrs[f"l{i}_v"] = v.float().numpy()
                arrs[f"l{i}_ring"] = np.array([int(cache.ring)], np.int64)
        tmp = path + ".tmp"
        np.savez(tmp, **arrs)
        os.replace(tmp + ".npz", path)

    @classmethod
    def load(cls, path: str, cfg: Gemma4Config, batch_size: int = 1,
             fp16_window: int = 2048):
        """Load caches from disk."""
        z = np.load(path)
        mgr = cls(cfg, batch_size, fp16_window)
        mgr.caches = []
        mgr._multi_res = []

        for i in range(cfg.num_layers):
            if f"l{i}_k_fp16" in z.files:
                # Multi-resolution global layer
                mr = MultiResGlobalKV(batch_size, cfg.global_head_dim, fp16_window)
                # FP16
                fp16_count = int(z[f"l{i}_fp16_count"][0])
                if fp16_count > 0:
                    k_fp16 = _cast(tc.tensor(z[f"l{i}_k_fp16"]))
                    v_fp16 = _cast(tc.tensor(z[f"l{i}_v_fp16"]))
                    with tc.no_grad():
                        tc.write_rows(mr.kb_fp16, k_fp16, 0)
                        tc.write_rows(mr.vb_fp16, v_fp16, 0)
                    mr.fp16_count = fp16_count
                # INT4
                int4_count = int(z[f"l{i}_int4_count"][0])
                if int4_count > 0:
                    mr.int4_count = int4_count
                    mr.int4_cap = _grow_cap(int4_count)
                    B = batch_size
                    D = cfg.global_head_dim
                    D_packed = D // 2
                    D_groups = D // 32
                    mr.kb_int4 = tc.zeros(B, 1, mr.int4_cap, D_packed, dtype="uint8")
                    mr.vb_int4 = tc.zeros(B, 1, mr.int4_cap, D_packed, dtype="uint8")
                    mr.scales_k = _zeros(B, 1, mr.int4_cap, D_groups)
                    mr.scales_v = _zeros(B, 1, mr.int4_cap, D_groups)
                    with tc.no_grad():
                        tc.write_rows(mr.kb_int4, tc.tensor(z[f"l{i}_k_int4"]), 0)
                        tc.write_rows(mr.vb_int4, tc.tensor(z[f"l{i}_v_int4"]), 0)
                        tc.write_rows(mr.scales_k, _cast(tc.tensor(z[f"l{i}_sk"])), 0)
                        tc.write_rows(mr.scales_v, _cast(tc.tensor(z[f"l{i}_sv"])), 0)

                # Bias
                bias = np.full((mr.fp16_cap, 1), -1e4, np.float32)
                if mr.fp16_count > 0:
                    bias[:mr.fp16_count] = 0.0
                mr.bias = _cast(tc.tensor(bias))

                mgr.caches.append(mr)
                mgr._multi_res.append(mr)
            else:
                # Standard KVRing
                k = _cast(tc.tensor(z[f"l{i}_k"]))
                v = _cast(tc.tensor(z[f"l{i}_v"]))
                ring = bool(int(z[f"l{i}_ring"][0])) if f"l{i}_ring" in z.files else False
                ring_cap = cfg.sliding_window if ring else None
                kv = KVRing(k, v, ring_cap=ring_cap)
                mgr.caches.append(kv)
                mgr._multi_res.append(None)

        return mgr, int(z["position_offset"][0])


# ==================================================================
# Graft adapter: bridge between GRM and KVManager
# ==================================================================
class GraftAdapter:
    """Adapts GRM graft artifacts for injection into KVManager.

    Handles:
      - Loading graft artifacts from disk
      - Format conversion to TensorCUDA tensors
      - Mounting/unmounting across all global layers
    """

    def __init__(self, kv_manager: KVManager, cfg: Gemma4Config):
        self.kv_mgr = kv_manager
        self.cfg = cfg
        self.mounted: Dict[str, Dict] = {}  # graft_id -> {layer_kvs, len}

    def load_graft(self, path: str) -> Dict[int, Tuple]:
        """Load a graft artifact from disk.

        Returns {layer_idx: (k_tensor, v_tensor)} for global layers.
        """
        z = np.load(path)
        graft = {}
        for i in range(self.cfg.num_layers):
            if not Gemma4Config.is_global(i):
                continue
            k_key = f"l{i}_k"
            v_key = f"l{i}_v"
            if k_key in z.files and v_key in z.files:
                k = _cast(tc.tensor(np.ascontiguousarray(z[k_key])))
                v = _cast(tc.tensor(np.ascontiguousarray(z[v_key])))
                graft[i] = (k, v)
        return graft

    def mount(self, graft_id: str, graft_path: str, scale: float = 1.0):
        """Mount a graft into all global layers."""
        graft = self.load_graft(graft_path)
        graft_len = 0
        for layer_idx, (k, v) in graft.items():
            self.kv_mgr.mount_graft(layer_idx, k, v, scale)
            graft_len = k.shape[2]
        self.mounted[graft_id] = {
            "path": graft_path,
            "len": graft_len,
            "scale": scale,
        }

    def unmount(self, graft_id: str):
        """Unmount a graft from all global layers."""
        info = self.mounted.pop(graft_id, None)
        if info is None:
            return
        graft_len = info["len"]
        for i in range(self.cfg.num_layers):
            if Gemma4Config.is_global(i):
                self.kv_mgr.unmount_graft(i, graft_len)

    def unmount_all(self):
        """Unmount all grafts."""
        for graft_id in list(self.mounted.keys()):
            self.unmount(graft_id)

    def get_mounted_tokens(self) -> int:
        """Total tokens consumed by all mounted grafts."""
        return sum(g["len"] for g in self.mounted.values())
