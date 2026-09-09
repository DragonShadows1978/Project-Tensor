"""APA-SP5 amendment 2 -- single-pass injection for the STREAMED ladder.

WHY THIS EXISTS. The H4 context ladder drives
`scripts/gpt_oss20b_stream_forward_smoke.py` as a SUBPROCESS with
`--attention-mode {standard,apa_selective}` (context_ladder.py parse_setting
emits only those two names). The smoke sets `block.self_attn.attention_mode`
from that flag, and the port's SOLE `apa_selective` branch is hard-wired to
the TWO-PASS entry:

    core/gpt_oss20b_tc.py:744
        attn = tc.apa_selective_attention_sink(q, k, kq, v, self.sinks,
                                               self.scaling, float(z), True)

which takes a z-score percentile and has NO delta parameter. There is no
single-pass seam anywhere in ladder, smoke or port (grep for
`apa_selective_attention_sp` in all three returns nothing), so a third mode
cannot be selected without editing the READ-ONLY port.

WHAT THIS DOES INSTEAD. Python imports `sitecustomize` automatically at
interpreter start if it is on sys.path. The smoke IMPORTS the port module
rather than exec-ing it, so `GptOssAttentionTC.__call__` is a live attribute
we can rebind IN THE SUBPROCESS'S OWN MEMORY. No file owned by the ladder,
the smoke or the port is modified on disk; this is the same in-process
monkey-patch the SP5 a1 harness uses (lead-accepted deviation 2).

It is armed ONLY when APA_SP5_SP_DELTA is set, so an un-instrumented run of
the ladder is byte-for-byte the stock two-pass path.

HONEST LABEL: results produced under this file are "the H4 ladder's
construction with the SP entry injected in-process", NOT "the ladder natively
supports the single pass". It does not.

Prior art: ordinary Python monkey-patching / sitecustomize (CPython docs);
SP5 a1 harness (this repo, 2026). The rule itself is BLASST's running-max
criterion (Yuan et al., arXiv 2512.12087) applied to precision, not sparsity.
"""
import os

_DELTA = os.environ.get('APA_SP5_SP_DELTA')

if _DELTA:
    import sys

    # The smoke HARD-CODES the stock engine at
    #   scripts/gpt_oss20b_stream_forward_smoke.py:24
    #     sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
    # and that build has NO single-pass entry (hasattr(tc._C,
    # 'apa_selective_attention_sp') is False on it). Since the smoke is
    # READ-ONLY too, the SP5 engine must win the import race from here:
    # import the SP5 tensor_cuda package + its _C FIRST, so the smoke's later
    # `import tensor_cuda` finds the already-loaded SP5 module in sys.modules
    # and its own sys.path.insert is a no-op. No file is edited.
    _SP5 = os.environ.get('APA_SP5_ENGINE_ROOT')
    _SP5_BUILD = os.environ.get('APA_SP5_ENGINE_BUILD')
    if _SP5 and _SP5_BUILD:
        sys.path.insert(0, _SP5_BUILD)
        sys.path.insert(0, _SP5)
        try:
            import tensor_cuda as _tc_preload
            if not hasattr(_tc_preload._C, 'apa_selective_attention_sp'):
                raise RuntimeError(
                    'apa_sp5_a2: preloaded engine lacks apa_selective_attention_sp; '
                    'refusing to run a mislabelled single-pass cell')
            sys.stderr.write(
                f'[apa_sp5_a2] engine preloaded: {_tc_preload._C.__file__}\n')
        except Exception as _e:
            sys.stderr.write(f'[apa_sp5_a2] engine preload FAILED: {_e}\n')
            raise

    def _install():
        try:
            import tensor_cuda as tc
            G = sys.modules.get('core.gpt_oss20b_tc')
            # The hook can fire while the port module is still executing, so
            # require the class to actually exist before touching it.
            if G is None or not hasattr(G, 'GptOssAttentionTC'):
                return False
            if not hasattr(G, 'F') or not hasattr(tc, '_C'):
                return False
        except Exception:
            return False
        if getattr(G, '_APA_SP5_SP_INSTALLED', False):
            return True
        from tensor_cuda.quant import _quantize_keys, _tables
        delta = float(_DELTA)
        original = G.GptOssAttentionTC.__call__

        def patched(self, x, cos, sin, position_offset=0, kv_cache=None):
            # Only full-attention layers in apa_selective mode are rerouted;
            # sliding layers and standard mode fall through untouched.
            if self.attention_mode != 'apa_selective' or self.sliding_window is not None:
                return original(self, x, cos, sin, position_offset, kv_cache)
            B, L, _ = x.shape
            q = self.q_proj(x).reshape([B, L, self.num_heads, self.head_dim]).transpose(1, 2)
            k = self.k_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
            v = self.v_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
            cs = cos.slice(0, position_offset, L)
            ss = sin.slice(0, position_offset, L)
            q = G.F.apply_rotary(q, cs, ss)
            k = G.F.apply_rotary(k, cs, ss)
            if kv_cache is not None:
                k = tc.cat([kv_cache[0], k], dim=2)
                v = tc.cat([kv_cache[1], v], dim=2)
            Rt, CB, BND = _tables(self.head_dim, self.bulk_bits,
                                  self.num_kv_heads, True, q.device.split(':')[0])
            kq = _quantize_keys(k, Rt, CB, BND)
            attn = tc._C.apa_selective_attention_sp(
                q, k, kq, v, self.scaling, delta, True, self.sinks, False)
            self.last_attention_backend = 'apa_sp_sink_fused_injected'
            out = attn.transpose(1, 2).reshape([B, L, self.num_heads * self.head_dim])
            return self.o_proj(out), (k, v)

        G.GptOssAttentionTC.__call__ = patched
        G._APA_SP5_SP_INSTALLED = True
        sys.stderr.write(f'[apa_sp5_a2] single-pass injected, delta={delta}\n')
        return True

    # The port is not importable at interpreter start (the smoke inserts its
    # own sys.path entries first). Deterministic, cheap approach: wrap the
    # builtin __import__ with a REENTRANCY GUARD -- _install() itself imports,
    # so an unguarded hook recurses pathologically (measured: a bare import
    # went from 0.1 s to >200 s). The guard makes the hook a no-op except on
    # the single import that first makes the port available, and the hook
    # uninstalls itself immediately afterwards.
    import builtins

    _real_import = builtins.__import__
    _busy = [False]

    def _hooked(name, globals=None, locals=None, fromlist=(), level=0):
        mod = _real_import(name, globals, locals, fromlist, level)
        if _busy[0]:
            return mod
        if 'core.gpt_oss20b_tc' in sys.modules and 'tensor_cuda' in sys.modules:
            _busy[0] = True
            try:
                if _install():
                    builtins.__import__ = _real_import
            finally:
                _busy[0] = False
        return mod

    builtins.__import__ = _hooked
