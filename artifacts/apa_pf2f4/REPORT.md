# APA-PF2-F4 report

**ACCEPT.** Both P1 tests now pin the packed-prefill route in APA §8. The complete CPU outcome comparison passed against both MAINCHK3 PF2 repeats.

## Gates

- **G1 PASS.** Both target IDs passed on CPU with Build F3.
- **G2 PASS.** The complete MAINCHK3-form run recorded 3,628 IDs. All 3,626 other IDs matched both PF2 repeats. The old P1-D ID went from failed to absent, its renamed replacement went from absent to passed, and the P1-F2 ID went from failed to passed. Pytest exited 1 with 1,074 passed, 2,131 failed, 407 skipped, 15 errors and 1 collection error. Those other outcomes are inherited. The run used the registered 23 GiB address-space limit with 12,049,129,472 bytes of startup headroom.
- **G3 PASS.** Both scratch mutations failed in the intended test's call phase. The restored P1-D refusal raised `ValueError: apa_sp2: prefill (L > 1) on packed keys is not supported; quantize with storage='dense' for prefill`. The P1-F2 fake raised `RuntimeError: PF2-F4 power: FakeSpo2.packed_prefill called`.
- **G4 PASS.** Across 32,540 scoped worktree paths, only the two authorized test files differed. Source boundary checks confirmed the two test definitions and the added fake method. Build F3 and all read-only inputs matched their pins. CUDA was hidden in every job. No product or other test changes, git writes, deletes, rebuilds or GPU runs.

Receipts for [G1](G1.json), [G2](G2.json), [G3](G3.json) and [G4](G4.json). The [dependency pins and rebindings](COPY_DEPS.json) identify the original scripts, frozen reference data, Build F3 and scratch layout.

**Not claimed fixed.** The inherited CPU failure, error and skip IDs remain in the per-ID receipt. For example, the existing selector-accuracy collection error is `RuntimeError: cudaMalloc failed: no CUDA-capable device is detected`. The route assertions exercise Python dispatch against recording fakes. The comparison does not measure GPU numerics or timing. [Full CPU receipt](cpu/G2/RESULT.json)

## Assertion changes

Renamed `test_prefill_on_packed_keys_is_refused_naming_dense` to `test_prefill_on_packed_keys_routes_to_packed_prefill`. `test_r1_pinned_route_refusals_still_precede_any_device_call` retains its ID.

| Site | Previous assertion | Current assertion |
| --- | --- | --- |
| P1-D | Prefill raises `ValueError` naming dense storage. | The public call returns the packed prefill sentinel. Exactly one `packed_prefill` call receives the query, exact keys, values, packed tuple, scale, causal flag and TQ4 format. |
| P1-D | No packed decode or dense quantization. | Also checks that dense attention does not run. |
| P1-D | Packed storage, eight query rows, prefill named in the fallback reason. | Packed storage, eight query rows, backend `spq_packed_prefill_tq4_row`, and a null fallback reason. The dense-copy assertion is retained. |
| P1-F2 | Prefill raises `ValueError` naming dense storage. | The public call returns the fake's packed-prefill output and records exactly one packed-prefill call for the eight-row query. |
| P1-F2 | The final call log has no packed decode or dense attention. | The earlier refusals produce no attention device calls. The later refusals produce no packed decode or dense attention and leave the one successful prefill call as the only packed-prefill entry. The eight remaining refusal assertions are retained. |

## Census summary

```json
{
  "prefill_grep_hits": 41,
  "prefill_grep_files": 16,
  "hits_by_category": {
    "dense dispatch": 12,
    "harness or receipt": 25,
    "packed helper": 1,
    "re-pin": 3
  },
  "packed_prefill_refusal_tests": [
    "tensor_cuda/tests/test_apa_p1d_policy.py::test_prefill_on_packed_keys_is_refused_naming_dense",
    "tensor_cuda/tests/test_apa_p1f2_cpu.py::test_r1_pinned_route_refusals_still_precede_any_device_call"
  ],
  "other_packed_prefill_refusal_tests": 0
}
```

The P1-F2 refusal line omits the word prefill and was identified from its eight-row query and the MAINCHK3 interaction. [CENSUS.json](CENSUS.json)

## Deviations

- The initial registration command hit a shell quoting error before it wrote a valid registration. Three misdirected receipt redirects hit `Read-only file system` at the filesystem root and wrote nothing there. The partial draft is retained as `REGISTRATION_DRAFT_FAILED.md`. The complete registration was pinned before any gate. Zero gate-script repairs.
- G2 used the pre-registered 23 GiB envelope when 24 GiB lacked the required startup headroom. The full suite and both per-ID comparisons used the registered criteria.

## Prior art

- Meszaros, *xUnit Test Patterns* (2007) supplies the recording test-double technique inherited from the P1 fixtures. PF2-F4 adds assertions for the shipped prefill call and route record.
- DeMillo, Lipton and Sayward, *Hints on Test Data Selection* (1978) supplies mutation testing. PF2-F4 applies the technique at the two sites specified by the order.
- MAINCHK3 (Project-Tensor, 2026) supplies the suite harness, phase folding, and scope manifests. PF2-F4 reuses them for a two-repeat per-ID comparison, an instance of differential testing (McKeeman, 1998), with SHA-256 pins (NIST FIPS 180-4, 2015).
- Unverified — lead to check “Meszaros xUnit Test Patterns Test Spy 2007”, “DeMillo Lipton Sayward Hints on Test Data Selection 1978”, and “McKeeman differential testing 1998”.

## Files written

The 46 worktree files and links are listed verbatim below. The 3,522 scratch files and links retained under `/tmp/pf2f4_run` are listed verbatim in [SCRATCH_WRITTEN.txt](SCRATCH_WRITTEN.txt). The pre-existing seat-launcher log `logs/apa_pf2f4_r1.log` is managed externally.

```text
artifacts/apa_pf2f4/CENSUS.json
artifacts/apa_pf2f4/COPY_DEPS.json
artifacts/apa_pf2f4/FILES_WRITTEN.txt
artifacts/apa_pf2f4/G1.json
artifacts/apa_pf2f4/G1_COMMAND.json
artifacts/apa_pf2f4/G1_FILES.json
artifacts/apa_pf2f4/G2.json
artifacts/apa_pf2f4/G2_COMMAND.json
artifacts/apa_pf2f4/G3.json
artifacts/apa_pf2f4/G3_FAKE.diff
artifacts/apa_pf2f4/G3_FAKE_COMMAND.json
artifacts/apa_pf2f4/G3_FAKE_FILES.json
artifacts/apa_pf2f4/G3_ROUTE.diff
artifacts/apa_pf2f4/G3_ROUTE_COMMAND.json
artifacts/apa_pf2f4/G3_ROUTE_FILES.json
artifacts/apa_pf2f4/G4.json
artifacts/apa_pf2f4/LEDGER.md
artifacts/apa_pf2f4/MODULES.json
artifacts/apa_pf2f4/PREFILL_GREP.txt
artifacts/apa_pf2f4/REGISTRATION.md
artifacts/apa_pf2f4/REGISTRATION.sha256
artifacts/apa_pf2f4/REGISTRATION_DRAFT_FAILED.md
artifacts/apa_pf2f4/REPORT.md
artifacts/apa_pf2f4/SCOPE_AFTER.json
artifacts/apa_pf2f4/SCOPE_BEFORE.json
artifacts/apa_pf2f4/SCRATCH_WRITTEN.txt
artifacts/apa_pf2f4/TESTS.diff
artifacts/apa_pf2f4/VERSIONS.txt
artifacts/apa_pf2f4/cpu/G1/RESULT.json
artifacts/apa_pf2f4/cpu/G1/junit.xml
artifacts/apa_pf2f4/cpu/G2/RESULT.json
artifacts/apa_pf2f4/cpu/G2/junit.xml
artifacts/apa_pf2f4/cpu/G3_FAKE/RESULT.json
artifacts/apa_pf2f4/cpu/G3_FAKE/junit.xml
artifacts/apa_pf2f4/cpu/G3_ROUTE/RESULT.json
artifacts/apa_pf2f4/cpu/G3_ROUTE/junit.xml
artifacts/apa_pf2f4/packages/PF2
artifacts/apa_pf2f4/trees/R6_FIXTURE
logs/apa_pf2f4_g1.log
logs/apa_pf2f4_g2.log
logs/apa_pf2f4_g3_fake.log
logs/apa_pf2f4_g3_route.log
scripts/apa_pf2f4_gates.py
scripts/apa_pf2f4_scope.py
tensor_cuda/tests/test_apa_p1d_policy.py
tensor_cuda/tests/test_apa_p1f2_cpu.py
```

## Diff

```diff
--- a/tensor_cuda/tests/test_apa_p1d_policy.py
+++ b/tensor_cuda/tests/test_apa_p1d_policy.py
@@ -571,16 +571,30 @@
 
 
 # ---------------------------------------------------------------- R-E prefill
-def test_prefill_on_packed_keys_is_refused_naming_dense(env, tmp_path):
-    """L3 found the transient reconstruction not bitwise equal: R-E refuses."""
+def test_prefill_on_packed_keys_routes_to_packed_prefill(env, tmp_path, monkeypatch):
+    """PF2 dispatches prefill directly on packed keys."""
+    # Recording test double (Meszaros, xUnit Test Patterns, 2007), as above.
+    prefill_out = object()
+
+    def packed_prefill(*args, **kwargs):
+        env.rec.calls.append(("packed_prefill", args, kwargs))
+        return prefill_out
+
+    monkeypatch.setattr(env.pkg._spo2, "packed_prefill", packed_prefill, raising=False)
     k = keys(B=1, KV=2, S=64, D=64)
     p = env.sp2.quantize_sp2_keys(k, 4)
     q = FakeTensor((1, 4, 8, 64))
-    with pytest.raises(ValueError, match="storage='dense'"):
-        env.sp2.apa_selective_attention_sp(q, k, p, FakeTensor((1, 2, 64, 64)), 1 / math.sqrt(64), .01,
-                                           bulk_bits=4, margins=packed_table(env, tmp_path), is_causal=True)
-    assert not calls(env, "packed_run") and not calls(env, "quantize_dense")
+    v = FakeTensor((1, 2, 64, 64))
+    scale = 1 / math.sqrt(64)
+    out = env.sp2.apa_selective_attention_sp(q, k, p, v, scale, .01, bulk_bits=4,
+                                            margins=packed_table(env, tmp_path), is_causal=True)
+    assert out is prefill_out
+    (_, args, kwargs), = calls(env, "packed_prefill")
+    assert args[:5] == (q, k, v, p._packed_tuple(), scale)
+    assert args[6] is True and kwargs == {"fmt": "tq4"}
+    assert not any(calls(env, kind) for kind in ("packed_run", "dense_epsilon", "quantize_dense"))
     route = env.packed.last_route("attention")
-    assert route["storage"] == "packed" and route["l"] == 8 and "prefill" in route["fallback_reason"]
+    assert route["storage"] == "packed" and route["l"] == 8
+    assert route["backend"] == "spq_packed_prefill_tq4_row" and route["fallback_reason"] is None
     # The object never gains a dense copy.
     assert not [t for t in _reachable_tensors(p) if t.shape[-1] == 64 and len(t.shape) == 4]
--- a/tensor_cuda/tests/test_apa_p1f2_cpu.py
+++ b/tensor_cuda/tests/test_apa_p1f2_cpu.py
@@ -93,6 +93,11 @@
         def packed_run(self, *args, **kwargs):
             calls.append(("packed_run", args, kwargs))
             return ("packed_out", "packed_mask") if args[8] else "packed_out"
+
+        def packed_prefill(self, *args, **kwargs):
+            # Recording test double (Meszaros, xUnit Test Patterns, 2007).
+            calls.append(("packed_prefill", args, kwargs))
+            return ("packed_prefill_out", "packed_mask") if args[8] else "packed_prefill_out"
 
     class FakeC:
         def apa_selective_attention_sp_epsilon(self, *args):
@@ -257,8 +262,12 @@
         call(q, k, p, v, .2, .01, bulk_bits=4, margins=m)
     with pytest.raises(ValueError, match="epsilon must be"):
         call(q, k, p, v, s, 0., bulk_bits=4, margins=m)
-    with pytest.raises(ValueError, match="storage='dense'"):
-        call(FakeTensor((1, 4, 8, 128)), k, p, v, s, .01, bulk_bits=4, margins=m, is_causal=True)
+    assert not device_calls(env) and not kinds(env, "packed_prefill")
+    prefill_q = FakeTensor((1, 4, 8, 128))
+    assert call(prefill_q, k, p, v, s, .01, bulk_bits=4, margins=m, is_causal=True) == "packed_prefill_out"
+    prefill_calls = kinds(env, "packed_prefill")
+    (_, args, _), = prefill_calls
+    assert args[0] is prefill_q
     p.invalidate()
     with pytest.raises(ValueError, match="stale packed keys"):
         call(q, k, p, v, s, .01, bulk_bits=4, margins=m)
@@ -268,7 +277,7 @@
     monkeypatch.delenv("TC_APA_SP")
     with pytest.raises(RuntimeError, match="TC_APA_SP=1"):
         call(q, k, p, v, s, .01, bulk_bits=4, margins=m)
-    assert not device_calls(env)
+    assert not device_calls(env) and kinds(env, "packed_prefill") == prefill_calls
 
 
 # ------------------------------------------------------------------ R3 (P1R2-F04)
```
