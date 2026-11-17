# Packed storage as the SP default: design for the implementing round

Design only. This round changes no Python/public API or shipped default. The implementing round must ship the packed build feature and choose packed storage by default when SP is enabled; it must also migrate real SP cache builders. An additional opt-in packed function by itself does not meet that requirement. `TC_APA_SP=1` remains the existing experimental-SP enable gate; choosing the default representation within enabled SP is a separate decision.

Evidence classes: current source census and integer storage arithmetic; historical finite GPU/model receipts where named; proposed implementation and unrun acceptance below. No new model-quality, GPU-concurrency, or promotion result is claimed here.

## (a) Entry today and the selection contract

The public entry is `tensor_cuda/tensor_cuda/apa_sp2.py:72`:

```python
apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *, bulk_bits,
                           margins, is_causal=False, sinks=None,
                           diagnostics=False)
```

It requires `TC_APA_SP=1`, a `RegisteredMargins` instance, rank-four queries, fp32, D64/D128, bulk bits 2/4, and the registered fp32 value of `scale=1/sqrt(D)` (`apa_sp2.py:53-59,80-89`). `quantize_sp2_keys(k, bulk_bits)` (`:61-69`) uses per-KV-head TurboQuant rotations/Lloyd-Max tables and returns a **dense fp32 reconstructed Kq**, with K's shape. The bit count describes the approximation; it does not describe Kq's resident width. The wrapper looks up the supplied table's measured `e_q` and calls raw `_C.apa_selective_attention_sp_epsilon(q,k,kq,v,scale,epsilon,e_q,...)` (`:88`). That native entry computes

`delta = upward-rounded-fp32(-log(epsilon) + 2*e_q)`

and forwards to the direct-delta SP entry (`src/apa_sp2.cuh:3-22`). The table has four measured entries with a frozen registration/hash and maximum-over-registered-valid-scores statistic. Its validation does not turn those finite measurements into an error bound on unseen tensors.

Precisely, let `b_j` be the computed bulk score, `x_j` the corresponding exact score, and `P_j=max(b_i: i<=j in this carry partition)`. SP refines iff `b_j >= P_j-delta`, including equality. The public dense decode carries within its fixed contiguous partition (`apa_sp1_1.cuh:2-3,66-82`); the packed wrappers default to `carry="partition"`, while EB3/EB4 model receipts use `carry="row"`. Those are distinct policies and cannot be silently interchanged. Every key still contributes to the merged-score softmax denominator; unrefined keys retain their bulk scores.

**Conditional guarantee (reasoning):** if `|b_i-x_i|<=e_q` for every relevant valid score, then an unrefined key obeys `x_j < M_j + log(epsilon)`, where `M_j` is the exact prefix maximum in that partition. Thus its unnormalized mass relative to that exact prefix maximum is below epsilon, and its exact full-row softmax probability is also below epsilon. This is a per-key selection implication. It is neither a total-tail-mass epsilon bound nor an attention-output/logit/PPL error bound. The existing measured table supplies evidence only for its registered regime.

`_apa_spq1_entry` and `_apa_spq4_entry` (`__init__.py:1586,1604`) accept `(q,k,v,packed,scale,delta,...)`, not epsilon/margins. Their packed tuple is `(uint8 codes [B,KV,C,D/2], fp32 norms [B,KV,C], fp32 rotations [KV,D,D], fp32 codebook[16])`. They use the same form of running-prefix criterion on their **own computed packed bulk scores**, with caller-supplied finite nonnegative delta. Without a packed-backend error margin, delta supplies only that computed-score selection rule. In particular, copying delta=8.1 or reusing the fp32 dense TQ margin does not preserve the epsilon implication. Rotation/reconstruction order, bf16 tensor-core score arithmetic, and row carry can differ. EB4 additionally chooses V precision by stable top-fraction block mass; epsilon does not govern that independent V approximation.

“Packed by default” therefore means the following for each argument:

| Argument | Required default behavior in the successor |
|---|---|
| `q` | Preserve tensor, dtype and head layout; no implicit fp32-to-bf16 conversion to enter EB3. Existing governed fp32 calls stay fp32. bf16 governance is a new registered regime, not inherited calibration. |
| `k` | Keep full-precision exact K resident; create/update packed derived state from this exact K, with explicit cache generation and append ownership. |
| `kq` | Accept the new owned packed-key object returned by the quantizer by default for supported 4-bit cases. Keep legacy dense Tensor input meaningful; do not discard arbitrary supplied Kq and regenerate a different approximation from K. Migrate production cache builders rather than leaving them holding dense Kq beside packed codes. |
| `v` | Keep exact V resident and use the SP/EB3 value path by default. SP-tk8's additional int8 V/scales increase storage and require separate opt-in quality evidence. |
| `scale` | Pass through unchanged, validate against the margin's actual scale regime. Existing governed calls retain their registered `1/sqrt(D)` restriction. |
| `epsilon` | Retain its explicit value and range `(0,1]`; derive delta using a backend-matching registered margin with the existing upward-rounding rule. Never reinterpret epsilon as delta. |
| `bulk_bits` | Honor 2 versus 4. Initial packed storage supports TQ4 only; 2-bit remains the explicit dense fallback until a separately tested packed-2 implementation exists. Never silently replace 2 with 4. |
| `margins` | Require the correct quantizer, dtype, backend arithmetic, scale and carry-policy identity. Old table applies to dense fp32 only. Add distinct measured packed entries and explicit fallback/rejection when unavailable. |
| `is_causal` | Preserve bottom-right visibility and existing validation. L1 decode sees the entire visible prefix; L>1 initially uses the unchanged dense fallback. |
| `sinks` | Preserve dtype/device/[H] validation, exact denominator participation, and one sink contribution. |
| `diagnostics` | Preserve output and mask schema and the same decision rule as non-diagnostic execution; diagnostic-only reconstructions must not become persistent normal cache storage. |
| packed `delta`, `fmt`, `carry`, `gqa_path` | Keep explicit raw controls for research/legacy callers. Public epsilon dispatch resolves them internally using matching calibration; old raw callers are not implicitly granted epsilon guarantees. |

## (b) Caller census and migration

The named roots searched are this worktree, `/mnt/ForgeRealm/GRAPA-Native-LLM`, `/mnt/ForgeRealm/GraftRepository`, and `/mnt/ForgeRealm/Project-Voice`. `GraftRepositoryMemory` is the service/seat name used for the GraftRepository project in this workspace (`docs/ENGINE_BUILD_LOG.md:49`); no separate checkout with that name was found at the checked explicit locations. The GRAPA root is a symlink to `$HOME/ForgeRealm-cold/GRAPA-Native-LLM`; the named root was searched without following nested symlinks. Other worktrees, generated builds, dependencies and binary/capture payloads are not application source. The initial broad census retained extra worktree/reference hits in `CALLER_CENSUS_COMMANDS.md`; the final commands explicitly exclude nested Fork/WorkTree copies and include Python, C/C++/CUDA, shell, notebooks, JS/TS and Rust source. No recursive search of `/`, `/mnt`, or `/mnt/Shared` occurred.

The final literal census has **560 matching source lines in this worktree, 11 in GRAPA's archived engine patch, zero in GraftRepository and zero in Project-Voice**. Those counts include definitions, exports, stubs, diagnostics, historical snapshots and negative controls; they are not caller counts. The static argument inventory finds **281 Python call sites, including 80 in current source/tests/harnesses and 201 in artifact/reference code**. In the 80 current rows: 67 call the direct/public SP name (including resolved `native`, `entry`, `original` aliases), three call epsilon raw (one is the public wrapper), two call `quantize_sp2_keys`, and eight call diagnostics/delta helpers. The full commands/output and every discovered Python call's positional/keyword expressions are pasted below; `CALLER_CENSUS_FINAL.json` retains machine-readable rows. This is source inspection, not runtime reachability or a guarantee about dynamically constructed names.

The two active quantizer consumers and the public callers require specific treatment:

| Site | What it passes / uses | Successor action |
|---|---|---|
| `scripts/apa_pidv.py:269,303` | Quantizes uploaded fp32 K at **2 bits**, immediately downloads Kq for layout/census; public call passes `q,k,self.kq_t,v_t,self.scale,epsilon=.05,bulk_bits=2,margins=self.margins,is_causal=caus`. Its margin table is a synthetic instrument, not production calibration. | Preserve dense 2-bit fallback and its instrument behavior; do not cite this synthetic table as packed calibration. |
| `scripts/apa_sp2_gpu.py:53,163,166` | Quantizes fp32 `kt` at variable 2/4 bits; public calls expand `q,k,kq,v,scale` plus epsilon, bits, supplied registered margins and causal flag; diagnostic call adds `diagnostics=True`. It also sends Kq to raw score/baseline diagnostics and direct-delta controls at `:73,169,171,179`. | Keep old dense-calibration control explicit with `storage="dense"`; add a separately registered packed-calibration lane. A packed handle must not be passed into a binding requiring a Tensor. |
| `tensor_cuda/tests/test_apa_sp2.py:175,178,181` | Public wrapper tests use repeated fake q tensors, scale .125, epsilon .001, bits2, valid/None margins, and disabled-entry negative control. `:125-126,146` cover direct and epsilon raw APIs. | Retain these old-governance/negative cases and add packed-governance cases. |
| `tensor_cuda/tensor_cuda/apa_sp2.py:88` | Forwards all operands, converted scale/epsilon, looked-up e_q, causal/sinks/diagnostics to epsilon raw. | Preserve this semantic adapter; dispatch to backend-matching epsilon logic, never bypass margin lookup. |
| Current raw model/capture/metric harnesses | Full per-site argument expressions are in the pasted table. They explicitly pass dense Kq and **delta**, often `MAX_DELTA` or an experiment's delta, with causal/sink/diagnostic controls. Aliases in `apa_sp3_a4_capture.py:123,125,136` and `apa_sp4g_a3_sweep.py:17,19,26` are included. | Model cache owners selected for production migrate to packed handles by default; exact-arithmetic and mutation controls retain explicit dense operands. Preserve each delta/carry/causal/sink meaning. |
| GRAPA `artifacts/bp_census_1/engine_patch/src/bindings.cpp:675,696,685,707,713` | Archived native forwarding of direct delta, epsilon/e_q, baseline diagnostics, score diagnostics and thresholds, respectively. Direct/epsilon forward q/k/kq/v `.data()`, scale and their existing control arguments. No current GRAPA application SP call was found. | Historical patch is evidence, not a production migration target. Do not edit this other repo in the default-switch round without new authorization. |
| GraftRepositoryMemory / Project-Voice | No names matched in executable source at the named roots. | No matching caller migration identified; re-run the census when implementing. |

Native chain sites, declarations and packed forwarding (these are not extra application callers):

| File:line | Arguments / forwarding |
|---|---|
| `tensor_cuda/src/bindings.cpp:2342` | Direct raw forwards q/k/kq/v `.data()`, scale, delta, causal, optional sinks and selected-mask pointer. |
| `tensor_cuda/src/bindings.cpp:2363` | Epsilon raw forwards the same tensors plus epsilon and e_q. |
| `tensor_cuda/src/apa_sp2.cuh:20-21` | Epsilon entry derives upward-rounded delta then calls direct SP with q/k/kq/v unchanged. |
| `tensor_cuda/src/kernels.cu:8418-8458` | Direct entry validates one CUDA device/dtype, shapes/scalars and bottom-right causal layout, allocates output/selected; L1 dispatches split-K and other L uses the original serial SP dispatch. |
| `tensor_cuda/tensor_cuda/__init__.py:1586-1593,1604-1611` | Packed wrappers unwrap workspace after format/visible-row checks and forward q/k/v, packed tuple, scale, delta, causal/sinks/diagnostics, fmt, carry and gqa_path (the latter in SPQ4). |
| `tensor_cuda/src/apa_spo/packed_launcher.cuh:45-50` | Packed route validates tuple storage, rotates q, calls packed core with exact k supplied as the placeholder kq and with explicit packed/code-scale/rotated-q/codebook operands. No dense reconstructed Kq is required by that route. |
| `tensor_cuda/src/apa_spo/spq4.cuh:448-462` | `r2` uses packed generic; `tc` validates/dispatches INT4 or TQ4 variant-B tensor-core paths. These choices are independent of public epsilon governance. |
| `tensor_cuda/include/tc/ops.h:14-28,316-326` | Existing declarations; signatures distinguish dense Kq from packed tuples/internal component arrays. |

## (c) Proposed API and default behavior

Keep the attention signature above. In the implementing round, extend the quantizer to `quantize_sp2_keys(k, bulk_bits, *, storage=None)`. `None` reads a single opt-out policy `TC_APA_SP_STORAGE`, whose shipped value/default is `packed`; explicit `storage="dense"` or `TC_APA_SP_STORAGE=dense` requests the old representation. Build/distribution configuration ships `TC_APA_PACKED=ON`; callers need no separate packed-feature switch. This proposal itself changes none of those settings.

For eligible four-bit inputs, return an owned `PackedSpKeys` object, not an ambiguously typed tuple pretending to be a Tensor. It contains exact-K generation/append identity, device ordinal, dtype, B/KV/D, capacity/visible rows, quantizer/table provenance, codes, norms, rotation/codebook and carry/backend identity. It owns buffers and append lifetime; table keys include the full device ordinal. Append is explicit, validated, and invalidates/rebuilds after non-append edits; stale keys, wrong devices/geometry, or wrong margins fail before launch. Device/stream guards must also be supplied for packing/rotation/append and V preparation in that later round: this round hardens EB3/EB4 ticket launches only. The current read-only preparation helpers' ambient-device assumptions are not newly certified.

The public wrapper recognizes a packed object, checks matching `bulk_bits`, exact-K provenance and calibration identity, derives delta from the correct packed margin, and selects the supported backend. Legacy dense Tensor Kq retains the old semantics and table; it is an explicit dense representation supplied by that caller. Do not silently regenerate Kq from K, keep both persistent representations, or ignore an unrecognized Kq. Existing raw `_C` Tensor entry points remain valid for explicit legacy/diagnostic controls; add clearly typed packed raw adapters for delta and epsilon, and migrate **every production SP cache builder** to those objects. Leaving the numerous raw model adapters on a dense cache and merely changing the public wrapper would not complete the default switch. The implementing order must authorize those caller changes; this round does not.

Backend/fallback order:

1. **Packed SP without tk8 is the memory default.** On bf16 L1 D=VD128, registered GQA groups 2..8 and valid compact storage, use EB3 with the registered matching arithmetic/carry policy. Existing public fp32 inputs must use the ordered packed-r2 path, not a cast into EB3. The generic packed path accepts fp32/bf16 D64/D128 decode. All must first acquire their own epsilon calibration; absence of calibration selects a governed dense fallback where the old table is valid, otherwise rejects the governed call.
2. **Unsupported shape or dtype:** two-bit bulk, fp16 raw calls, L>1 prefill, unsupported head/value dimensions/groups or missing packed build support use the unchanged dense SP route only where that old API supports them. Preserve the exact old Kq/margin/scale and causal/sink behavior. Packed-owned state may explicitly materialize a transient dense reconstruction for a supported fallback, with backend-specific calibration; it is released after the call and is not a second persistent cache. Until exact dense reconstruction and calibration parity exist, select dense preparation up front for that route. Shape fallback is not permission to invent a new quantizer or reuse a mismatched table.
3. **No silent model-policy change:** retain each model's eligibility mask, exact layer 0, epsilon/delta and head/causal policy. V stays full precision by default. EB4/tk8 remains an independently configured option because it stores an additional int8 V copy/scales and has a separate quality rule. L>1 Triton K1 from AB-1 is lab evidence; production does not acquire that backend by changing this default.
4. **Observability and opt-out:** expose backend/storage/fallback reason through existing diagnostics or an internal receipt, without a new attention output schema or routine user prompt. `TC_APA_SP_STORAGE=dense` and `storage="dense"` reproduce the retained dense route. Unsupported calls must not advertise packed memory savings. This remains a single legacy-default-stream contract, without graph capture or multi-stream support.

Storage arithmetic (D=VD128, compact live S, excluding tables, scratch, alignment/capacity slack and allocator reservation):

| Representation | Exact K | Exact V | Bulk codes/Kq + norm | Extra V int8/scales | B/key/KV head |
|---|---:|---:|---:|---:|---:|
| Dense raw SP bf16 | 256 | 256 | 256 | 0 | 768 |
| Packed SP bf16 / 2P-INT4 layout | 256 | 256 | 64+4 | 0 | 580 |
| SP-tk8 bf16 | 256 | 256 | 64+4 | 128+4 | 712 |
| Exact bf16 K/V comparator | 256 | 256 | 0 | 0 | 512 |
| Existing governed dense fp32 SP | 512 | 512 | 512 | 0 | 1536 |
| Proposed governed packed fp32 SP | 512 | 512 | 64+4 | 0 | 1092 |

The 580-byte layout is the requested 2P-INT4 memory level; it is not FLASH's 512 bytes. The fp32 proposal has different bytes and requires measurement, not a claim of 580. AB-1 `scripts/apa_ab1_common.py:237-250` provides these bf16 layout identities (tk8 712); survey S1 gives raw-bf16 versus public-fp32 distinctions. The layer-0-exact model totals are 129,376 B/token for Qwen3-1.7B and 166,496 for Qwen3-4B in AB-1. A1's P151 ruling records exact packed recurrence bytes and 13 layout cases on the production port. These are existing finite receipts; they do not measure the proposed new governor or fallback cache behavior.

Evidence required before shipping:

| Existing evidence | What remains / measurement |
|---|---|
| AB-1 measured cache bytes equal the declared layout for both models; GPU A1 P151 measured port bytes/layout. | In the actual production default, census every live allocation and copied prefix, including preparation, fallback, capacity and decode growth; measure allocator reserved/allocated and own-PID VRAM peaks/slope with a known-allocation instrument. Prove no persistent dense Kq beside codes. Check default and opt-out on the same inputs/session. |
| EB4 lead checks: T-fresh/EB2b bitwise output+flags on 24 real cases, forced-bf16/EB3 on eight, exact-tie mutant deterministically selects20. EB3 lead checks include B*KVH=1040 growth and chunk-count reuse. | Lead-run this round's actual concurrency and deterministic negative controls, additional device/append/fallback boundaries, and the forthcoming blind red team. Finite parity does not establish unseen model quality. |
| AB-1 decode next-token metrics on Qwen3-1.7B/4B, one corpus, 2,560 scored decode tokens per arm/model, eight free-run prompts; SP/tk8 near fp32-floor metrics. | Re-run engagement, per-token teacher-forced NLL/PPL/KL/top-1 plus free-run, prefill/decode latency and peak-memory on default versus retained dense and exact references at production operating points, with registered thresholds and attribution. Add unseen distributions, task/long-context checks as required by product scope. Neither a fixture nor SASS equality substitutes for this model gate. |
| AB-1 G1 RED (1.7B top-1 and chunking attribution) and G2 RED (torch allocator slope) remain RED; 4B32K reference OOM. | Retain those labels. Separately register a corrected/sufficient measurement when closing a gap; use a larger card for the unavailable reference point rather than inventing its result. No blanket certification from AB-1. |
| Old fp32 dense measured margin table. | Register and freeze packed margins by quantizer, dtype, dimension, score backend, scale and carry; measure **all valid bulk/exact score errors** in the registered corpus/geometry, hold out calibration rows, and retain distribution limits. Check upward rounding and epsilon implication with boundary fixtures. Do not select a new margin after observing a quality failure. |

Prior art for this proposal: TurboQuant (Zandieh et al., 2025) supplies packed rotated Lloyd-Max codes; BLASST (Yuan et al., 2025) supplies the inherited running-max criterion; Flash-Decoding (2023) supplies split/merge; KIVI (Liu et al., 2024) and ThriftAttention (2026) supply the optional EB4 value-precision lineage; existing Project-Tensor SP2/workspace APIs (2026) supply governance/append ownership. The conditional epsilon derivation is the inherited two-error triangle-inequality argument from SP2, not a new theorem or an unconditional error certificate. Taken: those mechanisms; proposed here: a typed default-storage adapter and explicit caller migration. No prior art known to me for this exact adapter assembly. Literature metadata not verified in this CPU order; lead search terms: “TurboQuant 2504.19874”, “BLASST 2512.12087”, “Flash-Decoding 2023”, “KIVI 2402.02750”, “ThriftAttention 2605.23081”.

## (d) Successor tests and staged GPU commands

The implementing round must freeze its own thresholds and build/source hashes first. CPU checks cover: shipped default/build configuration, both quantizer result kinds, storage opt-out precedence, full caller migration census, margin identity/epsilon rounding, stale append/device/type/shape rejection, unsupported-route dispatch, diagnostic schema, no persistent dense-Kq ownership, and exact existing per-ID outcome comparison. Existing guard/range errors stay errors. Negative controls must detect epsilon-as-delta, old-table reuse on packed arithmetic, silently changing two bits to four, stale packed keys, wrong causal offset/sink handling and retaining dense Kq in the nominal packed cache.

GPU coverage includes: default versus explicit packed/explicit dense on the same tensors; new margin selection masks; native fp32 packed-r2 and separately calibrated bf16 EB3; 2-bit/fp16/D64/L>1 fallbacks; all used GQA groups/value dimensions; B>1 and odd S; S=255/256/257 and 64-key boundaries; append/reuse/growth; sink/causal controls; device switching and two-host-thread legacy-stream concurrency; the present early-counter/tie negative controls; actual per-layer cache/temporary/peak bytes and model quality. Preserve S3's full-width, nonzero/nonconstant bitwise references. Carry policy and arithmetic changes need separate quality evidence; bitwise equality is required only where the registered same-arithmetic contract calls for it.

Illustrative command forms for that future round (not current runnable lines, not claimed dry-run):

```sh
forge-gpu-run --min 30 --max 590 -- timeout 560s python -B artifacts/apa_p1_default/run.py api-default-and-optout
forge-gpu-run --min 30 --max 590 -- timeout 560s python -B artifacts/apa_p1_default/run.py fallback-and-margin-controls
forge-gpu-run --min 30 --max 590 -- timeout 560s python -B artifacts/apa_p1_default/run.py append-device-and-negative-controls
forge-gpu-run --min 30 --max 590 -- timeout 560s python -B artifacts/apa_p1_default/run.py bytes-and-peak
forge-gpu-run --min 30 --max 590 -- timeout 560s python -B artifacts/apa_p1_default/run.py model-quality-shard --registered-shard ID
```

Model-quality shards must be pre-registered once-only units sized to each timeout, with a frozen aggregate scorer. The successor must replace every placeholder, supply the actual driver/build, register each mutant's named expected assertion, and dry-run every final line to78. The six **actual current-round** lines are `STAGED_GPU_LINES.sh` with `EXPECTED.json` and `DRYRUN.json`; no GPU execution was done here. After the implementing round, the lead arranges the blind cross-model red team over both patches, then handles promotion.

## Full caller arguments and raw search receipts

The following tables and command outputs are pasted from the census artifacts. Reference snapshots are explicitly labeled and must not be mistaken for active application routing.

| Repository / class | File:line | Entry | Arguments passed |
|---|---|---|---|
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_mf2_1/before/pkg/tensor_cuda/apa_sp2.py:88` | `apa_selective_attention_sp_epsilon` | `_C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale), float(epsilon), e_q, is_causal, sinks, diagnostics)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_mf2_2b/before/pkg/tensor_cuda/apa_sp2.py:88` | `apa_selective_attention_sp_epsilon` | `_C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale), float(epsilon), e_q, is_causal, sinks, diagnostics)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_mf2_2c/before/pkg/tensor_cuda/apa_sp2.py:88` | `apa_selective_attention_sp_epsilon` | `_C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale), float(epsilon), e_q, is_causal, sinks, diagnostics)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_p1/refs/lab/tensor_cuda/apa_sp2.py:88` | `apa_selective_attention_sp_epsilon` | `_C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale), float(epsilon), e_q, is_causal, sinks, diagnostics)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_p1/refs/production/tensor_cuda/apa_sp2.py:88` | `apa_selective_attention_sp_epsilon` | `_C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale), float(epsilon), e_q, is_causal, sinks, diagnostics)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_p1_a1/refs/lab/tensor_cuda/apa_sp2.py:88` | `apa_selective_attention_sp_epsilon` | `_C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale), float(epsilon), e_q, is_causal, sinks, diagnostics)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_p1_a1/refs/port/tensor_cuda/apa_sp2.py:88` | `apa_selective_attention_sp_epsilon` | `_C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale), float(epsilon), e_q, is_causal, sinks, diagnostics)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_p1_a1/refs/production/tensor_cuda/apa_sp2.py:88` | `apa_selective_attention_sp_epsilon` | `_C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale), float(epsilon), e_q, is_causal, sinks, diagnostics)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_p1_s3_r2/refs/after/tensor_cuda/apa_sp2.py:88` | `apa_selective_attention_sp_epsilon` | `_C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale), float(epsilon), e_q, is_causal, sinks, diagnostics)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_p1_s3_r2/refs/before/tensor_cuda/apa_sp2.py:88` | `apa_selective_attention_sp_epsilon` | `_C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale), float(epsilon), e_q, is_causal, sinks, diagnostics)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp1/sp1_1_before/scripts__apa_sp1_gpu.py:142` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(*tensors, scale, delta, causal)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp1/sp1_1_before/scripts__apa_sp1_gpu.py:145` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(*tensors, scale, delta, causal, diagnostics=True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp1/sp1_1_before/scripts__apa_sp1_gpu.py:212` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(*tt, 1 / np.sqrt(D), 0.5, True, sinks)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_baseline/scripts/apa_sp3_gpu.py:179` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 96 ** (-0.5), float(np.finfo(np.float32).max), True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_baseline/scripts/apa_sp3_gpu.py:194` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 96 ** (-0.5), 0.125, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_baseline/scripts/apa_sp3_model.py:249` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, causal, None, self.observe)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_baseline/tensor_cuda/tests/test_apa_sp3.py:76` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(t, k, k, k, 96 ** (-0.5), 1.0)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_baseline/tensor_cuda/tests/test_apa_sp3.py:79` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(t, k, k, k, 96 ** (-0.5), 1.0)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_baseline/tensor_cuda/tests/test_apa_sp3.py:81` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(t, k, k, k, 96 ** (-0.5), float('inf'))` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_final_verification/a4_mutations/ceiling_fit_bypass.py:79` | `apa_selective_attention_sp` | `model.tc._C.apa_selective_attention_sp(q.slice(2, 0, 128), k.slice(2, 0, 128), kq.slice(2, 0, 128), v.slice(2, 0, 128), scale, model.delta, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_final_verification/a4_mutations/mask_bitorder.py:125` | `apa_selective_attention_sp` | `native(q, k, kq, v, scale, self.delta, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_final_verification/a4_mutations/mask_bitorder.py:136` | `apa_selective_attention_sp` | `native(*args, scale, self.delta, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_final_verification/a4_mutations/native_parity_bypass.py:125` | `apa_selective_attention_sp` | `native(q, k, kq, v, scale, self.delta, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_final_verification/a4_mutations/native_parity_bypass.py:136` | `apa_selective_attention_sp` | `native(*args, scale, self.delta, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_final_verification/a4_mutations/no_file_integrity.py:125` | `apa_selective_attention_sp` | `native(q, k, kq, v, scale, self.delta, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_final_verification/a4_mutations/no_file_integrity.py:136` | `apa_selective_attention_sp` | `native(*args, scale, self.delta, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_final_verification/a4_mutations/range_row_skip.py:125` | `apa_selective_attention_sp` | `native(q, k, kq, v, scale, self.delta, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_final_verification/a4_mutations/range_row_skip.py:136` | `apa_selective_attention_sp` | `native(*args, scale, self.delta, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_mutations/ceiling_fit_bypass.py:79` | `apa_selective_attention_sp` | `model.tc._C.apa_selective_attention_sp(q.slice(2, 0, 128), k.slice(2, 0, 128), kq.slice(2, 0, 128), v.slice(2, 0, 128), scale, model.delta, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_mutations/mask_bitorder.py:125` | `apa_selective_attention_sp` | `native(q, k, kq, v, scale, self.delta, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_mutations/mask_bitorder.py:136` | `apa_selective_attention_sp` | `native(*args, scale, self.delta, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_mutations/native_parity_bypass.py:125` | `apa_selective_attention_sp` | `native(q, k, kq, v, scale, self.delta, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_mutations/native_parity_bypass.py:136` | `apa_selective_attention_sp` | `native(*args, scale, self.delta, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_mutations/no_file_integrity.py:125` | `apa_selective_attention_sp` | `native(q, k, kq, v, scale, self.delta, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_mutations/no_file_integrity.py:136` | `apa_selective_attention_sp` | `native(*args, scale, self.delta, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_mutations/range_row_skip.py:125` | `apa_selective_attention_sp` | `native(q, k, kq, v, scale, self.delta, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a4_mutations/range_row_skip.py:136` | `apa_selective_attention_sp` | `native(*args, scale, self.delta, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a5_baseline/scripts/apa_sp3_a4_capture.py:125` | `apa_selective_attention_sp` | `native(q, k, kq, v, scale, self.delta, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a5_baseline/scripts/apa_sp3_a4_capture.py:136` | `apa_selective_attention_sp` | `native(*args, scale, self.delta, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a5_baseline/scripts/apa_sp3_a4_jobs.py:79` | `apa_selective_attention_sp` | `model.tc._C.apa_selective_attention_sp(q.slice(2, 0, 128), k.slice(2, 0, 128), kq.slice(2, 0, 128), v.slice(2, 0, 128), scale, model.delta, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a5_baseline/scripts/apa_sp3_gpu.py:184` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 96 ** (-0.5), float(np.finfo(np.float32).max), True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a5_baseline/scripts/apa_sp3_gpu.py:199` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 96 ** (-0.5), 0.125, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a5_baseline/scripts/apa_sp3_model.py:249` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, causal, None, self.observe)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a5_baseline/tensor_cuda/tests/test_apa_sp3.py:76` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(t, k, k, k, 96 ** (-0.5), 1.0)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a5_baseline/tensor_cuda/tests/test_apa_sp3.py:79` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(t, k, k, k, 96 ** (-0.5), 1.0)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a5_baseline/tensor_cuda/tests/test_apa_sp3.py:81` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(t, k, k, k, 96 ** (-0.5), float('inf'))` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a5_mutations/shorten_measured_steps.py:249` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, causal, None, self.observe)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_baseline/scripts/apa_sp3_a4_capture.py:125` | `apa_selective_attention_sp` | `native(q, k, kq, v, scale, self.delta, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_baseline/scripts/apa_sp3_a4_capture.py:136` | `apa_selective_attention_sp` | `native(*args, scale, self.delta, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_baseline/scripts/apa_sp3_a4_jobs.py:79` | `apa_selective_attention_sp` | `model.tc._C.apa_selective_attention_sp(q.slice(2, 0, 128), k.slice(2, 0, 128), kq.slice(2, 0, 128), v.slice(2, 0, 128), scale, model.delta, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_baseline/scripts/apa_sp3_gpu.py:185` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 96 ** (-0.5), float(np.finfo(np.float32).max), True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_baseline/scripts/apa_sp3_gpu.py:200` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 96 ** (-0.5), 0.125, True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_baseline/scripts/apa_sp3_model.py:249` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, causal, None, self.observe)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_baseline/tensor_cuda/tests/test_apa_sp3.py:76` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(t, k, k, k, 96 ** (-0.5), 1.0)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_baseline/tensor_cuda/tests/test_apa_sp3.py:79` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(t, k, k, k, 96 ** (-0.5), 1.0)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_baseline/tensor_cuda/tests/test_apa_sp3.py:81` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(t, k, k, k, 96 ** (-0.5), float('inf'))` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_mutations/drop_cache.py:88` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_mutations/force_full_copy.py:88` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_mutations/force_layer_hook.py:88` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_mutations/pool_after_load.py:88` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_mutations/rail_boundary.py:88` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/a6_mutations/shorten_steps.py:88` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/capture_dtype_mutation_r3/mutations/final/off_by_one_targets.py:249` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, causal, None, self.observe)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/capture_dtype_mutation_r3/mutations/final/swapped_scale_delta.py:249` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, self.delta, scale, causal, None, self.observe)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/mutations/final/off_by_one_targets.py:221` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, causal, None, self.observe)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/mutations/final/swapped_scale_delta.py:221` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, self.delta, scale, causal, None, self.observe)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/mutations/off_by_one_targets.py:213` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, causal, None, self.observe)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/mutations/protocol2/off_by_one_targets.py:246` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, causal, None, self.observe)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/mutations/protocol2/swapped_scale_delta.py:246` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, self.delta, scale, causal, None, self.observe)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp3/mutations/swapped_scale_delta.py:213` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, self.delta, scale, causal, None, self.observe)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a1_baseline/scripts/apa_sp4g_gpu.py:33` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a1_baseline/scripts/apa_sp4g_gpu.py:34` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a1_baseline/scripts/apa_sp4g_metrics.py:49` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a1_baseline/scripts/apa_sp4g_model.py:95` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a1_baseline/scripts/apa_sp4g_model.py:108` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a1_baseline/scripts/apa_sp4g_model.py:109` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a1_baseline/tensor_cuda/tests/test_apa_sp4g.py:85` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, k, k, 1.0, 1.0)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a1_baseline/tensor_cuda/tests/test_apa_sp4g.py:87` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, k, k, 1.0, 1.0)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a1_baseline/tensor_cuda/tests/test_apa_sp4g.py:89` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, k, k, 1.0, delta)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a2_baseline/scripts/apa_sp4g_gpu.py:33` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a2_baseline/scripts/apa_sp4g_gpu.py:34` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a2_baseline/scripts/apa_sp4g_metrics.py:53` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a2_baseline/scripts/apa_sp4g_model.py:109` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a2_baseline/scripts/apa_sp4g_model.py:122` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a2_baseline/scripts/apa_sp4g_model.py:123` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a2_baseline/tensor_cuda/tests/test_apa_sp4g.py:86` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, k, k, 1.0, 1.0)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a2_baseline/tensor_cuda/tests/test_apa_sp4g.py:88` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, k, k, 1.0, 1.0)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/a2_baseline/tensor_cuda/tests/test_apa_sp4g.py:90` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, k, k, 1.0, delta)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations/margin_term.py:49` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations/percentile.py:49` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations/relaxed_match.py:33` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations/relaxed_match.py:34` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations/reverse_bracket.py:33` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations/reverse_bracket.py:34` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations/tail_denominator.py:49` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations/target_boundary.py:95` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations/target_boundary.py:108` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations/target_boundary.py:109` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations/wrong_target.py:95` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations/wrong_target.py:108` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations/wrong_target.py:109` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1/margin_term.py:53` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1/percentile.py:53` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1/relaxed_match.py:33` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1/relaxed_match.py:34` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1/reverse_bracket.py:33` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1/reverse_bracket.py:34` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1/tail_denominator.py:53` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1/target_boundary.py:109` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1/target_boundary.py:122` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1/target_boundary.py:123` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1/wrong_target.py:109` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1/wrong_target.py:122` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1/wrong_target.py:123` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1_final/margin_term.py:53` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1_final/percentile.py:53` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1_final/relaxed_match.py:33` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1_final/relaxed_match.py:34` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1_final/reverse_bracket.py:33` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1_final/reverse_bracket.py:34` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1_final/tail_denominator.py:53` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1_final/target_boundary.py:109` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1_final/target_boundary.py:122` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1_final/target_boundary.py:123` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1_final/wrong_target.py:109` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1_final/wrong_target.py:122` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a1_final/wrong_target.py:123` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/bf16_lowbits.py:88` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/bf16_lowbits.py:94` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/bf16_lowbits.py:96` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q.float(), k.float(), kq.float(), v.float(), scale, MAX_DELTA, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/bf16_lowbits.py:123` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/bf16_lowbits.py:124` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/fp32_arm.py:88` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/fp32_arm.py:94` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/fp32_arm.py:96` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q.float(), k.float(), kq.float(), v.float(), scale, MAX_DELTA, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/fp32_arm.py:123` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/fp32_arm.py:124` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/mask_ignored.py:91` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, rec['scale'], m['delta'], rec['causal'], None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/output_tolerance.py:91` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, rec['scale'], m['delta'], rec['causal'], None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/scale.py:88` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/scale.py:94` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/scale.py:96` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q.float(), k.float(), kq.float(), v.float(), scale, MAX_DELTA, True, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/scale.py:123` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a2/scale.py:124` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a3/native_tolerance.py:71` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a3/native_tolerance.py:72` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a3/refine_mask_ignored.py:71` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a3/refine_mask_ignored.py:72` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a3_final/native_tolerance.py:71` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a3_final/native_tolerance.py:72` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a3_final/refine_mask_ignored.py:71` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a3_final/refine_mask_ignored.py:72` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a4/cast_dropped.py:28` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a4/coverage_gap.py:28` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a4/input_dtype_ignored.py:28` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a4/output_dtype_ignored.py:28` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a4/return_wrong_arm.py:28` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a4_v2/cast_dropped.py:28` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a4_v2/coverage_gap.py:28` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a4_v2/input_dtype_ignored.py:28` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a4_v2/output_dtype_ignored.py:28` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a4_v2/return_wrong_arm.py:28` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a5/capture_wrong_layer.py:115` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a5/capture_wrong_layer.py:116` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a5/off_by_one.py:115` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a5/off_by_one.py:116` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a5/prefill_only.py:115` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a5/prefill_only.py:116` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a5/splitk_for_cached.py:115` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a5/splitk_for_cached.py:116` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a7/pool_off.py:109` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a7/pool_off.py:122` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_a7/pool_off.py:123` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_final/margin_term.py:49` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_final/percentile.py:49` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_final/relaxed_match.py:33` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_final/relaxed_match.py:34` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_final/reverse_bracket.py:33` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_final/reverse_bracket.py:34` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_final/tail_denominator.py:49` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_final/target_boundary.py:95` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_final/target_boundary.py:108` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_final/target_boundary.py:109` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_final/wrong_target.py:95` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_final/wrong_target.py:108` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_final/wrong_target.py:109` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_sealed/margin_term.py:49` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_sealed/percentile.py:49` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_sealed/relaxed_match.py:33` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_sealed/relaxed_match.py:34` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_sealed/reverse_bracket.py:33` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_sealed/reverse_bracket.py:34` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_sealed/tail_denominator.py:49` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_sealed/target_boundary.py:95` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_sealed/target_boundary.py:108` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_sealed/target_boundary.py:109` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_sealed/wrong_target.py:95` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_sealed/wrong_target.py:108` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp4g/mutations_sealed/wrong_target.py:109` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_sp5/a2_inject/sitecustomize.py:107` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, self.scaling, delta, True, self.sinks, False)` |
| Project-Tensor / artifact/reference/negative-control code | `artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/apa_sp2.py:88` | `apa_selective_attention_sp_epsilon` | `_C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale), float(epsilon), e_q, is_causal, sinks, diagnostics)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_pidv.py:269` | `quantize_sp2_keys` | `tc.quantize_sp2_keys(up(self.k, 'float32'), 2)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_pidv.py:303` | `apa_selective_attention_sp` | `tc.apa_selective_attention_sp(q, k, self.kq_t, v_t, self.scale, 0.05, bulk_bits=2, margins=self.margins, is_causal=caus)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp1_1_gpu.py:27` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(*tt, 1 / np.sqrt(D), 0.5, True, sinks, diagnostics=True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp1_1_gpu.py:29` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(*tt, 1 / np.sqrt(D), 0.5, True, sinks)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp1_1_gpu.py:46` | `apa_sp1_1_baseline_diagnostics` | `tc._C.apa_sp1_1_baseline_diagnostics(tt[0], tt[2])` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp1_gpu.py:145` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(*tensors, scale, delta, causal)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp1_gpu.py:148` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(*tensors, scale, delta, causal, diagnostics=True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp1_gpu.py:217` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(*tt, 1 / np.sqrt(D), 0.5, True, sinks)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp2_gpu.py:53` | `quantize_sp2_keys` | `tc.quantize_sp2_keys(kt, bits)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp2_gpu.py:73` | `apa_sp2_scores` | `tc._C.apa_sp2_scores(qt, kt, kqt, float(scale), 0, len(indices), shape['kind'] == 'prefill' or shape['D'] > 64)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp2_gpu.py:159` | `apa_sp2_delta` | `tc._C.apa_sp2_delta(epsilon, e_q)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp2_gpu.py:163` | `apa_selective_attention_sp` | `tc.apa_selective_attention_sp(*args, epsilon, bulk_bits=bits, margins=margins, is_causal=s['causal'])` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp2_gpu.py:166` | `apa_selective_attention_sp` | `tc.apa_selective_attention_sp(*args, epsilon, bulk_bits=bits, margins=margins, is_causal=s['causal'], diagnostics=True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp2_gpu.py:169` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(*args, delta, s['causal'])` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp2_gpu.py:171` | `apa_sp2_baseline_thresholds` | `tc._C.apa_sp2_baseline_thresholds(qt, kqt, float(scale), reg['baseline']['zthr'], s['causal'])` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp2_gpu.py:179` | `apa_sp2_scores` | `tc._C.apa_sp2_scores(qt, kt, kqt, float(scale), first, n, wc)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp3_a4_capture.py:125` | `apa_selective_attention_sp` | `native(q, k, kq, v, scale, self.delta, True, None, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp3_a4_capture.py:136` | `apa_selective_attention_sp` | `native(*args, scale, self.delta, True, None, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp3_a4_jobs.py:79` | `apa_selective_attention_sp` | `model.tc._C.apa_selective_attention_sp(q.slice(2, 0, 128), k.slice(2, 0, 128), kq.slice(2, 0, 128), v.slice(2, 0, 128), scale, model.delta, True, None, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp3_a6_decode.py:88` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp3_gpu.py:186` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 96 ** (-0.5), float(np.finfo(np.float32).max), True, None, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp3_gpu.py:201` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 96 ** (-0.5), 0.125, True, None, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp3_model.py:249` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, causal, None, self.observe)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a2_metrics.py:91` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, rec['scale'], m['delta'], rec['causal'], None, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a2_model.py:88` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, True, None, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a2_model.py:94` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, True, None, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a2_model.py:96` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q.float(), k.float(), kq.float(), v.float(), scale, MAX_DELTA, True, None, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a2_model.py:123` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a2_model.py:124` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a3_model.py:71` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a3_model.py:72` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a3_sweep.py:19` | `apa_selective_attention_sp` | `entry(q, k, kq, v, variant['scale'], MAX_DELTA, variant['causal'], sink, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a3_sweep.py:26` | `apa_selective_attention_sp` | `entry(q.slice(2, i, 1), k.slice(2, 0, n), kq.slice(2, 0, n), v.slice(2, 0, n), variant['scale'], MAX_DELTA, False, sink, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a4_metrics.py:94` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, rec['scale'], m['delta'], rec['causal'], None, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a4_model.py:28` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qf, kf, kqf, vf, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a5_model.py:115` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a5_model.py:116` | `apa_selective_attention_sp` | `self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, MAX_DELTA, causal, None, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_a6_metrics.py:95` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, rec['scale'], m['delta'], rec['causal'], None, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_gpu.py:33` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_gpu.py:34` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1.0, float(np.finfo(np.float32).max), L > 1, None, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_metrics.py:53` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, 1.0, m['delta'], True, None, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_model.py:109` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, delta, is_causal, None, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_model.py:122` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp4g_model.py:123` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta, is_causal, None, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp5_a1_margins.py:132` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, self.scaling, float(args.delta), True, self.sinks, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp5_a1_margins.py:181` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, c['scale'], float(args.delta), True, st, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp5_frac.py:47` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, self.scaling, float(args.delta), True, self.sinks, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp5_item2_sinkpin.py:125` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, float(dl), True, st, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp5_item2_sinkpin.py:127` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, float(dl), True, z0, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp5_item3_exactness.py:147` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, 1000000000.0, True, st, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp5_model.py:125` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, self.scaling, float(self._sp_delta), True, self.sinks, want_sel)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp5_shape_pin.py:117` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, big, True, st, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp5_shape_pin.py:128` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, big, True, zero, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp5_shape_pin.py:129` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, big, True, None, False)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_sp5_shape_pin.py:146` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, delta, True, st, True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_spd1_bench.py:141` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, kq, v, 1 / math.sqrt(shape['D']), shape['delta'], shape['causal'], diagnostics=True)` |
| Project-Tensor / current source/test/harness call | `scripts/apa_spd1_bench.py:227` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(*t, scale, shape['delta'], shape['causal'])` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tensor_cuda/apa_sp2.py:88` | `apa_selective_attention_sp_epsilon` | `_C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale), float(epsilon), e_q, is_causal, sinks, diagnostics)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp1_host.py:30` | `apa_selective_attention_sp` | `runtime._C.apa_selective_attention_sp(t, t, t, t, 1.0, 0.0)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp1_host.py:38` | `apa_selective_attention_sp` | `runtime._C.apa_selective_attention_sp(t, t, t, t, scale, delta)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp1_host.py:45` | `apa_selective_attention_sp` | `runtime._C.apa_selective_attention_sp(t, t, t, t, 1.0, 0.0)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp1_host.py:48` | `apa_selective_attention_sp` | `runtime._C.apa_selective_attention_sp(t, t, t, t, 1.0, 0.0)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp1_host.py:55` | `apa_selective_attention_sp` | `runtime._C.apa_selective_attention_sp(t, t, t, t, 1.0, 0.0)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp2.py:111` | `apa_sp2_delta` | `tc._C.apa_sp2_delta(epsilon, e)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp2.py:115` | `apa_sp2_delta` | `tc._C.apa_sp2_delta(eps, e)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp2.py:123` | `apa_sp2_delta` | `tc._C.apa_sp2_delta(epsilon, e)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp2.py:125` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(*args, delta)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp2.py:126` | `apa_selective_attention_sp_epsilon` | `tc._C.apa_selective_attention_sp_epsilon(*args, epsilon, e)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp2.py:146` | `apa_selective_attention_sp_epsilon` | `tc._C.apa_selective_attention_sp_epsilon(*args, 0.1, 0.5)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp2.py:175` | `apa_selective_attention_sp` | `tc.apa_selective_attention_sp(q, q, q, q, 0.125, 0.001, bulk_bits=2, margins=margins)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp2.py:178` | `apa_selective_attention_sp` | `tc.apa_selective_attention_sp(q, q, q, q, 0.125, 0.001, bulk_bits=2, margins=None)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp2.py:181` | `apa_selective_attention_sp` | `tc.apa_selective_attention_sp(q, q, q, q, 0.125, 0.001, bulk_bits=2, margins=margins)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp3.py:76` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(t, k, k, k, 96 ** (-0.5), 1.0)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp3.py:79` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(t, k, k, k, 96 ** (-0.5), 1.0)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp3.py:81` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(t, k, k, k, 96 ** (-0.5), float('inf'))` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp4g.py:86` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, k, k, 1.0, 1.0)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp4g.py:88` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, k, k, 1.0, 1.0)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp4g.py:90` | `apa_selective_attention_sp` | `tc._C.apa_selective_attention_sp(q, k, k, k, 1.0, delta)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp4g_a3.py:135` | `apa_selective_attention_sp` | `original(*args)` |
| Project-Tensor / current source/test/harness call | `tensor_cuda/tests/test_apa_sp4g_a3.py:205` | `apa_selective_attention_sp` | `original(*args)` |

### Project-Tensor

```sh
CUDA_VISIBLE_DEVICES= timeout 40s rg -n --sort path --hidden --no-ignore -g '*.{py,pyi,ipynb,cpp,cu,cuh,h,c,sh,js,ts,rs}' -g '!**/.git/**' -g '!**/__pycache__/**' -g '!**/.venv/**' -g '!**/node_modules/**' -g '!**/build*/**' -g '!**/Fork/**' -g '!**/WorkTree/**' '\b(apa_selective_attention_sp(_epsilon|_d1_packed|_d1)?|quantize_sp2_keys|apa_spq1_(dump|rotate)|apa_spo2_dump|apa_sp2_(delta|scores|baseline_thresholds)|apa_sp1_1_baseline_diagnostics)\b' .
```

Exit 0. Full output:

```text
./artifacts/apa_mf2_1/before/files/tensor_cuda__src__kernels.cu:7947:NDArray apa_selective_attention_sp(const NDArray& q, const NDArray& k,
./artifacts/apa_mf2_1/before/pkg/tensor_cuda/__init__.py:1287:from .apa_sp2 import RegisteredMargins, apa_selective_attention_sp, quantize_sp2_keys
./artifacts/apa_mf2_1/before/pkg/tensor_cuda/__init__.py:1288:__all__ += ["RegisteredMargins", "apa_selective_attention_sp", "quantize_sp2_keys"]
./artifacts/apa_mf2_1/before/pkg/tensor_cuda/apa_sp2.py:4:Callers must supply keys from quantize_sp2_keys and the calibrated input regime.
./artifacts/apa_mf2_1/before/pkg/tensor_cuda/apa_sp2.py:61:def quantize_sp2_keys(k, bulk_bits):
./artifacts/apa_mf2_1/before/pkg/tensor_cuda/apa_sp2.py:72:def apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *, bulk_bits,
./artifacts/apa_mf2_1/before/pkg/tensor_cuda/apa_sp2.py:77:    kq must come from quantize_sp2_keys(k, bulk_bits). Calibration does not certify
./artifacts/apa_mf2_1/before/pkg/tensor_cuda/apa_sp2.py:88:    return _C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale),
./artifacts/apa_mf2_1/snap/item2/tensor_cuda__src__kernels.cu:7961:NDArray apa_selective_attention_sp(const NDArray& q, const NDArray& k,
./artifacts/apa_mf2_1/snap/item3/tensor_cuda__src__kernels.cu:7961:NDArray apa_selective_attention_sp(const NDArray& q, const NDArray& k,
./artifacts/apa_mf2_2/before/tensor_cuda__src__kernels.cu:7961:NDArray apa_selective_attention_sp(const NDArray& q, const NDArray& k,
./artifacts/apa_mf2_2a/before/tensor_cuda__src__kernels.cu:7980:NDArray apa_selective_attention_sp(const NDArray& q, const NDArray& k,
./artifacts/apa_mf2_2b/before/pkg/tensor_cuda/__init__.py:1287:from .apa_sp2 import RegisteredMargins, apa_selective_attention_sp, quantize_sp2_keys
./artifacts/apa_mf2_2b/before/pkg/tensor_cuda/__init__.py:1288:__all__ += ["RegisteredMargins", "apa_selective_attention_sp", "quantize_sp2_keys"]
./artifacts/apa_mf2_2b/before/pkg/tensor_cuda/apa_sp2.py:4:Callers must supply keys from quantize_sp2_keys and the calibrated input regime.
./artifacts/apa_mf2_2b/before/pkg/tensor_cuda/apa_sp2.py:61:def quantize_sp2_keys(k, bulk_bits):
./artifacts/apa_mf2_2b/before/pkg/tensor_cuda/apa_sp2.py:72:def apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *, bulk_bits,
./artifacts/apa_mf2_2b/before/pkg/tensor_cuda/apa_sp2.py:77:    kq must come from quantize_sp2_keys(k, bulk_bits). Calibration does not certify
./artifacts/apa_mf2_2b/before/pkg/tensor_cuda/apa_sp2.py:88:    return _C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale),
./artifacts/apa_mf2_2c/before/pkg/tensor_cuda/__init__.py:1287:from .apa_sp2 import RegisteredMargins, apa_selective_attention_sp, quantize_sp2_keys
./artifacts/apa_mf2_2c/before/pkg/tensor_cuda/__init__.py:1288:__all__ += ["RegisteredMargins", "apa_selective_attention_sp", "quantize_sp2_keys"]
./artifacts/apa_mf2_2c/before/pkg/tensor_cuda/apa_sp2.py:4:Callers must supply keys from quantize_sp2_keys and the calibrated input regime.
./artifacts/apa_mf2_2c/before/pkg/tensor_cuda/apa_sp2.py:61:def quantize_sp2_keys(k, bulk_bits):
./artifacts/apa_mf2_2c/before/pkg/tensor_cuda/apa_sp2.py:72:def apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *, bulk_bits,
./artifacts/apa_mf2_2c/before/pkg/tensor_cuda/apa_sp2.py:77:    kq must come from quantize_sp2_keys(k, bulk_bits). Calibration does not certify
./artifacts/apa_mf2_2c/before/pkg/tensor_cuda/apa_sp2.py:88:    return _C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale),
./artifacts/apa_p1/refs/lab/tensor_cuda/__init__.py:1287:from .apa_sp2 import RegisteredMargins, apa_selective_attention_sp, quantize_sp2_keys
./artifacts/apa_p1/refs/lab/tensor_cuda/__init__.py:1288:__all__ += ["RegisteredMargins", "apa_selective_attention_sp", "quantize_sp2_keys"]
./artifacts/apa_p1/refs/lab/tensor_cuda/__init__.py:1303:    _C.apa_selective_attention_sp_d1 = _apa_spo2_entry
./artifacts/apa_p1/refs/lab/tensor_cuda/__init__.py:1304:    _C.apa_spo2_dump = _spo2.run
./artifacts/apa_p1/refs/lab/tensor_cuda/__init__.py:1352:    _C.apa_selective_attention_sp_d1_packed=_apa_spq1_entry
./artifacts/apa_p1/refs/lab/tensor_cuda/__init__.py:1353:    _C.apa_spq1_dump=_spo2.packed_run
./artifacts/apa_p1/refs/lab/tensor_cuda/__init__.py:1354:    _C.apa_spq1_rotate=_spo2.packed_rotate
./artifacts/apa_p1/refs/lab/tensor_cuda/__init__.py:1367:    _C.apa_selective_attention_sp_d1_packed=_apa_spq4_entry
./artifacts/apa_p1/refs/lab/tensor_cuda/apa_sp2.py:4:Callers must supply keys from quantize_sp2_keys and the calibrated input regime.
./artifacts/apa_p1/refs/lab/tensor_cuda/apa_sp2.py:61:def quantize_sp2_keys(k, bulk_bits):
./artifacts/apa_p1/refs/lab/tensor_cuda/apa_sp2.py:72:def apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *, bulk_bits,
./artifacts/apa_p1/refs/lab/tensor_cuda/apa_sp2.py:77:    kq must come from quantize_sp2_keys(k, bulk_bits). Calibration does not certify
./artifacts/apa_p1/refs/lab/tensor_cuda/apa_sp2.py:88:    return _C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale),
./artifacts/apa_p1/refs/production/tensor_cuda/__init__.py:1536:from .apa_sp2 import RegisteredMargins, apa_selective_attention_sp, quantize_sp2_keys
./artifacts/apa_p1/refs/production/tensor_cuda/__init__.py:1537:__all__ += ["RegisteredMargins", "apa_selective_attention_sp", "quantize_sp2_keys"]
./artifacts/apa_p1/refs/production/tensor_cuda/__init__.py:1554:    _C.apa_selective_attention_sp_d1 = _apa_spo2_entry
./artifacts/apa_p1/refs/production/tensor_cuda/__init__.py:1555:    _C.apa_spo2_dump = _spo2.run
./artifacts/apa_p1/refs/production/tensor_cuda/__init__.py:1603:    _C.apa_selective_attention_sp_d1_packed=_apa_spq1_entry
./artifacts/apa_p1/refs/production/tensor_cuda/__init__.py:1604:    _C.apa_spq1_dump=_spo2.packed_run
./artifacts/apa_p1/refs/production/tensor_cuda/__init__.py:1605:    _C.apa_spq1_rotate=_spo2.packed_rotate
./artifacts/apa_p1/refs/production/tensor_cuda/__init__.py:1618:    _C.apa_selective_attention_sp_d1_packed=_apa_spq4_entry
./artifacts/apa_p1/refs/production/tensor_cuda/apa_sp2.py:4:Callers must supply keys from quantize_sp2_keys and the calibrated input regime.
./artifacts/apa_p1/refs/production/tensor_cuda/apa_sp2.py:61:def quantize_sp2_keys(k, bulk_bits):
./artifacts/apa_p1/refs/production/tensor_cuda/apa_sp2.py:72:def apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *, bulk_bits,
./artifacts/apa_p1/refs/production/tensor_cuda/apa_sp2.py:77:    kq must come from quantize_sp2_keys(k, bulk_bits). Calibration does not certify
./artifacts/apa_p1/refs/production/tensor_cuda/apa_sp2.py:88:    return _C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale),
./artifacts/apa_p1_a1/refs/lab/tensor_cuda/__init__.py:1287:from .apa_sp2 import RegisteredMargins, apa_selective_attention_sp, quantize_sp2_keys
./artifacts/apa_p1_a1/refs/lab/tensor_cuda/__init__.py:1288:__all__ += ["RegisteredMargins", "apa_selective_attention_sp", "quantize_sp2_keys"]
./artifacts/apa_p1_a1/refs/lab/tensor_cuda/__init__.py:1303:    _C.apa_selective_attention_sp_d1 = _apa_spo2_entry
./artifacts/apa_p1_a1/refs/lab/tensor_cuda/__init__.py:1304:    _C.apa_spo2_dump = _spo2.run
./artifacts/apa_p1_a1/refs/lab/tensor_cuda/__init__.py:1389:    _C.apa_selective_attention_sp_d1_packed=_apa_spq1_entry
./artifacts/apa_p1_a1/refs/lab/tensor_cuda/__init__.py:1390:    _C.apa_spq1_dump=_spo2.packed_run
./artifacts/apa_p1_a1/refs/lab/tensor_cuda/__init__.py:1391:    _C.apa_spq1_rotate=_spo2.packed_rotate
./artifacts/apa_p1_a1/refs/lab/tensor_cuda/__init__.py:1404:    _C.apa_selective_attention_sp_d1_packed=_apa_spq4_entry
./artifacts/apa_p1_a1/refs/lab/tensor_cuda/apa_sp2.py:4:Callers must supply keys from quantize_sp2_keys and the calibrated input regime.
./artifacts/apa_p1_a1/refs/lab/tensor_cuda/apa_sp2.py:61:def quantize_sp2_keys(k, bulk_bits):
./artifacts/apa_p1_a1/refs/lab/tensor_cuda/apa_sp2.py:72:def apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *, bulk_bits,
./artifacts/apa_p1_a1/refs/lab/tensor_cuda/apa_sp2.py:77:    kq must come from quantize_sp2_keys(k, bulk_bits). Calibration does not certify
./artifacts/apa_p1_a1/refs/lab/tensor_cuda/apa_sp2.py:88:    return _C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale),
./artifacts/apa_p1_a1/refs/port/tensor_cuda/__init__.py:1536:from .apa_sp2 import RegisteredMargins, apa_selective_attention_sp, quantize_sp2_keys
./artifacts/apa_p1_a1/refs/port/tensor_cuda/__init__.py:1537:__all__ += ["RegisteredMargins", "apa_selective_attention_sp", "quantize_sp2_keys"]
./artifacts/apa_p1_a1/refs/port/tensor_cuda/__init__.py:1554:    _C.apa_selective_attention_sp_d1 = _apa_spo2_entry
./artifacts/apa_p1_a1/refs/port/tensor_cuda/__init__.py:1555:    _C.apa_spo2_dump = _spo2.run
./artifacts/apa_p1_a1/refs/port/tensor_cuda/__init__.py:1603:    _C.apa_selective_attention_sp_d1_packed=_apa_spq1_entry
./artifacts/apa_p1_a1/refs/port/tensor_cuda/__init__.py:1604:    _C.apa_spq1_dump=_spo2.packed_run
./artifacts/apa_p1_a1/refs/port/tensor_cuda/__init__.py:1605:    _C.apa_spq1_rotate=_spo2.packed_rotate
./artifacts/apa_p1_a1/refs/port/tensor_cuda/__init__.py:1618:    _C.apa_selective_attention_sp_d1_packed=_apa_spq4_entry
./artifacts/apa_p1_a1/refs/port/tensor_cuda/apa_sp2.py:4:Callers must supply keys from quantize_sp2_keys and the calibrated input regime.
./artifacts/apa_p1_a1/refs/port/tensor_cuda/apa_sp2.py:61:def quantize_sp2_keys(k, bulk_bits):
./artifacts/apa_p1_a1/refs/port/tensor_cuda/apa_sp2.py:72:def apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *, bulk_bits,
./artifacts/apa_p1_a1/refs/port/tensor_cuda/apa_sp2.py:77:    kq must come from quantize_sp2_keys(k, bulk_bits). Calibration does not certify
./artifacts/apa_p1_a1/refs/port/tensor_cuda/apa_sp2.py:88:    return _C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale),
./artifacts/apa_p1_a1/refs/production/tensor_cuda/__init__.py:1536:from .apa_sp2 import RegisteredMargins, apa_selective_attention_sp, quantize_sp2_keys
./artifacts/apa_p1_a1/refs/production/tensor_cuda/__init__.py:1537:__all__ += ["RegisteredMargins", "apa_selective_attention_sp", "quantize_sp2_keys"]
./artifacts/apa_p1_a1/refs/production/tensor_cuda/__init__.py:1554:    _C.apa_selective_attention_sp_d1 = _apa_spo2_entry
./artifacts/apa_p1_a1/refs/production/tensor_cuda/__init__.py:1555:    _C.apa_spo2_dump = _spo2.run
./artifacts/apa_p1_a1/refs/production/tensor_cuda/__init__.py:1603:    _C.apa_selective_attention_sp_d1_packed=_apa_spq1_entry
./artifacts/apa_p1_a1/refs/production/tensor_cuda/__init__.py:1604:    _C.apa_spq1_dump=_spo2.packed_run
./artifacts/apa_p1_a1/refs/production/tensor_cuda/__init__.py:1605:    _C.apa_spq1_rotate=_spo2.packed_rotate
./artifacts/apa_p1_a1/refs/production/tensor_cuda/__init__.py:1618:    _C.apa_selective_attention_sp_d1_packed=_apa_spq4_entry
./artifacts/apa_p1_a1/refs/production/tensor_cuda/apa_sp2.py:4:Callers must supply keys from quantize_sp2_keys and the calibrated input regime.
./artifacts/apa_p1_a1/refs/production/tensor_cuda/apa_sp2.py:61:def quantize_sp2_keys(k, bulk_bits):
./artifacts/apa_p1_a1/refs/production/tensor_cuda/apa_sp2.py:72:def apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *, bulk_bits,
./artifacts/apa_p1_a1/refs/production/tensor_cuda/apa_sp2.py:77:    kq must come from quantize_sp2_keys(k, bulk_bits). Calibration does not certify
./artifacts/apa_p1_a1/refs/production/tensor_cuda/apa_sp2.py:88:    return _C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale),
./artifacts/apa_p1_a1/source_before/tensor_cuda/src/apa_spo/eb2b.cuh:137:  NDArray pq=apa_spq1_rotate(q,rot,KVH,true);
./artifacts/apa_p1_a1/source_before/tensor_cuda/src/apa_spo/spq4.cuh:450:    NDArray pq=apa_spq1_rotate(q,r,k.shape[1],false);   // mutant: query NOT rotated
./artifacts/apa_p1_a1/source_before/tensor_cuda/src/apa_spo/spq4.cuh:452:    NDArray pq=apa_spq1_rotate(q,r,k.shape[1],true);
./artifacts/apa_p1_s3_r2/census.py:11:pattern=r'\b(apa_selective_attention_sp(_epsilon|_d1_packed|_d1)?|quantize_sp2_keys|apa_spq1_(dump|rotate)|apa_spo2_dump|apa_sp2_(delta|scores|baseline_thresholds)|apa_sp1_1_baseline_diagnostics)\b'
./artifacts/apa_p1_s3_r2/census.py:12:names=set(['apa_selective_attention_sp','apa_selective_attention_sp_epsilon','apa_selective_attention_sp_d1','apa_selective_attention_sp_d1_packed','quantize_sp2_keys','apa_spq1_dump','apa_spq1_rotate','apa_spo2_dump','apa_sp2_delta','apa_sp2_scores','apa_sp2_baseline_thresholds','apa_sp1_1_baseline_diagnostics'])
./artifacts/apa_p1_s3_r2/refs/after/tensor_cuda/__init__.py:1536:from .apa_sp2 import RegisteredMargins, apa_selective_attention_sp, quantize_sp2_keys
./artifacts/apa_p1_s3_r2/refs/after/tensor_cuda/__init__.py:1537:__all__ += ["RegisteredMargins", "apa_selective_attention_sp", "quantize_sp2_keys"]
./artifacts/apa_p1_s3_r2/refs/after/tensor_cuda/__init__.py:1554:    _C.apa_selective_attention_sp_d1 = _apa_spo2_entry
./artifacts/apa_p1_s3_r2/refs/after/tensor_cuda/__init__.py:1555:    _C.apa_spo2_dump = _spo2.run
./artifacts/apa_p1_s3_r2/refs/after/tensor_cuda/__init__.py:1603:    _C.apa_selective_attention_sp_d1_packed=_apa_spq1_entry
./artifacts/apa_p1_s3_r2/refs/after/tensor_cuda/__init__.py:1604:    _C.apa_spq1_dump=_spo2.packed_run
./artifacts/apa_p1_s3_r2/refs/after/tensor_cuda/__init__.py:1605:    _C.apa_spq1_rotate=_spo2.packed_rotate
./artifacts/apa_p1_s3_r2/refs/after/tensor_cuda/__init__.py:1618:    _C.apa_selective_attention_sp_d1_packed=_apa_spq4_entry
./artifacts/apa_p1_s3_r2/refs/after/tensor_cuda/apa_sp2.py:4:Callers must supply keys from quantize_sp2_keys and the calibrated input regime.
./artifacts/apa_p1_s3_r2/refs/after/tensor_cuda/apa_sp2.py:61:def quantize_sp2_keys(k, bulk_bits):
./artifacts/apa_p1_s3_r2/refs/after/tensor_cuda/apa_sp2.py:72:def apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *, bulk_bits,
./artifacts/apa_p1_s3_r2/refs/after/tensor_cuda/apa_sp2.py:77:    kq must come from quantize_sp2_keys(k, bulk_bits). Calibration does not certify
./artifacts/apa_p1_s3_r2/refs/after/tensor_cuda/apa_sp2.py:88:    return _C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale),
./artifacts/apa_p1_s3_r2/refs/before/tensor_cuda/__init__.py:1536:from .apa_sp2 import RegisteredMargins, apa_selective_attention_sp, quantize_sp2_keys
./artifacts/apa_p1_s3_r2/refs/before/tensor_cuda/__init__.py:1537:__all__ += ["RegisteredMargins", "apa_selective_attention_sp", "quantize_sp2_keys"]
./artifacts/apa_p1_s3_r2/refs/before/tensor_cuda/__init__.py:1554:    _C.apa_selective_attention_sp_d1 = _apa_spo2_entry
./artifacts/apa_p1_s3_r2/refs/before/tensor_cuda/__init__.py:1555:    _C.apa_spo2_dump = _spo2.run
./artifacts/apa_p1_s3_r2/refs/before/tensor_cuda/__init__.py:1603:    _C.apa_selective_attention_sp_d1_packed=_apa_spq1_entry
./artifacts/apa_p1_s3_r2/refs/before/tensor_cuda/__init__.py:1604:    _C.apa_spq1_dump=_spo2.packed_run
./artifacts/apa_p1_s3_r2/refs/before/tensor_cuda/__init__.py:1605:    _C.apa_spq1_rotate=_spo2.packed_rotate
./artifacts/apa_p1_s3_r2/refs/before/tensor_cuda/__init__.py:1618:    _C.apa_selective_attention_sp_d1_packed=_apa_spq4_entry
./artifacts/apa_p1_s3_r2/refs/before/tensor_cuda/apa_sp2.py:4:Callers must supply keys from quantize_sp2_keys and the calibrated input regime.
./artifacts/apa_p1_s3_r2/refs/before/tensor_cuda/apa_sp2.py:61:def quantize_sp2_keys(k, bulk_bits):
./artifacts/apa_p1_s3_r2/refs/before/tensor_cuda/apa_sp2.py:72:def apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *, bulk_bits,
./artifacts/apa_p1_s3_r2/refs/before/tensor_cuda/apa_sp2.py:77:    kq must come from quantize_sp2_keys(k, bulk_bits). Calibration does not certify
./artifacts/apa_p1_s3_r2/refs/before/tensor_cuda/apa_sp2.py:88:    return _C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale),
./artifacts/apa_p1_s3_r2/source_before/tensor_cuda/src/bindings.cpp:2334:  m.def("apa_selective_attention_sp", [](Tensor& q, Tensor& k, Tensor& kq,
./artifacts/apa_p1_s3_r2/source_before/tensor_cuda/src/bindings.cpp:2342:    NDArray result = tc::apa_selective_attention_sp(q.data(), k.data(),
./artifacts/apa_p1_s3_r2/source_before/tensor_cuda/src/bindings.cpp:2351:  m.def("apa_sp1_1_baseline_diagnostics", [](Tensor& q, Tensor& kq) {
./artifacts/apa_p1_s3_r2/source_before/tensor_cuda/src/bindings.cpp:2352:    auto result = tc::apa_sp1_1_baseline_diagnostics(q.data(),kq.data());
./artifacts/apa_p1_s3_r2/source_before/tensor_cuda/src/bindings.cpp:2356:  m.def("apa_sp2_delta", &tc::apa_sp2_delta, py::arg("epsilon"), py::arg("e_q"));
./artifacts/apa_p1_s3_r2/source_before/tensor_cuda/src/bindings.cpp:2357:  m.def("apa_selective_attention_sp_epsilon", [](Tensor& q, Tensor& k, Tensor& kq,
./artifacts/apa_p1_s3_r2/source_before/tensor_cuda/src/bindings.cpp:2363:    NDArray result = tc::apa_selective_attention_sp_epsilon(q.data(), k.data(),
./artifacts/apa_p1_s3_r2/source_before/tensor_cuda/src/bindings.cpp:2372:  m.def("apa_sp2_scores", [](Tensor& q, Tensor& k, Tensor& kq, double scale,
./artifacts/apa_p1_s3_r2/source_before/tensor_cuda/src/bindings.cpp:2374:    auto r = tc::apa_sp2_scores(q.data(),k.data(),kq.data(),(float)scale,first,count,wcoop);
./artifacts/apa_p1_s3_r2/source_before/tensor_cuda/src/bindings.cpp:2378:  m.def("apa_sp2_baseline_thresholds", [](Tensor& q, Tensor& kq, double scale,
./artifacts/apa_p1_s3_r2/source_before/tensor_cuda/src/bindings.cpp:2380:    return Tensor::make(tc::apa_sp2_baseline_thresholds(q.data(),kq.data(),
./artifacts/apa_p1_s3_r2/source_before/tensor_cuda/src/bindings.cpp:2758: m.def("packed_rotate",[](Tensor& q,Tensor& r,int kvh,bool tq){return Tensor::make(tc::apa_spq1_rotate(q.data(),r.data(),kvh,tq),false);});
./artifacts/apa_sp1/sp1_1_before/scripts__apa_sp1_gpu.py:142:    new=lambda:tc._C.apa_selective_attention_sp(*tensors,scale,delta,causal)
./artifacts/apa_sp1/sp1_1_before/scripts__apa_sp1_gpu.py:145:        diagnostic,mask=tc._C.apa_selective_attention_sp(*tensors,scale,delta,causal,diagnostics=True)
./artifacts/apa_sp1/sp1_1_before/scripts__apa_sp1_gpu.py:212:                got=tc._C.apa_selective_attention_sp(*tt,1/np.sqrt(D),0.5,True,sinks).numpy().astype(np.float32)
./artifacts/apa_sp2/before/tensor_cuda__include__tc__ops.h:15:NDArray apa_selective_attention_sp(const NDArray& q, const NDArray& k,
./artifacts/apa_sp2/before/tensor_cuda__include__tc__ops.h:18:std::pair<NDArray,NDArray> apa_sp1_1_baseline_diagnostics(const NDArray& q, const NDArray& kq);
./artifacts/apa_sp2/before/tensor_cuda__src__apa_sp1_1.cuh:200:std::pair<NDArray,NDArray> apa_sp1_1_baseline_diagnostics(const NDArray& q, const NDArray& kq) {
./artifacts/apa_sp2/before/tensor_cuda__src__apa_sp1_1.cuh:207:    throw std::runtime_error("apa_sp1_1_baseline_diagnostics: registered fp32 CUDA Part B shape only");
./artifacts/apa_sp2/before/tensor_cuda__src__bindings.cpp:664:  m.def("apa_selective_attention_sp", [](Tensor& q, Tensor& k, Tensor& kq,
./artifacts/apa_sp2/before/tensor_cuda__src__bindings.cpp:672:    NDArray result = tc::apa_selective_attention_sp(q.data(), k.data(),
./artifacts/apa_sp2/before/tensor_cuda__src__bindings.cpp:681:  m.def("apa_sp1_1_baseline_diagnostics", [](Tensor& q, Tensor& kq) {
./artifacts/apa_sp2/before/tensor_cuda__src__bindings.cpp:682:    auto result = tc::apa_sp1_1_baseline_diagnostics(q.data(),kq.data());
./artifacts/apa_sp2/before/tensor_cuda__src__kernels.cu:7441:NDArray apa_selective_attention_sp(const NDArray& q, const NDArray& k,
./artifacts/apa_sp3/a4_baseline/scripts/apa_sp3_gpu.py:179:            out,mask = tc._C.apa_selective_attention_sp(q,k,kq,v,96**-.5,
./artifacts/apa_sp3/a4_baseline/scripts/apa_sp3_gpu.py:194:            _,chosen=tc._C.apa_selective_attention_sp(q,k,kq,v,96**-.5,.125,True,None,True)
./artifacts/apa_sp3/a4_baseline/scripts/apa_sp3_model.py:249:        result = self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta,
./artifacts/apa_sp3/a4_baseline/tensor_cuda/tests/test_apa_sp3.py:35:    entry=s[s.index('NDArray apa_selective_attention_sp('):]
./artifacts/apa_sp3/a4_baseline/tensor_cuda/tests/test_apa_sp3.py:76:        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,1.)
./artifacts/apa_sp3/a4_baseline/tensor_cuda/tests/test_apa_sp3.py:79:        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,1.)
./artifacts/apa_sp3/a4_baseline/tensor_cuda/tests/test_apa_sp3.py:81:        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,float('inf'))
./artifacts/apa_sp3/a4_baseline/tensor_cuda/tests/test_apa_sp3.py:127:    obj.tc=SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=lambda *a: calls.append(a) or 'out'))
./artifacts/apa_sp3/a4_final_verification/a4_mutations/ceiling_fit_bypass.py:79:            other, mask = model.tc._C.apa_selective_attention_sp(
./artifacts/apa_sp3/a4_final_verification/a4_mutations/mask_bitorder.py:123:        native = self.tc._C.apa_selective_attention_sp
./artifacts/apa_sp3/a4_final_verification/a4_mutations/native_parity_bypass.py:123:        native = self.tc._C.apa_selective_attention_sp
./artifacts/apa_sp3/a4_final_verification/a4_mutations/no_file_integrity.py:123:        native = self.tc._C.apa_selective_attention_sp
./artifacts/apa_sp3/a4_final_verification/a4_mutations/range_row_skip.py:123:        native = self.tc._C.apa_selective_attention_sp
./artifacts/apa_sp3/a4_mutations/ceiling_fit_bypass.py:79:            other, mask = model.tc._C.apa_selective_attention_sp(
./artifacts/apa_sp3/a4_mutations/mask_bitorder.py:123:        native = self.tc._C.apa_selective_attention_sp
./artifacts/apa_sp3/a4_mutations/native_parity_bypass.py:123:        native = self.tc._C.apa_selective_attention_sp
./artifacts/apa_sp3/a4_mutations/no_file_integrity.py:123:        native = self.tc._C.apa_selective_attention_sp
./artifacts/apa_sp3/a4_mutations/range_row_skip.py:123:        native = self.tc._C.apa_selective_attention_sp
./artifacts/apa_sp3/a5_baseline/scripts/apa_sp3_a4_capture.py:123:        native = self.tc._C.apa_selective_attention_sp
./artifacts/apa_sp3/a5_baseline/scripts/apa_sp3_a4_jobs.py:79:            other, mask = model.tc._C.apa_selective_attention_sp(
./artifacts/apa_sp3/a5_baseline/scripts/apa_sp3_gpu.py:184:            out,mask = tc._C.apa_selective_attention_sp(q,k,kq,v,96**-.5,
./artifacts/apa_sp3/a5_baseline/scripts/apa_sp3_gpu.py:199:            _,chosen=tc._C.apa_selective_attention_sp(q,k,kq,v,96**-.5,.125,True,None,True)
./artifacts/apa_sp3/a5_baseline/scripts/apa_sp3_model.py:249:        result = self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta,
./artifacts/apa_sp3/a5_baseline/tensor_cuda/tests/test_apa_sp3.py:35:    entry=s[s.index('NDArray apa_selective_attention_sp('):]
./artifacts/apa_sp3/a5_baseline/tensor_cuda/tests/test_apa_sp3.py:76:        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,1.)
./artifacts/apa_sp3/a5_baseline/tensor_cuda/tests/test_apa_sp3.py:79:        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,1.)
./artifacts/apa_sp3/a5_baseline/tensor_cuda/tests/test_apa_sp3.py:81:        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,float('inf'))
./artifacts/apa_sp3/a5_baseline/tensor_cuda/tests/test_apa_sp3.py:127:    obj.tc=SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=lambda *a: calls.append(a) or 'out'))
./artifacts/apa_sp3/a5_baseline/tensor_cuda/tests/test_apa_sp3_a4.py:166:    obj.tc=SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=native))
./artifacts/apa_sp3/a5_baseline/tensor_cuda/tests/test_apa_sp3_a4.py:177:        obj.tc._C.apa_selective_attention_sp=bad
./artifacts/apa_sp3/a5_mutations/shorten_measured_steps.py:249:        result = self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta,
./artifacts/apa_sp3/a6_baseline/scripts/apa_sp3_a4_capture.py:123:        native = self.tc._C.apa_selective_attention_sp
./artifacts/apa_sp3/a6_baseline/scripts/apa_sp3_a4_jobs.py:79:            other, mask = model.tc._C.apa_selective_attention_sp(
./artifacts/apa_sp3/a6_baseline/scripts/apa_sp3_gpu.py:185:            out,mask = tc._C.apa_selective_attention_sp(q,k,kq,v,96**-.5,
./artifacts/apa_sp3/a6_baseline/scripts/apa_sp3_gpu.py:200:            _,chosen=tc._C.apa_selective_attention_sp(q,k,kq,v,96**-.5,.125,True,None,True)
./artifacts/apa_sp3/a6_baseline/scripts/apa_sp3_model.py:249:        result = self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta,
./artifacts/apa_sp3/a6_baseline/tensor_cuda/tests/test_apa_sp3.py:35:    entry=s[s.index('NDArray apa_selective_attention_sp('):]
./artifacts/apa_sp3/a6_baseline/tensor_cuda/tests/test_apa_sp3.py:76:        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,1.)
./artifacts/apa_sp3/a6_baseline/tensor_cuda/tests/test_apa_sp3.py:79:        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,1.)
./artifacts/apa_sp3/a6_baseline/tensor_cuda/tests/test_apa_sp3.py:81:        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,float('inf'))
./artifacts/apa_sp3/a6_baseline/tensor_cuda/tests/test_apa_sp3.py:127:    obj.tc=SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=lambda *a: calls.append(a) or 'out'))
./artifacts/apa_sp3/a6_baseline/tensor_cuda/tests/test_apa_sp3_a4.py:166:    obj.tc=SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=native))
./artifacts/apa_sp3/a6_baseline/tensor_cuda/tests/test_apa_sp3_a4.py:177:        obj.tc._C.apa_selective_attention_sp=bad
./artifacts/apa_sp3/a6_mutations/drop_cache.py:88:                    return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,
./artifacts/apa_sp3/a6_mutations/force_full_copy.py:88:                    return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,
./artifacts/apa_sp3/a6_mutations/force_layer_hook.py:88:                    return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,
./artifacts/apa_sp3/a6_mutations/pool_after_load.py:88:                    return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,
./artifacts/apa_sp3/a6_mutations/rail_boundary.py:88:                    return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,
./artifacts/apa_sp3/a6_mutations/shorten_steps.py:88:                    return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,
./artifacts/apa_sp3/capture_dtype_mutation_r3/mutations/final/off_by_one_targets.py:249:        result = self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta,
./artifacts/apa_sp3/capture_dtype_mutation_r3/mutations/final/swapped_scale_delta.py:249:        result = self.tc._C.apa_selective_attention_sp(q, k, kq, v, self.delta, scale,
./artifacts/apa_sp3/mutations/final/off_by_one_targets.py:221:        result = self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta,
./artifacts/apa_sp3/mutations/final/swapped_scale_delta.py:221:        result = self.tc._C.apa_selective_attention_sp(q, k, kq, v, self.delta, scale,
./artifacts/apa_sp3/mutations/off_by_one_targets.py:213:        result = self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta,
./artifacts/apa_sp3/mutations/protocol2/off_by_one_targets.py:246:        result = self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta,
./artifacts/apa_sp3/mutations/protocol2/swapped_scale_delta.py:246:        result = self.tc._C.apa_selective_attention_sp(q, k, kq, v, self.delta, scale,
./artifacts/apa_sp3/mutations/swapped_scale_delta.py:213:        result = self.tc._C.apa_selective_attention_sp(q, k, kq, v, self.delta, scale,
./artifacts/apa_sp4g/a1_baseline/scripts/apa_sp4g_gpu.py:33:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,True)
./artifacts/apa_sp4g/a1_baseline/scripts/apa_sp4g_gpu.py:34:                clean=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,False)
./artifacts/apa_sp4g/a1_baseline/scripts/apa_sp4g_metrics.py:49:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/a1_baseline/scripts/apa_sp4g_model.py:95:                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
./artifacts/apa_sp4g/a1_baseline/scripts/apa_sp4g_model.py:108:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/a1_baseline/scripts/apa_sp4g_model.py:109:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/a1_baseline/tensor_cuda/tests/test_apa_sp4g.py:60:    s=(R/'tensor_cuda/src/kernels.cu').read_text();entry=s[s.index('NDArray apa_selective_attention_sp('):]
./artifacts/apa_sp4g/a1_baseline/tensor_cuda/tests/test_apa_sp4g.py:85:    with pytest.raises(RuntimeError,match='default OFF'):tc._C.apa_selective_attention_sp(q,k,k,k,1.,1.)
./artifacts/apa_sp4g/a1_baseline/tensor_cuda/tests/test_apa_sp4g.py:87:    with pytest.raises(RuntimeError,match='share CUDA device'):tc._C.apa_selective_attention_sp(q,k,k,k,1.,1.)
./artifacts/apa_sp4g/a1_baseline/tensor_cuda/tests/test_apa_sp4g.py:89:        with pytest.raises(RuntimeError,match='finite nonnegative'):tc._C.apa_selective_attention_sp(q,k,k,k,1.,delta)
./artifacts/apa_sp4g/a1_baseline/tensor_cuda/tests/test_apa_sp4g.py:225:    calls=[];tc=SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=lambda *args:calls.append(args) or 'out'))
./artifacts/apa_sp4g/a2_baseline/scripts/apa_sp4g_gpu.py:33:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,True)
./artifacts/apa_sp4g/a2_baseline/scripts/apa_sp4g_gpu.py:34:                clean=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,False)
./artifacts/apa_sp4g/a2_baseline/scripts/apa_sp4g_metrics.py:53:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/a2_baseline/scripts/apa_sp4g_model.py:109:                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
./artifacts/apa_sp4g/a2_baseline/scripts/apa_sp4g_model.py:122:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/a2_baseline/scripts/apa_sp4g_model.py:123:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/a2_baseline/tensor_cuda/tests/test_apa_sp4g.py:61:    s=(R/'tensor_cuda/src/kernels.cu').read_text();entry=s[s.index('NDArray apa_selective_attention_sp('):]
./artifacts/apa_sp4g/a2_baseline/tensor_cuda/tests/test_apa_sp4g.py:86:    with pytest.raises(RuntimeError,match='default OFF'):tc._C.apa_selective_attention_sp(q,k,k,k,1.,1.)
./artifacts/apa_sp4g/a2_baseline/tensor_cuda/tests/test_apa_sp4g.py:88:    with pytest.raises(RuntimeError,match='share CUDA device'):tc._C.apa_selective_attention_sp(q,k,k,k,1.,1.)
./artifacts/apa_sp4g/a2_baseline/tensor_cuda/tests/test_apa_sp4g.py:90:        with pytest.raises(RuntimeError,match='finite nonnegative'):tc._C.apa_selective_attention_sp(q,k,k,k,1.,delta)
./artifacts/apa_sp4g/a2_baseline/tensor_cuda/tests/test_apa_sp4g.py:237:    calls=[];tc=SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=lambda *args:calls.append(args) or 'out'))
./artifacts/apa_sp4g/mutations/margin_term.py:49:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations/percentile.py:49:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations/relaxed_match.py:33:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,True)
./artifacts/apa_sp4g/mutations/relaxed_match.py:34:                clean=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,False)
./artifacts/apa_sp4g/mutations/reverse_bracket.py:33:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,True)
./artifacts/apa_sp4g/mutations/reverse_bracket.py:34:                clean=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,False)
./artifacts/apa_sp4g/mutations/tail_denominator.py:49:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations/target_boundary.py:95:                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations/target_boundary.py:108:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/mutations/target_boundary.py:109:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations/wrong_target.py:95:                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations/wrong_target.py:108:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/mutations/wrong_target.py:109:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_a1/margin_term.py:53:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations_a1/percentile.py:53:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations_a1/relaxed_match.py:33:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,True)
./artifacts/apa_sp4g/mutations_a1/relaxed_match.py:34:                clean=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,False)
./artifacts/apa_sp4g/mutations_a1/reverse_bracket.py:33:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,True)
./artifacts/apa_sp4g/mutations_a1/reverse_bracket.py:34:                clean=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,False)
./artifacts/apa_sp4g/mutations_a1/tail_denominator.py:53:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations_a1/target_boundary.py:109:                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_a1/target_boundary.py:122:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/mutations_a1/target_boundary.py:123:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_a1/wrong_target.py:109:                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_a1/wrong_target.py:122:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/mutations_a1/wrong_target.py:123:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_a1_final/margin_term.py:53:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations_a1_final/percentile.py:53:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations_a1_final/relaxed_match.py:33:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,True)
./artifacts/apa_sp4g/mutations_a1_final/relaxed_match.py:34:                clean=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,False)
./artifacts/apa_sp4g/mutations_a1_final/reverse_bracket.py:33:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,True)
./artifacts/apa_sp4g/mutations_a1_final/reverse_bracket.py:34:                clean=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,False)
./artifacts/apa_sp4g/mutations_a1_final/tail_denominator.py:53:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations_a1_final/target_boundary.py:109:                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_a1_final/target_boundary.py:122:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/mutations_a1_final/target_boundary.py:123:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_a1_final/wrong_target.py:109:                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_a1_final/wrong_target.py:122:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/mutations_a1_final/wrong_target.py:123:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_a2/bf16_lowbits.py:88:            d=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,True,None,False)
./artifacts/apa_sp4g/mutations_a2/bf16_lowbits.py:94:        d=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,True,None,False)
./artifacts/apa_sp4g/mutations_a2/bf16_lowbits.py:96:        d32=tc._C.apa_selective_attention_sp(q.float(),k.float(),kq.float(),v.float(),scale,MAX_DELTA,True,None,False)
./artifacts/apa_sp4g/mutations_a2/bf16_lowbits.py:123:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_a2/bf16_lowbits.py:124:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/mutations_a2/fp32_arm.py:88:            d=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,True,None,False)
./artifacts/apa_sp4g/mutations_a2/fp32_arm.py:94:        d=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,True,None,False)
./artifacts/apa_sp4g/mutations_a2/fp32_arm.py:96:        d32=tc._C.apa_selective_attention_sp(q.float(),k.float(),kq.float(),v.float(),scale,MAX_DELTA,True,None,False)
./artifacts/apa_sp4g/mutations_a2/fp32_arm.py:123:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_a2/fp32_arm.py:124:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/mutations_a2/mask_ignored.py:91:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,rec['scale'],m['delta'],rec['causal'],None,True)
./artifacts/apa_sp4g/mutations_a2/output_tolerance.py:91:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,rec['scale'],m['delta'],rec['causal'],None,True)
./artifacts/apa_sp4g/mutations_a2/scale.py:88:            d=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,True,None,False)
./artifacts/apa_sp4g/mutations_a2/scale.py:94:        d=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,True,None,False)
./artifacts/apa_sp4g/mutations_a2/scale.py:96:        d32=tc._C.apa_selective_attention_sp(q.float(),k.float(),kq.float(),v.float(),scale,MAX_DELTA,True,None,False)
./artifacts/apa_sp4g/mutations_a2/scale.py:123:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_a2/scale.py:124:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/mutations_a3/native_tolerance.py:71:    out = tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a3/native_tolerance.py:72:    checked, mask = tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,True)
./artifacts/apa_sp4g/mutations_a3/refine_mask_ignored.py:71:    out = tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a3/refine_mask_ignored.py:72:    checked, mask = tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,True)
./artifacts/apa_sp4g/mutations_a3_final/native_tolerance.py:71:    out = tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a3_final/native_tolerance.py:72:    checked, mask = tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,True)
./artifacts/apa_sp4g/mutations_a3_final/refine_mask_ignored.py:71:    out = tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a3_final/refine_mask_ignored.py:72:    checked, mask = tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,True)
./artifacts/apa_sp4g/mutations_a4/cast_dropped.py:28:        out=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a4/coverage_gap.py:28:        out=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a4/input_dtype_ignored.py:28:        out=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a4/output_dtype_ignored.py:28:        out=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a4/return_wrong_arm.py:28:        out=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a4_v2/cast_dropped.py:28:        out=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a4_v2/coverage_gap.py:28:        out=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a4_v2/input_dtype_ignored.py:28:        out=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a4_v2/output_dtype_ignored.py:28:        out=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a4_v2/return_wrong_arm.py:28:        out=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a5/capture_wrong_layer.py:33:    return dict(entry='tc._C.apa_selective_attention_sp',
./artifacts/apa_sp4g/mutations_a5/capture_wrong_layer.py:115:        out = self.tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a5/capture_wrong_layer.py:116:        checked,mask = self.tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,True)
./artifacts/apa_sp4g/mutations_a5/off_by_one.py:33:    return dict(entry='tc._C.apa_selective_attention_sp',
./artifacts/apa_sp4g/mutations_a5/off_by_one.py:115:        out = self.tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a5/off_by_one.py:116:        checked,mask = self.tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,True)
./artifacts/apa_sp4g/mutations_a5/prefill_only.py:33:    return dict(entry='tc._C.apa_selective_attention_sp',
./artifacts/apa_sp4g/mutations_a5/prefill_only.py:115:        out = self.tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a5/prefill_only.py:116:        checked,mask = self.tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,True)
./artifacts/apa_sp4g/mutations_a5/splitk_for_cached.py:33:    return dict(entry='tc._C.apa_selective_attention_sp',
./artifacts/apa_sp4g/mutations_a5/splitk_for_cached.py:115:        out = self.tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,False)
./artifacts/apa_sp4g/mutations_a5/splitk_for_cached.py:116:        checked,mask = self.tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,True)
./artifacts/apa_sp4g/mutations_a7/pool_off.py:109:                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_a7/pool_off.py:122:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/mutations_a7/pool_off.py:123:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_final/margin_term.py:49:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations_final/percentile.py:49:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations_final/relaxed_match.py:33:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,True)
./artifacts/apa_sp4g/mutations_final/relaxed_match.py:34:                clean=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,False)
./artifacts/apa_sp4g/mutations_final/reverse_bracket.py:33:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,True)
./artifacts/apa_sp4g/mutations_final/reverse_bracket.py:34:                clean=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,False)
./artifacts/apa_sp4g/mutations_final/tail_denominator.py:49:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations_final/target_boundary.py:95:                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_final/target_boundary.py:108:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/mutations_final/target_boundary.py:109:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_final/wrong_target.py:95:                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_final/wrong_target.py:108:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/mutations_final/wrong_target.py:109:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_sealed/margin_term.py:49:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations_sealed/percentile.py:49:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations_sealed/relaxed_match.py:33:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,True)
./artifacts/apa_sp4g/mutations_sealed/relaxed_match.py:34:                clean=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,False)
./artifacts/apa_sp4g/mutations_sealed/reverse_bracket.py:33:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,True)
./artifacts/apa_sp4g/mutations_sealed/reverse_bracket.py:34:                clean=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,False)
./artifacts/apa_sp4g/mutations_sealed/tail_denominator.py:49:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./artifacts/apa_sp4g/mutations_sealed/target_boundary.py:95:                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_sealed/target_boundary.py:108:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/mutations_sealed/target_boundary.py:109:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_sealed/wrong_target.py:95:                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
./artifacts/apa_sp4g/mutations_sealed/wrong_target.py:108:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./artifacts/apa_sp4g/mutations_sealed/wrong_target.py:109:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./artifacts/apa_sp5/a2_inject/sitecustomize.py:16:`apa_selective_attention_sp` in all three returns nothing), so a third mode
./artifacts/apa_sp5/a2_inject/sitecustomize.py:48:    # 'apa_selective_attention_sp') is False on it). Since the smoke is
./artifacts/apa_sp5/a2_inject/sitecustomize.py:60:            if not hasattr(_tc_preload._C, 'apa_selective_attention_sp'):
./artifacts/apa_sp5/a2_inject/sitecustomize.py:62:                    'apa_sp5_a2: preloaded engine lacks apa_selective_attention_sp; '
./artifacts/apa_sp5/a2_inject/sitecustomize.py:107:            attn = tc._C.apa_selective_attention_sp(
./artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/__init__.py:1301:from .apa_sp2 import RegisteredMargins, apa_selective_attention_sp, quantize_sp2_keys
./artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/__init__.py:1302:__all__ += ["RegisteredMargins", "apa_selective_attention_sp", "quantize_sp2_keys"]
./artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/apa_sp2.py:4:Callers must supply keys from quantize_sp2_keys and the calibrated input regime.
./artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/apa_sp2.py:61:def quantize_sp2_keys(k, bulk_bits):
./artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/apa_sp2.py:72:def apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *, bulk_bits,
./artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/apa_sp2.py:77:    kq must come from quantize_sp2_keys(k, bulk_bits). Calibration does not certify
./artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/apa_sp2.py:88:    return _C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale),
./artifacts/apa_ste5/receipts/INIT_BEFORE.py:1294:from .apa_sp2 import RegisteredMargins, apa_selective_attention_sp, quantize_sp2_keys
./artifacts/apa_ste5/receipts/INIT_BEFORE.py:1295:__all__ += ["RegisteredMargins", "apa_selective_attention_sp", "quantize_sp2_keys"]
./artifacts/bp_kernel_1/baseline/bindings.cpp:664:  m.def("apa_selective_attention_sp", [](Tensor& q, Tensor& k, Tensor& kq,
./artifacts/bp_kernel_1/baseline/bindings.cpp:672:    NDArray result = tc::apa_selective_attention_sp(q.data(), k.data(),
./artifacts/bp_kernel_1/baseline/bindings.cpp:681:  m.def("apa_sp1_1_baseline_diagnostics", [](Tensor& q, Tensor& kq) {
./artifacts/bp_kernel_1/baseline/bindings.cpp:682:    auto result = tc::apa_sp1_1_baseline_diagnostics(q.data(),kq.data());
./artifacts/bp_kernel_1/baseline/bindings.cpp:686:  m.def("apa_sp2_delta", &tc::apa_sp2_delta, py::arg("epsilon"), py::arg("e_q"));
./artifacts/bp_kernel_1/baseline/bindings.cpp:687:  m.def("apa_selective_attention_sp_epsilon", [](Tensor& q, Tensor& k, Tensor& kq,
./artifacts/bp_kernel_1/baseline/bindings.cpp:693:    NDArray result = tc::apa_selective_attention_sp_epsilon(q.data(), k.data(),
./artifacts/bp_kernel_1/baseline/bindings.cpp:702:  m.def("apa_sp2_scores", [](Tensor& q, Tensor& k, Tensor& kq, double scale,
./artifacts/bp_kernel_1/baseline/bindings.cpp:704:    auto r = tc::apa_sp2_scores(q.data(),k.data(),kq.data(),(float)scale,first,count,wcoop);
./artifacts/bp_kernel_1/baseline/bindings.cpp:708:  m.def("apa_sp2_baseline_thresholds", [](Tensor& q, Tensor& kq, double scale,
./artifacts/bp_kernel_1/baseline/bindings.cpp:710:    return Tensor::make(tc::apa_sp2_baseline_thresholds(q.data(),kq.data(),
./artifacts/bp_kernel_1/baseline/kernels.cu:7441:NDArray apa_selective_attention_sp(const NDArray& q, const NDArray& k,
./artifacts/bp_kernel_2/baseline/bindings.cpp:675:  m.def("apa_selective_attention_sp", [](Tensor& q, Tensor& k, Tensor& kq,
./artifacts/bp_kernel_2/baseline/bindings.cpp:683:    NDArray result = tc::apa_selective_attention_sp(q.data(), k.data(),
./artifacts/bp_kernel_2/baseline/bindings.cpp:692:  m.def("apa_sp1_1_baseline_diagnostics", [](Tensor& q, Tensor& kq) {
./artifacts/bp_kernel_2/baseline/bindings.cpp:693:    auto result = tc::apa_sp1_1_baseline_diagnostics(q.data(),kq.data());
./artifacts/bp_kernel_2/baseline/bindings.cpp:697:  m.def("apa_sp2_delta", &tc::apa_sp2_delta, py::arg("epsilon"), py::arg("e_q"));
./artifacts/bp_kernel_2/baseline/bindings.cpp:698:  m.def("apa_selective_attention_sp_epsilon", [](Tensor& q, Tensor& k, Tensor& kq,
./artifacts/bp_kernel_2/baseline/bindings.cpp:704:    NDArray result = tc::apa_selective_attention_sp_epsilon(q.data(), k.data(),
./artifacts/bp_kernel_2/baseline/bindings.cpp:713:  m.def("apa_sp2_scores", [](Tensor& q, Tensor& k, Tensor& kq, double scale,
./artifacts/bp_kernel_2/baseline/bindings.cpp:715:    auto r = tc::apa_sp2_scores(q.data(),k.data(),kq.data(),(float)scale,first,count,wcoop);
./artifacts/bp_kernel_2/baseline/bindings.cpp:719:  m.def("apa_sp2_baseline_thresholds", [](Tensor& q, Tensor& kq, double scale,
./artifacts/bp_kernel_2/baseline/bindings.cpp:721:    return Tensor::make(tc::apa_sp2_baseline_thresholds(q.data(),kq.data(),
./artifacts/bp_kernel_2/baseline/kernels.cu:7608:NDArray apa_selective_attention_sp(const NDArray& q, const NDArray& k,
./artifacts/bp_kernel_3/baseline/bindings.cpp:680:  m.def("apa_selective_attention_sp", [](Tensor& q, Tensor& k, Tensor& kq,
./artifacts/bp_kernel_3/baseline/bindings.cpp:688:    NDArray result = tc::apa_selective_attention_sp(q.data(), k.data(),
./artifacts/bp_kernel_3/baseline/bindings.cpp:697:  m.def("apa_sp1_1_baseline_diagnostics", [](Tensor& q, Tensor& kq) {
./artifacts/bp_kernel_3/baseline/bindings.cpp:698:    auto result = tc::apa_sp1_1_baseline_diagnostics(q.data(),kq.data());
./artifacts/bp_kernel_3/baseline/bindings.cpp:702:  m.def("apa_sp2_delta", &tc::apa_sp2_delta, py::arg("epsilon"), py::arg("e_q"));
./artifacts/bp_kernel_3/baseline/bindings.cpp:703:  m.def("apa_selective_attention_sp_epsilon", [](Tensor& q, Tensor& k, Tensor& kq,
./artifacts/bp_kernel_3/baseline/bindings.cpp:709:    NDArray result = tc::apa_selective_attention_sp_epsilon(q.data(), k.data(),
./artifacts/bp_kernel_3/baseline/bindings.cpp:718:  m.def("apa_sp2_scores", [](Tensor& q, Tensor& k, Tensor& kq, double scale,
./artifacts/bp_kernel_3/baseline/bindings.cpp:720:    auto r = tc::apa_sp2_scores(q.data(),k.data(),kq.data(),(float)scale,first,count,wcoop);
./artifacts/bp_kernel_3/baseline/bindings.cpp:724:  m.def("apa_sp2_baseline_thresholds", [](Tensor& q, Tensor& kq, double scale,
./artifacts/bp_kernel_3/baseline/bindings.cpp:726:    return Tensor::make(tc::apa_sp2_baseline_thresholds(q.data(),kq.data(),
./artifacts/bp_kernel_3/baseline/kernels.cu:7727:NDArray apa_selective_attention_sp(const NDArray& q, const NDArray& k,
./artifacts/bp_kernel_4/baseline/bindings.cpp:682:  m.def("apa_selective_attention_sp", [](Tensor& q, Tensor& k, Tensor& kq,
./artifacts/bp_kernel_4/baseline/bindings.cpp:690:    NDArray result = tc::apa_selective_attention_sp(q.data(), k.data(),
./artifacts/bp_kernel_4/baseline/bindings.cpp:699:  m.def("apa_sp1_1_baseline_diagnostics", [](Tensor& q, Tensor& kq) {
./artifacts/bp_kernel_4/baseline/bindings.cpp:700:    auto result = tc::apa_sp1_1_baseline_diagnostics(q.data(),kq.data());
./artifacts/bp_kernel_4/baseline/bindings.cpp:704:  m.def("apa_sp2_delta", &tc::apa_sp2_delta, py::arg("epsilon"), py::arg("e_q"));
./artifacts/bp_kernel_4/baseline/bindings.cpp:705:  m.def("apa_selective_attention_sp_epsilon", [](Tensor& q, Tensor& k, Tensor& kq,
./artifacts/bp_kernel_4/baseline/bindings.cpp:711:    NDArray result = tc::apa_selective_attention_sp_epsilon(q.data(), k.data(),
./artifacts/bp_kernel_4/baseline/bindings.cpp:720:  m.def("apa_sp2_scores", [](Tensor& q, Tensor& k, Tensor& kq, double scale,
./artifacts/bp_kernel_4/baseline/bindings.cpp:722:    auto r = tc::apa_sp2_scores(q.data(),k.data(),kq.data(),(float)scale,first,count,wcoop);
./artifacts/bp_kernel_4/baseline/bindings.cpp:726:  m.def("apa_sp2_baseline_thresholds", [](Tensor& q, Tensor& kq, double scale,
./artifacts/bp_kernel_4/baseline/bindings.cpp:728:    return Tensor::make(tc::apa_sp2_baseline_thresholds(q.data(),kq.data(),
./artifacts/bp_kernel_4/baseline/kernels.cu:7947:NDArray apa_selective_attention_sp(const NDArray& q, const NDArray& k,
./artifacts/p1t_fix7/preL1_sources/tensor_cuda/src/bindings.cpp:2178:  m.def("apa_selective_attention_sp", [](Tensor& q, Tensor& k, Tensor& kq,
./artifacts/p1t_fix7/preL1_sources/tensor_cuda/src/bindings.cpp:2186:    NDArray result = tc::apa_selective_attention_sp(q.data(), k.data(),
./artifacts/p1t_fix7/preL1_sources/tensor_cuda/src/bindings.cpp:2195:  m.def("apa_sp1_1_baseline_diagnostics", [](Tensor& q, Tensor& kq) {
./artifacts/p1t_fix7/preL1_sources/tensor_cuda/src/bindings.cpp:2196:    auto result = tc::apa_sp1_1_baseline_diagnostics(q.data(),kq.data());
./artifacts/p1t_fix7/preL1_sources/tensor_cuda/src/bindings.cpp:2200:  m.def("apa_sp2_delta", &tc::apa_sp2_delta, py::arg("epsilon"), py::arg("e_q"));
./artifacts/p1t_fix7/preL1_sources/tensor_cuda/src/bindings.cpp:2201:  m.def("apa_selective_attention_sp_epsilon", [](Tensor& q, Tensor& k, Tensor& kq,
./artifacts/p1t_fix7/preL1_sources/tensor_cuda/src/bindings.cpp:2207:    NDArray result = tc::apa_selective_attention_sp_epsilon(q.data(), k.data(),
./artifacts/p1t_fix7/preL1_sources/tensor_cuda/src/bindings.cpp:2216:  m.def("apa_sp2_scores", [](Tensor& q, Tensor& k, Tensor& kq, double scale,
./artifacts/p1t_fix7/preL1_sources/tensor_cuda/src/bindings.cpp:2218:    auto r = tc::apa_sp2_scores(q.data(),k.data(),kq.data(),(float)scale,first,count,wcoop);
./artifacts/p1t_fix7/preL1_sources/tensor_cuda/src/bindings.cpp:2222:  m.def("apa_sp2_baseline_thresholds", [](Tensor& q, Tensor& kq, double scale,
./artifacts/p1t_fix7/preL1_sources/tensor_cuda/src/bindings.cpp:2224:    return Tensor::make(tc::apa_sp2_baseline_thresholds(q.data(),kq.data(),
./artifacts/p1t_fix7/preL1_sources/tensor_cuda/tensor_cuda/__init__.py:1515:from .apa_sp2 import RegisteredMargins, apa_selective_attention_sp, quantize_sp2_keys
./artifacts/p1t_fix7/preL1_sources/tensor_cuda/tensor_cuda/__init__.py:1516:__all__ += ["RegisteredMargins", "apa_selective_attention_sp", "quantize_sp2_keys"]
./scripts/apa_pidv.py:269:            kq = tc.quantize_sp2_keys(up(self.k, "float32"), 2)
./scripts/apa_pidv.py:303:                    y = tc.apa_selective_attention_sp(q, k, self.kq_t, v_t, self.scale, 0.05, bulk_bits=2,
./scripts/apa_sp1_1_gpu.py:27:                out,mask=tc._C.apa_selective_attention_sp(*tt,1/np.sqrt(D),.5,True,sinks,diagnostics=True)
./scripts/apa_sp1_1_gpu.py:29:                plain=tc._C.apa_selective_attention_sp(*tt,1/np.sqrt(D),.5,True,sinks).numpy().astype(np.float32)
./scripts/apa_sp1_1_gpu.py:46:        dt,dm=tc._C.apa_sp1_1_baseline_diagnostics(tt[0],tt[2])
./scripts/apa_sp1_gpu.py:145:    new=lambda:tc._C.apa_selective_attention_sp(*tensors,scale,delta,causal)
./scripts/apa_sp1_gpu.py:148:        diagnostic,mask=tc._C.apa_selective_attention_sp(*tensors,scale,delta,causal,diagnostics=True)
./scripts/apa_sp1_gpu.py:217:                got=tc._C.apa_selective_attention_sp(*tt,1/np.sqrt(D),0.5,True,sinks).numpy().astype(np.float32)
./scripts/apa_sp2_gpu.py:53:    kqt = tc.quantize_sp2_keys(kt, bits)
./scripts/apa_sp2_gpu.py:73:        bulk, exact = tc._C.apa_sp2_scores(qt,kt,kqt,float(scale),0,len(indices),
./scripts/apa_sp2_gpu.py:159:    e_q=margins.lookup(bits,D,'float32',float(scale));delta=tc._C.apa_sp2_delta(epsilon,e_q)
./scripts/apa_sp2_gpu.py:163:    new=lambda:tc.apa_selective_attention_sp(*args,epsilon,bulk_bits=bits,margins=margins,is_causal=s['causal'])
./scripts/apa_sp2_gpu.py:166:        out,mask=tc.apa_selective_attention_sp(*args,epsilon,bulk_bits=bits,margins=margins,is_causal=s['causal'],diagnostics=True)
./scripts/apa_sp2_gpu.py:169:        direct=tc._C.apa_selective_attention_sp(*args,delta,s['causal']).numpy()
./scripts/apa_sp2_gpu.py:171:        thresholds=tc._C.apa_sp2_baseline_thresholds(qt,kqt,float(scale),reg['baseline']['zthr'],s['causal']).numpy()
./scripts/apa_sp2_gpu.py:179:        bdev,edev=tc._C.apa_sp2_scores(qt,kt,kqt,float(scale),first,n,wc)
./scripts/apa_sp3_a4_capture.py:123:        native = self.tc._C.apa_selective_attention_sp
./scripts/apa_sp3_a4_jobs.py:79:            other, mask = model.tc._C.apa_selective_attention_sp(
./scripts/apa_sp3_a6_decode.py:88:                    return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,
./scripts/apa_sp3_gpu.py:186:            out,mask = tc._C.apa_selective_attention_sp(q,k,kq,v,96**-.5,
./scripts/apa_sp3_gpu.py:201:            _,chosen=tc._C.apa_selective_attention_sp(q,k,kq,v,96**-.5,.125,True,None,True)
./scripts/apa_sp3_model.py:249:        result = self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta,
./scripts/apa_sp4g_a2_metrics.py:91:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,rec['scale'],m['delta'],rec['causal'],None,True)
./scripts/apa_sp4g_a2_model.py:88:            d=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,True,None,False)
./scripts/apa_sp4g_a2_model.py:94:        d=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,True,None,False)
./scripts/apa_sp4g_a2_model.py:96:        d32=tc._C.apa_selective_attention_sp(q.float(),k.float(),kq.float(),v.float(),scale,MAX_DELTA,True,None,False)
./scripts/apa_sp4g_a2_model.py:123:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./scripts/apa_sp4g_a2_model.py:124:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./scripts/apa_sp4g_a3_model.py:71:    out = tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,False)
./scripts/apa_sp4g_a3_model.py:72:    checked, mask = tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,True)
./scripts/apa_sp4g_a3_sweep.py:17:    entry = tc._C.apa_selective_attention_sp
./scripts/apa_sp4g_a4_metrics.py:94:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,rec['scale'],m['delta'],rec['causal'],None,True)
./scripts/apa_sp4g_a4_model.py:28:        out=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,causal,None,False)
./scripts/apa_sp4g_a5_model.py:33:    return dict(entry='tc._C.apa_selective_attention_sp',
./scripts/apa_sp4g_a5_model.py:115:        out = self.tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,False)
./scripts/apa_sp4g_a5_model.py:116:        checked,mask = self.tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,True)
./scripts/apa_sp4g_a6_metrics.py:95:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,rec['scale'],m['delta'],rec['causal'],None,True)
./scripts/apa_sp4g_gpu.py:33:                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,True)
./scripts/apa_sp4g_gpu.py:34:                clean=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,False)
./scripts/apa_sp4g_metrics.py:53:            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
./scripts/apa_sp4g_model.py:109:                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
./scripts/apa_sp4g_model.py:122:            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
./scripts/apa_sp4g_model.py:123:            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
./scripts/apa_sp5_a1_margins.py:132:            attn, sel = tc._C.apa_selective_attention_sp(
./scripts/apa_sp5_a1_margins.py:181:                o2, s2 = tc._C.apa_selective_attention_sp(
./scripts/apa_sp5_frac.py:47:        attn, sel = tc._C.apa_selective_attention_sp(
./scripts/apa_sp5_item2_sinkpin.py:125:            on, sel_on = tc._C.apa_selective_attention_sp(
./scripts/apa_sp5_item2_sinkpin.py:127:            off, sel_off = tc._C.apa_selective_attention_sp(
./scripts/apa_sp5_item3_exactness.py:147:                sp, sel = tc._C.apa_selective_attention_sp(
./scripts/apa_sp5_model.py:85:    # C/D/E: single pass. tc._C.apa_selective_attention_sp(..., sinks=...)
./scripts/apa_sp5_model.py:125:        res = tc._C.apa_selective_attention_sp(
./scripts/apa_sp5_shape_pin.py:117:        out_all = tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, big, True, st, False)
./scripts/apa_sp5_shape_pin.py:128:        out_zero = tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, big, True, zero, False)
./scripts/apa_sp5_shape_pin.py:129:        out_none = tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, big, True, None, False)
./scripts/apa_sp5_shape_pin.py:146:        out_d, sel = tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, delta, True, st, True)
./scripts/apa_spd1_bench.py:141:    out, mask = tc._C.apa_selective_attention_sp(q, k, kq, v,
./scripts/apa_spd1_bench.py:227:            funcs[name] = lambda t=engine[dtype]: tc._C.apa_selective_attention_sp(*t, scale, shape['delta'], shape['causal'])
./scripts/apa_spd1_common.py:72:                         entry="tensor_cuda._C.apa_selective_attention_sp; prefill / split-K decode; TC_APA_SP=1",
./scripts/apa_spd1_common.py:136:                                 'apa_selective_attention_sp', 'set_alloc_pooling']},
./tensor_cuda/include/tc/ops.h:15:NDArray apa_selective_attention_sp(const NDArray& q, const NDArray& k,
./tensor_cuda/include/tc/ops.h:18:std::pair<NDArray,NDArray> apa_sp1_1_baseline_diagnostics(const NDArray& q, const NDArray& kq);
./tensor_cuda/include/tc/ops.h:20:float apa_sp2_delta(double epsilon, double e_q);
./tensor_cuda/include/tc/ops.h:21:NDArray apa_selective_attention_sp_epsilon(const NDArray& q, const NDArray& k,
./tensor_cuda/include/tc/ops.h:24:std::pair<NDArray,NDArray> apa_sp2_scores(const NDArray& q, const NDArray& k,
./tensor_cuda/include/tc/ops.h:26:NDArray apa_sp2_baseline_thresholds(const NDArray& q, const NDArray& kq,
./tensor_cuda/include/tc/ops.h:317:NDArray apa_spq1_rotate(const NDArray&,const NDArray&,int,bool);
./tensor_cuda/src/apa_sp1_1.cuh:200:std::pair<NDArray,NDArray> apa_sp1_1_baseline_diagnostics(const NDArray& q, const NDArray& kq) {
./tensor_cuda/src/apa_sp1_1.cuh:207:    throw std::runtime_error("apa_sp1_1_baseline_diagnostics: registered fp32 CUDA Part B shape only");
./tensor_cuda/src/apa_sp2.cuh:3:float apa_sp2_delta(double epsilon, double e_q) {
./tensor_cuda/src/apa_sp2.cuh:16:NDArray apa_selective_attention_sp_epsilon(const NDArray& q, const NDArray& k,
./tensor_cuda/src/apa_sp2.cuh:20:  return apa_selective_attention_sp(q, k, kq, v, scale,
./tensor_cuda/src/apa_sp2.cuh:21:      apa_sp2_delta(epsilon, e_q), is_causal, sinks, selected);
./tensor_cuda/src/apa_sp2.cuh:73:std::pair<NDArray,NDArray> apa_sp2_scores(const NDArray& q, const NDArray& k,
./tensor_cuda/src/apa_sp2.cuh:91:  cuda_check_last("apa_sp2_scores");
./tensor_cuda/src/apa_sp2.cuh:95:NDArray apa_sp2_baseline_thresholds(const NDArray& q, const NDArray& kq,
./tensor_cuda/src/apa_sp2.cuh:113:  cuda_check_last("apa_sp2_baseline_thresholds");
./tensor_cuda/src/apa_spo/eb2b.cuh:137:  NDArray pq=apa_spq1_rotate(q,rot,KVH,true);
./tensor_cuda/src/apa_spo/packed_launcher.cuh:14:NDArray apa_spq1_rotate(const NDArray& q,const NDArray& r,int KVH,bool tq){
./tensor_cuda/src/apa_spo/packed_launcher.cuh:20: cuda_check_last("apa_spq1_rotate");return out;
./tensor_cuda/src/apa_spo/packed_launcher.cuh:49: NDArray pq=apa_spq1_rotate(q,r,k.shape[1],fmt=="tq4");
./tensor_cuda/src/apa_spo/spq4.cuh:456:    NDArray pq=apa_spq1_rotate(q,r,k.shape[1],false);   // mutant: query NOT rotated
./tensor_cuda/src/apa_spo/spq4.cuh:458:    NDArray pq=apa_spq1_rotate(q,r,k.shape[1],true);
./tensor_cuda/src/bindings.cpp:2334:  m.def("apa_selective_attention_sp", [](Tensor& q, Tensor& k, Tensor& kq,
./tensor_cuda/src/bindings.cpp:2342:    NDArray result = tc::apa_selective_attention_sp(q.data(), k.data(),
./tensor_cuda/src/bindings.cpp:2351:  m.def("apa_sp1_1_baseline_diagnostics", [](Tensor& q, Tensor& kq) {
./tensor_cuda/src/bindings.cpp:2352:    auto result = tc::apa_sp1_1_baseline_diagnostics(q.data(),kq.data());
./tensor_cuda/src/bindings.cpp:2356:  m.def("apa_sp2_delta", &tc::apa_sp2_delta, py::arg("epsilon"), py::arg("e_q"));
./tensor_cuda/src/bindings.cpp:2357:  m.def("apa_selective_attention_sp_epsilon", [](Tensor& q, Tensor& k, Tensor& kq,
./tensor_cuda/src/bindings.cpp:2363:    NDArray result = tc::apa_selective_attention_sp_epsilon(q.data(), k.data(),
./tensor_cuda/src/bindings.cpp:2372:  m.def("apa_sp2_scores", [](Tensor& q, Tensor& k, Tensor& kq, double scale,
./tensor_cuda/src/bindings.cpp:2374:    auto r = tc::apa_sp2_scores(q.data(),k.data(),kq.data(),(float)scale,first,count,wcoop);
./tensor_cuda/src/bindings.cpp:2378:  m.def("apa_sp2_baseline_thresholds", [](Tensor& q, Tensor& kq, double scale,
./tensor_cuda/src/bindings.cpp:2380:    return Tensor::make(tc::apa_sp2_baseline_thresholds(q.data(),kq.data(),
./tensor_cuda/src/bindings.cpp:2779: m.def("packed_rotate",[](Tensor& q,Tensor& r,int kvh,bool tq){return Tensor::make(tc::apa_spq1_rotate(q.data(),r.data(),kvh,tq),false);});
./tensor_cuda/src/kernels.cu:8418:NDArray apa_selective_attention_sp(const NDArray& q, const NDArray& k,
./tensor_cuda/tensor_cuda/__init__.py:1536:from .apa_sp2 import RegisteredMargins, apa_selective_attention_sp, quantize_sp2_keys
./tensor_cuda/tensor_cuda/__init__.py:1537:__all__ += ["RegisteredMargins", "apa_selective_attention_sp", "quantize_sp2_keys"]
./tensor_cuda/tensor_cuda/__init__.py:1554:    _C.apa_selective_attention_sp_d1 = _apa_spo2_entry
./tensor_cuda/tensor_cuda/__init__.py:1555:    _C.apa_spo2_dump = _spo2.run
./tensor_cuda/tensor_cuda/__init__.py:1603:    _C.apa_selective_attention_sp_d1_packed=_apa_spq1_entry
./tensor_cuda/tensor_cuda/__init__.py:1604:    _C.apa_spq1_dump=_spo2.packed_run
./tensor_cuda/tensor_cuda/__init__.py:1605:    _C.apa_spq1_rotate=_spo2.packed_rotate
./tensor_cuda/tensor_cuda/__init__.py:1618:    _C.apa_selective_attention_sp_d1_packed=_apa_spq4_entry
./tensor_cuda/tensor_cuda/apa_sp2.py:4:Callers must supply keys from quantize_sp2_keys and the calibrated input regime.
./tensor_cuda/tensor_cuda/apa_sp2.py:61:def quantize_sp2_keys(k, bulk_bits):
./tensor_cuda/tensor_cuda/apa_sp2.py:72:def apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *, bulk_bits,
./tensor_cuda/tensor_cuda/apa_sp2.py:77:    kq must come from quantize_sp2_keys(k, bulk_bits). Calibration does not certify
./tensor_cuda/tensor_cuda/apa_sp2.py:88:    return _C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale),
./tensor_cuda/tests/test_apa_pidv.py:185:    for exc in (ImportError("cannot import name 'quantize_sp2_keys' from 'tensor_cuda.apa_sp2'",
./tensor_cuda/tests/test_apa_sp1_host.py:30:        runtime._C.apa_selective_attention_sp(t,t,t,t,1.,0.)
./tensor_cuda/tests/test_apa_sp1_host.py:38:        runtime._C.apa_selective_attention_sp(t,t,t,t,scale,delta)
./tensor_cuda/tests/test_apa_sp1_host.py:45:        runtime._C.apa_selective_attention_sp(t,t,t,t,1.,0.)
./tensor_cuda/tests/test_apa_sp1_host.py:48:        runtime._C.apa_selective_attention_sp(t,t,t,t,1.,0.)
./tensor_cuda/tests/test_apa_sp1_host.py:55:        runtime._C.apa_selective_attention_sp(t,t,t,t,1.,0.)
./tensor_cuda/tests/test_apa_sp2.py:111:            actual=tc._C.apa_sp2_delta(epsilon,e)
./tensor_cuda/tests/test_apa_sp2.py:115:        with pytest.raises(RuntimeError):tc._C.apa_sp2_delta(eps,e)
./tensor_cuda/tests/test_apa_sp2.py:123:        delta=tc._C.apa_sp2_delta(epsilon,e)
./tensor_cuda/tests/test_apa_sp2.py:125:        for call in [lambda:tc._C.apa_selective_attention_sp(*args,delta),
./tensor_cuda/tests/test_apa_sp2.py:126:                     lambda:tc._C.apa_selective_attention_sp_epsilon(*args,epsilon,e)]:
./tensor_cuda/tests/test_apa_sp2.py:146:        tc._C.apa_selective_attention_sp_epsilon(*args,.1,.5)
./tensor_cuda/tests/test_apa_sp2.py:172:    monkeypatch.setattr(tc._C,'apa_selective_attention_sp_epsilon',lambda *a:calls.append(a) or 'ok')
./tensor_cuda/tests/test_apa_sp2.py:175:    assert tc.apa_selective_attention_sp(q,q,q,q,.125,.001,bulk_bits=2,margins=margins)=='ok'
./tensor_cuda/tests/test_apa_sp2.py:178:        tc.apa_selective_attention_sp(q,q,q,q,.125,.001,bulk_bits=2,margins=None)
./tensor_cuda/tests/test_apa_sp2.py:181:        tc.apa_selective_attention_sp(q,q,q,q,.125,.001,bulk_bits=2,margins=margins)
./tensor_cuda/tests/test_apa_sp3.py:35:    entry=s[s.index('NDArray apa_selective_attention_sp('):]
./tensor_cuda/tests/test_apa_sp3.py:76:        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,1.)
./tensor_cuda/tests/test_apa_sp3.py:79:        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,1.)
./tensor_cuda/tests/test_apa_sp3.py:81:        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,float('inf'))
./tensor_cuda/tests/test_apa_sp3.py:127:    obj.tc=SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=lambda *a: calls.append(a) or 'out'))
./tensor_cuda/tests/test_apa_sp3_a4.py:166:    obj.tc=SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=native))
./tensor_cuda/tests/test_apa_sp3_a4.py:177:        obj.tc._C.apa_selective_attention_sp=bad
./tensor_cuda/tests/test_apa_sp3_a6.py:93:          _C=NS(apa_selective_attention_sp=native))
./tensor_cuda/tests/test_apa_sp4g.py:61:    s=(R/'tensor_cuda/src/kernels.cu').read_text();entry=s[s.index('NDArray apa_selective_attention_sp('):]
./tensor_cuda/tests/test_apa_sp4g.py:86:    with pytest.raises(RuntimeError,match='default OFF'):tc._C.apa_selective_attention_sp(q,k,k,k,1.,1.)
./tensor_cuda/tests/test_apa_sp4g.py:88:    with pytest.raises(RuntimeError,match='share CUDA device'):tc._C.apa_selective_attention_sp(q,k,k,k,1.,1.)
./tensor_cuda/tests/test_apa_sp4g.py:90:        with pytest.raises(RuntimeError,match='finite nonnegative'):tc._C.apa_selective_attention_sp(q,k,k,k,1.,delta)
./tensor_cuda/tests/test_apa_sp4g.py:237:    calls=[];tc=SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=lambda *args:calls.append(args) or 'out'))
./tensor_cuda/tests/test_apa_sp4g_a2.py:135:    owner.tc=SimpleNamespace(matmul=mm,causal_softmax=lambda x:T(np.ones(x.shape,np.float32)/3),_C=SimpleNamespace(apa_selective_attention_sp=sp))
./tensor_cuda/tests/test_apa_sp4g_a3.py:58:    return SimpleNamespace(matmul=mm,causal_softmax=sm,_C=SimpleNamespace(apa_selective_attention_sp=sp),
./tensor_cuda/tests/test_apa_sp4g_a3.py:134:    tc=fake_tc();seen=[];original=tc._C.apa_selective_attention_sp
./tensor_cuda/tests/test_apa_sp4g_a3.py:136:    tc._C.apa_selective_attention_sp=sp
./tensor_cuda/tests/test_apa_sp4g_a3.py:203:    original=tc._C.apa_selective_attention_sp
./tensor_cuda/tests/test_apa_sp4g_a3.py:208:    tc._C.apa_selective_attention_sp=broken
./tensor_cuda/tests/test_apa_sp4g_a4.py:34:    return SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=sp),matmul=mm,causal_softmax=lambda x:x),seen
./tensor_cuda/tests/test_apa_sp4g_a5.py:229:    owner.tc=NS(_C=NS(apa_selective_attention_sp=sp))
```

### GRAPA

```sh
CUDA_VISIBLE_DEVICES= timeout 40s rg -n --sort path --hidden --no-ignore -g '*.{py,pyi,ipynb,cpp,cu,cuh,h,c,sh,js,ts,rs}' -g '!**/.git/**' -g '!**/__pycache__/**' -g '!**/.venv/**' -g '!**/node_modules/**' -g '!**/build*/**' -g '!**/Fork/**' -g '!**/WorkTree/**' '\b(apa_selective_attention_sp(_epsilon|_d1_packed|_d1)?|quantize_sp2_keys|apa_spq1_(dump|rotate)|apa_spo2_dump|apa_sp2_(delta|scores|baseline_thresholds)|apa_sp1_1_baseline_diagnostics)\b' /mnt/ForgeRealm/GRAPA-Native-LLM
```

Exit 0. Full output:

```text
/mnt/ForgeRealm/GRAPA-Native-LLM/artifacts/bp_census_1/engine_patch/src/bindings.cpp:667:  m.def("apa_selective_attention_sp", [](Tensor& q, Tensor& k, Tensor& kq,
/mnt/ForgeRealm/GRAPA-Native-LLM/artifacts/bp_census_1/engine_patch/src/bindings.cpp:675:    NDArray result = tc::apa_selective_attention_sp(q.data(), k.data(),
/mnt/ForgeRealm/GRAPA-Native-LLM/artifacts/bp_census_1/engine_patch/src/bindings.cpp:684:  m.def("apa_sp1_1_baseline_diagnostics", [](Tensor& q, Tensor& kq) {
/mnt/ForgeRealm/GRAPA-Native-LLM/artifacts/bp_census_1/engine_patch/src/bindings.cpp:685:    auto result = tc::apa_sp1_1_baseline_diagnostics(q.data(),kq.data());
/mnt/ForgeRealm/GRAPA-Native-LLM/artifacts/bp_census_1/engine_patch/src/bindings.cpp:689:  m.def("apa_sp2_delta", &tc::apa_sp2_delta, py::arg("epsilon"), py::arg("e_q"));
/mnt/ForgeRealm/GRAPA-Native-LLM/artifacts/bp_census_1/engine_patch/src/bindings.cpp:690:  m.def("apa_selective_attention_sp_epsilon", [](Tensor& q, Tensor& k, Tensor& kq,
/mnt/ForgeRealm/GRAPA-Native-LLM/artifacts/bp_census_1/engine_patch/src/bindings.cpp:696:    NDArray result = tc::apa_selective_attention_sp_epsilon(q.data(), k.data(),
/mnt/ForgeRealm/GRAPA-Native-LLM/artifacts/bp_census_1/engine_patch/src/bindings.cpp:705:  m.def("apa_sp2_scores", [](Tensor& q, Tensor& k, Tensor& kq, double scale,
/mnt/ForgeRealm/GRAPA-Native-LLM/artifacts/bp_census_1/engine_patch/src/bindings.cpp:707:    auto r = tc::apa_sp2_scores(q.data(),k.data(),kq.data(),(float)scale,first,count,wcoop);
/mnt/ForgeRealm/GRAPA-Native-LLM/artifacts/bp_census_1/engine_patch/src/bindings.cpp:711:  m.def("apa_sp2_baseline_thresholds", [](Tensor& q, Tensor& kq, double scale,
/mnt/ForgeRealm/GRAPA-Native-LLM/artifacts/bp_census_1/engine_patch/src/bindings.cpp:713:    return Tensor::make(tc::apa_sp2_baseline_thresholds(q.data(),kq.data(),
```

### GraftRepositoryMemory (GraftRepository checkout)

```sh
CUDA_VISIBLE_DEVICES= timeout 40s rg -n --sort path --hidden --no-ignore -g '*.{py,pyi,ipynb,cpp,cu,cuh,h,c,sh,js,ts,rs}' -g '!**/.git/**' -g '!**/__pycache__/**' -g '!**/.venv/**' -g '!**/node_modules/**' -g '!**/build*/**' -g '!**/Fork/**' -g '!**/WorkTree/**' '\b(apa_selective_attention_sp(_epsilon|_d1_packed|_d1)?|quantize_sp2_keys|apa_spq1_(dump|rotate)|apa_spo2_dump|apa_sp2_(delta|scores|baseline_thresholds)|apa_sp1_1_baseline_diagnostics)\b' /mnt/ForgeRealm/GraftRepository
```

Exit 1. Full output:

```text
(no matches)
```

### Project-Voice

```sh
CUDA_VISIBLE_DEVICES= timeout 40s rg -n --sort path --hidden --no-ignore -g '*.{py,pyi,ipynb,cpp,cu,cuh,h,c,sh,js,ts,rs}' -g '!**/.git/**' -g '!**/__pycache__/**' -g '!**/.venv/**' -g '!**/node_modules/**' -g '!**/build*/**' -g '!**/Fork/**' -g '!**/WorkTree/**' '\b(apa_selective_attention_sp(_epsilon|_d1_packed|_d1)?|quantize_sp2_keys|apa_spq1_(dump|rotate)|apa_spo2_dump|apa_sp2_(delta|scores|baseline_thresholds)|apa_sp1_1_baseline_diagnostics)\b' /mnt/ForgeRealm/Project-Voice
```

Exit 1. Full output:

```text
(no matches)
```
