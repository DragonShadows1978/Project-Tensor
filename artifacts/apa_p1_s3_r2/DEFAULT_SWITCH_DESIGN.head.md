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
