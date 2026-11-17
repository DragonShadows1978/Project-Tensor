# Prior art: exact receipt synthesis (HOUSE_RULES); no new algorithm.
from pathlib import Path
import json,datetime,hashlib,subprocess,shutil
A=Path('artifacts/apa_p1');root=Path.cwd();load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
g1=load(A/'g1.json');build=load(A/'build.json');sass=load(A/'sass_comparison.json');g3=load(A/'g3_comparison.json');g4=load(A/'g4_staging.json');exp=load(A/'EXPECTED.json')
assert all(x['status']=='PASS' for x in (g1,build,sass,g3,g4))
header='| Header | Lab/port common SHA-256 | Identity |\n|---|---|---|\n'
for r in g1['headers']:header+=f"| {Path(r['path']).name} | `{r['working_sha256']}` | PASS |\n"
(A/'HEADER_IDENTITY.md').write_text(header)
hunks='| Seam | Added lines | D2 source | Production bytes |\n|---|---:|---|---|\n'
for r in g1['hunks']:hunks+=f"| {r['path']} | {r['added_start']}–{r['added_end']} | {r['source']} | unchanged |\n"
(A/'HUNK_MAP.md').write_text(hunks)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (A/'LEDGER.md').open('a') as f:
    f.write(f'''\n- {now}: G4 staged YES. STAGED_GPU_LINES.sh has 51 foreground forge-gpu-run lines, each --max 600, inner timeout 540s, outer timeout 1750s. Six real captured model/S cases, one allocation line over six caches, 44 original production test files, 2078 collected IDs. EXPECTED.json frozen before any device execution; sha256 {g4['expected_sha256']}.\n- Input data: SPQ3 8K uint16 bf16 Q/K/V captures for the 2K/8K prefix checks and EB4 P0 32K captures for 32K, two model layers (1.7B L5/G2 and 4B L7/G4, KVH8/D128). All captures validated against aaa9b4ea's recorded SHA values. No model run or capture regeneration. INPUT_CENSUS.json.\n- Staged numerical contract: EB3 variant 0 and EB4 T-fresh, row carry, delta float32(8.1), registered layer fractions .02/.05, compact visible K/V allocations (no DS1 pitches), 9 append lengths, last four decode rows, diagnostic and nondiagnostic outputs. All saved arrays use unsigned integer views and exact equality; every reference must be nonzero/nonconstant. Shared tables, codes, norms, int8 V and scales also compared.\n- Staged bytes: actual Storage.nbytes from unique live token allocations for all 28/36 layers, 580 B/key/KV head beyond exact layer0 (512 B). Metadata tables and scratch excluded, matching AB-1 G2 arithmetic. SP-tk8 is 712 B and is not mislabeled 580 B.\n- CPU-built the original AB-1 read-only accessor against production headers in build-p1/interop (source SHA unchanged d2099ccd244f38b95f7d80c18b41d6e50957f2f494ec57cc7480c1d190658992), isolated from the lab accessor. CPU pointer/storage/alias smoke PASS; lab reference module import-only smoke PASS, no device use. interop_build.json, interop_cpu_smoke.json, lab_import_cpu_smoke.json.\n- Static staging checks PASS: 99 frozen code pins, capture pins, Python and bash syntax, complete command/ID/S/append census, restored port module SHA. g4_staging.json. GPU predictions P142/P143 remain unscored except P143's CPU part matches.\n- Prior art: AB-1 reference-pair and integer-bit checks (2026), SPQ3/EB4 post-norm/post-RoPE real captures (2026), pybind11 cross-module access (2016), CUDA memcpy/Storage metadata (NVIDIA), Numba CUDA Array Interface (2018), SHA-256 (NIST 2015), pytest report phases. These supply staging inspection/plumbing only; no selection/numerical algorithm introduced.\n- Final G1 rerun PASS; git ls-files --others --exclude-standard found no path outside authorized prefixes. No tests, quant.py, functional.py or production lifetime/DLPack lines changed. No git writes, subagents, GPU, file moves or deletions. Nothing to DELETE_ME. Retained build/scratch enumerated in FILES_WRITTEN.md.\n''')
report=f'''# APA-P1 report

ACCEPT under the registered CPU/staging rule: **G1 PASS; G2 PASS; G3 PASS; G4 staged YES.** GPU verdicts are pending the lead's run. P142 is unscored; P143's CPU part matched and its GPU part is unscored.

## Gate receipts

| Gate | Verdict | Evidence class and receipts |
|---|---|---|
| G1 scope | PASS | Static source identity: [g1.json](g1.json), [g1_tracked.diff](g1_tracked.diff), [g1_diff_stat.txt](g1_diff_stat.txt), [hunk_map.json](hunk_map.json) |
| G2 build + SASS | PASS | CUDA build: [build.json](build.json), [configure.log](configure.log), [build.log](build.log), [module.sha256](module.sha256); instruction census: [sass_comparison.json](sass_comparison.json), [sass_counts.tsv](sass_counts.tsv) |
| G3 CPU outcome parity | PASS | Exact per-ID comparison: [g3_comparison.json](g3_comparison.json), [g3_ids.tsv](g3_ids.tsv), [g3_production.json](g3_production.json), [g3_port.json](g3_port.json), both arm logs |
| G4 staging | YES, not executed | Static/CPU preparation: [STAGED_GPU_LINES.sh](STAGED_GPU_LINES.sh), [EXPECTED.json](EXPECTED.json), [EXPECTED.sha256](EXPECTED.sha256), [g4_staging.json](g4_staging.json), [INPUT_CENSUS.json](INPUT_CENSUS.json) |

Registration: immutable committed `orders/APA_P1_PORT_PACKED.md`, [REGISTRATION.md](REGISTRATION.md), [G4_REGISTRATION.md](G4_REGISTRATION.md). No changes to the order or registrations. This is the additive port slice only; acceptance does not switch a default or promote the path.

## Mechanism and precision

The port retains exact bf16 K and V. Its bulk is **stored** as packed TQ4 4-bit centroid codes plus one fp32 norm per key: 64+4 bytes at D128, or 4.25 stored bits/element. It is **computed** through the inherited variant-B split-bf16 hi/lo tensor-core products, with fp32 accumulation, norm and score scaling. The tail reads the exact bf16 K, uses bf16 MMA with fp32 accumulation, and all keys remain in the denominator. P rounds to bf16 before the inherited PV accumulation. No header arithmetic was edited.

SP token storage is `256 K + 256 V + 64 codes + 4 norm = 580 B/key/KV head`. SP-tk8 adds 128 B of int8 V and 4 B of V scale: **712 B/key/KV head**. Layer 0 remains exact bf16 K/V, 512 B/key/KV head. The 580-byte claim applies to SP token state, excludes shared tables and decode scratch, and makes no NVML-residency or performance claim. The symmetric-7 2P-INT4 comparator has equal token bytes but different codes/arithmetic.

Preflight read `docs/APA.md`, the paper's sections 2–3, and pinned survey S2/S5 at `e5538604`. These gates exercise packed bulk, exact tail, append and the additive production API seams. They do not evaluate model quality, performance, S3 hardening/promotion, the DS1 pitch ABI or a blind red team. Those are not claimed fixed or completed.

## Hunk map

{hunks}

The 16 new headers and five seams are the complete engine-source delta. All five production files retain their entire original byte prefix; **zero removed lines**. `git diff --stat 0e1dea3 -- tensor_cuda/` records 290 insertions in five tracked files; it omits the 16 untracked new headers, which are enumerated separately below. The only pre-existing non-engine committed delta from `0e1dea3` is the order file. The feature is explicitly default-off (`TC_APA_PACKED`); the companion initializer adds opt-in entries and does not redirect the existing SP/SP2 APIs. K1 AOT source/initializer/export is absent.

## Header identity table

Every reference is read with `git show aaa9b4ea:<path>`, never from lab working source files. All headers are under `tensor_cuda/src/apa_spo/`.

{header}

## Build and SASS

CUDA **12.6 V12.6.85**, sm89, Release, CMake equivalent of `tensor_cuda/build.sh 89`, output `tensor_cuda/build-p1/`. CMake and nvcc commands/exits are in build.json; the existing production output location is preserved. The build uses offline read-only pybind11 sources, so no dependency git operation occurs.

Defines applied consistently to CUDA and C++: `TC_APA_PACKED=1`, `TC_APA_EB2B=1`, `TC_APA_EB3=1`, `TC_APA_EB4=1`, `APA_EB2B_MUTANT=0`, `APA_FXTC1_VARIANT_DEFAULT=2`. The inherited EB3/EB4 non-mutant defaults are unchanged.

| Module | SHA-256 |
|---|---|
| Port | `{build['module_sha256']}` |
| Production reference | `{build['production_sha256']}` |
| AB-1 main reference | `{build['lab_sha256']}` |

The lab survey worktree's main-build receipt points to the actual read-only AB-1 build in `tensor_fork_apa_wt_ab1/artifacts/apa_ab1/build/main/`. That module's SHA matches the pinned `aaa9b4ea` G0 receipt. Both baseline modules remain untouched. The port module is restored in the package after G3 and matches the saved build-p1 copy.

**SASS: 213/213 ported kernels match. 428,344 instructions in each arm. Zero instruction-count differences, zero opcode-histogram differences, zero instruction-text differences.** The census covers all new-header namespaces (spo2/spo4/spo5, spq1/spq4/spq6, EB2b/EB3/EB4) plus baseline_dump, including predecessor/ablation instantiations. cuobjdump 12.6 selects sm89 and counts every address-prefixed instruction including alignment NOPs. Matched mangled symbols require no path normalization. Difference/cause list: **empty**. Raw disassemblies: sass_port.txt and sass_lab.txt. Source paths differ; emitted instruction text does not.

## G3 CPU outcomes

**689/689 IDs matched exactly**, including every setup/call/teardown outcome string; zero collection errors and zero outcome differences. Threads were 1; CUDA_VISIBLE_DEVICES was empty. No production test was edited, deselected, suppressed or given an extra skip. Per-arm module SHA provenance is recorded in each JSON. This is an outcome-parity PASS, **not a passing-suite claim**.

| Outcome | Production | Port |
|---|---:|---:|
| PASSED | 410 | 410 |
| SKIPPED | 139 | 139 |
| FAILED | 125 | 125 |
| ERROR (setup) | 15 | 15 |
| Total IDs | 689 | 689 |

{(A/'G3_TABLE.md').read_text()}

P/S/F/E means passed/skipped/failed/setup error. Exact failure messages are retained in the per-arm JSONs, including unchanged missing historical snapshots/manifests and CUDA-no-device errors. Representative literal failures:

```text
FileNotFoundError: [Errno 2] No such file or directory: '<worktree>/artifacts/apa_sp1/build/manifest.json'
RuntimeError: cudaMalloc failed: no CUDA-capable device is detected
RuntimeError: empty: the current-device query cudaGetDevice failed (cudaErrorNoDevice: no CUDA-capable device is detected); nothing was allocated
```

The first path is shortened here only; the receipt retains its full path. Historical `load_runtime` pins are not bypassed, so some host guard tests stop before reaching an imported module. Existing no-device failures and missing fixtures are **not claimed fixed**. All module imports are independently provenance-checked by the pytest receipt plugin.

## Staged GPU lines and expectations

The lead runs from the worktree root:

```bash
bash artifacts/apa_p1/STAGED_GPU_LINES.sh
```

There are **51 literal lease-wrapped lines**. Every one has `forge-gpu-run --max 600`, an inner `timeout 540s`, and an outer `timeout 1750s`. GPU visibility is enabled only inside the lease; the seat never executes any of them. The script stops below the registered free-space rails and writes exit/argv/log-hash receipts only after each process exits. A window/tool/timeout failure remains RED. Each new run has a separate timestamped directory and create-only child receipts.

| Lines | Registered operands | Expected |
|---|---|---|
| 6 bitwise | Qwen3-1.7B L5 (G2), Qwen3-4B L7 (G4), KVH8, D128; S=2048,8192,32768; both EB3/SP and EB4/SP-tk8 | Every saved integer-view element equal to the lab main build; no tolerance |
| Same six lines: append | Prefix lengths 1,63,64,65,S-4,S-3,S-2,S-1,S; compact K/V; last four decode outputs | Codes, fp32 norm bits, valid-row history, outputs, masks, flags, int8 V and scale bits equal |
| 1 bytes | All 28/36 model layers, KVH8, each S | Actual unique Storage.nbytes = `8*S*(512+(NL-1)*580)` |
| 44 regression | 43 original APA/APAMQ files + DLPack, 2078 frozen collected IDs | Same exact per-ID phase/outcome strings under paired production/port modules; every difference includes messages |

The numerical workers read real SPQ3 post-norm/post-RoPE captures for 2K/8K prefixes and EB4 P0 captures at 32K. SHA pins come from `aaa9b4ea` receipts. Each bf16 upload is checked against the captured integer bits. Every saved reference array must be nonzero and nonconstant; no vacuous equality is accepted. Diagnostic and nondiagnostic outputs must also agree within each worker. Delta float32(8.1), row carry, EB3 variant 0 and registered EB4 fractions .02/.05 are unchanged. Shared rotation/codebook state is also compared. Source/module/capture drift fails closed.

| S | Expected total SP token bytes, 1.7B | Expected total SP token bytes, 4B |
|---:|---:|---:|
| 2048 | 264,962,048 | 340,983,808 |
| 8192 | 1,059,848,192 | 1,363,935,232 |
| 32768 | 4,239,392,768 | 5,455,740,928 |

Reference arithmetic: pinned AB-1 G2 allocation records in reference_g2.json. This check covers actual Storage payload allocations, excludes shared tables/transient scratch, and does not assert the older AB-1 overall G2/NVML gate passed. Expected GPU test outcomes are generated by the paired production baseline in the same staged line; CPU skip/failure strings are not substituted for a GPU reference. G3's own CPU baseline remains g3_production.json.

EXPECTED.json SHA-256: `{g4['expected_sha256']}`. Full frozen test IDs, all command lines and all source/module pins are in that file. Literal staged commands follow:

```bash
{chr(10).join(x['line'] for x in exp['commands'])}
```

## Prior art

Taken unchanged: Project-Tensor SPQ1/SPQ4/SPQ6/EB2b/EB3/EB4 (2026). Git (2005) three-way merge supplies survey S5's dry-run method, cited at every seam; this order uses additive assembly and does not perform a merge. Header annotations retain TurboQuant (Zandieh et al., 2025), BLASST (Yuan et al., 2025, running-max criterion), Flash-Decoding (Dao et al., 2023), FlashInfer (Ye et al., 2025), KIVI (Liu et al., 2024), ThriftAttention (2026), NVIDIA threadFenceReduction, split-bf16 and prefetch lineage. External metadata is inherited and **unverified — lead to check**; search leads already at code sites include “CUTLASS swap AB small M”, “swapAB GEMM tensor core decode”, “TurboQuant 2504.19874”, “BLASST 2512.12087”, “ThriftAttention block precision”, “NVIDIA threadFenceReduction”. No new literature search is claimed.

Staging takes AB-1 reference-pair/raw-integer checks and SPQ3/EB4 captures (2026), pybind11 cross-module access (2016), CUDA memcpy/storage metadata, SHA-256 (NIST 2015) and pytest's report API. The production accessor is the exact AB-1 source rebuilt against production headers; its read-only CPU pointer/byte/alias smoke passed. No numerical or selection algorithm is introduced. **No prior art known to me for this exact hunk-selection/port assembly; no novelty claimed.** Code-site seam notes, original header annotations and LEDGER.md carry the lineage.

## Files written and leave-and-list

The complete per-file inventory is [FILES_WRITTEN.md](FILES_WRITTEN.md), with machine-readable inventory [FILES_WRITTEN.json](FILES_WRITTEN.json). It enumerates all 16 headers, five appended seams, package module, every build-p1 output, every apa_p1 artifact and retained /tmp/apa_p1_seat scratch entry. Nothing was moved or deleted; no DELETE_ME entry is needed. No direct writes to logs/. No git writes, subagents, background shell jobs, GPU calls with a visible device, service changes or baseline writes. Source tests and every other engine file remain untouched.

Budget: started 2026-11-02T18:06Z; report synthesized {now}, below the 3-hour rail. Every seat command ran with empty CUDA_VISIBLE_DEVICES and an explicit timeout below 30 minutes; running foreground commands were polled only at the requested maximum wait. Build/scratch remain in the authorized locations. The final default switch, promotion requirements, DS1 pitch ABI, red team, model quality and performance remain for later orders.
'''
(A/'REPORT.md').write_text(report)
commands='''# APA-P1 command receipt index

All seat shell commands used CUDA_VISIBLE_DEVICES='', TMPDIR=/tmp/apa_p1_seat after its creation, PYTHONDONTWRITEBYTECODE=1 for Python. Foreground commands had explicit timeouts. No git writes.

- Preflight named reads: /mnt/ForgeRealm/HOUSE_RULES.md; production/local AGENTS.md; docs/APA.md; docs/APA_PAPER_DRAFT.md sections 2–3; git show e5538604:artifacts/apa_2p/SURVEY.md (saved preflight_survey.txt).
- Pin/instrument reads: git show/log at 0e1dea3, aaa9b4ea and order HEAD; nvcc/pytest/g++ versions; production module sha256; pinned g0/g2, build scripts, interop source, SPQ3/EB4 capture metadata through git show. Actual modules/captures read only from named artifact paths.
- timeout 30s python artifacts/apa_p1/port.py
- timeout 30s python artifacts/apa_p1/check_scope.py (initial and final)
- timeout 1650s python artifacts/apa_p1/build_gate.py (its configure/build argv, exit codes and durations are build.json)
- timeout 90s /usr/local/cuda-12.6/bin/cuobjdump --dump-sass --gpu-architecture sm_89 <port/lab module> (separate outputs sass_port.txt/sass_lab.txt)
- timeout 30s python artifacts/apa_p1/compare_sass.py (expanded to full header namespace closure before final census)
- timeout 1400s python artifacts/apa_p1/cpu_gate.py (per-arm pytest argv/exits in g3_exits.json; fresh basetemps; port binary restored in finally)
- timeout 30s python artifacts/apa_p1/compare_g3.py
- timeout 45s python artifacts/apa_p1/prepare_stage.py; CPU g++ accessor build (argv/log in interop_build.json/interop_build.log)
- CPU-only accessor smoke (interop_cpu_smoke.json); pinned lab import smoke (lab_import_cpu_smoke.json)
- timeout 90s CPU pytest --collect-only for 44 original GPU test files (gpu_collection_cpu.json/log); no tests/device calls executed by collection
- timeout 30s python artifacts/apa_p1/freeze_stage.py (created G4 registration, script, EXPECTED.json)
- timeout 30s python artifacts/apa_p1/verify_stage.py; bash -n STAGED_GPU_LINES.sh (static/CPU only)
- git ls-files --others --exclude-standard (untracked_scope.json); no untracked path outside authorized prefixes
- timeout 30s python artifacts/apa_p1/finish_report.py; final inventory/receipt verification

Failed setup attempts (before product writes): nested Python string SyntaxError while generating port.py; immediate subsequent invocation found no port.py. Corrected quoting, then successful port.py execution. Ledger records both; no gate was weakened.
'''
(A/'COMMANDS.md').write_text(commands)
print('Report, ledger, hunk map, header identity table and command index written.')
