# APA-SP3 amendment 4 — CPU delivery PASS; GPU execution UNRUN

**89 CPU tests passed; final mutation gate 8/8 killed, zero invalid.**
All 147 local existing PASS receipts remain **source-compatible**, with zero
source-driven reruns required. All 148 existing job receipts, the base
registration and the order are unchanged. No GPU result is claimed by this seat.

The original `capture_b4_B_8192` **WORKER_TIMEOUT is not claimed fixed by GPU
evidence**. Split execution is implemented and CPU-validated. Full 32768
captures also need more disk capacity than this filesystem currently has.

## 1. New/changed cells (ids, kinds, deps, estimates); capture split scheme; T arm entry point and SDPA backend; ceiling grid

The dry-run contains **859 cells**, including **172 new/changed cells**:
167 added IDs and five existing capture IDs changed to aggregations.
Every exact ID, dependency, estimate and rail appears in [A4_CELLS.md](A4_CELLS.md)
and [a4_lead_list_final.json](a4_lead_list_final.json).

| Kind | Count | Identity / scheme | Worker planning estimate |
|---|---:|---|---|
| `capture_range` | 144 | `capture_b{bits}_{arm}_{S}_r{lo:02d}_{hi:02d}` | 30–188 s; final 8192 range 30–172 s |
| `capture_aggregate` | 7 | Five existing 8192 capture IDs, plus B/C bulk4 at 32768 | 5–280 s |
| `torch_reference` | 3 | `ppl_T_1024`, `ppl_T_8192`, `ppl_T_32768`; depend on G0 | 30–480 s |
| `ceiling` | 15 | `ceiling_b4_{A,B,C}_{4096,8192,16384,24576,32768}` | 30–280 s |
| `ppl_long` | 3 | `ppl_b4_{B,C,D}_32768` | 60–480 s |

8192 captures cover B/C bulk4, B/C bulk8 and E bulk4. Each uses
**[0,16), [16,32), [32,48), [48,62)**. The two 32768 captures use 62
single-layer ranges apiece. Each range depends on its original prerequisites
and its predecessor, restores the pinned hidden activation in its original
dtype, and executes only its assigned layers. No earlier layers are replayed.
Every layer still attends over its full input prefix.

Planning formula: **60 + 8 × (S/8192)² × range_layers seconds**, using the
lead's 60 layers / 480 seconds observation and a conservative quadratic
extrapolation for 32768. Maximum is 188 seconds, below 300. These are estimates,
not measured completion times. Ranges and ceiling probes have **290-second
worker TERM rails plus five seconds of child termination grace**. Other jobs
retain 480 seconds; outer calls remain below ten minutes.

Captures stream diagnostic rows into the original `layerNN/{q,k,kq}.npy`,
`selected.pack.npy`, `capture.json` layout. Native output and diagnostic replay
must match **bitwise**. Aggregation requires all 62 layers, rehashes each layer
manifest, checks every array's exact stat identity since its completed SHA256,
and SHA-pins the manifest set. Margin workers retain their IDs/dependencies
and rehash their arrays. All new/changed receipts use `jobs_a4/`, preserving
legacy capture receipts including RED results. Existing partial capture
directories must be preserved/archived by the lead before the first range.

T entry point: `scripts/apa_sp3_a4_torch.py::reference`, through the same
leased worker. It loads the **same pinned MiniCPM3-4B HF snapshot**, bf16
weights, `trust_remote_code=True`, `attn_implementation="sdpa"`.
`FusedSDPA` tests actual Q/K width 96 and V width 64 tensors on sm_89; it forces
**FLASH_ATTENTION when eligible, otherwise EFFICIENT_ATTENTION**, using
`torch.nn.attention.sdpa_kernel`. Neither eligible means RED. **Actual backend
is UNRUN**, not inferred from device name. The receipt records every layer's
full-S shape and selected backend; math fallback is disabled.

The 2024 remote code requires compatibility handling on installed Transformers
5.12: byte-identical local staging for relative imports, the removed FX
availability import, tied-weight metadata, standard AutoModel registration,
and reconstruction of nonpersistent rotary buffers using the unchanged remote
`_init_rope`. The CPU gate checks every tiny checkpoint parameter after bf16
loading exactly and verifies full versus restricted output logits bitwise.
Original snapshot files are untouched. Installed Torch is 2.11.0.

T scores six independent 1024 windows, 512 targets each, and one prefix-0
last-512 window at each long length. New T/32K scorers project only required
LM-head rows; **attention is never chunked**. The original engine model and
`last512`/`score_windows` source bytes are unchanged. OOM produces explicit
non-fit evidence without a PPL value; no chunked retry is substituted.

B/C 32768 require their own successful `fit=true` ceiling receipt. D 32768
requires T and compares identical target hashes; its INT4-minus-bf16 gap is
reported without an equality threshold. D keeps float32-max delta and adds a
layer-0 first-128-query all-selected/native-parity check. This is **not an
all-layer 32768 diagnostic mask measurement**. T/engine gaps at 1024 are also
reported, never treated as determinism REDs. No 32768 margin cells or changes
to the finite E calibration envelope were introduced.

## 2. Fingerprint amendment: which existing receipts stay valid and why (per-kind closure), which do not

[amendment_006_fingerprint.json](amendment_006_fingerprint.json), bound to the
order and registration SHAs, records the reviewed r3→a4 endpoints and closure
for every kind. [amendment_008_validation_followup.json](amendment_008_validation_followup.json)
extends it immutably after the CPU timing finding; it adds per-action validation
memoization. The next action rechecks identities. An ancestor changed between
actions is a tested rejection. Unknown source transitions fail closed.

| Kinds | Additional import closure beyond common receipt/driver/control/runner code and registration |
|---|---|
| baseline, ppl, match, capture, decode, kernel | Model seam; pinned build, diagnostic sources, adapter imports, weight identity; match also metrics |
| margin | Metrics; pinned runtime/build/diagnostic/adapter/weight artifacts |
| calibration | Metrics; dependency receipts |
| parity, freeze, eq, eq_check | Common aggregation logic and dependency receipts |
| capture_range | New job/registry/range modules plus model/runtime closure |
| capture_aggregate | New job/registry/range modules and model fraction-summary helper |
| torch_reference | New job/registry/T modules, unchanged scorer, weight identity and remote-source manifest |
| ceiling, ppl_long | New job/registry modules plus engine closure; long PPL also compact scorer in T module |

The exact path lists are in amendment 006. Source/build manifests recursively
pin existing engine and external adapter sources. Protocol, corpus and token
bytes are independently checked. Renderer, test/mutation code, protocol
creation script and unexecuted new-kind algorithms are outside legacy numeric
closures. Installed T package versions are recorded in runtime receipts.

Machine checks establish that the legacy execute AST is identical after
removing the new-kind dispatch, the base registry AST is identical after its
rename, and the **entire model/scorer file is byte-identical**. This is a
reviewed exact-transition bridge, not a general semantic-equivalence proof.

[a4_receipt_audit.json](a4_receipt_audit.json) lists every local receipt:
**147 PASS eligible; zero source-invalidated; one original RED retained**.
The in-flight lead chain was not read; the same closure rule applies when the
lead audits its additional receipts after merge.

**Keep the lead's original build directory and compiled modules.** This seat
rebuilt locally for CPU API gates. The local build identity differs, and no
runtime-equivalence waiver was issued. Accordingly, local live validation
reports zero reusable receipts under that different build, separately from
source eligibility. A CPU-only simulation supplying the known original lead
build-manifest hash validated all 147 receipts in **0.734 s**, and B8192 range
preflight in **0.436 s**. It changed no build bytes and is not GPU evidence.

## 3. CPU gate results; blocked-report; exact lead commands

Final receipt: [A4_CPU_GATES_FINAL.json](A4_CPU_GATES_FINAL.json).

- **89 tests passed**, two upstream deprecation warnings; exact scoring,
  checkpoint continuation, packed masks, missing/changed capture rejection,
  native-output negative controls, fused-only dispatch, OOM/fit forks,
  reference identity, compiled CPU API guards and validation-cache lifetime.
- **8/8 final semantic mutants killed**, zero invalid, threshold ≥0.80;
  [mutation receipt](a4_final_verification/a4_mutations/results.json).
- Host CUDA compile/link PASS; no production kernel changes. Source pins and
  all 107 registered kernel-body pins checked; shell syntax PASS.
- Dry-run and leased-shell list agree on all 859 cells. All 148 original
  receipt hashes and the immutable registration/order hashes remain intact.

[A4_BLOCKED_FINAL.json](A4_BLOCKED_FINAL.json) records GPU and storage limits.
**First 32768 capture range needs 490,783,899,647 bytes free** under the disk
rail; about 392 GB was free at inspection. The lead must provide capacity for
these unsampled captures. This does not block the T/ceiling/PPL jobs on disk.

After merge, preserving the lead's build and captures:

```bash
cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3
timeout 30s python3 scripts/apa_sp3_a4_audit.py
timeout 30s bash scripts/apa_sp3_lead_gpu.sh list
timeout 30s bash scripts/apa_sp3_lead_gpu.sh summary
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run capture_b4_B_8192_r00_16
```

Then use [lead_commands.txt](lead_commands.txt): exact dependency order,
per-worker wall estimates and explicit conditional B/C 32768 commands.
Each invocation runs at most one foreground leased job. No automatic rerun of
PASS or replacement of RED receipts. The refreshed summary distinguishes
reference quality, memory shape, captures and unknown/unrun results.

## 4. Prior art; deviations; RED; process safety; model id and effort

### Prior art

OpenBMB MiniCPM3 (2024) supplies unchanged model/block arithmetic.
[PyTorch SDPA](https://pytorch.org/blog/accelerated-pytorch-2/) (2023),
[FlashAttention-2, Tri Dao](https://arxiv.org/abs/2307.08691) (2023), and the
existing xFormers efficient implementation supply fused attention.
[sdpa_kernel documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.attention.sdpa_kernel.html)
supports explicit backend restriction. These primary sources were checked.
This work adds harness/receipt wiring, not a new attention algorithm.

Existing APA, TurboQuant and BLASST selection arithmetic is reused unchanged;
see original registration and code annotations. PROTOCOL-2 teacher-forced NLL
is reused unchanged (standard NLL; Shannon 1948 attribution retained).
Checkpoint/restart, packed bitmaps, memory maps, restricted LM-head projection,
content hashes, fixed-grid sweeps, stat-identity checks and compatibility
adapters are standard techniques; **no prior art known to me for this specific
SP3 range layout**. Unverified leads to check: Kahn (1962), “Topological sorting
of large networks”; Feldman/Make (1979); Dolstra et al./Nix (2004);
DeMillo/Lipton/Sayward (1978), “Hints on Test Data Selection”. Reuse versus new
wiring is annotated at code sites and in the ledger.

Deviations and RED findings are preserved in [the A4 ledger](../../docs/APA_SP3_A4_LEDGER.md)
and failure logs: HF import/metadata/buffer/loading compatibility failures;
initial repeated validation timeout; the amended worker-rail test update;
one-byte disk-estimate oracle correction. No scientific tolerance was relaxed.
D/T uses an explicit cross-weight-format gap rather than D/A equality;
32K D diagnostics have the limited scope stated above. Lead-owned blind review
and every actual GPU result remain pending.

Seat **gpt-6-astra, reasoning effort xhigh**, recorded in
`logs/apa_sp3_a4_r1.log`. No git, subagents, shell background jobs, foreign
process signals, GPU leases, service changes or access to the lead's live
worktree. Only bounded host commands were run in this isolated checkout.
