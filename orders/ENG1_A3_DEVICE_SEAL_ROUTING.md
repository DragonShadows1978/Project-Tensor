# ENG1-A3: route the staged GPU lines to an up-to-date seal (CPU only; no GPU)

**Dispatched 2026-11-14.** This order follows `orders/ENG1_A2_REV2_FIXES.md` (A2 ACCEPT, receipts `97edd1fd`). The
ENG1 orders before it stand unchanged. This round changes no evaluation, test, census, engine or expected result. It
only makes the device-mode entry point check the seal that matches the tool files on disk.

## What happened

- **The lead's G5 attempt** (11-14 05:39, wrapper `artifacts/eng1/lead/eng1_g5.sh`, log
  `artifacts/eng1/lead/g5_1114/`) refused on all 24 labels within 4 seconds.
  - Every `start_gpu_run.py` raised `AssertionError: tool seal mismatch: ids_a1_regression.txt` (`evidence.py:66`).
  - No run root was created and the GPU was never leased.
- **The cause:**
  - `artifacts/eng1/tool/evidence.py:23-24` hard-codes `ORDER = "orders/ENG1_A1_REV1_FIXES.md"` and
    `SEAL_NAME = "staging_seal_a1.json"`.
  - A2 changed two sealed tool files, `ids_a1_regression.txt` and `source_reading_tests.json`, and wrote
    `staging_seal_a2.json`.
  - A2's `a2/sitecustomize.py` routes to the A2 seal only for the CPU audit. It asserts `CUDA_VISIBLE_DEVICES == ''`,
    `ENG1_AUDIT == '1'` and `RUNS == a2/audit`.
  - So no device-mode path verifies against the current files.
- **Fault:** this is a staging defect that A2's gates did not catch. A2's audit exercised only the audit route. It is
  not a seat-honesty issue, and no result exists.

## The task

1. **Make the device-mode path verify against a seal that matches the current tool files,** and the CPU audit path too.
   - **Allowed changes:**
     - (a), preferred: `evidence.py`'s `ORDER` / `SEAL_NAME` constants (lines 23-24), pointing at this order and a
       new A3 seal. `start_gpu_run.py:5`, `run_gpu_line.py:19` and `run_gpu_tests.py:26` import `SEAL_NAME` by value
       from `evidence`, so route (a) reaches every reader with no other edit;
     - or (b), only if (a) is impossible (say why): a new routing module, activated by one line in the staged
       command, that sets only those two names before any tool imports `evidence`.
     - In both cases, write a new seal `artifacts/eng1/tool/staging_seal_a3.json` with the existing staging routine,
       following `a2/staging.py`, which binds this order.
     - Edit `STAGED_GPU_COMMANDS.sh` only if (b) needs an activation line.
   - **Forbidden changes:**
     - `score_gpu.py`, `run_gpu_tests.py`, `run_gpu_line.py`, `runtime_evidence.py`, `device_evidence.py`;
     - `gpu_lines.json`, every `ids_*` and `collect_*` file, the arms and their maps;
     - every expected exit, bound and label;
     - `tensor_cuda/` (source, tests, builds) and every earlier receipt or seal.
     If the fix seems to need one of these, stop and report.
2. **Gates (CPU only, no device, no lease):**
   - **G1, device-mode dry run.** For each of the 24 labels, in `gpu_lines.json` order, run the exact command the
     lead's wrapper runs, with `FORGE_GPU_DRY_RUN=1` added:
     `env -u PYTHONOPTIMIZE -u ENG1_AUDIT RUNS=<root> CUDA_VISIBLE_DEVICES=0 TMPDIR=/tmp/eng1_a3_dry FORGE_GPU_DRY_RUN=1 bash artifacts/eng1/tool/STAGED_GPU_COMMANDS.sh <label>`
     - Use one fresh run root that `evidence.guard` accepts, named `a3_dry_<n>`, never `g5_*`.
     - **Pass** when, on every label, `start_gpu_run.py` verifies the seal and every arm, and `forge-gpu-run` reports
       admission and returns 78 before the lease (GW3).
     - **Before the first G1 line:** show, by reading the code path, that nothing before the governor initializes
       CUDA (`start_gpu_run.py`, `evidence.py`, `run_gpu_line.py` up to the governor call). If something would, stop
       and report rather than run.
   - **G2, audit route.** The CPU plumbing audit (`STAGED_GPU_COMMANDS.sh --audit`, the A2 routing or its successor)
     passes against the A3 seal, with `RUNS` under `artifacts/eng1/a3/audit`.
   - **G3, scope.** `git diff 97edd1fd -- . ':!artifacts/eng1/a3' ':!orders' ':!logs'` shows only the allowed
     files. Every forbidden file listed above is byte-identical to `97edd1fd`, and both engine pins are unchanged
     (`73ededa0…523e` baseline, `3e6d5805…ad53` candidate).
   - **G4, mutant M17.** With the routing left on the A1 seal, G1's first label refuses with "tool seal mismatch". The
     lead's 05:39 log is one observed instance. Re-show it once in a scratch copy, never in the live tool directory.
3. **Ledger and report.** Append to `artifacts/eng1/a3/LEDGER.md`, and write `artifacts/eng1/a3/REPORT.md` and
   `artifacts/eng1/a3/FILES_WRITTEN.json` (sha256 of every file you wrote or changed).

**Acceptance:** ACCEPT when G1 to G4 pass. The lead then runs G5 on the GPU under a fresh `g5_*` root. G5's bar is the
original order's, unchanged.

## Mechanism & scope

- **Measures:** whether the staged device-mode entry point verifies the seal of the files on disk, and reaches the GPU
  governor without touching the device.
- **Mechanism doc:**
  - `orders/ENG1_ENGINE_DEFECTS.md` (G4, G5);
  - `artifacts/eng1/a2/REPORT.md` ("Staging, seal and audit");
  - `artifacts/eng1/tool/evidence.py:23-69`;
  - `/mnt/ForgeRealm/SystemOps/docs/GPU_WINDOW.md` (GW3 dry run).
- **Exercises:** `STAGED_GPU_COMMANDS.sh` for all 24 labels in dry-run device mode, plus the audit route.
- **Does not cover:**
  - any GPU execution or result;
  - the engine fixes themselves (G1–G4 of earlier rounds);
  - G5's verdict;
  - promotion.
- **Operand census:** the 24 labels of `artifacts/eng1/tool/gpu_lines.json`, plus the audit route.
- **Baseline isolation:** the dry-run root is fresh and is never reused for G5.
- **Required evidence:**
  - G1's per-label records (seal verified, governor rc 78);
  - G2's audit receipt;
  - G3's diff and hash list;
  - G4's refusal text.
- **Forbidden configuration:**
  - a real lease or any device work;
  - editing a forbidden file;
  - reusing a `g5_*` root;
  - deleting anything.

## Where to work

**Seat:** Claude Opus 5.5, high effort (Agent tool).

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng1`
(branch `eng1`, at the commit carrying this order). Edits and CPU runs are AUTHORIZED only in these paths:
- `artifacts/eng1/tool/evidence.py` (the two constants), or a new routing module under `artifacts/eng1/tool/`;
- `artifacts/eng1/tool/staging_seal_a3.json` (new);
- `artifacts/eng1/tool/STAGED_GPU_COMMANDS.sh` (one activation line, only under route (b));
- `artifacts/eng1/tool/gpu_runs/a3_dry_*` (new);
- everything under `artifacts/eng1/a3/` (new);
- scratch under `/tmp/eng1_a3_*`. Point `TMPDIR`, `PYTHONPYCACHEPREFIX` and `XDG_CACHE_HOME` there.

**Read-only:** everything else, including `artifacts/eng1/lead/`, every earlier `a1/`, `a2/`, `rev*/` receipt, and
`tensor_cuda/`.

**Rules:**
- No git, not even read-only. No subagents, no background waits.
- No GPU: `FORGE_GPU_DRY_RUN=1` on every staged line, and never call `forge-gpu-run` without it.
- **No deletes.** Use `delete-me` if anything must go. Never run `rm`, `rmdir`, `unlink`, `git clean`,
  `find -delete` or an inline delete call.
- Never kill a process you did not start. Never walk `/`, `/mnt` or `/mnt/Shared` recursively.

## Registration checklist (order-lint)

- **L01:** the instruments are the unchanged ENG1 tools. Run `forge-gpu-run --help` and `python3 --version` before
  G1 to show they start.
- **L02:** G1 is the smoke. No device run happens in this round.
- **L16:** the cheapest passing result is routing that skips the seal check. It is excluded: G1 must show the seal
  verified and every arm verified (`verify_arm`) on each label, and G4 shows the check still refuses a stale seal.
- **L41:** M17 is shown refusing inside G1's own command path.
- **L47:** the expected values have receipts:
  - the refusal text and the 24 rc=1 lines (`artifacts/eng1/lead/g5_1114/eng1_g5.log`, `eng1_g5_lines.out`);
  - the differing tool files (`staging_seal_a1.json` vs `staging_seal_a2.json`: `ids_a1_regression.txt`,
    `source_reading_tests.json`).
- **L56 (b):** a failed G1 or G2 is a harness defect in this round. At most 3 repairs, each ledgered verbatim.
- **L65:** no deletes, as stated.
- **L67:** this round writes under 50 MB (receipts only). Free space at dispatch: see the lead's `df -B1` line in the
  dispatch ledger entry. Stop if `/mnt/ForgeRealm` falls below 25 GB.
- **L09:** G3's identity rule compares whole files by sha256 of their bytes, with no masks, against `97edd1fd`.
  Logs, stdout, stderr and this round's own receipts are excluded. The engine pins are compared as recorded in the
  seals (`engine_sha256`).
- **L48:** G1 runs `STAGED_GPU_COMMANDS.sh <label>` exactly as written: the same script, argv and environment as the
  lead's wrapper, as a plain subprocess. The one addition is `FORGE_GPU_DRY_RUN=1`, the governor's own documented
  dry-run switch, which rebinds no config, registration or pin. The first unmet need is the lease (rc 78). The steps
  after the device call (child receipts, parent evaluation, `score_gpu.py`) are unchanged since A2. Under route (a)
  they read the A3 seal at `evidence.py:88`, `:245` and `:348`, and G2's audit route drives them on A2's constructed
  shape classes. G2's receipt lists each post-device step it drove, with a terminal label. No trailing command may
  mask a line's exit.
- **L60:** n/a. No engine entry point or guard is added or changed. The "entry point" here is the tool's staged
  command, and the only change is which seal file it reads.
- **L82:** `grep -rn -E 'SEAL_NAME|staging_seal_a[0-9]?\.json|staging_seal\.json|evidence\.ORDER|^ORDER'
  artifacts/eng1/tool/*.py artifacts/eng1/tool/*.sh artifacts/eng1/a2/*.py` gives 21 hits.
  - Writable: `evidence.py:23-24`, plus a new `artifacts/eng1/a3/staging.py` that writes the A3 seal the way
    `a2/staging.py:25,36` does.
  - Read-only, because they import or read the name and need no edit:
    - `start_gpu_run.py:5,15`, `run_gpu_line.py:19,76`, `run_gpu_tests.py:26,105`;
    - `evidence.py:62,88,245,348`, which read the constant.
  - Read-only, as earlier rounds' tools:
    - `stage.py:7,23-24,240-241`, the A1 staging routine;
    - `a2/sitecustomize.py:12-13`, `a2/staging.py:25,36`, `a2/verify.py:145`.
- **L90:** the roles in the L82 census: harness, 21 of 21. All are ENG1 GPU-lane tool files; none is product,
  test or fixture code. The treatment per harness file:
  - `evidence.py:23-24`: edited, route (a);
  - the 3 importers and 4 readers: unchanged;
  - earlier rounds' staging and audit tools: unchanged, read-only.
- **L79:** CPU dry runs of staged GPU lines must stop before the governor's lease: `FORGE_GPU_DRY_RUN=1` returns 78
  before the lease, and G1 first shows that nothing earlier initializes CUDA.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- A failed gate is a result, reported with receipts.
- **Prior Art Directive:** annotate any routing or sealing idea with known prior art, or write "no prior art known to
  me".

## Done

Your final message gives, verbatim:
- the route chosen and every changed byte (diff);
- the A3 seal's sha256;
- G1's table (label, seal verified, arms verified, governor rc);
- G2's audit result;
- G3's diff summary and the engine pins;
- G4's refusal text;
- `FILES_WRITTEN.json`'s sha256;
- a "Prior art" heading.
