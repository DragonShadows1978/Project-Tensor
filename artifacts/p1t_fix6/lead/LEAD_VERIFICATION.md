# P1T-FIX6 lead verification (Project-Tensor lead, 2026-10-13, about 10:00 EDT)

All runs were fresh processes under `flock -w 900 /tmp/forge-gpu.lock`, on native driver 595.99.02.

- **Before:** the acd42fa build in `wt_fix5`, whose `git status` was clean before and after every run.
- **After:** this worktree's build. Both hashes match the seat's report, and both binaries are newer than every source
  file:
  - core `e7d4a3a313562d80af0401f1deba819260fa95a6ea86abb3a53443c3e17b5a10`;
  - `_tc_signal` `ef44b9c455f26a7b40d3429d70b7e08ebd4b35444689811de90428e063f2f197`.
- **Harnesses:** I re-ran the seat's own runners (`run_rev6_probes.sh`, `run_rev5_probes.sh`) with my own output
  directories. They rebuild every probe module from each arm's sources.

## Checks

| Check | Result | Receipt |
|---|---|---|
| Scope | Every changed path is in the order's Exercises, plus `docs/P1T_FIX6_*`, `artifacts/p1t_fix6/**` and `logs/**`. `tensor_cuda/src/bindings.cpp` hunks touch only the P1-T bindings: `__dlpack__`, `__dlpack_device__`, `data_ptr`, `fill_`, `mark_modified`, `revision`, `_C.empty`, `_C._from_dlpack_capsule`, `_C._dlpack_*`. No pre-existing engine binding changed. | `git status`, `git diff acd42fa -- tensor_cuda/src/bindings.cpp` |
| Probe files | `artifacts/p1t_rev{,2,3,4,5,6}` are byte-identical to HEAD, with no untracked additions. | — |
| New core `__global__` | 0 | — |
| Round-6 `sync_lock_inversion`, no GPU / GPU | before rc 1 / rc 1 (rescue needed); after rc 0 / rc 0 (`rescue=False`) | `probe_runs/rev6_{before,after}_summary.txt` |
| Round-6 `sync_late_release` | before rc 1 (`waited_for_later=True`, 0.700 s); after rc 0 (`waited_for_later=False`) | same |
| Round-6 `core_cast_exit_gpu` bool / index | before rc 139 / rc 139; after rc 0 / rc 0 (refused, method never called) | same |
| Round-6 `core_cast_exit_gpu` controls | rc 0 on both builds | same |
| Round-6 `core_host_cpu casts_exit` bool / index, regenerated | before rc -6 / -6; after rc 0 / 0. The before-arm slices equal the reviewer's pins (`all_identical= True`). | `probe_runs/rev6_before/slices_selfcheck_before.txt` |
| Round-6 `release_trace_cpu diagnostic_oom_{query,sync,device}` | before rc 1 ×3; after rc 0 ×3 | `probe_runs/rev6_*_summary.txt` |
| Round-6 other CPU checks | query_once, multidevice_restore, prior_oom, inside_callback, deferred_exit, arithmetic, fix2/fix3 harnesses, cpu_native ×2, operand_census: rc 0 on both builds | same |
| Round-5 `exit_gate full`, no GPU / GPU, each with control | rc 0 ×4 (after) | `probe_runs/rev5_after_summary.txt` |
| Round-5 `exit_gate gpu` and control | rc 0 / rc 0 (after). The daemon's later producer call is refused after the close (the documented after-close rule, U6). | same |
| Round-5 `exit_gate trace` as written | rc 1 (after), as the FIX6 order's Required fix 1 anticipated: its child's hook is `synchronize()` alone, which no longer waits. The exit-entry copy (hook = `synchronize(); close(5.0)`) passes, rc 0. | same |
| Round-5 others (after) | `query_error_cpu`, `query_error_single_gpu`, `fill_grid_single_gpu`, `device_error_sweep` (40 passed), `sync_existing_error`, `addon_lifetime` (12/12), `core_edges`, `core_exit_overrun` during / bounded (3.3 s / 7.8 s), `numerics_and_streams`: rc 0. The two-GPU probes exit 3 (INCONCLUSIVE). The as-written forms of `handle_insert_oom_cpu` / `fill_grid_cpu` embed 1475fae code (rc 1, as in FIX5); their regenerated forms give rc 0. | same |
| FIX2, FIX4, FIX5 and FIX6 tests (real build) | 184 passed | `probe_runs/after_fix456_tests.log` |
| Mutant (a), V6-01 wait restored (`eacd0df5…`) | 6 failed of 6, KILLED. Run from the mutant tree, as the seat did. My first run used this worktree's test path, so its trace module was built from the fixed source and one case passed (`mut_a_tests.log`). | `probe_runs/mut_a_tests_mutant_tree.log` |
| Mutant (b), V6-02 C-level reading removed (`436788ff…`) | 30 failed, 17 passed, KILLED (the 17 are accepted-value cases that call no method) | `probe_runs/mut_b_tests.log` |
| Mutant (c), V6-03 persistence moved after the allocation point (`a896fe33…`) | 7 failed of 7, KILLED | `probe_runs/mut_c_tests.log` |
| Each mutant tree differs from this worktree in exactly one source file, and its test files are byte-identical | yes | `diff -rq` |
| Voice runner, T1–T6 plus T1R and T4S | PASS 249/249 (T1 11, T2 144, T3 28, T4 12, T5 28, T6 13, T1R 1, T4S 12), 69.8 s; adapter sha 600f38b4, unchanged; Project-Voice `git status` clean | `voice_runner.log`, `p1t_all_lead.json` |
| Combined core plus tc_signal session | 37 failed, 2886 passed, 47 skipped, 15 errors (625.5 s). The compare finds 52 = 52 baseline ids, 0 new and 0 no longer failing: **G1 PASS**. The 2886 passes are FIX5's 2765 plus the 121 new FIX6 tests. | `combined_session.log`, `combined_compare.txt` |

## Gate of the seat report (David, 2026-10-13: "bugs don't get to be post-hoc accepted")

**Residuals.** The report lists no residual defect. FIX5's R7 is fixed and tested, and R4 went away with the removed
wait. R2, R5, R6 and the environment items are in a "Not a defect" table, each with a citation.

**"Not a defect" items.** I checked each one. Four needed the lead to check a citation or tighten the reasoning:
- **N1 (a deleter that calls `pthread_exit`):** accepted. The DLPack deleter is a destructor that must return. The
  seat's receipt shows the same deleter kills torch 2.11 (rc 134) and hangs NumPy 2.4.2.
- **N5 (a host function the application enqueued):** citation checked:
  `citations/cuda126_cudaLaunchHostFunc.txt`. A host function is stream work, and it "must not perform any
  synchronization that may depend on outstanding CUDA work not mandated to run earlier". P1-T enqueues none.
- **N6 (a deleter that needs a lock the releasing thread holds):** accepted, with the reasoning corrected. The pending
  list is module-global (`dlpack_operand.cpp:837`), so a deleter queued by thread A's call can run on thread B, at B's
  next call or `synchronize()`. The README documents this ("released in FIFO order, on the calling host thread … at the
  start of the next tc_signal call"). So "on that thread" in N6 is too narrow, but the conclusion holds. A DLPack
  deleter is a finalizer, and Python's data model says a finalizer "can be invoked when arbitrary code is being
  executed, including from any arbitrary thread. If __del__() needs to take a lock … it may deadlock":
  `citations/python312_datamodel_del_warning.txt`. torch's own deleters run wherever gc collects the tensor.
- **N8 (free-threaded CPython and subinterpreters):** citations checked:
  `citations/python_free_threading_and_subinterpreters.txt`.
  - A free-threaded build needs its own "t"-suffixed binaries. A module that does not declare GIL-free support
    re-enables the GIL.
  - An own-GIL subinterpreter (`check_multi_interp_extensions=1`) refuses single-phase-init modules.
  - A legacy shared-GIL subinterpreter can import the module, but CPython documents "mixing multiple interpreters and
    the PyGILState_* API is unsupported". That limit already applies to baseline core, whose existing code uses
    `gil_scoped_acquire` (`bindings.cpp:137`). So this is a configuration CPython itself does not support, not a P1-T
    defect.

N2, N3, N4, N7 and N9 are accepted as written.

**User-visible changes U1–U9.** I accept each as justified:
- U1, U6 and U7 are the V6-01 fix and the documented after-close rule.
- U2 and U3 (strict C-level argument types) follow the array API types. torch 2.11's own `__dlpack__` refuses a
  non-int `stream`: `torch/_tensor.py:1784`, `type(stream) is not int`.
  - The newly refused values (`numpy.float32` for `fill_`, numpy integer scalars for `stream`) are only for P1-T
    methods that are new in this merge, so baseline behaviour is not affected.
  - Code read: `CoreCallArgs` reads `float` / `int` subclasses through `PyFloat_AsDouble` / `PyLong_*`, which for
    exact types and subclasses read the value without calling `__float__` / `__index__`. It matches keywords with
    `PyUnicode_CompareWithASCIIString`, with no `__eq__`. It checks `self` with `PyObject_TypeCheck` before casting.
- U4 and U5 are the "a CUDA failure is reported exactly once" rule (FIX2 N1).
- U8 and U9 are internal.

Nothing goes back to the seat.

## Verdict

FIX6 is verified, so the round-7 independent review can start on this commit.
