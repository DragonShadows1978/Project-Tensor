# P1T-FIX9 ledger

Seat: Opus 5.5 (high), the resumed FIX7/FIX8 seat. Order: `orders/P1T_FIX9_ERROR_RECORD_INTEGRITY.md` (f3ab493),
immutable. Writable target: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix9`
(branch voice-fix9, fd5fcb6 + f3ab493), plus scratch `/tmp/p1t_fix9_*`. Environment: RTX 4070 SUPER (one GPU), native
driver, Python 3.12.3.

## 2026-10-13 18:50 Registration (written before any source change, build, gate run or probe run)

- Before arm: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev9` (fd5fcb6,
  read-only), the e4cf1d6 build: core `d6b1b7ad1d87e0d7d5eed70cff94b33e469e6f18b857d78679110e6b2dce3c24`, tc_signal
  `30aee4e67fbe80320863cf9d0d4b7748dca46c0db7a3689f70bcc9d9e8dc782b`. The round-9 probes find their package through
  `Path.cwd()` / `parents[3]`; probes that write into `artifacts/p1t_rev9/` or `/tmp/p1t_rev9_build` run as path-only
  copies that build under `/tmp/p1t_fix9_rev9/<arm>` and write under `artifacts/p1t_fix9/`.
- Reviewer files registered before any change: `artifacts/p1t_fix9/probe_files_before.sha256` (146 probe files of rounds
  1-9) and `artifacts/p1t_fix9/rev9_all_files_before.sha256` (98 files under `artifacts/p1t_rev9`). They must stay
  byte-identical (G5).
- FIX8's runners and mutant tooling were copied (not edited in place) from wt_fix8 to `artifacts/p1t_fix9/from_fix8/`.
- Gates: as the order states them (G1-G5). Budget: the combined session, Voice's runner and the mutants run once, at the
  end, on the final build.
- Design decisions, registered before implementation:
  - **Core: one record store, one snapshot-and-acknowledge API, keyed by sequence numbers.** Every record gets a
    sequence number under the store mutex. A snapshot is taken once under the mutex: the count, the records still held
    by name (a fixed ring of the latest kCoreNamedRecords), and the sequence bound `hi`. A report is built only from the
    snapshot, and anything retained (the exit report) is that snapshot's summary. The acknowledgement advances the
    acknowledged bound to exactly `hi`, so a record that arrives later keeps its own sequence number and name and is
    reported later. A lock failure never loses a count: writers fall back to an atomic count of unnamed records, and
    acknowledgements to an atomic "acknowledge up to" bound that the next lock holder applies.
  - Every core raise on a P1-T CUDA failure records the failed call (static site text, device, error) BEFORE the next
    CUDA call or allocation. It consumes the error slot only once the record exists, builds its report, and
    acknowledges the snapshot it reported, which includes any earlier unreported record. If the report cannot be built
    (bad_alloc), the records stay for the next `tensor_cuda.synchronize()` or the exit reader. This includes the export
    fence (V9-01), `fill_`, the label proof, `from_dlpack` and `empty`.
  - **Add-on:** the store stays GIL-protected `PendingState`, made exact by a report window.
    - While a report is being built (synchronize, the exit logger), any record write goes to a second record set,
      merged after the report's records are cleared. The report and its clear therefore cover the same snapshot even
      if something wrote in between.
    - The raise helpers (`cuda_guard.h`, `conv.cu`, `acquire_event`, `RetainUntilComplete`) persist their failure first,
      as a tagged record in an in-flight slot of the store, and consume the error slot. The slot is acknowledged
      (cleared) once the RuntimeError exists, or committed to the reported records if the report cannot be built.
  - **Registered conflict.** `core_fence_cpu.py`'s single-failure OOM control asserts `slot == 1`: the failure must
    stay in the runtime's error slot, which was FIX8's mechanism. The order requires persistence as a named record
    (census: "persisted and then reported by the next synchronize() or at exit, naming its call and device, exactly
    once"). A failure kept only in the slot loses its call and device (a later reader reports only the error code), and
    the next failing CUDA call overwrites it (sweep (m)). Keeping it in BOTH the slot and a record would report it
    twice. So on this build that control is expected to read `slot 0, records 1`. It will be reported as written,
    with this reason, for the lead's ruling, and never weakened.
- Rules: no git writes, no subagents, no deletes, never kill a process I did not start, GPU under the flock, absolute
  output directories, no defect residuals, prior art at code sites + here + the report.
