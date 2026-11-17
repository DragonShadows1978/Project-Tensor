# ENG3 r2: resume with Amendment 1 (the G1 surface, E4's scope, bool numpy)

**Dispatched 2026-11-16** on branch `eng3`, at the commit carrying this file. r1 stopped correctly under L35 before
editing any product file. Its census and Build B (module `9d8be988…`, receipts under `artifacts/eng3/build/`) carry
over.

**The plan is `orders/ENG3_HOST_TENSOR_CORE.md` (commit `114d6269`), unchanged except for Amendment 1 below.** Every
ruling, gate, rule, machine limit and checklist answer in it binds this run. The writable target is the same, plus
A1.1's narrow addition.

## Amendment 1 (lead, 2026-11-16; registered before any gate runs)

r1 found three clauses that could not hold as written. All three are lead faults: the order named Python entries
without checking that the public API has them, and its E1 census and E4 scope reached files outside the writable set.

- **A1.1, the G1 surface (r1's option b).**
  - **What changes:** `tensor_cuda/src/bindings.cpp` becomes writable for one thing only. You may add private,
    underscore-prefixed test entries to the main module that call the NDArray core members directly: `_core_full`,
    `_core_clone`, `_core_to` and `_core_zeros`. You may use one dispatching entry instead; name it in the report.
  - **How G1 reaches each member:**
    - `full`, `clone`, `to` (host to host) and `zeros`: through those private entries;
    - `from_host`: through `tc.tensor`;
    - `to_host`: through `.numpy()`;
    - `astype`: through `.astype`.
  - **What stays unchanged:** the public API. No public name is added. Each private entry carries a comment saying it
    is a test hook, with its prior-art line.
- **A1.2, E4's scope (r1's option i).** E4 covers NDArray core members only. By r1's census
  (`artifacts/eng3/HOST_CENSUS.json`), no core member beyond the E2 set reaches CUDA on a host tensor, and `fill_`
  already refuses. So this round adds no E4 refusal, and G1's E4 clause holds on the census.
  - The 172 compute entries that take a host tensor's pointer into engine kernels are out of scope. The lead registers
    them as a successor item (HOST-COMPUTE-HMM).
  - In the report, list the 14 entries r1 found with a device check, and the rest as unchecked.
- **A1.3, bool `.numpy()` (r1's item 3).** Bool is excluded from G1's `.numpy()` value check on every device.
  - `tensor_to_numpy` (`bindings.cpp:103-109`) has no Bool branch; r1 found this by reasoning. The lead records it as a
    suspected defect (BOOL-NUMPY1) and will measure it.
  - Do not change `tensor_to_numpy` in this round. The A1.1 grant does not cover it.
- **A1.4, the dtype pairs (E3, G2).** "Every dtype pair the engine supports" means every pair `NDArray::astype`
  dispatches today: the `DISPATCH_FLOAT` types in both directions, and the uint8 branches. List them from the code, at
  file:line.

Nothing else changes. If any other clause cannot hold as written, stop and report RED with the reason (L35). Only the
lead amends.

## Mechanism & scope

- **Measures:** as the order: whether a host tensor's core members work with no GPU and no HMM, bit-exact to the
  device casts, with device results unchanged.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3/orders/ENG3_HOST_TENSOR_CORE.md`
  (the plan), with r1's `artifacts/eng3/HOST_CENSUS.json`.
- **Exercises:** the order's E2 and E3, through A1.1's private entries and the public entries named there; Builds B
  and E plus mutant builds; G1 to G5.
- **Does not cover:** as the order, plus the 172 compute entries (A1.2) and bool `.numpy()` (A1.3).
- **Required evidence:** the order's list, plus the private entries' diff and the A1.4 pair list.
- **Operand census:** the E2 members, the A1.4 dtype pairs, and G2's value census.
- **Baseline isolation:** Build B (`9d8be988…`) is the reference for G3, as built in r1.
- **Forbidden configuration:** the order's list, plus any public API change, any change to `tensor_to_numpy`, and any
  edit to `bindings.cpp` beyond A1.1's private entries.

## Registration checklist (order-lint)

- Every answer is the order's, unchanged, except as below.
- **L01:** cmake, nvcc, pytest, python3 with NumPy and `forge-gpu-run`, each run with `--version` (or
  `--status --min 30 --max 600`) first in this run. r1's `preflight_L01.txt` recorded them.
- **L16:** the cheapest passing work would make the private entries call host code that the public paths never use,
  or fix only the members G1 reaches through public entries. Excluded: the private entries call the NDArray core
  members themselves (show it in the diff), and `m03` targets `clone`, which only the private entry reaches.
- **L41:** each mutant's killing test must execute the mutated line with the mutant module loaded. Show this with a
  counter or a log line in the test's receipt.
- **L69:** one reference per operand class: host casts -> the device cast on the same bits (G2); device results ->
  Build B (G3); test outcomes -> Build B per ID (G3). "Nothing else changes" is checked by the diff.
- **L35:** A1.1 to A1.4 are the lead's amendments, registered here before any gate runs, not seat registrations.
- **L37:** the private entries are test hooks, not knobs. Their path is `bindings.cpp`, which A1.1 makes writable for
  that alone, then NDArray's core members.
- **L56:** (a) a slip in running your own tools goes under Deviations: re-run and continue. (b) An exception in your own
  scripts is repaired at most 3 times, each recorded. The STOP list is exhaustive: an L35 clause that cannot hold, the
  8-hour wall, and the disk floor.
- **L59:** n/a. No seal time is claimed in this resume beyond the order's: registrations are written before the gates
  they govern, and the report quotes each file's mtime beside the first gate command's `date`.
- **L99:** the lead's readings for this amendment: r1's report lists the missing public entries (`tc.full`,
  `Tensor.clone`, `Tensor.to`), the 217-entry Python census (172 compute, 14 with a device check), and
  `bindings.cpp:103-109` with no Bool branch. Its `HOST_CENSUS.json` sha256 is `604632d7…f646`.

## Done

As the order's "Done", plus the private entries' names and the A1.4 pair list.
