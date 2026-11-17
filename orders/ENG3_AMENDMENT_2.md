# ENG3 Amendment 2: fix BOOL-NUMPY1 in this round (supersedes A1.3)

Sent by the lead to the running seat by message on 2026-11-16, at about 07:45. It is filed here after the round, so
that nothing was written into the worktree while the seat ran. The ledger records it at 07:37 (inspect-kalead,
commit `74ee2bda1`).

**Measured by the lead on GPU,** with ENG3's Build B (`9d8be988`), through `forge-gpu-run`:
- `tc.tensor([True,False,True,True,False], dtype='bool').numpy()` returns dtype float32, shape [5], with values
  [2.37e-38, 4.45e-41, 7.40e19, 4.47e-41, 0.0];
- an 8-element numpy bool input returns similar garbage.

So `tensor_to_numpy` has no Bool branch, and bool output is wrong on every device. This is a production defect, MAJOR.
Receipt: inspect-kalead `docs/lead_checks/` (commit `b823ffd44`).

- **A2.1.** `bindings.cpp`'s `tensor_to_numpy` (around lines 103-109) becomes writable for a Bool branch only. It
  returns a numpy bool array with the tensor's values, one byte per element, as the engine stores bool.
- **A2.2.** E5 is waived for this one row only: the device-tensor result of bool `.numpy()` changes from garbage to
  correct. Every other device result stays bitwise unchanged.
- **A2.3, new gate G1b.** Bool `.numpy()` returns exactly the input values on the GPU (a GPU line), and, after E2, on
  host tensors with no visible GPU (CPU). Cover shapes of 1, 5, 8, 33 and 1,000 elements, all-True, all-False, and a
  seeded random pattern.
- **A2.4, new mutant `m05_bool_numpy_float`.** The Bool branch is removed again. It must fail G1b.
- **A2.5.** G3's intended-change list gains this row: any existing test that relied on the old bool output.

Everything else in the order and Amendment 1 is unchanged.
