# P1T-REV6 ledger

- Read mirrored HOUSE_RULES, project AGENTS, REV5 mechanism and approved alternative plan. User explicitly permits read-only git; mirror rule followed. No scope mismatch: actual core/add-on sources are reviewed. Memory quick pass yielded no relevant review history; no technical conclusions borrowed.
- Registered CPU-only rails before gates. Saved full/FIX5 diffs and source hashes. Early report and JSON created.
- Prior art: standard contracts in registration; original contribution is review scenarios only.

- Built exact-source add-on copy under /tmp/p1t_rev6_build with CUDA12.6/pybind11v2.12, foreground timeouts; build exit0, logs/p1t_rev6_{configure,build}.log. No GPU execution.
- V6-01 full-extension CPU lock-inversion witness requires rescue after1s; record and standard fix above. Prior art: Coffman et al1971 circular wait, CPython3.12 lock/GIL; scenario is this review contribution.

- CPU driver completed18 cases (not an all-green gate): lock inversion rc1; core bool/index exit slices rc-11 each, controls rc0; three current-drain diagnostic-OOM combinations rc1; query-once/prior-OOM/multidevice-restore/self-exclusion/deferred-exit controls pass. 18 checked arithmetic and17 accounting UBSan cases pass; native real OOM and throwing deleters pass;17 operand census cases pass. Exact slice extraction hashes stored. V6-02 MAJOR, V6-03 MINOR recorded. Prior art and fix distinctions in findings JSON/report.
