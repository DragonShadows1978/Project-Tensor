# Execution amendments and retained negatives

The order and REGISTRATION.md are unchanged. This note records implementation
corrections and extra controls; it does not change G1–G6, P86/P87, any tolerance,
or the D2/D6 acceptance predicates.

- First G4 run is retained as RED: the missing-ID control expected `missing test
  calls`, but the real parent/scorer rejected it with `JUnit/summary outcomes
  disagree ID by ID` and `baseline-passing IDs no longer pass`. The control now
  checks the immutable D6 rule directly. No failed run is relabelled PASS.
- Extra controls were written before each fresh final-seal run: exact REV1
  error wordings, GroupNorm-origin refusal sensitivity, setup-error regressions,
  and legitimate failed+error dual JUnit entries (based on a native pytest probe).
- D6 applies to setup/teardown errors as well as call failures in regression
  runs. Focus/mutation setup errors remain invalid. This is the order’s original
  no-regression predicate, not a new allowance for mutant kills.
- All older seals and hostile runs remain under .attempt* or previous-seal
  names. Final G4_hostile.json is the 48-control run under the final seal.

Prior art: GN1-REV1 fault injection; pytest/JUnit actual output; DeMillo et al.
(1978), mutation validity. No new general algorithm claimed.
