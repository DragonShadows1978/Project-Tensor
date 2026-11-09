# REV1 → A2 hostile-control adaptations

Prior art: REV1 hostile_scorer.py; Meszaros (2007), test stubs/fault injection;
DeMillo, Lipton and Sayward (1978), mutation validity. Literature unverified
— lead to check. The stand-in never calls forge-gpu-run or any GPU API.

All runs use A2’s actual sealed run_gpu_line.py, score_gpu.py and evidence.py
in their own directory. REV1 needed a byte-identical /tmp mirror because A1
was read-only. A2 needs no mirror: its authorized hostile_runs/ is writable.
RUNS, TMPDIR, census files and script locations now point at A2. The stand-in
adds the bindings A2 requires (nn/engine/seal hashes, arm/suite, root/attempt,
exception class/origin) and real JUnit ID/outcome rows. It writes the installed
forge runner’s nine-field tab-separated log format. These additions keep
unrelated missing evidence from masking the targeted attack.

| REV1 control | A2 adaptation and expected result |
|---|---|
| H1 | Forge parent-only receipts under attempt-0001 instead of the old flat directory; omit START and children exactly as before. Scorer rejects missing START. |
| H1b | A /tmp/gn1_a2_* outside root, using the same real scorer. Guard rejects before writing a summary outside A2. |
| H2 | Exact old message on every ID: RuntimeError: CUDA error at reduce_lastdim: an illegal memory access was encountered. Bound/JUnit-consistent session is void. Separately inject each of all 12 D2 patterns into a reference failure while a real non-reference assertion also fails: any session device fault vetoes a kill. |
| H2c | Exact old cudaErrorIllegalAddress wording on every ID. Still void. |
| H3 | Use M07fast (the old M07 fast site, split by D4) as the registered mutant label. Summary arm=baseline, values fail with the old reshape message; other IDs pass. The arm/suite binding rejects it. |
| H3b | M07fast command with child claiming a passing candidate/focus session and its focus census. Both wrong exit and binding reject. |
| H4 | Preserve all-pass candidate summary plus the old foreign x failure in JUnit; reject unregistered JUnit ID. An additional control changes exactly one registered ID’s JUnit outcome to pin exact ID-by-ID comparison. |
| H5 | Both 75/no-window and 76/no-lock receive numbered not_admitted attempts. A second valid baseline-focus attempt in the same root must pass and be the scorer’s selected admitted attempt, with both attempts listed. |
| H6 | Bound shortened from 5s to 1s; stand-in writes the passing completed child files then sleeps 30s. The real foreground timeout terminates that child tree. Receipt must fail despite the earlier summary; missing forge outcome is reported as missing, not mislabelled timeout. |

Additional controls: reference-only failure; generic RuntimeError and foreign
TypeError rejected as kills; positive GroupNorm-origin TypeError admitted;
wrong nn/engine hashes; setup and collection failures; a one-byte engine-copy
corruption after START (saved under /tmp and restored in finally); explicit
forge timeout and window-closed 124 outcomes; full valid synthetic run;
parent-killer tamper, expected-exit tamper and START-seal tamper; unchanged
legacy regression failure accepted, new failure/changed exception class/
missing pass/skipped pass rejected. Every control is followed by the real
scorer. An incomplete run is rejected overall even when its selected valid
control line passes; the line-level result is inspected for the positive
controls. No synthetic result is presented as device execution.

D6 also covers a matching legacy setup error (accepted), a new setup error on a baseline-passing ID (rejected), and a changed setup exception class (rejected). This preserves the regression rule while focus/mutation setup faults remain invalid.

A native pytest fixture demonstrated legitimate call-failure + teardown-error duplicate JUnit entries (junit_real_shape.json). Added that matching legacy regression positive and a duplicate-pass negative. Parent verifies the failed+error pair against independent phase reports before reducing its per-ID outcome to error.
