# STAGED_LINES.md

**HOLD — RED: SCOPE MISMATCH.** Zero GPU lines staged and zero dry runs. The
required G2 norm reproduction includes eight G=4 cases that dispatch to
`spq1::bulk_shared` and bypass the sole authorized edit at `packed.cuh:203`.
This seat stopped at the pre-build mechanism check. There are no launcher
commands to run. See `REPORT.md` for the dispatch evidence.
