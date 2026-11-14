# ENG1-A3 lead ruling (2026-11-14): RED as registered on G1 (lead order fault); routing verified; G5 proceeds

**As registered:** G1 FAIL. Only 6 of 24 labels reach the governor's dry-run exit 78. G2, G3 and G4 PASS.

**The cause is the order, not the seat.** The order's G1 asked every label to reach the governor in a dry run. The
lead did not read `tool/run_gpu_line.py:39-42`, which refuses on purpose:
- every mutant label until `candidate_eng1` has passed;
- each candidate regression or terrain label until its baseline has passed.

A dry run passes nothing, so 18 labels cannot reach the governor before G5. The seat predicted this from the code,
ledgered the prediction before running, and correctly attempted no repair.

**Lead verification on disk:**
- `git diff 97edd1fd -- . ':!artifacts/eng1/a3' ':!orders' ':!logs'` changes, in the tool, only `evidence.py`
  lines 23-24 (ORDER and SEAL_NAME re-pointed to A3). The other files in the diff are the lead's own commit
  `46676057`.
- The A3 seal's sha256 is `e0d48543…de44`, and `FILES_WRITTEN.json`'s is `f5ada065…9704`, both as reported.
- The dependency asserts are at `run_gpu_line.py:39-42`, as quoted.

**What the round established:**
- In device mode the seal verifies on 24 of 24 labels, and all 20 arms verify on every label.
- The 6 root labels return 78 at the governor.
- Each label's exact runner argv, given to the governor with the dry-run switch, returns 78 on all 24
  (`S1_GOVERNOR_ARGV.json`).
- The audit route passes with the A3 seal, including every post-device step.
- M17 (the A1 seal) refuses with the 05:39 text.

**Ruling:**
- The routing fix is accepted for use. G5 runs the labels in `gpu_lines.json` order, which satisfies every
  dependency, under the fresh root `gpu_runs/g5_1114`. The 05:39 attempt never created it.
- G5's bar is the original order's, unchanged.
- The registered verdict of A3 stays RED on G1, and is recorded as such.

**Lesson (lead):** before registering a dry-run gate over staged lines, read the runner for inter-line dependency
checks. A line that needs an earlier line to pass cannot be dry-run on its own.

**Other notes:**
- The harness refused the seat's write of `REPORT.md`. The lead copied the final message verbatim into it.
- The seat counted 22 L82 grep hits, against the order's 21. The order's itemised lines also sum to 22, so the "21"
  was a lead miscount.
