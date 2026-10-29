# APA-DK-STE5B ledger

- Read house-rules mirror, AGENTS.md, STE5B/STE5/STE4 and REV1 mechanism evidence. No git, GPU or subagents. Registration precedes gate execution.
- Source/data inspection: all six real capture.json records omit effective_pins_sha256; lead/PREMISE.json contains it. STAGING.json is command staging, not capture pin evidence.
- Prior art: no prior art known to me. No new algorithm; repair capture/reader schema and preserve strict pin check.
- Baseline source snapshots and complete lead file hashes retained in scratch/ste5b/.
- Reproduced original failure with CPU verify5 --premise: rc1, KeyError('effective_pins_sha256'), gates {}. Receipt scratch/ste5b/before-premise.{json,md,log,rc}.
- Cause correction: actual failing reader is verify5.py:76 (pre-fix loop meta[key]), requiring effective_pins_sha256 at :72. common.py:154 reads a saved premise report that already has the key. capture5.py:77 lacks the writer; scratch/amend_test.py:28 literal replacement omitted spaces and did not match it (amendment 001 capture hash before == after). Original tests never reached validate_capture with a real record.
- Added capture pin writer and explicit missing-binding assertion; no fallback/default or check removal. Old capture driver hashes also differ after repair; re-capture is mandatory. Captures remain read-only.
- INPUT_PINS_AMENDMENT_002.json records only three harness/test pin changes, before B-gates. Original pins, amendment 001 and registrations untouched. Prior art: no prior art known to me; no new algorithm.
- B-G1 PASS: 39 passed in 11.03s (CPU author suite), rc0, scratch/ste5b/cpu.log and cpu.xml. Five new real-schema CLI cases: accepted bound CPU export, missing binding, wrong binding, wrong input pin, one-bit tensor G-G1 FAIL. New writer keys are checked directly against capture5 AST; all real tensors/governor rows are read unchanged. Fixture provenance is explicitly test-only.
- Existing base premise rerun rc1, explicit AssertionError re-capture required, gates {}. No premise PASS or G-G1 FAIL claimed. B-G2 takes registered re-capture exception; all six records lack binding. Two original governor ran/0 rows validated.
- stage5 --check rc0. GPU_COMMANDS.sh and STAGING.json unchanged; no restage needed.
- Preserved prior mutable S5_CPU.json/cpu.xml/cpu.log in scratch/ste5b/previous-* and refreshed receipts consumed by bundle5.complete to avoid a stale test-source hash. No bundle or engine write.
- Final integrity PASS: all 18 lead files byte-identical (six JSON captures, nine NPZ files, governor, old premise JSON/MD); GPU_COMMANDS.sh/STAGING.json/common.py unchanged; complete original pin set plus amendments checked. Receipts scratch/ste5b/integrity.json. No deletions or DELETE_ME moves.
- Registered outcome: STE5B ACCEPT under explicit re-capture exception. Existing base captures DO NOT COUNT; STE5 G-G1..G-G7 not accepted. Lead must re-capture with fixed writer before judging premise. No GPU by seat.
