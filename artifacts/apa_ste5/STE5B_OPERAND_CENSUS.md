# STE5B effective_pins_sha256 operand census

Production readers/writers (all occurrences inspected; only capture5/verify5 changed):

| File:line | Role | Record and enforced binding |
|---|---|---|
| capture5.py:71 | writer (new) | capture.json: digest of original INPUT_PINS plus all immutable amendments |
| verify5.py:68-70 | schema reader (new) | missing capture binding explicitly rejects; never defaults |
| verify5.py:77-80 | reader (unchanged equality) | capture metadata must equal current effective digest |
| verify5.py:210 | reader | timing receipt must equal current effective digest |
| verify5.py:232 | writer | current verifier report, including premise report |
| common.py:154 | reader | saved lead/PREMISE.json must equal current effective digest; unchanged |
| common.py:162-165 | digest provider | ordered original pin digest plus every amendment digest, SHA-256; unchanged |
| timing5.py:44 | writer | timing receipt effective digest; unchanged |
| bundle5.py:98 | reader/copy writer | verifier report copied to manifest g1 evidence; unchanged, not executed |
| tests/test_verify5.py real-schema test | test-only writer/mutator | faithful exported metadata, new binding / missing / wrong binding cases |

Historical scratch scripts (read, not executed or modified):
- scratch/amend_test.py:24 adds common reader; :28 attempts capture/timing writer; :31-33 adds verifier readers/writer; :36 copies bundle field. Literal replacement :28 has no spaces after colons; real capture5 uses spaces, so replacement does not match. Amendment 001 shows capture5 before == after.
- scratch/cpu_receipts.py:19 writes CPU summary binding (historical 34-test helper, not rerun).
- scratch/reference_preflight.py:20 writes CPU oracle preflight binding.
- scratch/report5.py:24 writes status summary binding.

Real data census: six capture records, nine NPZ files, three fresh PIDs per suite, two original ran/rc0 governor rows. Every capture omits the key. The failed lead/PREMISE.json already contains the key (f8d7623066a51a8d28fa2ebd4baa766fb6f085a817246c43167d8e8671b27de4); it is not the crashing operand. STAGING.json has command-file provenance only and is not a capture record. CPU summary/preflight/status receipts also contain historical bindings and are not capture admission evidence.

No capture-time effective binding can be reconstructed from the old input_pins_sha256 alone: it names only the original pin file, not the amendment set. Re-capture required; no fallback permitted. The revised capture driver hash also changes and the verifier continues to require its exact hash.

Prior art: no prior art known to me; no new algorithm. Original pin/digest implementation is unchanged.
