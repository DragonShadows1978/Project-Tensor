# ENG2-REV1 ledger

Read the local HOUSE_RULES mirror, project AGENTS, ENG2 mechanism order, and ENG2-REV1 order. User order allows read-only git. No subagents. Initial REPORT.md and findings.json written before checks. No product edits authorized.

Registration: six questions, exact module-ID-only SASS normalization, nonzero/nonconstant array checks, paired CPU comparisons in one process, four traced mutant killers. CUDA hidden; one thread; nice 19; timeout under ten minutes; TMPDIR=/tmp/eng2_rev1_bootstrap; bytecode disabled. No registered gates run.

L01 instruments: Python 3.12.3; NumPy 2.4.2; pytest 9.0.2; CUDA cuobjdump 12.6 V12.6.77; git 2.43.0.

Deviations: `python3 -m numpy --version` failed because NumPy has no __main__; repaired by `python3 -c "import numpy; print(numpy.__version__)"`. No check affected.

Memory quick pass: no ENG2-specific prior context; all review claims will use live source and receipts.

Prior art: differential testing (McKeeman, 1998) and mutation testing (DeMillo, Lipton and Sayward, 1978), unverified -- lead to check using those author/year search terms. Taken: compare baseline/candidate and execute seeded faults; ours: ENG2-specific probes. Float32 accumulation: Micikevicius et al., 2018, Mixed Precision Training, unverified -- lead to check. No novel algorithm claimed.

- source_reads: exit 0; `source_reads.command.json`, `source_reads.log`.

- tool_reads: exit 0; `tool_reads.command.json`, `tool_reads.log`.

- focused_reads: exit 0; `focused_reads.command.json`, `focused_reads.log`.

- evidence_reads: exit 0; `evidence_reads.command.json`, `evidence_reads.log`.

- edge_probes: exit 0; `edge_probes.command.json`, `edge_probes.log`.

Findings E2R-F01/F02/F03 reproduced and recorded before further review.

Deviations: focused_reads included an unmatched build/*.py rg glob; direct relevant source reads succeeded, and no claim relies on that failed glob. The _factory spy returned a nonempty array with device=cpu; the engine from_host path attempted CUDA and refused with no CUDA-capable device. No GPU execution occurred. Repair 1 for probe_edges.py: use zero-byte spy results in future; CUDA remains hidden.

- tape_reads: exit 0; `tape_reads.command.json`, `tape_reads.log`.

- cpu_review: exit 0; `cpu_review.command.json`, `cpu_review.log`.

- sass_review: exit 0; `sass_review.command.json`, `sass_review.log`.

- norm_setup_reads: exit 0; `norm_setup_reads.command.json`, `norm_setup_reads.log`.

- norm_review: exit 1; `norm_review.command.json`, `norm_review.log`.

Deviation / norm_review.py repair 1 of at most 3: L19 liveness guard stopped on the baseline all-nonfinite (4,16) gradient. Classify an entirely nonfinite result as invalid rather than treating it as an empty/vacuous numeric pass; keep finite nonzero/nonconstant checks. Original failure: norm_review.log. Re-run as norm_review_r1.

- norm_review_r1: exit 0; `norm_review_r1.command.json`, `norm_review_r1.log`.

- collection_reads: exit 0; `collection_reads.command.json`, `collection_reads.log`.

- static_tool_review: exit 1; `static_tool_review.command.json`, `static_tool_review.log`.

Deviation / static_tool_review.py repair 1 of at most 3: COPY_DEPS explicitly registers e6_record.py as binned with sha256_after=null; initial script attempted to hash the absent file. Check registered absence instead. Original traceback in static_tool_review.log.

- static_tool_review_r1: exit 0; `static_tool_review_r1.command.json`, `static_tool_review_r1.log`.

- census_review: exit 1; `census_review.command.json`, `census_review.log`.

Deviation / census_review.py repair 1 of at most 3: exact JSON comparison encountered two Tensor.astype TypeError messages containing different Python object addresses. Only `object at 0x[0-9a-f]+` is normalized for receipt comparison; outcomes/classes/messages otherwise exact. Raw census JSON retained. Original assertion: census_review.log.

- census_review_r1: exit 0; `census_review_r1.command.json`, `census_review_r1.log`.

- cross_arm_review: exit 0; `cross_arm_review.command.json`, `cross_arm_review.log`.

E2R-F04 reproduced through real collection/_record and exact scorer loop with clearly labeled equal constructed payloads; E2R-F05 recorded for omitted entries/private fallback. No production/tool change made.

- completion_checks: exit 0; `completion_checks.command.json`, `completion_checks.log`.

- final_checks: exit 0; `final_checks.command.json`, `final_checks.log`.

- nonstring_random_review: exit 0; `nonstring_random_review.command.json`, `nonstring_random_review.log`.

- final_versions: exit 0; `final_versions.command.json`, `final_versions.log`.

Prior art addendum: completion_checks.py uses Higham (2002) gamma_n bounds and IEEE 754 rounding accounting; unverified -- lead to check "Higham Accuracy Stability Numerical Algorithms gamma n". Proposed identifier corrections reuse evidence.norm_id (this repository, 2026); type validation reuses Python/NumPy refusal practice. No novel algorithm claimed.

Final verdict RED: 2 BLOCKER, 2 MAJOR, 1 MINOR; no NIT. Findings remain open because product/tool edits are forbidden. Every question answered in REPORT.md and answers.json. findings.json SHA-256 1938c6e080a4d729c1a214ba0ebf041eb282f1b18307c4c0145fc0643db71a67.
