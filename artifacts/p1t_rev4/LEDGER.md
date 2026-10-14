# P1T-REV4 ledger

Registration and report created before gates. House rules and Project-Tensor AGENTS read. Memory registry has no relevant P1T review history; not used.

Source review of core DLPack, storage, checked helpers, all add-on production code/plan functions/build/wrappers preceded fix-report consultation. Full/fix3 diffs saved; working_vs_target empty.
CPU build PASS: scratch unchanged add-on, core syntax, current core helper/lifecycle slices, actual Operand and accounting harnesses. Host checks 20 subprocess rc0; operand census 17 checks / 7 once-only managers; ck_census 18/18; accounting 17/17 UBSan clean; two actual entry-point aggregate refusals with once-only deleters.
Lifecycle ordinary exit default (finite 5.5 s deleter) and short both rc0 with completion before finalizing; diagnostics while waiting.
Live real metadata: manifest 139 cases, 61 NPZ weight headers; native plans match 129 geometries. Other 10 Resize/Pad cases need scale/pad values/numeric GPU probe.
New findings: V4-01 MAJOR exporter device mismatch (trace tensor_device=0 recorded_device=1); V4-02 NIT stale wait docstring; V4-03 MINOR ignored core query failure (real runtime cudaErrorNoDevice 100, import succeeds/consumes); V4-04 MINOR preexisting synchronous error slot (source reasoning; actual-source CPU trace, GPU witness pending). Prior art at probe sites: DLPack2017+/Array API2021+, CUDA12.6 event/device/error rules; fix recommendation is standard guard/error check, nothing novel.
Core trace first compilation failed because the probe build driver read only the includes line from python3-config and missed linker flags. Corrected driver only; failed log preserved.
Lead receipt audit: 236 PASS entries claimed by supplied JSON; 52 failure IDs in lead and FIX3 combined logs match each other. Historical evidence, not this seat running GPU tests.

Foreground bounded commands: build_cpu.sh timeout1100 (each nested build timeout180/240); host_checks timeout120; operand_census/checked/accounting timeout40; lifecycle default timeout25/short timeout15; core_interop build+run timeout400; aggregate entrypoints subprocess timeout20; scratch build/input hashes and ldd/nm/objdump captured. Every returned receipt was inspected; no GPU launch executed here.
Procedural deviation: after an orchestration tool yielded during the foreground build's maximum PTY wait, one functions.wait used 1000ms instead of the required maximum. The next wait used300000ms; no process was terminated, and the foreground build finished normally. No status was sent between these waits.

Final verification PASS: required 14 closure IDs and R1-R6 present; findings counts 0 BLOCKING/1 MAJOR/2 MINOR/1 NIT; tables embedded verbatim; all 48 submitted hashes unchanged; copied prior probes byte-identical; receipt assertions pass. Final report/JSON complete, verdict REJECT. See final_verification.json.
