| Path under artifacts/p1t_rev8/ | Decides / result |
|---|---|
| probes/build_cpu.sh | Build source-linked release/UBSan harnesses and syntax-check core bindings. PASS in logs/p1t_rev8_cpu_build.log. |
| probes/build_core_host.py | Generate/hash actual core DLPack/lifecycle/dispatch spans and compile host module. PASS in logs/p1t_rev8_core_build.log. |
| probes/run_cpu_checks.py | 14 inherited release, OOM, accounting and concurrent-sync CPU checks. PASS; cpu_results.json. |
| probes/argument_census_cpu.py | 1067 argument reads/refusals, 29 add-on bindings plus selected core dispatch; zero user conversion calls. PASS. |
| probes/operand_census_cpu.py | 17 actual-source scalar/empty/huge/offset/read-only/copied/consumed/stride/grid descriptor checks without launch. PASS. |
| probes/knob_exit_cpu.py | Full add-on finite deleter completes before finalization despite huge diagnostic knob. PASS; --control retained. |
| probes/core_knob_exit_cpu.py | Exact-source core hook/lifecycle huge diagnostic knob; callback completes before finalization. PASS; --control retained. |
| probes/query_diagnostic_cpu.py | Query failure after successful stream sync names cudaEventQuery. PASS. |
| probes/fix7_cpu_census.py | 36 no-GPU children: 22 add-on knob classes, exceptions/stderr/pre-frame interrupt/deleted module, both raw exit entries/runners. PASS. |
| probes/attribution_cpu.py | 22 actual FIX7 host trace assertions including guard/spill/record/query/destroy and message OOM. PASS after correcting AST skip-guard filter; first harness attempt failed NameError. |
| probes/lock_checks_cpu.py | Core initial/relock retries and add-on polling fallback all diagnose and wait for finite callback. Three CPU stagings PASS. |
| probes/diagnostic_errors_cpu.cpp + probes/diagnostic_errors_cpu.py | V8-01: include current product source; one-shot diagnostic create/record/query/destroy errors must be reported. RED: 4/4 erased. |
| probes/exit_log_oom_cpu.cpp + probes/verify_exit_log_cpu.py | V8-02 add-on: final logger must name the saved query failure even under C++ allocation failure. OOM arm RED, normal control PASS; exit_log_oom_cpu.py is child. |
| probes/build_core_host_oom.py + probes/core_exit_log_oom_cpu.py | V8-02 core: actual native runner must emit its saved failure under message/snapshot allocation failure. OOM child has no CUDA stderr, control does; source-hashed generated module. Both child rc0 after stats-key correction; stderr comparison is the gate. |
| probes/diagnostic_event_shim.cpp + probes/diagnostic_events_gpu.py | LEAD ONLY, single GPU: real built add-on must report four one-shot marker errors; failed setup must still diagnose finite stream wait. Shim compiles; GPU probe NOT RUN; exit3 without GPU. |
| probes/gpu_regressions.py fix7 / fix56 / semantics / multidevice | LEAD ONLY: reviewed binaries run original FIX7/FIX5-6 or finite arithmetic/stream tests, zero-skips gate. Two-device lane exits3 on one GPU; fix7 and semantics are single-GPU forms. NOT RUN here. |
