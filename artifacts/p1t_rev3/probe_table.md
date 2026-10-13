| Path under artifacts/p1t_rev3/ | State | Pass/fail decision |
|---|---|---|
| probes/host_checks.py | CPU executed | Clean None/overflow/refusal/scalar checks, duplicate-capsule refusal, no-device sync error; dispatches core_cpu and cpu_native.20 cases rc0. |
| probes/core_cpu.cpp | CPU executed | Current verbatim core helpers/empty: normal empty succeeds; prefix/suffix overflow refuses without UBSan.3 cases rc0. |
| probes/cpu_native.cpp | CPU executed | Real release-path host allocation failure and throwing deleter do not abort.2 cases rc0 (included in host20). |
| probes/lifecycle_timeout.py | CPU executed | Finite admitted deleter must complete before normal finalization, exit0.Uses lifecycle_slice.cpp. Control rc0;0.1s and default5s overrun rc-6. |
| probes/pending_bytes.cpp | CPU executed | Per-entry and queue-byte accumulation must avoid signed overflow for individually validated descriptors. Both modes rc1 under UBSan; entry statement copied verbatim, note_push actual source. |
| probes/operand_census.py | CPU executed | Offsets/singleton strides/0-d/read-only-copied flags/duplicate consumption/zero-neighbor overflow obey contract;17 cases pass,7 consumed deleters called once. |
| probes/real_metadata_plans.py | CPU executed | Recorded real fixture output geometry matches reviewed plans;129 cases pass. Metadata only, placeholders never launched. |
| probes/core_exit_overrun.py | LEAD GPU ONLY, pending | Full real imported Storage + foreign pthread, during and bounded modes: ordinary exit0, no abort/hang, finite5.5s deleter across default5s wait. No os._exit escape. |
| probes/addon_lifetime.py | LEAD GPU ONLY, pending | 12 finite non-default-stream last-owner trials: all outputs correct and every launch increments retained entries. No universal race-freedom claim. |
| probes/core_device_drain.py | LEAD GPU ONLY, pending | Two devices: producer deleter observes device0 work complete after drop on current device1; caller device restored. Fewer than2 devices => exit3 inconclusive. |
| probes/core_edges.py | LEAD GPU ONLY, pending | Reject rank/count/byte wraps; scalar/empty/offset/singleton-stride interop and stream0 refusal obey contract. |
| probes/numerics_and_streams.py | LEAD GPU ONLY, pending | Finite conv/LSTM determinism and parity, handle settings, L=1 norm, special rint, nonlegacy producer ordering; all assertions pass. |
| probes/cuda_errors_multidevice.py | LEAD GPU ONLY, pending | Two-device FIFO releases once, outputs correct, caller device restored; synchronize raises for staged preexisting launch error. Exact runtime last-error behavior unresolved until run; fewer than2 devices => exit3. |
| probes/release_failures.py | LEAD GPU ONLY, pending | All existing FIX1/FIX2 add-on regressions execute without skips, including partial launch, queue OOM, event recycle, fallback/bound failures, throwing deleters and pending exit. Exit0 plus nonempty passing suite required. |
