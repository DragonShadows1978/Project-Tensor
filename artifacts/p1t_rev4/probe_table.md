| Path under artifacts/p1t_rev4/probes/ | Decides |
|---|---|
| core_export_device.py | Two GPUs: decides V4-01 wrong-device export dependency; explicit mismatch refusal or correct producer completion and values must pass. timeout60; exit3 needs two GPUs. |
| core_import_query_error.py | Full rebuilt core, CUDA_VISIBLE_DEVICES empty: decides V4-03 refusal before ownership transfer. timeout30. |
| sync_existing_error.py | One GPU: decides V4-04 preexisting synchronous launch-error reporting, with no device-memory fault. timeout30; exit3 if staging is inconclusive. |
| addon_lifetime.py | F01 side-stream capsule-only manager retention and finite reclamation/reuse stress. timeout60; all 12 outputs correct and entries retained. |
| core_edges.py | Core scalar/zero/offset/singleton/used-capsule/stream and shape-refusal cases. timeout60; every assertion passes. |
| core_device_drain.py | Two GPUs: F05 last-use completion before deleter and caller-device restoration. timeout60; exit3 needs two GPUs. |
| core_exit_overrun.py | Full imported Storage foreign release at ordinary exit: run during and bounded, optionally nested/after. timeout25 per process; completed deleter before finalizing, rc0. |
| numerics_and_streams.py | Finite ONNX fp64 parity, repeatability, dual handles, L=1/eps=0, special rounding and nondefault producer ordering. timeout90. |
| release_failures.py | Actual FIX1/FIX2 OOM, producer exceptions, fallback/bound errors, partial launch and pending-at-exit tests. timeout650; rc0 and no skips. |
| cuda_errors_multidevice.py | Multi-device FIFO retention, exact output, restore caller device plus existing-error census. timeout90; exit3 needs two GPUs. |
| host_checks.py / operand_census.py / real_metadata_plans.py | CPU rebuilt add-on: 20 subprocess cases, 17 ownership/metadata cases and 129 live fixture geometries. timeout120/40/40; inspect each result, not collector rc alone. |
| lifecycle.cpp / lifecycle_timeout.py | CPU current core admission/GIL/wait slice: default and short ordinary-finalization cases. timeout25/15; finite deleter completes first, still_in_flight=0, rc0. |
| core_interop_cpu.cpp / build_core_interop_cpu.sh | CPU actual core import with real runtime (V4-03 witness); export call tracing (V4-01 mechanism only). timeout400; trace reproduces wrong-device record and failed-query consumption. |
| sync_slot_cpu.cpp | CPU actual add-on sync with explicit successful API returns and a staged slot; decides whether source checks the slot (it does not). Trace only, not real CUDA validation. timeout15. |
| core_cpu.cpp / build_cpu.sh | CPU builds, core zero-extent source helpers and core syntax. Includes actual FIX2 ck_census and FIX3 accounting harness compilation; run only CPU modes, never dtor/sweep here. timeout1100 build. |
