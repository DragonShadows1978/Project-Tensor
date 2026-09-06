## A5 production-pool decode — kernel sweep / in-model timing

Pool ON after raw persistent weight loading, including in-process controls and all decode forwards. 32 teacher-forced tokens, per-token CUDA-synchronized wall time. Prefill excluded from tokens/s; setup, controls and prefill included in worker TERM 290s (+5s grace). 32K plans use same-arm/bit pool-on 8192: setup + guard + 16*prefill + 4*decode_work + 15 seconds. Missing measurement blocks; estimate >=290s is terminal non-fit before lease.

| Bits | S | Arm | Status/outcome | tokens/s | ms/token | Pool reserved peak MiB | Planning seconds |
|---:|---:|---|---|---:|---:|---:|---|
| 4 | 2048 | A | UNRUN | — | — | — | [60, 280] |
| 4 | 2048 | B | UNRUN | — | — | — | [60, 280] |
| 4 | 2048 | C | UNRUN | — | — | — | [60, 280] |
| 4 | 8192 | A | UNRUN | — | — | — | [60, 280] |
| 4 | 8192 | B | UNRUN | — | — | — | [60, 280] |
| 4 | 8192 | C | UNRUN | — | — | — | [60, 280] |
| 8 | 2048 | A | UNRUN | — | — | — | [60, 280] |
| 8 | 2048 | B | UNRUN | — | — | — | [60, 280] |
| 8 | 2048 | C | UNRUN | — | — | — | [60, 280] |
| 8 | 8192 | A | UNRUN | — | — | — | [60, 280] |
| 8 | 8192 | B | UNRUN | — | — | — | [60, 280] |
| 8 | 8192 | C | UNRUN | — | — | — | [60, 280] |
| 4 | 32768 | A | UNRUN | — | — | — | — |
| 4 | 32768 | B | UNRUN | — | — | — | — |
| 4 | 32768 | C | UNRUN | — | — | — | — |
| 8 | 32768 | A | UNRUN | — | — | — | — |
| 8 | 32768 | B | UNRUN | — | — | — | — |
| 8 | 32768 | C | UNRUN | — | — | — | — |

Peak source: CUDA default-pool ReservedMemHigh and UsedMemHigh, reset before measured prefill. `peak_resident_mib` is a compatibility alias for reserved pool high water, **not whole-device resident peak**; raw weights and driver/context allocations are excluded. Legacy pool-off decode receipts retain exact intercepted-allocation evidence and are never scheduled by default.
P5 (bulk4, C/B at starting S=32768): `{"ratio": null, "source_kind": "decode_pool", "status": "UNASSESSED", "threshold": 2.0}`. Only valid `decode_pool` receipts qualify. No pool-off fallback. Contexts 32769–32800 exceed the trained window. **G2/G3 rows establish nothing about model quality by themselves.**

32K captures remain registered but are OFF by default. Explicit `run CELL` is lead cell-list inclusion; `APA_SP3_INCLUDE_32K_CAPTURES=1` opts into resume/command generation. The separately generated `lead_commands_32k_captures.txt` is a lead opt-in list. Existing a4 disk preflight is unchanged; its conservative 490,783,899,647-byte first-range requirement exceeds the lead-reported free disk. This amendment does not relax that rail.

