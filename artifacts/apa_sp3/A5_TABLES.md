## A5 production-pool decode — kernel sweep / in-model timing

Pool ON after raw persistent weight loading, including in-process controls and all decode forwards. 32 teacher-forced tokens, per-token CUDA-synchronized wall time. Prefill excluded from tokens/s; setup, controls and prefill included in worker TERM 290s (+5s grace). 32K plans use same-arm/bit pool-on 8192: setup + guard + 16*prefill + 4*decode_work + 15 seconds. Missing measurement blocks; estimate >=290s is terminal non-fit before lease.

| Bits | S | Arm | Status/outcome | tokens/s | ms/token | Pool reserved peak MiB | Planning seconds |
|---:|---:|---|---|---:|---:|---:|---|
| 4 | 2048 | A | MEASURED | 4.983881818602905 | 200.6468123436207 | 1152.0 | [60, 280] |
| 4 | 2048 | B | MEASURED | 3.276394157771093 | 305.2135829348117 | 1312.0 | [60, 280] |
| 4 | 2048 | C | MEASURED | 2.5849036484968444 | 386.8616149702575 | 512.0 | [60, 280] |
| 4 | 8192 | A | UNRUN | — | — | — | [60, 280] |
| 4 | 8192 | B | MEASURED | 0.8980462154039196 | 1113.5284385673003 | 1824.0 | [60, 280] |
| 4 | 8192 | C | MEASURED | 0.9541114291473313 | 1048.0956096434966 | 1824.0 | [60, 280] |
| 8 | 2048 | A | MEASURED | 4.977119800800429 | 200.91941524879076 | 1152.0 | [60, 280] |
| 8 | 2048 | B | MEASURED | 3.2417339711597206 | 308.476885795244 | 1248.0 | [60, 280] |
| 8 | 2048 | C | MEASURED | 2.587737450832818 | 386.43796714313794 | 512.0 | [60, 280] |
| 8 | 8192 | A | UNRUN | — | — | — | [60, 280] |
| 8 | 8192 | B | MEASURED | 0.8988825587292588 | 1112.4923832248896 | 1824.0 | [60, 280] |
| 8 | 8192 | C | MEASURED | 0.9550237713907892 | 1047.0943550899392 | 1824.0 | [60, 280] |
| 4 | 32768 | A | UNRUN | — | — | — | — |
| 4 | 32768 | B | NON_FIT_PLANNED_RAIL | — | — | — | 874.1740921866149 |
| 4 | 32768 | C | NON_FIT_PLANNED_RAIL | — | — | — | 593.8294479208998 |
| 8 | 32768 | A | UNRUN | — | — | — | — |
| 8 | 32768 | B | NON_FIT_PLANNED_RAIL | — | — | — | 886.2879916499369 |
| 8 | 32768 | C | NON_FIT_PLANNED_RAIL | — | — | — | 606.1457218374126 |

Peak source: CUDA default-pool ReservedMemHigh and UsedMemHigh, reset before measured prefill. `peak_resident_mib` is a compatibility alias for reserved pool high water, **not whole-device resident peak**; raw weights and driver/context allocations are excluded. Legacy pool-off decode receipts retain exact intercepted-allocation evidence and are never scheduled by default.
P5 (bulk4, C/B at starting S=32768): `{"ratio": null, "source_kind": "decode_pool", "status": "UNASSESSED", "threshold": 2.0}`. Only valid `decode_pool` receipts qualify. No pool-off fallback. Contexts 32769–32800 exceed the trained window. **G2/G3 rows establish nothing about model quality by themselves.**

32K captures remain registered but are OFF by default. Explicit `run CELL` is lead cell-list inclusion; `APA_SP3_INCLUDE_32K_CAPTURES=1` opts into resume/command generation. The separately generated `lead_commands_32k_captures.txt` is a lead opt-in list. Existing a4 disk preflight is unchanged; its conservative 490,783,899,647-byte first-range requirement exceeds the lead-reported free disk. This amendment does not relax that rail.

