# SP2 error margins — BLOCKED G2

Evidence class: kernel sweep; this establishes nothing about model quality.
No GPU samples or measured margins exist from this seat.

| Bulk bits | D | Max | p99.9 | Mean | Registered e_q |
|---|---|---|---|---|---|
| 2 | 64 | BLOCKED | BLOCKED | BLOCKED | not frozen |
| 2 | 128 | BLOCKED | BLOCKED | BLOCKED | not frozen |
| 4 | 64 | BLOCKED | BLOCKED | BLOCKED | not frozen |
| 4 | 128 | BLOCKED | BLOCKED | BLOCKED | not frozen |

Rule: maximum over all valid registered calibration scores, pooled by (bits,D),
rounded upward to fp32. The maximum covers observed scores; a percentile would
knowingly exclude some. Neither establishes an arbitrary-input bound. G2 uses
2 seeded q/K draws per shape, 32 sampled queries per prefill draw or the one
decode query, and all eligible keys. The GPU quantizer is the existing rotated
Lloyd-Max 2/4-bit TurboQuant path, reconstructed fp32. Scale is fp32(1/sqrt(D)).

Machine-readable blocked values and exact commands: `GPU_BLOCKED.json` and
`lead_commands.txt`. After all 64 G2 jobs, `run freeze` creates immutable
`e_q_table.json` and `e_q_table.sha256`; `summary` puts the measured table in
`EPSILON_CURVE.md`. This initial blocked report remains historical.
