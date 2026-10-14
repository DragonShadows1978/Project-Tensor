# T4S timing check: FIX4 vs FIX5 (lead, 2026-10-13 07:09–07:10 EDT)

Voice saw T4S take 161.7 s on FIX5, against 32.6 s on FIX4, with host load around 25. The lead ran T4S alone,
alternating builds, under the GPU lock. The load average was about 29 throughout.

| Run | FIX4 (1475fae) | FIX5 (acd42fa) |
|---|---|---|
| r1 | 15.3 s, PASS 12/12 | 9.0 s, PASS 12/12 |
| r2 | 11.7 s, PASS 12/12 | 9.3 s, PASS 12/12 |

FIX5 shows no slowdown. The difference Voice saw is host-load variance, not the new cross-thread wait in
`synchronize()`, which returns at once when no other thread has an admitted release. These are not performance
measurements: the host was loaded.
