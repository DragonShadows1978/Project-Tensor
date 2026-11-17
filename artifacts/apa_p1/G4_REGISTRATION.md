# G4 staged registration (immutable, no device data collected)

51 lease-wrapped lines: six real captured numerical cases (two models, three S), one bytes line (six full caches), and 44 production regression files (2078 frozen collected IDs).

Every line: foreground timeout 1750s; forge-gpu-run --min 120 --max 600; inner timeout 540s. Numerical child timeouts 240s per arm; byte child 80s per cache; regression child 240s per module. Commands, source/module SHA pins, exact expected arithmetic, complete IDs and nonvacuity guards are frozen in EXPECTED.json before the lead's first device execution. Each command exit/output is receipted only after it exits.

No seat GPU execution. Lead invokes STAGED_GPU_LINES.sh exactly as written from root. If the window/tool times out the line records that failure; it is never counted as equality. No scripts edit another run's receipts. GPU worker output creation is exclusive.
