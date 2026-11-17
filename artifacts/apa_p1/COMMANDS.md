# APA-P1 command receipt index

All seat shell commands used CUDA_VISIBLE_DEVICES='', TMPDIR=/tmp/apa_p1_seat after its creation, PYTHONDONTWRITEBYTECODE=1 for Python. Foreground commands had explicit timeouts. No git writes.

- Preflight named reads: /mnt/ForgeRealm/HOUSE_RULES.md; production/local AGENTS.md; docs/APA.md; docs/APA_PAPER_DRAFT.md sections 2–3; git show e5538604:artifacts/apa_2p/SURVEY.md (saved preflight_survey.txt).
- Pin/instrument reads: git show/log at 0e1dea3, aaa9b4ea and order HEAD; nvcc/pytest/g++ versions; production module sha256; pinned g0/g2, build scripts, interop source, SPQ3/EB4 capture metadata through git show. Actual modules/captures read only from named artifact paths.
- timeout 30s python artifacts/apa_p1/port.py
- timeout 30s python artifacts/apa_p1/check_scope.py (initial and final)
- timeout 1650s python artifacts/apa_p1/build_gate.py (its configure/build argv, exit codes and durations are build.json)
- timeout 90s /usr/local/cuda-12.6/bin/cuobjdump --dump-sass --gpu-architecture sm_89 <port/lab module> (separate outputs sass_port.txt/sass_lab.txt)
- timeout 30s python artifacts/apa_p1/compare_sass.py (expanded to full header namespace closure before final census)
- timeout 1400s python artifacts/apa_p1/cpu_gate.py (per-arm pytest argv/exits in g3_exits.json; fresh basetemps; port binary restored in finally)
- timeout 30s python artifacts/apa_p1/compare_g3.py
- timeout 45s python artifacts/apa_p1/prepare_stage.py; CPU g++ accessor build (argv/log in interop_build.json/interop_build.log)
- CPU-only accessor smoke (interop_cpu_smoke.json); pinned lab import smoke (lab_import_cpu_smoke.json)
- timeout 90s CPU pytest --collect-only for 44 original GPU test files (gpu_collection_cpu.json/log); no tests/device calls executed by collection
- timeout 30s python artifacts/apa_p1/freeze_stage.py (created G4 registration, script, EXPECTED.json)
- timeout 30s python artifacts/apa_p1/verify_stage.py; bash -n STAGED_GPU_LINES.sh (static/CPU only)
- git ls-files --others --exclude-standard (untracked_scope.json); no untracked path outside authorized prefixes
- timeout 30s python artifacts/apa_p1/finish_report.py; final inventory/receipt verification

Failed setup attempts (before product writes): nested Python string SyntaxError while generating port.py; immediate subsequent invocation found no port.py. Corrected quoting, then successful port.py execution. Ledger records both; no gate was weakened.
