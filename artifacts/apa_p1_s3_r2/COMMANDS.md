# Commands and source records

Cwd for all actual runs: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod`. Every command used `CUDA_VISIBLE_DEVICES=`. No GPU job, git command, subagent, background job, retirement/move or seat deletion. All long commands were awaited to completion before another command started. Command timeouts were <=600 seconds, except the build's <=1780-second bound. Running exec sessions used the maximum 300000 ms process wait.

Read first: `/mnt/ForgeRealm/HOUSE_RULES.md` (present local mirror), `/mnt/ForgeRealm/Project-Tensor/AGENTS.md`, `docs/APA.md`, APA2P `artifacts/apa_2p/SURVEY.md` S3/S5, `lead/LEAD_RULING_A1_GPU.md`, `artifacts/APA_P1_S3_LEAD_RULING.md`, original immutable r2b order. The accepted r1 boundary/stop records were reused and extended. A1 source/build/SASS/outcome artifacts stayed read-only. The explicitly named EB4 lead fixture was read-only. Other project reads were for the caller census only. A direct read of the GRM checkout config identified its canonical checkout. No recursive root/share search.

Executed instrument commands: `cuobjdump --version`, `nvcc --version`, `python -B -m pytest --version`, and `FORGE_GPU_DRY_RUN=1 forge-gpu-run --min 30 --max 60 -- timeout 45s true`. Exits 0/0/0/78 are in `INSTRUMENTS.json`. A1's build script was executed through the scoped path/test-flag adapter `BUILD.py`; original A1 execution would write its read-only tree, so only the isolated adapter was used (`BUILD_PATH_ADAPTER.diff`, `BUILD.json`, `BUILD_COMPLETED.json`). Offline read-only pybind11 source is `/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src`.

Meaningful gate commands (all with empty visibility; exact pytest/child argv also in receipts):

```sh
CUDA_VISIBLE_DEVICES= timeout 550s python -B artifacts/apa_p1_s3_r2/regression.py before
CUDA_VISIBLE_DEVICES= timeout 1780s python -B artifacts/apa_p1_s3_r2/BUILD.py
CUDA_VISIBLE_DEVICES= timeout 1750s cmake --build artifacts/apa_p1_s3_r2/build -j2
CUDA_VISIBLE_DEVICES= timeout 350s python -B artifacts/apa_p1_s3_r2/sass_compare.py
CUDA_VISIBLE_DEVICES= timeout 550s python -B artifacts/apa_p1_s3_r2/regression.py after
CUDA_VISIBLE_DEVICES= timeout 190s python -B artifacts/apa_p1_s3_r2/census.py
CUDA_VISIBLE_DEVICES= timeout 30s python -B artifacts/apa_p1_s3_r2/verify.py
```

The one isolated build tree had an initial compile error and two incremental resumes: one corrected an unintended bulk-exit insertion; one changed the test hook's timeout broadcast from static shared storage to `__syncthreads_or`. No build flags/numerical thresholds were changed. Original failure logs/dump/comparison remain. SASS was rerun only after the broadcast fix; no instruction/encoding/control-word normalization.

New test execution used `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONPATH=artifacts/apa_p1_s3_r2/refs/after:tensor_cuda/tests`, `TMPDIR=/tmp/apap1s3r2_cpu_tests`, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, `timeout 180s python -B artifacts/apa_p1_a1/pytest_receipt.py ... -q --tb=short -ra -p no:cacheprovider --basetemp /tmp/apap1s3r2_cpu_tests/pytest_verified` and the three new test files. The initial marker-extraction failures and corrected runs are retained separately. Both existing regression arms ran all 74 frozen test files with the same seed 142 and exact per-ID phase scoring. Every observed external file open from the test process is recorded in each `OUTCOMES_*.json` (including installed dependencies and authorized /tmp test files); external project reads included the two GRM source/test files and the parent Project-Tensor/fork pyproject paths. No visibility-unmasking command exists in the existing test files; the existing DLPack subprocess test explicitly keeps visibility empty.

Actual staged lines are in `STAGED_GPU_LINES.sh`. Each was passed verbatim to bash from the stated cwd with `FORGE_GPU_DRY_RUN=1`, using the real `forge-gpu-run`, no executable replacement or command wrapper. All six returned78; child programs were not run. The output names the scheduler's halted state as the first unmet live-run need. `EXPECTED.json` preceded staging; `STAGING_AMENDMENT_1.json` preserves the initial and corrected source pins without changing cases, thresholds, or expected outcomes. Initial staging/dry-run records remain.

Full caller grep commands/output are pasted in `CALLER_CENSUS_FINAL_COMMANDS.md` and `DEFAULT_SWITCH_DESIGN.md`; all argument expressions are in `CALLER_ARGUMENTS.md`. `TICKET_CENSUS.txt` is the complete current writable-file counters/L2 census; `TICKET_BOUNDARY.json` describes ambient/operand device, first use, growth and all caller sites. `ALL_CHANGES.diff` includes the engine/binding edits and all new tests. Historical edit recipes `implement.py` and `add_hooks.py` remain for provenance; they are not rerunnable entry points.
