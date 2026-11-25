import subprocess
if subprocess.run(["forge-gpu-lease-check", "--require-held"]).returncode != 0: raise SystemExit("not inside a held forge-gpu-run lease")
"""ENG5-S-T GPU step script: the only script a staged ENG5-S-T GPU line runs inside forge-gpu-run. Its first statement
(line 2, after `import subprocess`) is the held-lease check (order L125), before anything else runs.

  forge-gpu-run --min 120 --max 540 --label eng5st-<arm> -- timeout 480s python3 -B -P scripts/eng5st_gpu_step.py <arm>

Arms: S, B, s01, s02, s03, s04 (artifacts/eng5s/t/GPU_LINES_T.json). The step (scripts/eng5st_common.py run_arm):
refuses if artifacts/eng5s/t/gpu/<arm>/ exists (write-once, exit 5); refuses if CUDA_VISIBLE_DEVICES is empty in the
lease's environment (exit 7; it never changes the variable); refuses if the arm's module sha256 differs from its build
receipt (L103, exit 3; the same check also runs before the lease, in `scripts/eng5st_line.py run`); then runs
tensor_cuda/tests/test_eng5s_status_checks.py under pytest as a child, inside the lease, with the arm's package first
on PYTHONPATH, and writes junit.xml, pytest.log, VERDICT.json (from the junit, per row: scripts/eng5st_verdict.py),
STEP_RESULT.json (pytest exit, junit sha256, marker lines) and INNER_EXIT under the line's directory. A mutant arm's
marker file (ENG5S_MUTANT_MARKER) is written there by the mutated module.

Prior art: ENG5-S's step script (this repo), whose lease-check layout and refusals this copies; the house
forge-gpu-run contract. No other prior art known to me.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eng5st_common as cm


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in cm.ARMS:
        raise SystemExit("usage: eng5st_gpu_step.py {" + "|".join(cm.ARMS) + "}")
    return cm.run_arm(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
