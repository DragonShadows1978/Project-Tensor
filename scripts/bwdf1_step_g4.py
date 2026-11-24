import subprocess
if subprocess.run(["forge-gpu-lease-check"]).returncode != 0: raise SystemExit("not inside a forge-gpu-run lease")
# BWD-F1 G4 GPU half (staged GPU line): one budgeted chunk of production's suite for one build, continuing a series
# (scripts/bwdf1_suite.py, ENG4's chunking). Usage: python3 -B -P scripts/bwdf1_step_g4.py <series gpu_B|gpu_E>
# <arm B|E> <budget seconds>. Writes artifacts/bwdf1/g4/<series>/part_<n>/ (write-once per part) and the series'
# progress.json / outcomes.json; installs the arm's files and modules into the worktree and restores Build E after.
# Exits 0 at once when the series is complete. Prior art: ENG4's eng4_gpu_line.py g4 steps (this repo).
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bwdf1_suite  # noqa: E402

series, arm, budget = sys.argv[1], sys.argv[2], sys.argv[3]
assert series == f"gpu_{arm}", (series, arm)
sys.argv = ["bwdf1_suite.py", series, arm, "--budget", budget]
sys.exit(bwdf1_suite.main())
