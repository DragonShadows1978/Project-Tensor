import subprocess
if subprocess.run(["forge-gpu-lease-check", "--require-held"]).returncode != 0: raise SystemExit("not inside a held forge-gpu-run lease")
# BWD-F2 K3 GPU half / R11 restage (staged GPU line): one budgeted chunk of production's suite for one build,
# continuing a series (scripts/bwdf2_suite.py, a copy of BWD-F1's runner). Usage: python3 -B -P
# scripts/bwdf2_step_g4.py <series gpu_B|gpu_F2> <arm B|F2> <budget seconds>. Writes artifacts/bwdf1/f2/g4/<series>/
# part_<n>/ (write-once per part) and the series' progress.json / outcomes.json; installs the arm's files and modules
# into the worktree and restores Build F2 after. Exits 0 at once when the series is complete.
# Prior art: BWD-F1's and ENG4's g4 steps (this repo).
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bwdf2_suite  # noqa: E402

series, arm, budget = sys.argv[1], sys.argv[2], sys.argv[3]
assert series == f"gpu_{arm}", (series, arm)
sys.argv = ["bwdf2_suite.py", series, arm, "--budget", budget]
sys.exit(bwdf2_suite.main())
