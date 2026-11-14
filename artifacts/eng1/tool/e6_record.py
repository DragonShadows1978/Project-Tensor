"""ENG1 L46 write-detection record around a terrain-file run (the tool's own instrument, independent of
test_eng1.py's E6 test).

The target is the directory the terrain file's `repo_root / "artifacts"` writes would reach:
Path(terrain).resolve().parents[2] / "artifacts". The record holds:
- `ordered`: `git status --porcelain --ignored -- artifacts/` exactly as the order names it (informational: it lists
  ENG1's own ignored receipts, which the run writes, so it is not compared);
- `uall`: `git status --porcelain --ignored -uall -- <target>` (an arm copy's target is an untracked directory, which
  plain git status would collapse into one line);
- `stat`: size and mtime_ns of every file under the target, plus sha256 for .ppm files (all 17 receipts are tracked,
  and a rewrite with identical bytes is invisible to git status; ledger F4).
The comparison is `uall` and `stat`. Prior art: git status porcelain v1 (git 1.7+); POSIX stat mtime. Ours: the
record's shape.
"""
import hashlib
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def _git(*args):
    p = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--ignored", *args], capture_output=True,
                       text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    return p.stdout


def snapshot(terrain):
    target = Path(terrain).resolve().parents[2] / "artifacts"
    rel = os.path.relpath(target, ROOT)
    stat = {}
    if target.is_dir():
        for dirpath, dirnames, filenames in os.walk(target):
            if rel == "artifacts" and Path(dirpath) == target:
                dirnames[:] = [d for d in dirnames if d != "eng1"]
            for f in filenames:
                fp = Path(dirpath) / f
                st = fp.stat()
                row = [st.st_size, st.st_mtime_ns]
                if f.endswith(".ppm"):
                    row.append(hashlib.sha256(fp.read_bytes()).hexdigest())
                stat[str(fp.relative_to(target))] = row
    exclude = [":(exclude)artifacts/eng1"] if rel == "artifacts" else []
    return dict(target=rel, uall=_git("-uall", "--", rel, *exclude), stat=stat, ordered=_git("--", "artifacts/"))


def compare(before, after):
    """True when the run wrote into the target: the compared fields differ."""
    return before["uall"] != after["uall"] or before["stat"] != after["stat"]
