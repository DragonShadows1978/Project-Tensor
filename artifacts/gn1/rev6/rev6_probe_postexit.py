#!/usr/bin/env python3
"""GN1-REV6 (re-run of REV5) probe (D4): a post-exit step outside A8's protected block.
run_gpu_line.py builds the receipt with seal_sha256=sha(staging_seal.json) after
the try/except. The reviewer makes ITS OWN mirror's seal unreadable while the
child runs (mode restored right after the parent returns) and records whether a
parent receipt is still published. Prior art: fault injection (Meszaros 2007);
no prior art known to me for the specific case.
"""
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hostile_rev6 as H  # noqa: E402

H.BASE.mkdir(parents=True)
H.make_tools()
runs = H.start("PROBE_seal_unreadable_during_child")
seal = H.MA8 / "staging_seal.json"
mode = os.stat(seal).st_mode & 0o777
p = H.parent(runs, "candidate_focus", [*H.SR(), *H.CANON, "--", *H.INNER], popen=True,
             REV2_MODE="hang_after_result", REV3_SLEEP="4")
time.sleep(2.0)
os.chmod(seal, 0)
try:
    out, err = p.communicate(timeout=120)
finally:
    os.chmod(seal, mode)
r = H.collect(runs, "candidate_focus", dict(exit=p.returncode, stdout=out[-1200:], stderr=err[-2500:]))
sc = H.score(runs, "candidate_focus")
res = dict(parent_exit=r["exit"], parent_stderr_json=r["stderr_json_line"], traceback=r["traceback"],
           receipt=r["parent"], attempt_files=r["last_attempt_files"], seal_mode_restored=oct(os.stat(seal).st_mode & 0o777),
           scorer_line=sc.get("line"))
print(json.dumps(res, indent=1, default=str)[:2500])
(Path(__file__).resolve().parent / "rev6_probe_postexit.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
