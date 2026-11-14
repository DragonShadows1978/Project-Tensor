#!/usr/bin/env python3
"""ENG2-A3 L38 ID diff: collect tensor_cuda/tests/test_eng1.py's test IDs (CUDA hidden) with an arm's package preloaded,
by the same collector the sealed generator uses (tool/stage.py COLLECT, --import-mode=prepend, as its regnorm suite
collects test_eng1.py from the worktree). Writes a3/ids_test_eng1_<TAG>.txt (sorted, one ID per line).
Usage: CUDA_VISIBLE_DEVICES= TMPDIR=/tmp/eng2_a3_* collect_eng1_ids.py ARM TAG
Prior art: pytest --collect-only; the ENG2 tool's own collector. No new algorithm."""
import json, os, subprocess, sys
from pathlib import Path
A3 = Path(__file__).resolve().parent
TOOL = A3.parent / "tool"
ROOT = A3.parents[2]
sys.path.insert(0, str(TOOL))
import stage  # noqa: E402  (only its COLLECT text and norm() are used)
from arm_origin import bound_environment  # noqa: E402
arm, tag = sys.argv[1:3]
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and Path(os.environ["TMPDIR"]).name.startswith("eng2_a3_")
pkg = TOOL / "arms" / arm / "tensor_cuda"
out = Path(os.environ["TMPDIR"]) / f"collect_eng1_{arm}_{tag}.json"
env = bound_environment(pkg, dict(os.environ, ENG2_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
                                  ENG2_IMPORT_MODE="--import-mode=prepend"))
p = subprocess.run([sys.executable, "-B", "-c", stage.COLLECT, str(pkg), str(out), str(ROOT / "tensor_cuda/tests/test_eng1.py")],
                   env=env, cwd=os.environ["TMPDIR"], capture_output=True, text=True, timeout=500)
data = json.loads(out.read_text())
ids = sorted(stage.norm(i) for i in data["ids"])
(A3 / f"ids_test_eng1_{tag}.txt").write_text("".join(i + "\n" for i in ids))
print(json.dumps(dict(arm=arm, tag=tag, rc=data["rc"], n=len(ids), errors=len(data["errors"]))))
sys.exit(0 if data["rc"] == 0 and not data["errors"] else 1)
