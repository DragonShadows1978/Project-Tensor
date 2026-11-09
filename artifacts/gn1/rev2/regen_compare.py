#!/usr/bin/env python3
"""GN1-REV2 Q6: run A2's own regenerate_arms.py into a fresh /tmp mirror and
compare sha256 maps with A2's on-disk arms, arm_maps.json, the seal and the
mutant sources.

Method: the mirror holds byte copies (sha256-checked) of exactly the inputs the
script reads: regenerate_arms.py, build_mutants.py, order_commit.txt,
engine_pin.json, d5_edits.json, staging_seal.json (for --verify-seal), the
worktree nn.py (equality assert only) and the pinned module. Git is read-only:
GIT_DIR points at this worktree's git dir; only `git ls-tree` and `git show`
run (inside the script). Nothing outside /tmp/gn1_rev2_* and rev2/ is written.

Prior art: SHA-256 (FIPS 180-4) content maps; reproducible-build comparison by
content digest (Reproducible Builds project, ~2013+, unverified - lead to
check: "reproducible builds sha256 comparison"). Ours: the file selection.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REAL = Path(__file__).resolve().parents[3]
A2 = REAL / "artifacts/gn1/a2"
OUT = Path(__file__).resolve().parent
MROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/gn1_rev2_regen/W")
MA2 = MROOT / "artifacts/gn1/a2"
GIT_DIR = subprocess.check_output(["git", "-C", str(REAL), "rev-parse", "--absolute-git-dir"], text=True).strip()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
INPUTS = ["regenerate_arms.py", "build_mutants.py", "order_commit.txt", "engine_pin.json", "d5_edits.json",
          "staging_seal.json"]


def tree_map(base):
    base = Path(base)
    out = {}
    for p in sorted(base.rglob("*")):
        if p.is_symlink():
            out[str(p.relative_to(base))] = "SYMLINK"
        elif p.is_file() and "__pycache__" not in p.parts:
            out[str(p.relative_to(base))] = sha(p)
    return out


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not MROOT.exists(), "fresh mirror required: " + str(MROOT)
    MA2.mkdir(parents=True)
    seal = json.loads((A2 / "staging_seal.json").read_text())
    pin = json.loads((A2 / "engine_pin.json").read_text())
    copied = {}
    for f in INPUTS:
        shutil.copyfile(A2 / f, MA2 / f)
        copied[f] = dict(sha256=sha(MA2 / f), sealed=seal["tool_files"].get(f), source=sha(A2 / f))
    for rel in ("tensor_cuda/tensor_cuda/nn.py", pin["module"]):
        (MROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL / rel, MROOT / rel)
        copied[rel] = dict(sha256=sha(MROOT / rel), source=sha(REAL / rel))
    env = dict(os.environ, GIT_DIR=GIT_DIR, PYTHONDONTWRITEBYTECODE="1", TMPDIR="/tmp/gn1_rev2_tmp")
    r = subprocess.run([sys.executable, "-B", str(MA2 / "regenerate_arms.py"), "--verify-seal"], cwd=MA2, env=env,
                       capture_output=True, text=True, timeout=600)
    result = dict(mirror=str(MROOT), git_dir_readonly=GIT_DIR, inputs=copied, exit=r.returncode,
                  stdout=r.stdout[-2000:], stderr=r.stderr[-3000:])
    if r.returncode == 0:
        regen_arms = {a.name: tree_map(a) for a in sorted((MA2 / "arms").iterdir())}
        disk_arms = {a.name: tree_map(a) for a in sorted((A2 / "arms").iterdir())}
        sealed_arms = {arm: {"tensor_cuda/" + k: v for k, v in files.items()} for arm, files in seal["arms"].items()}
        regen_maps = json.loads((MA2 / "arm_maps.json").read_text())
        result["arm_count"] = dict(regenerated=len(regen_arms), on_disk=len(disk_arms), sealed=len(sealed_arms))
        result["regenerated_equals_on_disk"] = regen_arms == disk_arms
        result["regenerated_equals_seal"] = regen_arms == sealed_arms
        result["regen_arm_maps_json_equals_a2"] = sha(MA2 / "arm_maps.json") == sha(A2 / "arm_maps.json")
        result["regen_arm_maps_equals_seal"] = regen_maps == seal["arms"]
        result["mutants_json_equals_a2"] = sha(MA2 / "mutants.json") == sha(A2 / "mutants.json")
        result["mutants_json_equals_seal"] = sha(MA2 / "mutants.json") == seal["tool_files"]["mutants.json"]
        result["mutant_sources_equal_a2"] = tree_map(MA2 / "mutants") == {
            k: v for k, v in tree_map(A2 / "mutants").items() if k.endswith(("nn.py", "nn.diff"))}
        extra_in_a2_mutants = sorted(k for k in tree_map(A2 / "mutants") if not k.endswith(("nn.py", "nn.diff")))
        result["a2_mutants_extra_files"] = extra_in_a2_mutants
        diffs = {}
        for arm in sorted(set(regen_arms) | set(disk_arms)):
            a, b = regen_arms.get(arm, {}), disk_arms.get(arm, {})
            d = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            if d:
                diffs[arm] = d
        result["per_arm_differences"] = diffs
        result["symlinks_regenerated"] = sum(v == "SYMLINK" for m in regen_arms.values() for v in m.values())
        result["symlinks_on_disk"] = sum(v == "SYMLINK" for m in disk_arms.values() for v in m.values())
        # product nn.py at the A2 commit equals the regenerated candidate
        a2_commit_nn = subprocess.check_output(["git", "-C", str(REAL), "show", "bd30498a:tensor_cuda/tensor_cuda/nn.py"])
        result["candidate_nn_equals_commit_bd30498a"] = (
            hashlib.sha256(a2_commit_nn).hexdigest() == sha(MA2 / "arms/candidate/tensor_cuda/nn.py"))
        result["engine_copies_equal_pin"] = all(
            m["tensor_cuda/" + Path(pin["module"]).name] == pin["sha256"] for m in regen_arms.values())
        (OUT / "regen_maps_regenerated.json").write_text(json.dumps(regen_arms, indent=1, sort_keys=True) + "\n")
        (OUT / "regen_maps_a2_disk.json").write_text(json.dumps(disk_arms, indent=1, sort_keys=True) + "\n")
    (OUT / "regen_compare.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("inputs",)}, indent=1))
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
