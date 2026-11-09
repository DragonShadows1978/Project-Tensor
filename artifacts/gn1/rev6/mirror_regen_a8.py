#!/usr/bin/env python3
"""GN1-REV6: independent arm regeneration with A8's own regenerate_arms.py in a
fresh /tmp/gn1_rev6_* mirror, sha256-map comparison against A8's on-disk arms,
the seal and arm_maps.json; then completion of the mirror (byte copies of every
sealed input) so A8's REAL parent, child and scorer run there with RUNS under
the mirror's a8/ (A8's guard refuses RUNS outside a8/, and a8/ is read-only here).

Method: REV5 mirror_regen_a5.py (2026; itself from REV4 mirror_regen_a4.py), retargeted at a8/. Git is read-only:
GIT_DIR points at this worktree's git dir; only `git ls-tree`/`git show` run
(inside A8's script). Writes only under /tmp/gn1_rev6_* and rev6/.

Prior art: SHA-256 (FIPS 180-4) content maps; reproducible-build comparison by
content digest (Reproducible Builds project, ~2013+, unverified - lead to check:
"reproducible builds sha256 comparison"). Ours: the file selection.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REAL = Path(__file__).resolve().parents[3]
A8 = REAL / "artifacts/gn1/a8"
OUT = Path(__file__).resolve().parent
MROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/gn1_rev6_m/W")
MA8 = MROOT / "artifacts/gn1/a8"
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
    assert str(MROOT).startswith("/tmp/gn1_rev6_")
    assert not MROOT.exists(), "fresh mirror required: " + str(MROOT)
    MA8.mkdir(parents=True)
    seal = json.loads((A8 / "staging_seal.json").read_text())
    pin = json.loads((A8 / "engine_pin.json").read_text())
    copied = {}
    for f in INPUTS:
        shutil.copyfile(A8 / f, MA8 / f)
        copied[f] = dict(sha256=sha(MA8 / f), sealed=seal["tool_files"].get(f), source=sha(A8 / f))
    for rel in ("tensor_cuda/tensor_cuda/nn.py", pin["module"]):
        (MROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL / rel, MROOT / rel)
        copied[rel] = dict(sha256=sha(MROOT / rel), source=sha(REAL / rel))
    env = dict(os.environ, GIT_DIR=GIT_DIR, PYTHONDONTWRITEBYTECODE="1", TMPDIR="/tmp/gn1_rev6_tmp")
    r = subprocess.run([sys.executable, "-B", str(MA8 / "regenerate_arms.py"), "--verify-seal"], cwd=MA8, env=env,
                       capture_output=True, text=True, timeout=900)
    result = dict(mirror=str(MROOT), inputs=copied, exit=r.returncode,
                  stdout=r.stdout[-2000:], stderr=r.stderr[-3000:])
    if r.returncode == 0:
        regen_arms = {a.name: tree_map(a) for a in sorted((MA8 / "arms").iterdir())}
        disk_arms = {a.name: tree_map(a) for a in sorted((A8 / "arms").iterdir())}
        sealed_arms = {arm: {"tensor_cuda/" + k: v for k, v in files.items()} for arm, files in seal["arms"].items()}
        regen_maps = json.loads((MA8 / "arm_maps.json").read_text())
        result["arm_count"] = dict(regenerated=len(regen_arms), on_disk=len(disk_arms), sealed=len(sealed_arms))
        result["regenerated_equals_on_disk"] = regen_arms == disk_arms
        result["regenerated_equals_seal"] = regen_arms == sealed_arms
        result["regen_arm_maps_json_equals_a8"] = sha(MA8 / "arm_maps.json") == sha(A8 / "arm_maps.json")
        result["regen_arm_maps_equals_seal"] = regen_maps == seal["arms"]
        result["mutants_json_equals_a8"] = sha(MA8 / "mutants.json") == sha(A8 / "mutants.json")
        result["mutants_json_equals_seal"] = sha(MA8 / "mutants.json") == seal["tool_files"]["mutants.json"]
        a8m = {k: v for k, v in tree_map(A8 / "mutants").items() if k.endswith(("nn.py", "nn.diff"))}
        result["mutant_sources_equal_a8"] = tree_map(MA8 / "mutants") == a8m
        diffs = {}
        for arm in sorted(set(regen_arms) | set(disk_arms)):
            a, b = regen_arms.get(arm, {}), disk_arms.get(arm, {})
            d = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            if d:
                diffs[arm] = d
        result["per_arm_differences"] = diffs
        result["symlinks_regenerated"] = sum(v == "SYMLINK" for m in regen_arms.values() for v in m.values())
        result["symlinks_on_disk"] = sum(v == "SYMLINK" for m in disk_arms.values() for v in m.values())
        a3_commit_nn = subprocess.check_output(["git", "-C", str(REAL), "show", "cf922891:tensor_cuda/tensor_cuda/nn.py"])
        result["candidate_nn_equals_commit_cf922891"] = (
            hashlib.sha256(a3_commit_nn).hexdigest() == sha(MA8 / "arms/candidate/tensor_cuda/nn.py"))
        result["engine_copies_equal_pin"] = all(
            m["tensor_cuda/" + Path(pin["module"]).name] == pin["sha256"] for m in regen_arms.values())
        (OUT / "regen_maps_regenerated.json").write_text(json.dumps(regen_arms, indent=1, sort_keys=True) + "\n")
        (OUT / "regen_maps_a8_disk.json").write_text(json.dumps(disk_arms, indent=1, sort_keys=True) + "\n")
        proof = {}
        for f, digest in seal["tool_files"].items():
            if not (MA8 / f).exists():
                shutil.copyfile(A8 / f, MA8 / f)
            proof["a8/" + f] = sha(MA8 / f) == digest
        for group in ("test_files", "import_modules"):
            for f, digest in seal[group].items():
                (MROOT / f).parent.mkdir(parents=True, exist_ok=True)
                if not (MROOT / f).exists():
                    shutil.copyfile(REAL / f, MROOT / f)
                proof[f] = sha(MROOT / f) == digest
        for rel in ("orders/GN1_A8_CONTRACT.md", "artifacts/gn1/engine_sources_after.json"):
            (MROOT / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REAL / rel, MROOT / rel)
            proof[rel] = sha(MROOT / rel) == sha(REAL / rel)
        code = ("import sys; sys.path.insert(0, %r); import evidence; s = evidence.verify_seal(); "
                "[evidence.verify_arm(a, s) for a in s['arms']]; print('seal+arms verified', len(s['arms']))" % str(MA8))
        v = subprocess.run([sys.executable, "-B", "-c", code], cwd=MA8, capture_output=True, text=True, timeout=300)
        result["mirror_all_identical"] = all(proof.values())
        result["mirror_files_checked"] = len(proof)
        result["mirror_mismatches"] = sorted(k for k, ok in proof.items() if not ok)
        result["mirror_verify_seal_exit"] = v.returncode
        result["mirror_verify_out"] = v.stdout.strip()
        result["mirror_verify_err"] = v.stderr[-1500:]
    (OUT / "regen_compare.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("inputs",)}, indent=1))
    return 0 if r.returncode == 0 and result.get("mirror_verify_seal_exit") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
