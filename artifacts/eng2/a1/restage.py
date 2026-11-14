#!/usr/bin/env python3
"""ENG2-A1 re-staging (CPU only; orders/ENG2_A1_REV1_FIXES.md, A-G3). Drives the sealed generator
artifacts/eng2/tool/stage.py (imported, unchanged) and replaces only the three steps that cannot run as written:

  arms   stage.make_arm refuses an existing arm ("arms are written once") and deleting is forbidden, so each arm is
         REWRITTEN IN PLACE with make_arm's exact recipe (same sources, same mutant edit via stage.apply_mutant, same
         hunk check), then its file set is checked equal to the recipe's (nothing stale can remain).
  lines  stage.step_lines verbatim, except that the baseline arm's CPU outcomes come from this round's
         artifacts/eng2/a1/baseline_cpu.json (artifacts/eng2/g2/ is read-only and predates the new CPU IDs).
  seal   stage.step_seal verbatim, except mode "w" (the file exists) and the registration list: a1/baseline_cpu.json
         replaces g2/baseline_cpu.json, and orders/ENG2_A1_REV1_FIXES.md is added.
census and script are stage.step_census / stage.step_script, called unchanged.
Usage: CUDA_VISIBLE_DEVICES= TMPDIR=/tmp/eng2_a1_stage restage.py STEP [STEP ...]   (arms | census | lines | seal | script)
Prior art: ENG1/ENG2 stage.py (2026), whose steps this reuses; SHA-256 pins (NIST FIPS 180-4). No new algorithm.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

A1 = Path(__file__).resolve().parent
ENG2 = A1.parent
TOOL = ENG2 / "tool"
ROOT = ENG2.parents[1]
sys.path.insert(0, str(TOOL))
import stage  # noqa: E402  (the sealed generator, unchanged)

A1_ORDER = "orders/ENG2_A1_REV1_FIXES.md"
BASELINE_CPU = A1 / "baseline_cpu.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rewrite_arm(arm, package_src, module_src, edit=None):
    dest = TOOL / "arms" / arm / "tensor_cuda"
    assert dest.is_dir(), "re-staging rewrites existing arms only: " + arm
    pkg, tests = dest / "tensor_cuda", dest / "tests"
    expected = set()
    for src in sorted(Path(package_src).rglob("*.py")):
        if "__pycache__" in src.parts:
            continue
        target = pkg / src.relative_to(package_src)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(src.read_bytes())
        expected.add(str(target.relative_to(dest.parent)))
    if not (pkg / stage.MODULE).exists() or sha(pkg / stage.MODULE) != sha(module_src):
        shutil.copyfile(module_src, pkg / stage.MODULE)
    os.chmod(pkg / stage.MODULE, 0o755)
    expected.add(str((pkg / stage.MODULE).relative_to(dest.parent)))
    for name in ("test_eng2.py", "test_eng1.py"):
        shutil.copyfile(ROOT / "tensor_cuda/tests" / name, tests / name)
        expected.add(str((tests / name).relative_to(dest.parent)))
    if edit is not None:
        path = dest.parent / edit["file"]
        mutated, hunk = stage.apply_mutant(path.read_text(), edit)
        assert hunk == edit["hunk_lines"], (arm, hunk)
        path.write_text(mutated)
    from evidence import arm_files
    files = set(arm_files(arm))
    assert files == expected, (arm, sorted(files ^ expected))
    print("rewrote", arm, len(files), "files; module", sha(pkg / stage.MODULE)[:12])


def step_arms():
    before = json.loads((ENG2 / "worktree_before.json").read_text())["files"]
    base_pkg = ENG2 / "base_pkg" / "tensor_cuda"
    for src in base_pkg.rglob("*.py"):  # the baseline package is bc02d3a1's (ENG2's pre-edit map)
        assert sha(src) == before["tensor_cuda/tensor_cuda/" + str(src.relative_to(base_pkg))], src
    assert sha(ROOT / "tensor_cuda/tests/test_eng1.py") == before["tensor_cuda/tests/test_eng1.py"]
    rewrite_arm("baseline", base_pkg, stage.BASE_MODULE)
    rewrite_arm("candidate", ROOT / "tensor_cuda/tensor_cuda", stage.CAND_MODULE)
    for m in stage.registry():
        rewrite_arm(m["id"], ROOT / "tensor_cuda/tensor_cuda", stage.CAND_MODULE, edit=m)


def step_lines():
    """stage.step_lines, verbatim but for the baseline CPU receipt's path (BASELINE_CPU)."""
    assert json.loads((TOOL / "regression_files.json").read_text()) == stage.regression_files()
    (TOOL / "source_reading_tests.json").write_text("[]\n")
    rows = [dict(label="baseline_eng2", arm="baseline", suite="eng2", role="baseline", census="eng2",
                 expected_child_exits=[1], expected_parent_exit=0, **stage.bounds(stage.ENG2_INNER)),
            dict(label="candidate_eng2", arm="candidate", suite="eng2", role="candidate", census="eng2",
                 expected_child_exits=[0], expected_parent_exit=0, **stage.bounds(stage.ENG2_INNER))]
    for suite, inner in (("regnorm", stage.REGNORM_INNER), ("regdtype", stage.REGDTYPE_INNER)):
        rows.append(dict(label="baseline_" + suite, arm="baseline", suite=suite, role="regression", census=suite,
                         expected_child_exits=[0, 1], expected_parent_exit=0, **stage.bounds(inner)))
        rows.append(dict(label="candidate_" + suite, arm="candidate", suite=suite, role="regression",
                         census=suite + "_candidate", expected_child_exits=[0, 1], expected_parent_exit=0,
                         **stage.bounds(inner)))
    for m in stage.registry():
        rows.append(dict(label=m["id"], arm=m["id"], suite="mutation", role="mutant", census="eng2",
                         expected_child_exits=[1], expected_parent_exit=0, **stage.bounds(stage.ENG2_INNER)))
    (TOOL / "gpu_lines.json").write_text(json.dumps(rows, indent=2) + "\n")
    ids = (TOOL / "ids_eng2.txt").read_text().split()
    cpu = json.loads(BASELINE_CPU.read_text())
    emu = set(json.loads((ENG2 / "N1_EMULATION.json").read_text())["predicted_failures"]["baseline"])
    fail = []
    for i in ids:
        name = i.split("::", 1)[1]
        if "cpu" in name.split("[")[0]:
            assert i in cpu["outcomes"], i
            if cpu["outcomes"][i] != "passed":
                fail.append(i)
        elif i in emu:
            fail.append(i)
    (TOOL / "baseline_expected_failures.json").write_text(json.dumps(sorted(fail), indent=1) + "\n")
    print(len(rows), "lines;", len(fail), "registered baseline failures of", len(ids))


def step_seal():
    """stage.step_seal, verbatim but for mode "w" and the registration list (module docstring)."""
    arms = sorted(p.name for p in (TOOL / "arms").iterdir())
    from evidence import arm_files
    reg = json.loads((TOOL / "regression_files.json").read_text())
    seal = dict(order=stage.ORDER, order_commit=stage.ORDER_COMMIT, order_sha256=sha(ROOT / stage.ORDER),
                base_commit=stage.BASE_COMMIT, module_name=stage.MODULE,
                tool_files={p.name: sha(p) for p in sorted(TOOL.iterdir())
                            if p.is_file() and not p.name.startswith(("staging_seal", "collect_"))},
                registration_files={str(p.relative_to(ROOT)): sha(p) for p in
                                    [ROOT / "tensor_cuda/tests/test_eng2.py", ROOT / "tensor_cuda/tests/test_eng1.py",
                                     ENG2 / "mutants/mutants.json", ENG2 / "N1_EMULATION.json",
                                     BASELINE_CPU, ENG2 / "L38_PINS.json", ROOT / A1_ORDER] +
                                    [ROOT / f for suite in sorted(reg) for f in reg[suite]]},
                arms={a: arm_files(a) for a in arms},
                engine_sha256={a: sha(TOOL / "arms" / a / "tensor_cuda/tensor_cuda" / stage.MODULE) for a in arms})
    for a in arms:
        for name in ("test_eng2.py", "test_eng1.py"):
            assert sha(TOOL / "arms" / a / "tensor_cuda/tests" / name) == sha(ROOT / "tensor_cuda/tests" / name), (a, name)
    labels = {r["arm"] for r in json.loads((TOOL / "gpu_lines.json").read_text())}
    missing = sorted(labels - set(arms))
    assert not missing, missing
    seal["arms_pending"] = missing
    (TOOL / stage.SEAL_NAME).write_text(json.dumps(seal, indent=1, sort_keys=True) + "\n")
    print("sealed arms:", arms, "seal", sha(TOOL / stage.SEAL_NAME))


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert Path(os.environ["TMPDIR"]).name.startswith("eng2_a1_")
    for step in sys.argv[1:]:
        dict(arms=step_arms, census=stage.step_census, lines=step_lines, seal=step_seal,
             script=stage.step_script)[step]()
