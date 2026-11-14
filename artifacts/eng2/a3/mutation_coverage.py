#!/usr/bin/env python3
"""ENG2-A3 A3-G1 mutation coverage (orders/ENG2_A3_ENG1_ROWS_AND_CONSUMERS.md R1, L30, L03): per mutant, not a share.

For each of ENG1's 16 mutants (artifacts/eng1/mutants/mutants.json + artifacts/eng1/a1/MUTANTS.json) and ENG2's 15
(artifacts/eng2/mutants/mutants.json):
  1. the registered killer patterns that match any REMOVED ID (the 10 test_eng1.py float64 IDs: a3/ids_test_eng1_diff,
     equal to the pre-A3 L38_PINS.json expected_changes keys), each side matched by ITS OWN tool's matcher, unchanged
     (ENG1: artifacts/eng1/tool/evidence.matches; ENG2: artifacts/eng2/tool/evidence.matches);
  2. the remaining killers: IDs the mutant's G5 line collected, minus the removed IDs, that match a registered pattern;
  3. from the recorded G5 receipts (ENG1: the lead's run root artifacts/eng1/tool/gpu_runs/g5_1114 in the eng1 worktree,
     where the run wrote it -- this worktree holds only its git-tracked subset; ENG2: this worktree's
     artifacts/eng2/tool/gpu_runs/g5_1114), the admitted attempt's child_result.json, junit.xml and parent_result.json,
     each sha256 confirmed against the lead's manifest (artifacts/eng{1,2}/lead/g5_1114/eng{1,2}_g5_MANIFEST.sha256,
     whose own sha256 is quoted from the run log's post_run_manifest line); per-ID outcomes by the tool's own
     summary_outcomes() and junit_outcomes(), which must agree ID by ID (as evaluate() requires);
  4. verdict: at least one remaining killer FAILED on the mutant's G5 line. A mutant whose only killers are removed
     IDs is a RED (order R1).
Writes a3/MUTATION_COVERAGE.json and prints the table.
Prior art: DeMillo, Lipton and Sayward (1978), a mutant is killed by a failing test; the per-mutant "killing test set"
view as in mutation-testing tools (e.g. PIT's per-mutant killing tests; unverified -- lead to check "PIT mutation
killingTests"); GNU sha256sum -c for the manifest check. Ours: the removed-ID intersection and the receipt binding.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys

A3 = Path(__file__).resolve().parent
ENG2 = A3.parent
ROOT = ENG2.parents[1]
ENG1 = ROOT / "artifacts/eng1"
ENG1_WT = ROOT.parent / "tensor_fork_apa_wt_eng1"
RUN_ROOTS = {"eng1": ENG1_WT / "artifacts/eng1/tool/gpu_runs/g5_1114", "eng2": ENG2 / "tool/gpu_runs/g5_1114"}
MANIFESTS = {"eng1": ENG1 / "lead/g5_1114/eng1_g5_MANIFEST.sha256", "eng2": ENG2 / "lead/g5_1114/eng2_g5_MANIFEST.sha256"}
RUN_LOGS = {"eng1": ENG1_WT / "artifacts/eng1/lead/g5_1114/eng1_g5.log", "eng2": ENG2 / "lead/g5_1114/eng2_g5.log"}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def load_evidence(tool_dir, name):
    """Import a tool's evidence.py (and its sibling imports) under its own sys.path, then forget the sibling modules so
    the other tool's same-named modules load from their own directory."""
    sibling = ("evidence", "device_evidence", "e6_record", "arm_origin", "runtime_evidence")
    for m in sibling:
        sys.modules.pop(m, None)
    sys.path.insert(0, str(tool_dir))
    mod = importlib.import_module("evidence")
    sys.path.remove(str(tool_dir))
    for m in sibling:
        sys.modules.pop(m, None)
    sys.modules[name] = mod
    return mod


ev1 = load_evidence(ENG1 / "tool", "eng1_evidence")
ev2 = load_evidence(ENG2 / "tool", "eng2_evidence")
EV = {"eng1": ev1, "eng2": ev2}

before = (A3 / "ids_test_eng1_before_candidate.txt").read_text().split()
after = (A3 / "ids_test_eng1_after_candidate.txt").read_text().split()
removed = sorted(set(before) - set(after))
pins = sorted(read(A3 / "before/artifacts__eng2__L38_PINS.json")["expected_changes"])
assert removed == pins and len(removed) == 10 and not set(after) - set(before), "removed IDs != the 10 L38 pins"

mutants = {"eng1": read(ENG1 / "mutants/mutants.json") + read(ENG1 / "a1/MUTANTS.json"),
           "eng2": read(ENG2 / "mutants/mutants.json")}
assert len(mutants["eng1"]) == 16 and len(mutants["eng2"]) == 15

manifest_rows, manifest_sha = {}, {}
for side, path in MANIFESTS.items():
    rows = {}
    for line in path.read_text().splitlines():
        h, rel = line.split("  ", 1)
        rows[rel[2:] if rel.startswith("./") else rel] = h
    manifest_rows[side] = rows
    logged = [l for l in RUN_LOGS[side].read_text().splitlines() if "post_run_manifest:" in l and "manifest sha256" in l]
    manifest_sha[side] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), files=len(rows),
                              run_log=str(RUN_LOGS[side]), run_log_quote=logged[-1] if logged else None,
                              matches_run_log=bool(logged) and logged[-1].endswith(sha(path)))
    # the eng1 manifest copy in the eng1 worktree is the same file
    if side == "eng1":
        twin = ENG1_WT / "artifacts/eng1/lead/g5_1114/eng1_g5_MANIFEST.sha256"
        manifest_sha[side]["eng1_worktree_copy_sha256"] = sha(twin)


def confirm(side, rel):
    p = RUN_ROOTS[side] / rel
    want = manifest_rows[side].get(rel)
    got = sha(p)
    return dict(file=rel, manifest_sha256=want, actual_sha256=got, ok=want == got)


summaries = {s: read(RUN_ROOTS[s] / "G5_SUMMARY.json") for s in RUN_ROOTS}
rows, all_ok, red = [], True, []
for side in ("eng1", "eng2"):
    ev = EV[side]
    for m in mutants[side]:
        label, patterns = m["id"], m["killers"]
        hit_removed = sorted(p for p in patterns if any(ev.matches(r, [p]) for r in removed))
        attempts = sorted((RUN_ROOTS[side] / label).glob("attempt-*"))
        admitted = [a for a in attempts if read(a / "parent_result.json").get("admission") == "admitted"]
        assert len(admitted) == 1, (side, label, [a.name for a in attempts])
        att = admitted[0]
        rel = str(att.relative_to(RUN_ROOTS[side]))
        checks = [confirm(side, rel + "/" + n) for n in ("child_result.json", "junit.xml", "parent_result.json")]
        data, parent = read(att / "child_result.json"), read(att / "parent_result.json")
        ids = data["collected_ids"]
        outcomes = ev.summary_outcomes(data)
        dual = {i for i in ids
                if any(r["id"] == i and r["when"] == "call" and r["outcome"] == "failed" for r in data["reports"])
                and any(r["id"] == i and r["when"] == "teardown" and r["outcome"] == "failed" for r in data["reports"])}
        xml, _ = ev.junit_outcomes(att / "junit.xml", ids, dual)
        junit_agrees = xml == outcomes
        norm = {ev.norm_id(i): o for i, o in outcomes.items()}
        collected = set(norm)
        removed_collected = sorted(collected & set(removed))
        remaining_ids = sorted(i for i in collected - set(removed) if ev.matches(i, patterns))
        remaining_patterns = sorted(p for p in patterns if any(ev.matches(i, [p]) for i in remaining_ids))
        failed_remaining = sorted(i for i in remaining_ids if norm[i] in ("failed", "error"))
        failed_removed_killers = sorted(i for i in removed_collected if norm[i] in ("failed", "error")
                                        and ev.matches(i, patterns))
        scored = summaries[side]["lines"][label]
        g5_actual = sorted(ev.norm_id(i) for i in scored.get("actual_killers") or [])
        ok = (bool(failed_remaining) and all(c["ok"] for c in checks) and junit_agrees
              and parent.get("passed") is True and scored.get("passed") is True
              and set(g5_actual) <= set(failed_remaining) | set(failed_removed_killers))
        if not failed_remaining:
            red.append(label)
        all_ok &= ok
        rows.append(dict(side=side, mutant=label, file=m["file"], registered_patterns=len(patterns),
                         patterns_matching_removed=hit_removed, removed_ids_collected_on_line=removed_collected,
                         remaining_patterns=len(remaining_patterns), remaining_killer_ids=len(remaining_ids),
                         g5_attempt=rel, g5_child_exit=data["pytest_exit"], g5_parent_passed=parent.get("passed"),
                         g5_failed_remaining_killers=failed_remaining, g5_scorer_actual_killers=len(g5_actual),
                         g5_scorer_killers_all_remaining=set(g5_actual) <= set(failed_remaining),
                         junit_agrees=junit_agrees, receipts=checks, ok=ok))
# Control (no vacuous "none"): constructed patterns that DO name removed IDs must match under each side's matcher.
controls = {"eng1 'test_e2_pow_grad[float64-*]'": sum(ev1.matches(r, ["test_e2_pow_grad[float64-*]"]) for r in removed),
            "eng1 'test_e2_pow_grad_bits[float64-p+2]'": sum(ev1.matches(r, ["test_e2_pow_grad_bits[float64-p+2]"]) for r in removed),
            "eng2 'tensor_cuda/tests/test_eng1.py::test_e2_pow_grad*[float64-*]'":
                sum(ev2.matches(r, ["tensor_cuda/tests/test_eng1.py::test_e2_pow_grad*[float64-*]"]) for r in removed)}
controls_ok = controls == {"eng1 'test_e2_pow_grad[float64-*]'": 5, "eng1 'test_e2_pow_grad_bits[float64-p+2]'": 1,
                           "eng2 'tensor_cuda/tests/test_eng1.py::test_e2_pow_grad*[float64-*]'": 10}
all_ok &= controls_ok
out = dict(order="orders/ENG2_A3_ENG1_ROWS_AND_CONSUMERS.md", removed_ids=removed, manifests=manifest_sha,
           matcher_controls=dict(counts=controls, ok=controls_ok),
           run_roots={k: str(v) for k, v in RUN_ROOTS.items()},
           g5_summary_sha256={k: sha(RUN_ROOTS[k] / "G5_SUMMARY.json") for k in RUN_ROOTS},
           matchers={"eng1": "artifacts/eng1/tool/evidence.py matches (sha256 %s)" % sha(ENG1 / "tool/evidence.py"),
                     "eng2": "artifacts/eng2/tool/evidence.py matches (sha256 %s)" % sha(ENG2 / "tool/evidence.py")},
           rows=rows, mutants=len(rows), red_only_killer_removed=red,
           all_ok=bool(all_ok and not red and len(rows) == 31
                       and all(v["matches_run_log"] for v in manifest_sha.values())))
(A3 / "MUTATION_COVERAGE.json").write_text(json.dumps(out, indent=1) + "\n")
print("manifests:", json.dumps(manifest_sha, indent=1))
print("| side | mutant | registered | match removed | remaining patterns | G5 failed remaining killers | receipts sha ok | ok |")
for r in rows:
    print("| %s | %s | %d | %s | %d | %d (e.g. %s) | %s | %s |" % (
        r["side"], r["mutant"], r["registered_patterns"], r["patterns_matching_removed"] or "none",
        r["remaining_patterns"], len(r["g5_failed_remaining_killers"]),
        (r["g5_failed_remaining_killers"][:1] or ["-"])[0].split("::")[-1], all(c["ok"] for c in r["receipts"]), r["ok"]))
print("red:", red, "all_ok", out["all_ok"])
sys.exit(0 if out["all_ok"] else 1)
