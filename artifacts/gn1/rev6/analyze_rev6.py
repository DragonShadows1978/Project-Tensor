#!/usr/bin/env python3
"""GN1-REV6 analysis (reads receipts only; runs nothing):
1. REV5 re-run table: each hostile_rev6_{q,c,n,p} and probe case, REV5's A5 outcome,
   A8's line/root verdict and whether the REV5-target rule itself fired.
2. D4 check: the 32 class (iii) targets with the reviewer's reading of the control
   code that produced them, and the class (i)/(ii) refusal census from A8's G4
   receipts (including roots whose only refusal was the device rule).
Prior art: none beyond receipt cross-tabulation; no prior art known to me.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
GN1 = OUT.parent
A8 = GN1 / "a8"


def load(p):
    return json.loads(Path(p).read_text())


# ---- 1. REV5 re-run table
TARGET_RULE = {
    "Q2_wrapped_runner_command0": ("R5-F01", "runner command[0] differs from pinned runner"),
    "Q2_path_shadow_inner_timeout": ("R5-F02", "program pin mismatch: timeout"),
    "NEW_parent_result_deep_nesting": ("R5-F03", "maximum recursion depth"),
    "NEW_start_json_deep_nesting": ("R5-F03", "maximum recursion depth"),
    "NEW_regression_deep_child_result": ("R5-F03", "maximum recursion depth"),
}
rev5 = {}
for part in "qncp":
    for r in load(GN1 / "rev5" / ("hostile_rev5_%s.json" % part))["results"]:
        rev5[r["case"]] = r["outcome_as_expected"]
table = []
for part in "qcnp":
    for r in load(OUT / ("hostile_rev6_%s.json" % part))["results"]:
        sc = r.get("scorer") or {}
        line = sc.get("line") or {}
        reasons = line.get("reasons") or []
        txt = json.dumps(r)
        f, rule = TARGET_RULE.get(r["case"], (None, None))
        table.append(dict(part=part, case=r["case"], rev5_as_expected_under_a5=rev5.get(r["case"]),
                          rev6_as_expected_by_rev5_criterion=r["outcome_as_expected"],
                          a8_line_passed=line.get("passed"), a8_line_mode=line.get("mode"), a8_root_errors=sc.get("errors"),
                          a8_line_reasons=reasons, rev5_finding=f, target_rule=rule,
                          target_rule_fired=(rule in txt) if rule else None))
for probe in ("rawexit", "postexit"):
    table.append(dict(part="probe", case="rev6_probe_" + probe, result=load(OUT / ("rev6_probe_%s.json" % probe))))

# ---- 2. D4 check
d4 = load(A8 / "D4_classification.json")
iii_reading = {
    "Stored_parent_tamper": "a6/hostile_scorer.py:150-152 edits parent_result.json actual_killers after a completed run",
    "Expected_exit_tamper": "a6/hostile_scorer.py:155-157 edits parent_result.json expected_child_exits after the run",
    "START_seal_tamper": "a6/hostile_scorer.py:158-160 edits START.json seal_sha256 after the run",
    "Stored_timeout_cause_tamper": "a6/extra_controls.py: edits stored timeout_cause in a completed parent receipt",
    "Import_helper_pin_parent": "a6/extra_controls.py:38-53 edits staging_seal.json import_modules digest (a sealed INPUT, not a run receipt); live analogue run by REV6 (rev6_cases_i.json)",
    "Import_helper_pin_child": "same edit as Import_helper_pin_parent, observed at the child",
    "Import_helper_pin_scorer": "same edit as Import_helper_pin_parent, observed at the scorer",
    "D7_outer": "a6/a4_controls.py:232-252 edits outer_bound_seconds in a completed parent_result.json",
    "D7_maximum": "a6/a4_controls.py:232-252 edits --max value in the recorded command",
    "D7_minimum": "a6/a4_controls.py:232-252 edits --min value in the recorded command",
    "D7_inner": "a6/a4_controls.py:232-252 edits inner duration in the recorded command",
    "D7_expected_exits": "a6/a4_controls.py:232-252 edits expected_child_exits in the receipt",
    "raw_exit_minus9_exit0": "a6/a6_controls.py:128-: edits raw_exit in a completed receipt (REV5 F04 repro was itself an edited receipt)",
    "raw_exit_removed": "a6/a6_controls.py:128-: removes raw_exit",
    "raw_exit_bool": "a6/a6_controls.py:128-: raw_exit True",
    "raw_exit_string": "a6/a6_controls.py:128-: raw_exit '0'",
    "D0_control_receipt_promotion": "a6/D0_MODE_PROMOTION.json: mode-field edits after the run (closed by the lead's manifest, LEAD_RULING_A7)",
}
for k in ("runner", "timeout", "interpreter", "runner_interpreter", "parent_interpreter", "outer_timeout", "shebang_env"):
    iii_reading["Recorded_pin_" + k] = "a6/a6_controls.py:203 production_operand edits the recorded program pin in parent_result.json; live analogues: Q2_wrapped_runner_command0, D0c_* (class i) and REV6 F01/F02 cases"
for k in ("PATH", "FORGE_GPU_SUMMARY", "FORGE_GPU_WINDOW_FILE", "FORGE_GPU_LOCK", "FORGE_GPU_KILL_GRACE", "FORGE_GPU_POLL",
          "FORGE_GPU_RUN_LOG", "FORGE_GPU_UNKNOWN"):
    iii_reading["Recorded_env_" + k] = "a6/a6_controls.py:208 production_operand edits the recorded environment in parent_result.json; live analogues: Actual_env_* (class i, control root) and REV6 Q2_live_gpu_hidden_* / F02_new_live_* (production)"
iii = [c for c in d4["controls"] if c["classification"] == "iii"]
iii_rows = [dict(source=c["source"], control=c["control"], seal_or_receipt=("staging seal (pre-run input)" if c["control"].startswith("Import_helper_pin") else "run receipt"),
                 reviewer_reading=iii_reading.get(c["control"], "UNREAD")) for c in iii]
g4 = load(A8 / "G4_summary.json")
run_rows = [c for c in g4["controls"] if c["classification"] in ("i", "ii")]
roots = load(A8 / "G4_root_refusals.json")["controls"]
device_only = [r for r in roots if r["errors"] == ["not a device run: every line requires positive child device evidence"]]
d4_res = dict(
    a6_targets=len(d4["controls"]), counts=d4["a6_counts"], iii=len(iii), iii_rows=iii_rows,
    iii_receipt_edits=sum(r["seal_or_receipt"] == "run receipt" for r in iii_rows),
    iii_seal_edits=[r["control"] for r in iii_rows if r["seal_or_receipt"] != "run receipt"],
    iii_unread=[r["control"] for r in iii_rows if r["reviewer_reading"] == "UNREAD"],
    i_ii_in_g4=len(run_rows), i_ii_all_passed=all(c["passed"] is True for c in run_rows),
    i_ii_not_run=[c["control"] for c in run_rows if c["execution"] != "CPU replay"],
    a7_rows=len(d4["a7_added_controls"]), a7_counts=d4["a7_counts"],
    a7_class_i=[c for c in d4["a7_added_controls"] if c["classification"] == "i"],
    root_refusals=len(roots), roots_accepted=sum(bool(r["root_accepted"]) for r in roots),
    roots_refused_only_by_device_rule=len(device_only),
    roots_device_only_with_all_line_checks_passing=[r["root"] for r in device_only if r.get("line_checks_passed") and r.get("line_checks_passed") == r.get("lines", r.get("line_checks_passed"))][:400],
)
res = dict(rev5_rerun=table, d4_check=d4_res)
(OUT / "analysis.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
for row in table:
    if row["part"] != "probe":
        print(row["part"], row["case"], "| A5", row["rev5_as_expected_under_a5"], "| A8 line", row["a8_line_passed"],
              row["a8_line_mode"], "| target", row["rev5_finding"], row["target_rule_fired"])
print(json.dumps({k: v for k, v in d4_res.items() if k not in ("iii_rows", "roots_device_only_with_all_line_checks_passing")}, indent=1))
