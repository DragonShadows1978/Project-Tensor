#!/usr/bin/env python3
"""Recompute every ENG2 verdict from START, seal, parent, child and JUnit receipts; no GPU calls (copied from ENG1's
tool as A3 left it). ENG2 changes (ledgered): ENG1's M09/C_UBSAN rule dropped (ENG2 has no controls); the N1 census
rows are read from candidate_eng2 (n1census_*.json) and baseline_eng2, and test_eng1's E3 rows from the line-(c)
regnorm lines, both reported (not gated); cross-arm identity is gated: every f1bytes_* and p302bits_* row of
candidate_eng2 must equal baseline_eng2's row for the same test ID (P302 and the F1 supported-dtype bytes, compared as
sha256 of the uint views the test wrote), with both sets non-empty and equal in IDs.
(ENG1 text follows.) Recompute every ENG1 verdict from START, seal, parent, child and JUnit receipts; no GPU calls (adapted from
GN1 A8 score_gpu.py). Prior art: DeMillo et al. (1978), mutation adequacy; paired regression (GN1 D6).
ENG1 changes (ledgered): M09 counts only with a passing C_UBSAN control; the E3 census rows are collected from
candidate_eng1 and reported (not gated); the per-mutant killer IDs are listed.
"""
import json
import os
import sys
from pathlib import Path
from evidence import TOOL, guard, verify_start, lines, recompute_line, summary_outcomes, norm_id


# ENG2-A1 (REV1 E2R-F04): the registered row census, per kind: the test functions that write the rows (test_eng2.py
# _record calls) and the order's expected row counts (orders/ENG2_A1_REV1_FIXES.md, operand census: 35 and 20).
CROSS_ROWS = {"f1bytes": (("test_f1_tensor_supported", "test_f1_factory_supported", "test_f1_default_float64_data",
                           "test_f1_factory_default"), 35),
              "p302bits": (("test_n1_float32_bits",), 20)}


def keyed_rows(kind, rows):
    """{norm_id(test_id): sha256} for one arm's rows of `kind`, with census errors: a row of another kind, a duplicate
    normalised ID, or an ID set different from the registered census (ids_eng2.txt filtered by CROSS_ROWS[kind]).
    Prior art: keyed equi-join on a normalised key with a uniqueness check (relational practice; no specific source
    known to me)."""
    functions, count = CROSS_ROWS[kind]
    expected = {i for i in (TOOL / "ids_eng2.txt").read_text().split()
                if i.split("::", 1)[1].split("[", 1)[0] in functions}
    keyed, errors = {}, []
    if len(expected) != count:
        errors.append("%s: registered census has %d IDs, the order registers %d" % (kind, len(expected), count))
    for r in rows:
        nid = norm_id(r["test_id"])
        if r.get("kind") != kind:
            errors.append("%s: row of kind %r (%s)" % (kind, r.get("kind"), nid))
        if nid in keyed:
            errors.append("%s: duplicate normalised ID %s" % (kind, nid))
        keyed[nid] = r["sha256"]
    if rows and set(keyed) != expected:
        errors.append("%s: row IDs differ from the census: missing %r, extra %r" % (
            kind, sorted(expected - set(keyed))[:5], sorted(set(keyed) - expected)[:5]))
    return keyed, errors


def main():
    runs = guard(os.environ["RUNS"])
    summary, errors = {}, []
    start = {}
    try:
        seal, start = verify_start(runs)
        for label in lines():
            try:
                summary[label] = recompute_line(runs, label, seal, start)
            except Exception as exc:
                summary[label] = dict(passed=False, reasons=[str(exc) or type(exc).__name__])
    except Exception as exc:
        errors.append(str(exc) or type(exc).__name__)
    if start.get("audit"):
        errors.append("audit run cannot establish GPU acceptance")
    if summary and any(row.get("mode") != "device" for row in summary.values()):
        errors.append("not a device run: every line requires positive child device evidence")
    def rows(label, pattern):
        attempt = summary.get(label, {}).get("selected_attempt")
        return [json.loads(p.read_text()) for p in sorted(Path(attempt).glob(pattern))] if attempt else []
    census = dict(candidate_eng2=rows("candidate_eng2", "n1census_*.json"),
                  baseline_eng2=rows("baseline_eng2", "n1census_*.json"),
                  e3_baseline_regnorm=rows("baseline_regnorm", "census_*.json"),
                  e3_candidate_regnorm=rows("candidate_regnorm", "census_*.json"))
    cross = {}
    for kind in ("f1bytes", "p302bits"):
        # ENG2-A1 (REV1 E2R-F04): rows are keyed by evidence.norm_id on BOTH arms (test_eng2.py's _record stores
        # request.node.nodeid, which carries artifacts/eng2/tool/arms/<arm>/), duplicates refused, and each arm's ID set
        # must equal the registered census exactly (35 f1bytes, 20 p302bits rows). Hashes are compared unchanged.
        base, base_errors = keyed_rows(kind, rows("baseline_eng2", kind + "_*.json"))
        cand_rows, cand_errors = keyed_rows(kind, rows("candidate_eng2", kind + "_*.json"))
        differ = sorted(i for i in base if cand_rows.get(i) != base[i])
        cross[kind] = dict(baseline=len(base), candidate=len(cand_rows), differing=differ,
                           ids_equal=sorted(base) == sorted(cand_rows), census_errors=base_errors + cand_errors)
        if summary and (not base or sorted(base) != sorted(cand_rows) or differ or base_errors or cand_errors):
            errors.append("cross-arm %s: %d baseline rows, %d candidate rows, IDs equal %s, differing %r, census %r" % (
                kind, len(base), len(cand_rows), sorted(base) == sorted(cand_rows), differ[:10],
                (base_errors + cand_errors)[:4]))
    source_outcomes = {}
    for label in ("candidate_regnorm", "candidate_regdtype"):  # ENG2: two regression suites (was candidate_regression)
        regression = summary.get(label, {}).get("selected_attempt")
        if regression:
            selectors = {r["test_id"] for r in json.loads((TOOL / "source_reading_tests.json").read_text())}
            source_outcomes.update({norm_id(k): v for k, v in summary_outcomes(json.loads(
                (Path(regression) / "child_result.json").read_text())).items()
                               if norm_id(k).split("[", 1)[0] in selectors})
    passed = not errors and bool(summary) and all(row.get("passed") for row in summary.values())  # after cross
    result = dict(passed=passed, errors=errors, lines=summary, cross_arm=cross,
                  killers={k: v.get("actual_killers") for k, v in summary.items() if lines()[k]["role"] == "mutant"},
                  n1_census=census,
                  source_reading_tests=json.loads((TOOL / "source_reading_tests.json").read_text()),
                  source_reader_outcomes=source_outcomes)
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "G5_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=passed, errors=errors,
                          lines={k: (v.get("passed"), v.get("mode"), v.get("reasons", [])[:3]) for k, v in summary.items()})))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
