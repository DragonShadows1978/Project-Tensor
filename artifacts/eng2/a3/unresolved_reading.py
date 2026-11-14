#!/usr/bin/env python3
"""ENG2-A3 R2: attach the seat's source-reading note to every unresolved site on an entry ENG2 changes
(a3/CONSUMER_CENSUS.json unresolved_changed). Evidence class: reasoning (reading the consumer source, cross-file where
needed), NOT a mechanical trace; the lead decides per site. Writes a3/UNRESOLVED_CHANGED_READING.json.
Prior art: none (a lookup table of reading notes); no prior art known to me."""
import json
from pathlib import Path

A3 = Path(__file__).resolve().parent
d = json.loads((A3 / "CONSUMER_CENSUS.json").read_text())
NOTES = {
    "causal": "the argument is <t>.dtype of a query/scores tensor on a tensor_cuda attention path (an SDPA argument, or "
              "the result of tc.matmul / tensor arithmetic); Tensor.dtype returns one of the six supported names "
              "(tensor_cuda/src/bindings.cpp:1806 via kernels.cu dtype_name), so ENG2 would not refuse it",
    "voice": "t1_empty's dtype comes from engine/tensor_requests/p1t_tests/run_p1t.py (:605 `for dt in ('float32', "
             "'float16')`; literals 'float16'/'float32' at :510/:545/:553/:568) and r4_cases.py:119 ('float32'); "
             "adapter_api.py:56 states 'float32' or 'float16': supported",
    "trinity": "_compute_dtype() returns BlockTC.COMPUTE_DTYPE, assigned 'bfloat16' in the same file (:165, :951); the "
               "tracer does not follow function returns: supported",
    "compute": "the value is BlockTC.COMPUTE_DTYPE (core/mistral7b_tc.py:558 class default 'float16'; model loaders set "
               "'bfloat16', e.g. core/gpt_oss20b_tc.py:1342, core/deepseek_v2_lite_tc.py:829, core/gemma4_tc.py:1086, "
               "core/minicpm3_tc.py:437); every literal COMPUTE_DTYPE assignment in GraftRepository is 'bfloat16', "
               "'float16' or 'float32'. Dynamic setters exist only in scripts/glc_p0_audit.py:161/166 (its dtype "
               "parameter, under a CPU stand-in), scripts/trinity_nope_graft_width_sweep.py:573/575 and "
               "scripts/trinity_t1_floor_reverb.py:619/623 (str compute_dtype, default 'float32'): supported unless a "
               "caller of those scripts passes something else; not proven",
}


def note(r):
    if r["entry"] == "functional._causal_mask":
        return "causal"
    if r["repo"] == "Project-Voice":
        return "voice"
    if "trinity_nano_tc" in r["file"]:
        return "trinity"
    return "compute"


sites = [dict(site="%s %s:%d" % (r["repo"], r["file"], r["line"]), role=r["role"], live=r["live"], group=r["group"],
              entry=r["entry"], arg=r["arg"], tracer=r["values"], reading=note(r), reading_says_refused=False)
         for r in d["unresolved_changed"]]
out = dict(evidence_class="reasoning (source reading by the seat), not a mechanical trace", notes=NOTES, sites=sites,
           counts={k: sum(s["reading"] == k for s in sites) for k in NOTES}, total=len(sites))
(A3 / "UNRESOLVED_CHANGED_READING.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out["counts"]), out["total"])
