"""ENG5-S CPU drive of the G2 child code (L48), CUDA hidden, injection shim preloaded, before any lease:

  python3 -B -P scripts/eng5s_gpu_cpudrive.py <out json>

(a) wiring: every G2 case's child (scripts/eng5s_gpu_step.py G2_CHILD) runs on Builds B and S with each "cuda" device
    string replaced by "cpu" and tc.synchronize() by `None`, so the case code, the attempt/arm/fired bookkeeping, the
    value checks and the JSON row run end to end (a slip in a case shows up here, not in the lead's lease). Site 9's
    setup needs a CUDA key store, so on this lane its child reports the refusal; that is recorded, not a wiring error.
(b) site 2 for real: the two OOM-drain cases run on the CPU lane with "cuda" kept (only tc.synchronize() becomes the no-op expression
    `None`, as there is no device to synchronize) (no GPU is needed to reach the allocator's
    retry: the shim makes the allocation return cudaErrorMemoryAllocation and the drain fail). Build S must raise in
    the S1 form; Build B continues to the retried allocation.
The verdict code is then applied to (a)'s rows (wiring of judge_line; not a GPU verdict). The lease-check lines (the
children's first statements) are left out here only: this drive never runs inside a lease and touches no GPU.

Prior art: ENG5-F2's scripts/eng5_f2_gpu_cpudrive.py (this repo), whose stripping approach this copies. No other prior
art known to me.
"""
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def strip_lease(code):
    return "\n".join(l for l in code.split("\n") if "forge-gpu-lease-check" not in l)


def hostify(code):
    return (strip_lease(code).replace('"cuda"', '"cpu"').replace("'cuda'", "'cpu'")
            .replace("tc.synchronize()", "None"))


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU drive runs with CUDA hidden (L58)"
    src = open(os.path.join(ROOT, "scripts", "eng5s_gpu_step.py")).read().split("\n")
    assert "forge-gpu-lease-check" in src[1]
    step = types.ModuleType("eng5s_gpu_step_cpudrive")
    step.__file__ = os.path.join(ROOT, "scripts", "eng5s_gpu_step.py")
    exec(compile("\n".join([src[0], "pass"] + src[2:]), step.__file__, "exec"), step.__dict__)
    import eng5s_gpu_verdict as verdict
    inj = step.build_shim()
    os.makedirs(step.TMP, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=step.TMP, LD_PRELOAD=inj)
    out = {"note": "CPU drive: wiring check (a) and site 2 on the CPU lane (b); not a GPU verdict", "wiring": {},
           "site2_cpu_lane": {}}
    for short in ("B", "S"):
        rows = step.g2_rows(short, env, child_code=hostify(step.G2_CHILD))
        out["wiring"][short] = {
            "rows": rows,
            "child_failed": [[r["case"], r.get("stderr", "")[-300:]] for r in rows if r.get("outcome") == "child_failed"],
            "verdict_wiring": verdict.judge_g3(rows, short)["verdict"]}
        for case in ("oom_drain", "oom_drain_pooled"):
            code = strip_lease(step.G2_CHILD).replace("tc.synchronize()", "None")  # no device to synchronize here
            rec, info = step.child([code, step.pkg(step.ARMS[short]), case], env, "G2ROW", 60)
            out["site2_cpu_lane"][f"{short}:{case}"] = rec if rec else {"outcome": "child_failed", **info}
    s = out["site2_cpu_lane"]
    checks = {}
    for case in ("oom_drain", "oom_drain_pooled"):
        si, bi = (s.get(f"S:{case}") or {}).get("injected") or {}, (s.get(f"B:{case}") or {}).get("injected") or {}
        checks[f"S:{case} raises in the S1 form"] = (verdict.fired_all(si) and si.get("outcome") == "raised" and
                                                     (si.get("message") or "").startswith(verdict.PREFIX[case]))
        checks[f"B:{case} continues to the retried allocation"] = (verdict.fired_all(bi) and
                                                                   (bi.get("message") or "").startswith("cudaMalloc"))
    out["site2_checks"] = checks
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for short in ("B", "S"):
        w = out["wiring"][short]
        print(short, "rows", len(w["rows"]), "child_failed", [c for c, _ in w["child_failed"]],
              "verdict wiring", w["verdict_wiring"])
    print(json.dumps(checks, indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
