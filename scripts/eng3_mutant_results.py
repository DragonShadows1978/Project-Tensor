"""ENG3 G4: MUTANT_RESULTS.json from the mutant runs (artifacts/eng3/g3/mut_*), against MUTANTS.json.
For each mutant: the named test's outcome in its named-only run, the marker 'ENG3-MUTANT <id> executed' in that run's
junit (L41: the killing test executed the mutated line with the mutant module loaded), the counter evidence where the
test prints one, and every failing ID of the full runs. Prior art: DeMillo, Lipton and Sayward (1978)."""
import json, re
M = json.load(open("artifacts/eng3/MUTANTS.json"))
G = "artifacts/eng3/g3/"
runs = {"m01_from_host_checks_cuda": ("mut_named_m01_from_host_checks_cuda", ["mut_cpu_m01_from_host_checks_cuda"]),
        "m02_host_astype_truncates": ("mut_gpu_named_m02", ["mut_gpu_m02", "mut_cpu_m02_host_astype_truncates"]),
        "m03_host_clone_cudamemcpy": ("mut_named_m03_host_clone_cudamemcpy", ["mut_cpu_m03_host_clone_cudamemcpy"]),
        "m04_u8_no_clamp": ("mut_gpu_named_m04", ["mut_gpu_m04", "mut_cpu_m04_u8_no_clamp"]),
        "m05_bool_numpy_float": ("mut_named_m05_bool_numpy_float", ["mut_cpu_m05_bool_numpy_float", "mut_gpu_m05_g1b"])}
out = []
for m in M["mutants"]:
    mid = m["id"]
    named_run, full = runs[mid]
    d = json.load(open(G + named_run + "/outcomes.json"))
    named = m["named_assertion"].split(" -- ")[0]
    oc = d["outcomes"].get(named, "absent")
    junit = open(G + named_run + "/junit.xml").read()
    marker = junit.count(f"ENG3-MUTANT {mid} executed")
    text = d["failures"].get(named, "")
    counter = re.findall(r"'host_counts': \{[^}]*\}", text)
    fails = {}
    for r in full:
        o = json.load(open(G + r + "/outcomes.json"))
        fails[r] = {"module_sha256": o["module_sha256"], "counts": o["counts"],
                    "failed_ids": sorted(k.split("::")[1] for k, v in o["outcomes"].items() if v in ("failed", "error"))}
    out.append({"id": mid, "gate": m["gate"], "named": named, "named_run": named_run,
                "named_module_sha256": d["module_sha256"], "named_outcome": oc, "marker_in_named_junit": marker,
                "counter_evidence": counter, "named_failure_tail": text[-300:], "full_runs": fails,
                "killed": oc in ("failed", "error") and marker >= 1})
json.dump({"mutants": out, "killed": sum(x["killed"] for x in out), "total": len(out)},
          open("artifacts/eng3/MUTANT_RESULTS.json", "w"), indent=1)
for x in out:
    print(x["id"], x["gate"], x["named_outcome"], "marker", x["marker_in_named_junit"], "killed", x["killed"],
          {r: v["counts"] for r, v in x["full_runs"].items()}, x["counter_evidence"])
