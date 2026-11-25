"""BWD-F1 staged GPU lines: the table, the pre-lease gate, GPU_LINES.json and the dry runs.

  python3 -B -P scripts/bwdf1_gpu_lines.py gate <line>     # pre-lease (L103/L119): exit 0 ok, 3 refused
  python3 -B -P scripts/bwdf1_gpu_lines.py write            # artifacts/bwdf1/GPU_LINES.json
  python3 -B -P scripts/bwdf1_gpu_lines.py dryrun           # each line once, exactly as written, FORGE_GPU_DRY_RUN=1

Each line is one shell command, run from the worktree root:
  python3 -B -P scripts/bwdf1_gpu_lines.py gate <line> && forge-gpu-run --min <m> --max 540 --label bwdf1-<line> --
      timeout <t>s [compute-sanitizer --tool memcheck --log-file <log>] python3 -B -P <step> <args>
The gate reads files only (no engine import, no CUDA call): the module sha256 of every build the line loads against
its receipt (frozen package and build copy), and the line's write-once paths, which must not exist yet.
Prior art: ENG4's and BWD-AUD1's staged-line patterns (this repo); no other prior art known to me.
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bwdf1_gpu_common as G  # noqa: E402

GPU = "artifacts/bwdf1/gpu"
G4_CHUNKS = 12


def lines():
    L = {}
    for arm in ("B", "E"):
        for tag, script in (("f01", f"artifacts/bwdf1/gpu_lines/{arm}/f01_halfprec_zero_inner.py"),
                            ("f03", f"artifacts/bwdf1/gpu_lines/{arm}/f03_zero_batch.py")):
            L[f"g2_{tag}_{arm}"] = dict(gate="G2", arm=arm, mods=[("tc", arm)], memcheck=True, min=240, timeout=500,
                                        argv=[script], out=f"{GPU}/g2_{tag}_{arm}", peak_host_bytes=4 << 30)
        L[f"g2_direct_{arm}"] = dict(gate="G2", arm=arm, mods=[("tc", arm)], memcheck=True, min=300, timeout=500,
                                     argv=["scripts/bwdf1_step_g2_direct.py", arm], out=f"{GPU}/g2_direct_{arm}",
                                     peak_host_bytes=4 << 30)
        L[f"copy_{arm}"] = dict(gate="C3 (G2)", arm=arm, mods=[("tc", arm)], memcheck=True, min=240, timeout=500,
                                argv=["scripts/bwdf1_step_copy.py", arm], out=f"{GPU}/copy_{arm}",
                                peak_host_bytes=4 << 30)
    for arm in ("B", "E", "m03_reduce_gt1"):
        L[f"g3_{arm}"] = dict(gate="G3" if arm in "BE" else "G5 m03", arm=arm, mods=[("tc", arm)], memcheck=False,
                              min=300, timeout=500, argv=["scripts/bwdf1_step_g3.py", arm], out=f"{GPU}/g3_{arm}",
                              peak_host_bytes=3 << 30)
    for arm in ("B", "E"):
        for i in range(1, G4_CHUNKS + 1):
            L[f"g4_{arm}_{i:02d}"] = dict(gate="G4 GPU", arm=arm, mods=[("tc", arm), ("ts", arm)], memcheck=False,
                                          min=480, timeout=500, argv=["scripts/bwdf1_step_g4.py", f"gpu_{arm}", arm,
                                                                      "330"],
                                          out=f"artifacts/bwdf1/g4/gpu_{arm}/part_<n> (created by the runner, "
                                              f"write-once per part)", peak_host_bytes=6 << 30, series=True)
    for k, v in L.items():
        mc = (["compute-sanitizer", "--tool", "memcheck", "--log-file", f"{GPU}/{k}.memcheck.log"]
              if v["memcheck"] else [])
        v["command"] = (f"python3 -B -P scripts/bwdf1_gpu_lines.py gate {k} && forge-gpu-run --min {v['min']} "
                        f"--max 540 --label bwdf1-{k} -- timeout {v['timeout']}s "
                        + " ".join(mc + ["python3", "-B", "-P"] + v["argv"]))
        v["write_once"] = ([] if v.get("series") else [v["out"]]) + ([f"{GPU}/{k}.memcheck.log"] if v["memcheck"]
                                                                     else [])
    return L


def gate(line):
    v = lines()[line]
    for pkg, arm in v["mods"]:
        try:
            got = G.check_arm(pkg, arm)
        except (SystemExit, OSError) as e:
            print(f"bwdf1 gate {line}: {e}", file=sys.stderr)
            return 3
        print(f"bwdf1 gate {line}: {pkg} {arm} sha256 {got} matches its receipt")
    for p in v["write_once"]:
        if (ROOT / p).exists():
            print(f"bwdf1 gate {line}: {p} exists; move it aside before a retry", file=sys.stderr)
            return 3
    if v.get("series") and json.loads((ROOT / f"artifacts/bwdf1/g4/gpu_{v['arm']}/progress.json").read_text()
                                      if (ROOT / f"artifacts/bwdf1/g4/gpu_{v['arm']}/progress.json").exists()
                                      else "{}").get("complete"):
        print(f"bwdf1 gate {line}: series gpu_{v['arm']} complete; the step exits 0 at once")
    return 0


def write():
    L = lines()
    doc = {"order": "BWD-F1 (+ Amendment 1 C3)", "cwd": "worktree root", "lines": L,
           "lead_records_per_line": "forge-gpu-run exit, the inner timeout exit (124 = timeout, never a pass), the run "
                                    "log's sha256 (L45)",
           "exit_semantics": "gate 3 = refused before the lease (module sha or write-once path); forge-gpu-run 75/76 "
                             "= no window or lease (nothing ran); 124 = timeout; step 0 = completed, verdict in its "
                             "results.json and last stdout line; a G4 step exits 0 at once when its series is "
                             "complete",
           "rlimit_as": "none set (L114); the lead runs each line in a scope with MemoryMax=12G, OOMPolicy=continue",
           "post_device_followups_cpu": [
               "python3 -B -P scripts/bwdf1_gpu_post.py line <line>   (each G2/G3/copy line: memcheck summary + "
               "verdict -> artifacts/bwdf1/gpu/<line>.post.json)",
               "python3 -B -P scripts/bwdf1_gpu_post.py controls       (G2 nonzero controls, B vs E bitwise)",
               "python3 -B -P scripts/bwdf1_gpu_post.py g4             (G4 GPU per-ID, B vs E)"],
           "lease_check_lines": {s: f"{s}:1-2" for s in (
               "scripts/bwdf1_step_g2_direct.py", "scripts/bwdf1_step_g3.py", "scripts/bwdf1_step_copy.py",
               "scripts/bwdf1_step_g4.py", "artifacts/bwdf1/gpu_lines/B/f01_halfprec_zero_inner.py",
               "artifacts/bwdf1/gpu_lines/B/f03_zero_batch.py", "artifacts/bwdf1/gpu_lines/E/f01_halfprec_zero_inner.py",
               "artifacts/bwdf1/gpu_lines/E/f03_zero_batch.py")},
           "copied_audit_scripts": "artifacts/bwdf1/gpu_lines/<arm>/f01_halfprec_zero_inner.py and f03_zero_batch.py "
                                   "are BWD-AUD1's with two lines changed (ROOT one level deeper; OUT under "
                                   "artifacts/bwdf1/gpu/); aud1_verdicts.py is byte-identical; aud1_gpu_load.py keeps "
                                   "the audit's API with the module pin changed to this round's Build B / Build E"}
    (ROOT / "artifacts/bwdf1/GPU_LINES.json").write_text(json.dumps(doc, indent=1))
    print("lines", len(L))


def dryrun():
    out = []
    for k, v in lines().items():
        env = dict(os.environ, FORGE_GPU_DRY_RUN="1")
        p = subprocess.run(["bash", "-c", v["command"]], cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=120)
        out.append({"line": k, "command": v["command"], "exit": p.returncode, "stdout": p.stdout[-600:],
                    "stderr": p.stderr[-600:]})
        print(k, p.returncode, flush=True)
    d = ROOT / "artifacts/bwdf1/gpu_dry"
    d.mkdir(exist_ok=True)
    (d / "DRY_RUN.json").write_text(json.dumps(out, indent=1))
    created = [v["out"] for v in lines().values() if not v.get("series") and (ROOT / v["out"]).exists()]
    print("dry-run exits:", sorted({r["exit"] for r in out}), "write-once paths created:", created)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "gate":
        sys.exit(gate(sys.argv[2]))
    if mode == "write":
        write()
    elif mode == "dryrun":
        dryrun()
