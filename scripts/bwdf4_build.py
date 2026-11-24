"""BWD-F4 build helper: a copy of scripts/bwdf3_build.py (tensor_fork_apa_wt_bwdf1) with receipts under
artifacts/bwdf1/f4/build, TMPDIR /tmp/bwdf4_tmp, trees /tmp/bwdf4_src_<arm> (HEAD's tensor_cuda/ plus this round's
FILES), the build directory tensor_cuda/build-bwdf4/<arm>, tensor_cuda only (tc_signal is unchanged) and arms F4* /
h0* only. BWD-F3's text follows in substance: configure + build one arm out of tree, each step in its own systemd
scope (MemoryMax=12G, OOMPolicy=continue, CPUWeight=50, nice 19, -j4), copy the module beside the build and write a
build receipt with the module sha256 and a sha256 map of the package's engine sources.

Usage (from the worktree root):
  python3 -B -P scripts/bwdf4_build.py tree <arm>
  python3 -B -P scripts/bwdf4_build.py tc <arm> <source tree root: holds tensor_cuda/>

Refuses to start a step with MemAvailable < 12 GB or /mnt/ForgeRealm free < 40 GB.
Prior art: CMake out-of-tree builds; git archive (read-only git); content-hash manifests (reproducible-builds
practice); scripts/bwdf3_build.py and ENG4's eng4_build.py (this repo), copied. No other prior art known to me.
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYB = "/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src"
PKGS = {"tc": ("tensor_cuda", "_tensor_cuda.cpython-312-x86_64-linux-gnu.so", ("src", "include", "tensor_cuda"))}
SCOPE = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=12G", "-p", "OOMPolicy=continue",
         "-p", "CPUWeight=50", "--setenv=TMPDIR=/tmp/bwdf4_tmp", "--setenv=CUDACXX=/usr/local/cuda-12.6/bin/nvcc",
         "--setenv=CUDA_VISIBLE_DEVICES=", "nice", "-n", "19"]
RDIR = os.path.join(ROOT, "artifacts/bwdf1/f4/build")
FILES = ["tensor_cuda/src/kernels.cu", "tensor_cuda/src/ops.cpp"]
ARM_PREFIXES = ("F4", "h0")
RECEIPTS = os.path.join(RDIR, "build_receipts.json")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guard(step):
    with open("/proc/meminfo") as f:
        avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable"))
    free = shutil.disk_usage("/mnt/ForgeRealm").free
    if avail < 12 * 1024 * 1024:
        raise SystemExit(f"STOP before {step}: MemAvailable {avail} kB < 12 GB")
    if free < 40 * 10 ** 9:
        raise SystemExit(f"STOP before {step}: /mnt/ForgeRealm free {free} B < 40 GB")
    return avail, free


def run(name, argv, bound):
    avail, free = guard(name)
    os.makedirs(RDIR, exist_ok=True)
    log_path = os.path.join(RDIR, f"{name}.log")
    start = now()
    with open(log_path, "ab") as log:
        log.write(f"\n=== {name} {start}\n".encode())
        log.flush()
        p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            rc, to = p.wait(timeout=bound), False
        except subprocess.TimeoutExpired:
            p.terminate()  # our own child only
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            rc, to = 124, True
    rows = json.load(open(RECEIPTS)) if os.path.exists(RECEIPTS) else []
    rows.append({"name": name, "command": argv, "bound_seconds": bound, "start": start, "end": now(), "exit": rc,
                 "timeout": to, "log": os.path.relpath(log_path, ROOT), "mem_available_kB_at_start": avail,
                 "disk_free_B_at_start": free, "cwd": ROOT,
                 "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "TMPDIR")}})
    with open(RECEIPTS, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"{name}: exit {rc} timeout {to}", flush=True)
    if rc != 0:
        raise SystemExit(rc)


def source_map(tree, pkg):
    name, _, roots = PKGS[pkg]
    base = os.path.join(tree, name)
    files = {}
    for root in roots:
        for d, dirs, names in os.walk(os.path.join(base, root)):
            dirs[:] = sorted(x for x in dirs if x != "__pycache__")
            for n in sorted(names):
                p = os.path.join(d, n)
                if n.endswith((".so", ".pyc")) or os.path.islink(p):
                    continue
                files[name + "/" + os.path.relpath(p, base)] = sha(p)
    combined = hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
    return files, combined


def make_tree(arm):
    """Build F4's source tree: HEAD's tensor_cuda/ (git archive, read-only git), then this round's
    changed engine FILES copied from the worktree. Refuses if the tree exists."""
    dst = f"/tmp/bwdf4_src_{arm}"
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists")
    os.makedirs(dst)
    a = subprocess.run(["git", "-C", ROOT, "archive", "HEAD", "tensor_cuda"], capture_output=True,
                       check=True).stdout
    subprocess.run(["tar", "-x", "-C", dst], input=a, check=True)
    for f in FILES:
        shutil.copyfile(os.path.join(ROOT, f), os.path.join(dst, f))
        assert sha(os.path.join(ROOT, f)) == sha(os.path.join(dst, f)), f
    print("TREE", dst, {f: sha(os.path.join(dst, f)) for f in FILES})


def main():
    if sys.argv[1] == "tree":
        return make_tree(sys.argv[2])
    pkg, arm, tree = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
    if not arm.startswith(ARM_PREFIXES):
        raise SystemExit(f"arm {arm}: this round builds F4* and h0* arms only")
    name, so, _ = PKGS[pkg]
    src = os.path.join(tree, name)
    build = os.path.join(ROOT, name, "build-bwdf4", arm)
    if not os.path.exists(os.path.join(build, "CMakeCache.txt")):
        run(f"configure_{pkg}_{arm}", SCOPE + ["cmake", "-S", src, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                                               "-DCMAKE_CUDA_ARCHITECTURES=89",
                                               f"-DFETCHCONTENT_SOURCE_DIR_PYBIND11={PYB}",
                                               "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"], 600)
    if "--configure-only" in sys.argv:  # BWD-F2: configure and build as two tool calls (each under 600 s)
        return
    run(f"build_{pkg}_{arm}", SCOPE + ["cmake", "--build", build, "-j4"], 3300)
    built = os.path.join(src, name, so)
    mod_dir = os.path.join(build, "module")
    os.makedirs(mod_dir, exist_ok=True)
    dst = os.path.join(mod_dir, so)
    if os.path.exists(dst):  # never overwrite a receipted module: keep the old one beside it
        os.replace(dst, dst + "." + datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
    shutil.copy2(built, dst)
    files, combined = source_map(tree, pkg)
    rec = {"package": name, "arm": arm, "time": now(), "source_tree": tree, "built_module": built,
           "module_copy": os.path.relpath(dst, ROOT), "module_sha256": sha(dst), "module_bytes": os.path.getsize(dst),
           "source_files": len(files), "source_map_sha256": combined, "sources": files}
    out = os.path.join(RDIR, f"BUILD_{pkg}_{arm}.json")
    if os.path.exists(out):
        os.replace(out, out[:-5] + "_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + ".json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print("BUILT", pkg, arm, rec["module_sha256"], "sources", len(files), combined, flush=True)


if __name__ == "__main__":
    main()
