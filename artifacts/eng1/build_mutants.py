#!/usr/bin/env python3
"""ENG1 mutant builds (CPU only; no device). One registered edit per mutant (mutants/mutants.json), applied to a
scratch copy of the candidate sources under /tmp/eng1_mut_src (or /tmp/eng1_ubsan_src for the UBSan pair), built
with the arms' exact CMake arguments and CUDA 12.6, incrementally after one full build of the unmutated copy; the
module is copied to artifacts/eng1/mutants/<id>/ with a receipt (sha256, exit, timeout, the edited file's sha256).
The pristine file is restored after each mutant (receipted by sha256).

Usage (each call stays under 20 minutes): build_mutants.py STEP [STEP ...]
  STEP = prepare | M01..M13 | ubsan_prepare | M09
Prior art: DeMillo, Lipton and Sayward (1978) mutation testing (one syntactic change per mutant); UBSan
(-fsanitize=integer-divide-by-zero, GCC 4.9, 2014; unverified -- lead to check) as the instrument that makes a host
division by zero observable. Ours: the incremental scratch-build arrangement.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "eng1" / "mutants"
MUT = {m["id"]: m for m in json.loads((OUT / "mutants.json").read_text())}
CUDA = "/usr/local/cuda-12.6"
PYBIND = "/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src"
TREES = {"module": Path("/tmp/eng1_mut_src"), "ubsan": Path("/tmp/eng1_ubsan_src")}
UBSAN = "-fsanitize=integer-divide-by-zero -fno-sanitize-recover=integer-divide-by-zero"
MODULE = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
ENV = dict(os.environ, CUDACXX=CUDA + "/bin/nvcc", PATH=CUDA + "/bin:" + os.environ["PATH"], CUDA_VISIBLE_DEVICES="",
           TMPDIR="/tmp/eng1_tmp")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(name, cmd, bound, logdir):
    logdir.mkdir(parents=True, exist_ok=True)
    start, timed_out = utc(), False
    with (logdir / (name + ".log")).open("w") as log:
        try:
            rc = subprocess.run(["nice", "-n", "19", *cmd], stdout=log, stderr=subprocess.STDOUT, env=ENV,
                                timeout=bound).returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = 124, True
    rec = dict(name=name, command=cmd, start=start, end=utc(), exit=rc, timeout=timed_out, bound_seconds=bound)
    (logdir / (name + ".json")).write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec), flush=True)
    return rc


def prepare(kind):
    tree = TREES[kind]
    src = tree / "tensor_cuda"
    assert not tree.exists(), f"{tree} exists; scratch trees are never reused or deleted by this script"
    src.mkdir(parents=True)
    shutil.copy2(ROOT / "tensor_cuda" / "CMakeLists.txt", src / "CMakeLists.txt")
    for d in ("src", "include"):
        shutil.copytree(ROOT / "tensor_cuda" / d, src / d)
    shutil.copytree(ROOT / "tensor_cuda" / "tensor_cuda", src / "tensor_cuda",
                    ignore=shutil.ignore_patterns("__pycache__", "*.so"))
    args = ["cmake", "-S", str(src), "-B", str(tree / "build"), "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CUDA_ARCHITECTURES=89", "-DFETCHCONTENT_SOURCE_DIR_PYBIND11=" + PYBIND,
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"]
    if kind == "ubsan":
        args += ["-DCMAKE_CXX_FLAGS=" + UBSAN, "-DCMAKE_CUDA_FLAGS=" + " ".join("-Xcompiler=" + f for f in UBSAN.split()),
                 "-DCMAKE_MODULE_LINKER_FLAGS=" + UBSAN, "-DCMAKE_SHARED_LINKER_FLAGS=" + UBSAN]
    logdir = OUT / ("_" + kind + "_prepare")
    rc = run("configure", args, 600, logdir)
    rc = rc or run("build", ["cmake", "--build", str(tree / "build"), "-j2"], 1100, logdir)
    if rc == 0:
        label = "C_UBSAN" if kind == "ubsan" else "_unmutated_copy"
        dest = OUT / label
        dest.mkdir(exist_ok=True)
        shutil.copy2(src / "tensor_cuda" / MODULE, dest / MODULE)
        (dest / "receipt.json").write_text(json.dumps(dict(
            id=label, kind=kind, module_sha256=sha(dest / MODULE), sources_from=str(ROOT / "tensor_cuda"),
            flags=(UBSAN if kind == "ubsan" else ""), built=utc()), indent=2) + "\n")
    return rc


def mutant(mid):
    m = MUT[mid]
    assert m["kind"] in TREES, mid
    tree = TREES[m["kind"]]
    target = tree / m["file"]
    pristine = (ROOT / m["file"]).read_text()
    assert target.read_text() == pristine, "scratch copy differs from the candidate before mutation"
    assert pristine.count(m["old"]) == 1
    mutated = pristine.replace(m["old"], m["new"], 1)
    target.write_text(mutated)
    logdir = OUT / mid
    rc = run("build", ["cmake", "--build", str(tree / "build"), "-j2"], 1100, logdir)
    rec = dict(id=mid, decision=m["decision"], kind=m["kind"], file=m["file"], exit=rc,
               mutated_file_sha256=sha(target), pristine_file_sha256=hashlib.sha256(pristine.encode()).hexdigest())
    if rc == 0:
        shutil.copy2(tree / "tensor_cuda" / "tensor_cuda" / MODULE, logdir / MODULE)
        rec["module_sha256"] = sha(logdir / MODULE)
    target.write_text(pristine)
    rec["restored_sha256"] = sha(target)
    assert rec["restored_sha256"] == rec["pristine_file_sha256"]
    rec["built"] = utc()
    (logdir / "receipt.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec), flush=True)
    return rc


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    status = 0
    for step in sys.argv[1:]:
        if step in ("prepare", "ubsan_prepare"):
            status |= prepare("ubsan" if step == "ubsan_prepare" else "module")
        else:
            status |= mutant(step)
    sys.exit(1 if status else 0)
