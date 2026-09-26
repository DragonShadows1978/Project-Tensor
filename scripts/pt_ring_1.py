"""PT-RING-1 create-only, sequential lead GPU gates; import is CPU safe.

Prior art: CUDA event timing/stream concurrency (NVIDIA 12.6, 2024), CheckFreq
(Mohan et al., FAST 2021) snapshot/drain, Apex multi_tensor_apply (2018 onward),
PT-DET/PT-TF32 receipts (2026), SHA256 (NIST, 2001), POSIX child timeouts.
Taken: mechanisms. Ours: engine-specific hazard, exact geometry and acceptance.
No lock acquisition/inspection, GPU discovery, live imports or unowned signals.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/pt_ring_1"
LANES = ("correctness", "hazard", "overlap", "ring", "limits", "goldens", "engine_suite")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write("\n")


def lead_guard(authorized):
    if not authorized or os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"):
        raise RuntimeError("BLOCKED: lead GPU slot required; --lead-gpu missing or CUDA_VISIBLE_DEVICES empty")


def verify_manifest():
    m = json.loads((ART / "SOURCE_MANIFEST.json").read_text())
    for name, expected in m["files"].items():
        if sha(ROOT/name) != expected: raise ValueError("source/binary drift: " + name)
    return m


def import_engine():
    sys.path.insert(0, str(ROOT / "tensor_cuda"))
    import tensor_cuda as tc
    if not Path(tc._C.__file__).resolve().is_relative_to(ROOT): raise RuntimeError("wrong engine import")
    return tc


def pytest_worker(gate, out):
    # Prior art: PEP 578 audit hooks (Python 3.8, 2019) and pytest.main. Taken:
    # import boundary enforcement; ours: the immutable live-engine exclusion.
    # Preload this fork BEFORE legacy test modules can prepend their old paths.
    import_engine()
    forbidden = "/mnt/ForgeRealm/Project-Tensor/"
    def audit(event, args):
        path = args[0] if event == "open" and args else (args[1] if event == "import" and len(args) > 1 else None)
        if isinstance(path, (str, bytes)):
            path = os.fsdecode(path)
            if os.path.abspath(path).startswith(forbidden):
                raise RuntimeError("live engine access refused by PT-RING-1: " + path)
    sys.addaudithook(audit)
    import pytest
    tests = ["tensor_cuda/tests/test_async_copy.py"] if gate == "correctness" else [str(x.relative_to(ROOT)) for x in sorted((ROOT/"tensor_cuda/tests").glob("test_*.py"))]
    # One existing bench-in-test-file reads sys.argv[1] as an integer at import.
    # Programmatic pytest leaves its default intact, without changing assertions.
    sys.argv = ["pt_ring_1_pytest"]
    return pytest.main(["-q", "-o", "cache_dir="+str(out.parent/"pytest-cache"), *tests])


def finite(values, count=7):
    return (isinstance(values, list) and len(values) == count and
            all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values))


def assess_overlap(result):
    try:
        rows = result["samples"]
        if result["bytes"] != 2_940_000_000 or len(rows) != 7: return "RED"
        if not finite([result["calibrated_compute_ms"]], 1) or result["calibrated_compute_ms"] < 400: return "RED"
        if not finite([r["solo_wall_ms"] for r in rows]) or not finite([r["concurrent_wall_ms"] for r in rows]): return "RED"
        ratios = [r["concurrent_wall_ms"] / r["solo_wall_ms"] for r in rows]
        for r in rows:
            c0, c1, d0, d1 = [r[k] for k in ("compute_start_ms", "compute_end_ms", "copy_start_ms", "copy_end_ms")]
            if not all(math.isfinite(v) for v in (c0, c1, d0, d1)): return "RED"
            if not (c0 >= 0 and d0 >= 0 and c1 > c0 and d1 > d0 and min(c1, d1) > max(c0, d0)): return "RED"
        return "GREEN" if statistics.median(ratios) <= 1.03 else "RED"
    except (KeyError, TypeError, ZeroDivisionError): return "RED"


def assess_ring(result):
    try:
        if result["count"] != 1110 or result["bytes"] != 2_939_091_552 or result["slots"] != 3: return "RED"
        if result["host_slots_bitwise"] is not True or result["pinned_bytes"] < 3*result["bytes"]: return "RED"
        if not finite([result["compute_ms"]], 1) or result["compute_ms"] < 400: return "RED"
        # Compare both, do not silently switch the default after observing timings.
        for method in ("memcpy", "kernel"):
            r = result["methods"][method]
            if not finite(r["host_ms"]) or not finite(r["legacy_ms"]): return "RED"
        r = result["methods"]["memcpy"]
        return "GREEN" if max(statistics.median(r["host_ms"]), statistics.median(r["legacy_ms"])) <= 50.9 else "RED"
    except (KeyError, TypeError): return "RED"


def timed_event(tc, stream=None):
    return tc.Event(enable_timing=True).record(stream)


def compute_iterations(tc, scratch):
    iterations = 10000
    while True:
        start = timed_event(tc)
        tc._C._ring_compute(scratch, iterations)
        end = timed_event(tc); end.synchronize()
        elapsed = start.elapsed_time(end)
        if elapsed >= 400 or iterations >= 10_000_000: return iterations, elapsed
        iterations = min(10_000_000, max(iterations+1, int(iterations*450/max(elapsed, .01))))


def overlap(tc):
    nbytes = 2_940_000_000
    source = tc._C._ring_empty((nbytes,), "uint8")
    host = tc.pinned_empty((nbytes,), "uint8")
    scratch = tc._C._ring_empty((65536,), "uint8")
    stream = tc.Stream(); legacy = tc.legacy_stream()
    # Resident scratch and operands; initialization/pre-touch is outside timing.
    tc.copy_to_host_async([source], [host], stream).synchronize()
    iterations, calibrated = compute_iterations(tc, scratch)
    def solo():
        legacy.synchronize(); t = time.perf_counter()
        tc._C._ring_compute(scratch, iterations)
        timed_event(tc).synchronize()
        return (time.perf_counter()-t)*1000
    def concurrent():
        legacy.synchronize(); stream.synchronize()
        origin = timed_event(tc); stream.wait(origin)
        c0 = timed_event(tc)
        t = time.perf_counter()
        tc._C._ring_compute(scratch, iterations)
        c1 = timed_event(tc)
        d0 = timed_event(tc, stream)
        done = tc.copy_to_host_async([source], [host], stream, after=origin)
        d1 = timed_event(tc, stream)
        c1.synchronize(); wall = (time.perf_counter()-t)*1000
        done.synchronize(); d1.synchronize()
        return dict(concurrent_wall_ms=wall, compute_start_ms=origin.elapsed_time(c0),
                    compute_end_ms=origin.elapsed_time(c1), copy_start_ms=origin.elapsed_time(d0),
                    copy_end_ms=origin.elapsed_time(d1))
    for _ in range(2): solo(); concurrent()
    rows = []
    for i in range(7):
        # Adjacent interleaved A/B; reverse every other pair to limit drift bias.
        if i % 2: r = concurrent(); s = solo()
        else: s = solo(); r = concurrent()
        rows.append(dict(r, solo_wall_ms=s))
    result = dict(bytes=nbytes, iterations=iterations, calibrated_compute_ms=calibrated, samples=rows,
                  evidence_class="synthetic compute/D2H microbenchmark, not a training step")
    result["status"] = assess_overlap(result)
    return result


def hazard(tc):
    import numpy as np
    legacy = tc.legacy_stream(); stream = tc.Stream()
    before = np.arange(1 << 18, dtype=np.float32)
    live = tc.tensor(before); updated = tc.tensor(before+13)
    stage = tc._C._ring_empty(live.shape, live.dtype)
    host = tc.pinned_empty(live.shape, live.dtype)
    rows = []
    for use_wait in (True, False):
        with tc.no_grad():
            live.copy_(tc.tensor(before))
            origin = timed_event(tc)
            tc.copy_many_([stage], [live]); staged = timed_event(tc)
            tc._C._ring_delay(200, stream)
            stream.wait(staged); start_copy = timed_event(tc, stream)
            done = tc.copy_to_host_async([stage], [host], stream, after=staged)
            live.copy_(updated)
            if use_wait: legacy.wait(done)
            tc.copy_many_([stage], [live]); overwritten = timed_event(tc)
        done.synchronize(); overwritten.synchronize()
        actual = host.numpy().tobytes()
        rows.append(dict(wait=use_wait, equals_preupdate=actual == before.tobytes(),
                         equals_update=actual == (before+13).tobytes(),
                         overwritten_relative_to_copy_start_ms=origin.elapsed_time(overwritten)-origin.elapsed_time(start_copy),
                         sha256=hashlib.sha256(actual).hexdigest()))
    # A valid negative control either observes corruption, or records timing
    # evidence that this device serialized the overwrite after the copy began.
    positive, negative = rows
    observed = not negative["equals_preupdate"]
    explanation = None if observed else (
        "Race not observed in this bounded delayed-copy trial; overwrite timestamp relative to copy start "
        f"is {negative['overwritten_relative_to_copy_start_ms']:.6f} ms. This does not prove safety without the edge.")
    return dict(status="GREEN" if positive["equals_preupdate"] else "RED", samples=rows,
                negative_race_observed=observed, negative_explanation=explanation,
                evidence_class="constructed concurrent staging hazard; not a training replay")


def ring(tc):
    import numpy as np
    schema = json.loads((ART/"GEOMETRY.json").read_text())
    legacy = tc.legacy_stream(); stream = tc.Stream()
    # Synthetic values with the exact checkpoint schema; no checkpoint or model
    # imported on GPU. One source + one staging + 3 pinned states, allocated once.
    live = [tc._C._ring_from_bytes(np.zeros(r["shape"], r["dtype"]).tobytes(), r["shape"], r["dtype"])
            for r in schema["tensors"]]
    stage = [tc.empty_like(t) for t in live]
    hosts = [[tc.pinned_empty(t.shape, t.dtype) for t in live] for _ in range(3)]
    scratch = tc._C._ring_empty((65536,), "uint8")
    iterations, duration = compute_iterations(tc, scratch)
    results = {m: dict(host_ms=[], legacy_ms=[]) for m in ("memcpy", "kernel")}
    previous = None
    for iteration in range(9):
        for method in (("memcpy", "kernel") if iteration % 2 == 0 else ("kernel", "memcpy")):
            slot = iteration % 3
            # Complete the synthetic step. Production has ~5.09 seconds; this
            # gate provides >=400ms of compute and reports the calibrated value.
            tc._C._ring_compute(scratch, iterations)
            timed_event(tc).synchronize()
            start = timed_event(tc); wall = time.perf_counter()
            if previous is not None: legacy.wait(previous)
            with tc.no_grad(): tc.copy_many_(stage, live, method=method)
            stop = timed_event(tc)
            previous = tc.copy_to_host_async(stage, hosts[slot], stream, after=stop)
            host_ms = (time.perf_counter()-wall)*1000
            stop.synchronize()
            # Host observation also proves receipt operands were actually copied.
            if iteration >= 2:
                results[method]["host_ms"].append(host_ms)
                results[method]["legacy_ms"].append(start.elapsed_time(stop))
    previous.synchronize()
    bitwise = all(not np.any(h.numpy().view(np.uint8)) for slot in hosts for h in slot)
    # Preserve raw sample evidence; the default memcpy method owns the 1% gate.
    result = dict(count=schema["count"], bytes=schema["bytes"], slots=3, methods=results,
                  pinned_bytes=tc.pinned_bytes(), compute_ms=duration, host_slots_bitwise=bitwise,
                  evidence_class="exact run-7 geometry synthetic ring microbenchmark; no model throughput claim")
    result["status"] = assess_ring(result) if bitwise else "RED"
    return result


def limits(tc):
    total = json.loads((ART/"GEOMETRY.json").read_text())["bytes"] * 3
    initial = tc.pinned_bytes(); limit = tc.pinned_memory_limit()
    result = dict(requested_ring_bytes=total, memlock_bytes=list(resource.getrlimit(resource.RLIMIT_MEMLOCK)),
                  driver_allocation_succeeded=False, explicit_limit_refused=False)
    start = time.perf_counter()
    buffers = []
    try:
        for _ in range(3): buffers.append(tc.pinned_empty((total//3,), "uint8"))
        result.update(driver_allocation_succeeded=True, held_bytes=tc.pinned_bytes())
    except RuntimeError as e: result["driver_error"] = str(e)
    finally: buffers.clear(); gc.collect()
    result["allocation_seconds"] = time.perf_counter()-start
    tc.set_pinned_memory_limit(tc.pinned_bytes())
    try:
        t = time.perf_counter()
        try: tc.pinned_empty((1,), "uint8")
        except RuntimeError as e:
            result.update(explicit_limit_refused="no allocation attempted" in str(e), limit_error=str(e))
        result["limit_refusal_seconds"] = time.perf_counter()-t
    finally: tc.set_pinned_memory_limit(limit)
    result["released_bytes"] = tc.pinned_bytes() == initial
    result["status"] = "GREEN" if all(result[k] for k in ("driver_allocation_succeeded", "explicit_limit_refused", "released_bytes")) else "RED"
    result["evidence_class"] = "bounded ring-size driver allocation and explicit engine budget refusal; not physical exhaustion"
    return result


def golden(binary, out):
    import numpy as np
    spec = importlib.util.spec_from_file_location("_tensor_cuda", binary)
    c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
    rows = {}
    rng = np.random.default_rng(914)
    a0 = rng.normal(size=(17, 19)).astype(np.float32)
    b0 = rng.normal(size=(19, 13)).astype(np.float32)
    for dtype in ("float32", "bfloat16"):
        a = c.tensor(a0, "cuda", False).astype(dtype)
        b = c.tensor(b0, "cuda", False).astype(dtype)
        # Cast first, then make leaves: the existing engine intentionally frees
        # non-leaf gradients during backward. Test actual retained leaf grads.
        a.requires_grad = True; b.requires_grad = True
        product = c.matmul(a, b, 1., False)
        nonlinear = product.gelu().sum([], False)
        nonlinear.backward()
        for name, tensor in (("product", product), ("loss", nonlinear), ("a_grad", a.grad), ("b_grad", b.grad)):
            if tensor is None: raise ValueError("missing golden gradient: " + name)
            array = tensor.numpy()
            if not np.isfinite(array).all(): raise ValueError("nonfinite golden")
            rows[dtype+"/"+name] = dict(shape=list(array.shape), host_dtype=str(array.dtype),
                                       sha256=hashlib.sha256(array.tobytes()).hexdigest())
    write_json(out, dict(binary_sha256=sha(binary), values=rows,
                         representation="bf16 exported exactly as fp32; fp32 native bytes; finite values"))


def child(argv, out, seconds, env):
    with Path(out).open("x") as log:
        t = time.monotonic()
        # subprocess.run owns and kills only its immediate child on timeout.
        # Workers do not start persistent processes or use shell background jobs.
        try:
            rc = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=seconds).returncode
            return dict(argv=argv, returncode=rc, seconds=time.monotonic()-t,
                        status="GREEN" if rc == 0 else "RED")
        except subprocess.TimeoutExpired:
            return dict(argv=argv, returncode=None, seconds=time.monotonic()-t, status="BLOCKED_TIMEOUT")


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lead-gpu", action="store_true")
    p.add_argument("--gate", choices=(*LANES, "all"), default="all")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--golden-binary", help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    try: lead_guard(args.lead_gpu)
    except RuntimeError as e:
        # Guard precedes engine import and any output/lock/device access.
        print(str(e)); return 2
    manifest = verify_manifest()
    args.out = args.out.resolve()
    if not args.out.is_relative_to(ROOT): raise ValueError("receipts must stay in the registered fork")
    if args.golden_binary:
        if Path(args.golden_binary).resolve() not in [ROOT/manifest[k] for k in ("baseline_binary", "binary")]:
            raise ValueError("golden binary must be one of the two sealed fork artifacts")
        golden(args.golden_binary, args.out); return 0
    if args.worker:
        if args.gate in ("correctness", "engine_suite"):
            return pytest_worker(args.gate, args.out)
        result = globals()[args.gate](import_engine())
        write_json(args.out, result)
        print(json.dumps(result, allow_nan=False)); return 0 if result["status"] == "GREEN" else 1
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"tmp").mkdir()
    env = dict(os.environ, PYTHONPATH=str(ROOT/"tensor_cuda"), PT_RING_1_LEAD_GPU="1",
               PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(args.out.resolve()/"tmp"))
    lanes = LANES if args.gate == "all" else (args.gate,)
    receipts = {}; deadline = started + 890  # 10 seconds reserved for receipts
    for lane in lanes:
        budget = min(300 if lane == "engine_suite" else 90, deadline-time.monotonic())
        if budget <= 0: receipts[lane] = dict(status="BLOCKED_TIMEOUT"); break
        if lane in ("correctness", "engine_suite"):
            tests = ["tensor_cuda/tests/test_async_copy.py"] if lane == "correctness" else [str(x.relative_to(ROOT)) for x in sorted((ROOT/"tensor_cuda/tests").glob("test_*.py"))]
            argv = [sys.executable, "-B", str(Path(__file__).resolve()), "--lead-gpu", "--worker", "--gate", lane,
                    "--out", str(args.out.resolve()/(lane+".json"))]
            rec = child(argv, args.out/(lane+".log"), budget, env)
            rec["test_files"] = tests
        elif lane == "goldens":
            rec = dict(status="GREEN", processes=[])
            lane_deadline = min(deadline, time.monotonic()+budget)
            for key in ("baseline_binary", "binary"):
                argv = [sys.executable, "-B", str(Path(__file__).resolve()), "--lead-gpu", "--golden-binary", str(ROOT/manifest[key]), "--out", str(args.out.resolve()/(key+".json"))]
                r = child(argv, args.out/(key+".log"), max(.001, lane_deadline-time.monotonic()), env)
                rec["processes"].append(r)
                if r["status"] != "GREEN": rec["status"] = r["status"]; break
            if rec["status"] == "GREEN":
                a, b = [json.loads((args.out/(key+".json")).read_text())["values"] for key in ("baseline_binary", "binary")]
                rec.update(bitwise=a == b and len(a) == 8, status="GREEN" if a == b and len(a) == 8 else "RED")
        else:
            argv = [sys.executable, "-B", str(Path(__file__).resolve()), "--lead-gpu", "--worker", "--gate", lane, "--out", str(args.out.resolve()/(lane+".json"))]
            rec = child(argv, args.out/(lane+".log"), budget, env)
        receipts[lane] = rec
        write_json(args.out/(lane+"_process.json"), rec)
        if rec["status"] != "GREEN":
            for remaining in lanes[lanes.index(lane)+1:]: receipts[remaining] = dict(status="BLOCKED_PRIOR_GATE")
            break
    status = "GREEN" if len(receipts) == len(lanes) and all(r["status"] == "GREEN" for r in receipts.values()) else "RED_OR_BLOCKED"
    write_json(args.out/"SUMMARY.json", dict(status=status, lanes=receipts, manifest_sha256=sha(ART/"SOURCE_MANIFEST.json"),
                                           complete_acceptance=lanes == LANES and status == "GREEN"))
    return 0 if status == "GREEN" else 1


if __name__ == "__main__": raise SystemExit(main())
