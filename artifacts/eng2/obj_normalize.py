#!/usr/bin/env python3
"""ENG2 G3: show that the two arms' object files differ only in nvcc's PID-derived names.
For each object pair (base build, cand build) under CMakeFiles/_tensor_cuda.dir: raw equality; then, in a copy of the
base bytes, every `tmpxft_<8 hex>_` temp-file tag and every per-compilation module-ID suffix `_<8 hex>_<digits>`
(read from __nv_module_id) is replaced, in order of occurrence, by the candidate's same-length value; the copy is
then compared with the candidate's bytes.
Prior art: reproducible-builds normalization of nondeterministic fields (diffoscope / reproducible-builds.org;
unverified -- lead to check); nvcc temp names (tmpxft_*) are documented nvcc behaviour (unverified -- lead to check
"nvcc tmpxft reproducible build"). Ours: this check.
Usage: obj_normalize.py BASE_BUILD CAND_BUILD OUT.json
"""
import hashlib, json, re, subprocess, sys
from pathlib import Path
bb, cb, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = []
for b in sorted((bb / "CMakeFiles/_tensor_cuda.dir").rglob("*.o")):
    c = cb / b.relative_to(bb)
    B, C = b.read_bytes(), c.read_bytes()
    row = dict(object=str(b.relative_to(bb)), base_sha256=hashlib.sha256(B).hexdigest(),
               cand_sha256=hashlib.sha256(C).hexdigest(), raw_equal=B == C)
    if B != C:
        tb = re.findall(rb"tmpxft_[0-9a-f]{8}_", B)
        tc = re.findall(rb"tmpxft_[0-9a-f]{8}_", C)
        def mids(p):
            s = subprocess.run(["readelf", "-p", "__nv_module_id", "-W", str(p)], capture_output=True).stdout
            return re.findall(rb"_cu(_[0-9a-f]{8}_\d+)\s*$", s, re.M)
        mb, mc = mids(b), mids(c)
        subs = {}
        for x, y in list(zip(tb, tc)) + list(zip(mb, mc)):
            assert len(x) == len(y)
            subs[x] = y
        N = B
        for x, y in subs.items():
            N = N.replace(x, y)
        row.update(tmpxft_tags=len(tb), module_id_suffixes=len(mb), substitutions={x.decode(): y.decode() for x, y in subs.items()},
                   equal_after_substitution=N == C,
                   differing_bytes_after=None if len(N) != len(C) else sum(1 for i in range(len(N)) if N[i] != C[i]))
    rows.append(row)
summary = dict(objects=len(rows), raw_equal=sum(r["raw_equal"] for r in rows),
               equal_after_substitution=sum(1 for r in rows if not r["raw_equal"] and r.get("equal_after_substitution")),
               still_different=[r["object"] for r in rows if not r["raw_equal"] and not r.get("equal_after_substitution")])
out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=1) + "\n")
print(json.dumps(summary))
