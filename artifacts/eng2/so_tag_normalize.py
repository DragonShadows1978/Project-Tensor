#!/usr/bin/env python3
"""ENG2 G3: explain the baseline/candidate module byte difference. nvcc (relocatable device code) gives some
translation units a per-compilation module ID "_<hash8>_<number>" (section __nv_module_id); this script reads both
modules' IDs, maps base -> cand per TU (same length), substitutes them in a copy of the base bytes, and counts the
bytes that still differ, per ELF section.
Prior art: reproducible-builds practice of diffing after normalizing known nondeterministic fields (diffoscope,
reproducible-builds.org; unverified -- lead to check "diffoscope reproducible builds"). Ours: this normalization.
Usage: so_tag_normalize.py BASE.so CAND.so OUT.json
"""
import hashlib, json, re, subprocess, sys
from pathlib import Path
base, cand, out = map(Path, sys.argv[1:4])
B, C = base.read_bytes(), cand.read_bytes()
def ids(path):
    # The IDs exactly as recorded in the __nv_module_id section (in the mangled kernel names the trailing number
    # runs into the name's length digits, so the names themselves are not parsed).
    sec = subprocess.run(["readelf", "-p", "__nv_module_id", "-W", str(path)], capture_output=True).stdout
    return sorted(set(re.findall(rb"(_[0-9a-f]{8}_\d+_[A-Za-z_]+_cu_[0-9a-f]{8}(?:_\d+)?)\s*$", sec, re.M)))
def tu_map(path):
    m = {}
    for s in ids(path):
        mm = re.match(rb"(_[0-9a-f]{8}_\d+_[A-Za-z_]+_cu)(_[0-9a-f]{8}(?:_\d+)?)$", s)
        m.setdefault(mm.group(1), set()).add(mm.group(2))
    return m
mb, mc = tu_map(base), tu_map(cand)
assert mb, "no module IDs read"
assert mb.keys() == mc.keys(), "different TU sets"
subs = {}
for tu in mb:
    assert len(mb[tu]) == 1 and len(mc[tu]) == 1, tu
    a, c = next(iter(mb[tu])), next(iter(mc[tu]))
    if a != c:
        assert len(a) == len(c), (a, c)
        subs[a] = c
N = B
for a, c in subs.items():
    N = N.replace(a, c)
diff = [i for i in range(min(len(N), len(C))) if N[i] != C[i]] if len(N) == len(C) else None
secs = []
for line in subprocess.run(["readelf", "-S", "-W", str(cand)], capture_output=True, text=True).stdout.splitlines():
    m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+[0-9a-f]+\s+([0-9a-f]+)\s+([0-9a-f]+)", line)
    if m:
        secs.append((m.group(1), int(m.group(2), 16), int(m.group(3), 16)))
per = {}
for i in diff or []:
    name = next((s for s, o, z in secs if o <= i < o + z), "outside-sections")
    per[name] = per.get(name, 0) + 1
rec = dict(base=str(base), cand=str(cand), base_sha256=hashlib.sha256(B).hexdigest(), cand_sha256=hashlib.sha256(C).hexdigest(),
           same_length=len(B) == len(C), raw_differing_bytes=sum(1 for i in range(min(len(B), len(C))) if B[i] != C[i]),
           tu_ids_base={k.decode(): sorted(v.decode() for v in s) for k, s in mb.items()},
           tu_ids_cand={k.decode(): sorted(v.decode() for v in s) for k, s in mc.items()},
           substitutions={a.decode(): c.decode() for a, c in subs.items()},
           differing_bytes_after_substitution=None if diff is None else len(diff), per_section_after=per,
           normalized_sha256=hashlib.sha256(N).hexdigest())
Path(out).write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(dict((k, rec[k]) for k in ("raw_differing_bytes", "substitutions", "differing_bytes_after_substitution", "per_section_after"))))
