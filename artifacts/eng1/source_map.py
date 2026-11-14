#!/usr/bin/env python3
"""sha256 map of the engine's build inputs (CMakeLists, src, include) and the Python package sources.
Usage: source_map.py OUT.json. Prior art: SHA-256 (NIST FIPS 180-2, 2002); GN1 engine_sources maps. No new algorithm."""
import hashlib, json, sys
from pathlib import Path
TC = Path(__file__).resolve().parents[2] / "tensor_cuda"
files = [TC / "CMakeLists.txt"] + sorted((TC / "src").rglob("*")) + sorted((TC / "include").rglob("*")) \
    + sorted(p for p in (TC / "tensor_cuda").rglob("*.py") if "__pycache__" not in p.parts)
m = {str(p.relative_to(TC.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
Path(sys.argv[1]).write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
print(len(m), hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest())
