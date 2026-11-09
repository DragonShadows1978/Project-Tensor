"""Import tracing via Python profile during real suite collection.
Prior art: Python sys.setprofile (PSF); content seals SHA-256/FIPS 180-4.
A3 records every imported module origin; local origins become child pins.
External interpreter/dependency origins are recorded with hashes as well.
"""
import hashlib, json, os, sys
from pathlib import Path
TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[2]
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
seen={}
def tracer(frame,event,arg):
    if event=="call" and frame.f_code.co_name=="<module>":
        p=Path(frame.f_code.co_filename)
        if p.is_file():seen[str(p.resolve())]=frame.f_globals.get("__name__", "")
sys.setprofile(tracer)
sys.path.insert(0,str(TOOL/"arms/candidate"))
sys.path.insert(1,str(ROOT))
import tensor_cuda
import pytest
class Collect:
    def pytest_collection_finish(self, session):self.ids=[i.nodeid for i in session.items]
plugin=Collect()
files=json.loads((TOOL/"nn_import_test_files.json").read_text())
rc=int(pytest.main(["-q","--collect-only","-p","no:cacheprovider","--import-mode=importlib","--rootdir="+str(ROOT),*[str(ROOT/f) for f in files]],plugins=[plugin]))
sys.setprofile(None)
# Include origins already loaded before the tracing hook and binary modules.
for name,mod in list(sys.modules.items()):
    f=getattr(mod,"__file__",None)
    if f and Path(f).is_file():seen.setdefault(str(Path(f).resolve()),name)
rows=[]
for f,name in sorted(seen.items()):
    p=Path(f)
    rows.append(dict(module=name,path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p).replace(str(Path.home())+"/", "$HOME/", 1),local=p.is_relative_to(ROOT),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
result=dict(exit=rc,collected_ids=plugin.ids,modules=rows,method="module execution profile plus sys.modules origin census")
(TOOL/"import_trace.json").write_text(json.dumps(result,indent=2)+"\n")
assert rc==0
assert any(r["path"]=="scripts/apa_om135_core.py" for r in rows)
print("import-traced",len(rows),"modules",len(plugin.ids),"IDs")
