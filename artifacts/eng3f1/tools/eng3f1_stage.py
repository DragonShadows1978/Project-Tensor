"""Pinned G2/G3 launch and CPU prefix; never obtains a lease itself.
Prior art: inherited ENG3 same-process cast/per-ID harnesses (2026), pytest
hooks, NIST SHA-256 (2001). Ours: fail-before-lease pin, isolated E/F package
workers and explicit stop immediately before the common first device call.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/"artifacts/eng3f1"
LINES={"g2":[("F","own","")],
       "g3-cpu":[("E","cpu",""),("F","cpu","")],
       "g3-g1":[("E","suite","g1"),("F","suite","g1")],
       "g3-g2":[("E","suite","g2"),("F","suite","g2")],
       "g3-device":[("E","devpath",""),("F","devpath","")]}


def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2))


def pin(arm):
    rec=json.loads((ART/f"build/BUILD_{arm}.json").read_text())
    module=ROOT/rec["module"]
    got=hashlib.sha256(module.read_bytes()).hexdigest()
    if got!=rec["module_sha256"]:
        raise RuntimeError(f"REFUSED BEFORE LEASE: {arm} module {got} != {rec[module_sha256]}")
    return rec,module


def worker(arm,kind,split,label,prefix):
    rec,module=pin(arm)
    package=str(module.parent.parent)
    os.environ.update(PYTHONPATH=package,ENG3F1_PACKAGE_PARENT=package,ENG3F1_MODULE=str(module),
                      ENG3_EXPECT=rec["module_sha256"],ENG3_LABEL=label,
                      ENG3_RECORD_DIR=str(ART/"g3"/label))
    sys.path.insert(0,package)
    import numpy as np
    import tensor_cuda as tc
    assert Path(tc._C.__file__).resolve()==module.resolve()
    # This is the actual common CPU prefix used by every real worker. The
    # next tensor call is its first device allocation. Prefix exits before it.
    src=np.array([0.0,1.0,-2.0],np.float32)
    host=tc.tensor(src,device="cpu")
    assert host.dtype=="float32" and host.shape==src.shape
    for t in (host,tc._C._core_clone(host),tc._C._core_to(host,"cpu")):
        got=t.numpy().view(np.uint32);ref=src.view(np.uint32)
        assert got.dtype==ref.dtype and got.shape==ref.shape
        assert got.any() and ref.any() and (got!=got[0]).any() and (ref!=ref[0]).any()
        assert np.array_equal(got,ref)
    if prefix:
        result={"arm":arm,"kind":kind,"module_sha256":rec["module_sha256"],
                "cpu_prefix_pass":True,"stopped_before":"production CPU suite" if kind=="cpu" else "tc.tensor(src,device=cuda)","gpu_run":False}
        print(json.dumps(result),flush=True)
        return 0
    if kind!="cpu":
        device=tc.tensor(src,device="cuda")  # FIRST DEVICE CALL; prefix stops above
        assert device.shape==src.shape
    if kind=="own":
        import pytest
        out=ART/"g3"/label;out.mkdir(parents=True,exist_ok=True)
        args=["-q","-s","-p","no:cacheprovider","--basetemp=/tmp/eng3f1_tmp/pt_"+label,
              "--junitxml="+str(out/"junit.xml"),"-k","g2 or (g1b and cuda) or f1_gpu",
              str(ROOT/"tensor_cuda/tests/test_eng3_host.py"),str(ART/"tools/eng3f1_checks.py")]
        return int(pytest.main(args))
    if kind in ("suite","cpu"):
        if kind=="suite":
            files=json.loads((ROOT/"artifacts/eng3/g3/SPLIT.json").read_text())[split]
            os.environ["ENG3_FILES"]=" ".join(files)
        else:
            os.environ.pop("ENG3_FILES",None)
        ns=runpy.run_path(str(ART/"tools/eng3f1_suite.py"),run_name="eng3f1_suite")
        return ns["main"]()
    ns=runpy.run_path(str(ART/"tools/eng3f1_devpath.py"),run_name="eng3f1_devpath")
    return ns["main"](label)


def compare(line):
    if line in ("g3-cpu", "g3-g1", "g3-g2"):
        a=json.loads((ART/"g3"/(line+"_E")/"outcomes.json").read_text())
        b=json.loads((ART/"g3"/(line+"_F")/"outcomes.json").read_text())
        ids=sorted(set(a["outcomes"])|set(b["outcomes"]))
        differences=[{"id":i,"E":a["outcomes"].get(i),"F":b["outcomes"].get(i)}
                     for i in ids if a["outcomes"].get(i)!=b["outcomes"].get(i)]
        result={"n_ids":len(ids),"differences":differences,"intended_changes":[],
                "collection_file_differences":sorted(set(a["collection_errors"])^set(b["collection_errors"])),
                "suite_exits":[a["pytest_exit"],b["pytest_exit"]],
                "writes":{"E":a["worktree_writes_by_tests"],"F":b["worktree_writes_by_tests"]}}
        ok=bool(ids) and not differences and not result["collection_file_differences"] and not any(result["writes"].values())
    else:
        import numpy as np
        paths=[ART/"g3"/(line+"_"+arm) for arm in ("E","F")]
        meta=[json.loads(Path(str(p)+".meta.json").read_text()) for p in paths]
        assert meta[0]==meta[1], "device output dtype/shape metadata differ"
        dumps=[np.load(str(p)+".npz") for p in paths]
        assert set(dumps[0].files)==set(dumps[1].files)==set(meta[0])
        groups={}
        for key in sorted(meta[0]):
            a,b=dumps[0][key],dumps[1][key]
            assert a.dtype==b.dtype and a.shape==b.shape and a.dtype.kind=="u",key
            groups.setdefault(str(a.dtype),[]).append(key)
        result={"keys":len(meta[0]),"groups":{},"differences":[],"intended_changes":[]}
        for width,keys in groups.items():
            a,b=[np.concatenate([d[key].ravel() for key in keys]) for d in dumps]
            for x in (a,b):
                assert x.any() and (x!=x[0]).any(), (width,"uninformative actual output corpus")
            bad=np.flatnonzero(a!=b)
            result["groups"][width]={"elements":int(a.size),"mismatches":int(bad.size),"keys":keys}
            if bad.size:
                result["differences"].append({"width":width,"offsets":bad[:20].tolist()})
        ok=bool(result["keys"]) and not result["differences"]
    result["pass"]=ok
    dump(ART/"staged"/line/"COMPARE.json",result)
    return ok


def execute(line,prefix):
    results=[]
    for arm,kind,split in LINES[line]:
        rec,module=pin(arm)
        label=line+"_"+arm
        env=dict(os.environ,TMPDIR="/tmp/eng3f1_tmp",PYTHONDONTWRITEBYTECODE="1",
                 OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        if prefix or kind=="cpu":
            env["CUDA_VISIBLE_DEVICES"]=""
        cmd=["timeout","1700s","nice","-n","19",sys.executable,"-B","-P",__file__,"worker",arm,kind,split,label,"1" if prefix else "0"]
        p=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
        print(p.stdout,flush=True);sys.stderr.write(p.stderr)
        row={"arm":arm,"kind":kind,"split":split,"argv":cmd,"exit":p.returncode,
             "stdout":p.stdout,"stderr":p.stderr,"module_sha256":rec["module_sha256"]}
        results.append(row)
        if p.returncode:
            break
    ok=len(results)==len(LINES[line]) and all(r["exit"]==0 for r in results)
    if ok and not prefix and line.startswith("g3-"):
        ok=compare(line)
    dump(ART/"staged"/line/("CPU_PREFIX.json" if prefix else "INNER_RESULT.json"),
         {"line":line,"prefix_only":prefix,"pass":ok,"inner_exit":0 if ok else 1,"results":results,"time":datetime.datetime.now(datetime.timezone.utc).isoformat()})
    return 0 if ok else 1


def finish(line,rc):
    out=ART/"staged"/line
    inner=out/"INNER_RESULT.json"
    log=out/"GPU.log"
    inner_result=json.loads(inner.read_text()) if inner.exists() else None
    dump(out/"RESULT.json",{"line":line,"forge_exit":int(rc),
         "inner_exit":inner_result.get("inner_exit") if inner_result else (124 if int(rc)==124 else None),
         "inner_result":inner_result,
         "log_sha256":hashlib.sha256(log.read_bytes()).hexdigest(),"gpu_run":True})
    return int(rc)


if __name__=="__main__":
    mode=sys.argv[1]
    if mode=="worker":
        sys.exit(worker(*sys.argv[2:6],prefix=sys.argv[6]=="1"))
    if mode=="finish":
        sys.exit(finish(sys.argv[2],sys.argv[3]))
    assert mode in ("prefix","run")
    sys.exit(execute(sys.argv[2],mode=="prefix"))
