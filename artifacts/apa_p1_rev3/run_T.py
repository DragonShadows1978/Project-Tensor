"""Run the three authorized CPU files on sha-verified BuildF2. Prior art: pytest report API;
existing P1 per-ID receipt runner (Project-Tensor 2026), imported unchanged. No new technique.
"""
import hashlib,json,os,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/"artifacts/apa_p1_rev3"
label=sys.argv[1]
r=json.loads((OUT/"MODULES.json").read_text())["F2"]
module=pathlib.Path(r["path"])
with module.open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==r["sha256"],"REFUSED: module sha mismatch"
env=os.environ.copy()
for key in ("TC_APA_SP_STORAGE","LD_PRELOAD","APA_P1D_SRC","APA_P1F1_SRC","APA_P1F2_SRC"):env.pop(key,None)
env.update(PYTHONPATH=str(module.parent.parent)+":"+str(ROOT/"tensor_cuda/tests")+":"+str(ROOT/"artifacts/apa_p1"),
 PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",APA_P1F1_PACKED_MODULE=str(module),CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",
 APA_P1F2_BASE_TREE="/tmp/apap1f2_treeF0",TMPDIR="/tmp/p1_rev3_r1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
files=["tensor_cuda/tests/"+n for n in ("test_apa_p1d_policy.py","test_apa_p1f1_cpu.py","test_apa_p1f2_cpu.py")]
cmd=[sys.executable,"-B","-P",str(ROOT/"artifacts/apa_p1/pytest_receipt.py"),str(OUT/f"T_{label}.raw.json"),"-q","--tb=short","-ra","-p","no:cacheprovider","--basetemp",f"/tmp/p1_rev3_r1/bt_{label}",*files]
print("sha",r["sha256"],"argv",json.dumps(cmd),flush=True)
r=subprocess.run(["timeout","1100s",*cmd],cwd=ROOT,env=env)
print("PYTEST_EXIT",r.returncode,flush=True)
sys.exit(r.returncode)
