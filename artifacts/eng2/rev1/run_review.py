"""Receipt wrapper. Prior art: subprocess completion/exit-status receipts, Python subprocess (PEP 324, 2003), unverified -- lead to check; no novel method."""
from pathlib import Path
import subprocess, sys, json, time, os
root=Path(__file__).resolve().parent
name, command=sys.argv[1:]
start=time.time()
r=subprocess.run(["timeout", "590s", "nice", "-n", "19", "bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(root/(name+".log")).write_text(r.stdout)
(root/(name+".command.json")).write_text(json.dumps({"command":command,"exit_code":r.returncode,"seconds":time.time()-start,"TMPDIR":os.environ.get("TMPDIR"),"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),"OMP_NUM_THREADS":os.environ.get("OMP_NUM_THREADS")},indent=2)+"\n")
with (root/"LEDGER.md").open("a") as f: f.write(f"\n- {name}: exit {r.returncode}; `{name}.command.json`, `{name}.log`.\n")
print(r.stdout,end=""); print(f"\nRECEIPT {name}: exit {r.returncode}")
sys.exit(r.returncode)
