"""Prior art: NVIDIA cuobjdump binary inspection, taken; BK4 symbol receipts ours. No GPU calls."""
from pathlib import Path
import subprocess,re,json,hashlib
r=Path(__file__).resolve().parents[2];a=r/'artifacts/bp_kernel_4'
b=json.loads((a/'engine_build_receipt.json').read_text())['binaries'][0]
run=subprocess.run(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(r/b['path'])],capture_output=True,text=True,check=True,timeout=60)
parts=re.split(r'Function : ',run.stdout);found=[]
for part in parts[1:]:
 name=part.splitlines()[0]
 if 'bk4' not in name:continue
 mma=[line.strip() for line in part.splitlines() if 'HMMA' in line]
 found.append(dict(symbol=name,mma_instructions=len(mma),examples=mma[:3]))
with (a/'mma_build_inspection.json').open('x') as f:json.dump(dict(evidence_class='offline compiled SASS inspection; no GPU execution',binary=b,functions=found),f,indent=2)
assert len([v for v in found if 'forward' in v['symbol'] and v['mma_instructions']>0])==2
print('SASS: h normal and diagnostic both contain HMMA')
