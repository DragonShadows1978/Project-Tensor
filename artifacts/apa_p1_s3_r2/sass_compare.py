"""Compare exact SASS function bodies, retaining every instruction/control word.
Prior art: NVIDIA cuobjdump, A1's bijective anonymous-symbol correspondence
(Project-Tensor, 2026). Only compiler-generated symbol IDs and the test-only
trailing hook argument are mapped; instruction text is never normalized.
"""
from pathlib import Path
import hashlib,json,re,subprocess
A=Path('artifacts/apa_p1_s3_r2')
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def read(path):
    rows={};key=None;lines=[]
    for line in path.open():
        m=re.search(r'Function : (\S+)',line)
        if m:key=m[1];lines=[]
        elif key and line.strip()=='..........':
            assert key not in rows,('duplicate symbol',key)
            rows[key]=''.join(lines);key=None
        elif key:lines.append(line)
    return rows

def names(rows):
    symbols=list(rows)
    p=subprocess.run(['timeout','10s','c++filt'],input='\n'.join(symbols)+'\n',text=True,capture_output=True,check=True)
    ds=p.stdout.splitlines();assert len(ds)==len(symbols)
    out={}
    for symbol,demangled in zip(symbols,ds):
        canonical=re.sub(r'(_(?:gemm_apa|paint_raster)_cu_)[0-9a-f]{8}_[0-9]+',r'\1BUILD',symbol)
        touched=('tc::eb3::select_pv<' in demangled or 'tc::eb4::select_mixed<' in demangled)
        if touched:
            canonical=demangled.replace(', tc::eb3::EarlyCounterHook)',')')
        assert canonical not in out,('non-bijective kernel correspondence',canonical)
        out[canonical]=dict(symbol=symbol,demangled=demangled,body=rows[symbol],touched=touched)
    return out

if __name__=='__main__':
    b=json.loads((A/'BUILD_COMPLETED.json').read_text());module=b['module'];dump=A/'after_broadcast.sass'
    cmd=['/usr/local/cuda-12.6/bin/cuobjdump','-sass','--gpu-architecture','sm_89',module]
    with dump.open('x') as log:subprocess.run(['timeout','300s',*cmd],stdout=log,stderr=subprocess.PIPE,check=True)
    baseline=Path('artifacts/apa_p1_a1/fixed.sass')
    x,y=names(read(baseline)),names(read(dump));rows=[]
    for key in sorted(set(x)|set(y)):
        a,c=x.get(key),y.get(key)
        same=a is not None and c is not None and a['body']==c['body']
        touched=bool((a or c)['touched'])
        rows.append(dict(kernel=key,before_symbol=a['symbol'] if a else None,after_symbol=c['symbol'] if c else None,demangled=(a or c)['demangled'],equal=same,allowed_apa_select=touched,before_body_sha256=sha_bytes(a['body'].encode()) if a else None,after_body_sha256=sha_bytes(c['body'].encode()) if c else None))
    changed=[r for r in rows if not r['equal']]
    valid=bool(rows) and set(x)==set(y) and all(r['allowed_apa_select'] for r in changed)
    r=dict(status='PASS' if valid else 'FAIL',kernels=len(rows),outside_select=sum(not r['allowed_apa_select'] for r in rows),outside_select_differences=sum(not r['equal'] and not r['allowed_apa_select'] for r in rows),changed_kernels=len(changed),changed=changed,rows=rows,baseline_dump=str(baseline),baseline_sha256=sha_bytes(baseline.read_bytes()),after_dump=str(dump),after_module=module,after_module_sha256=b['module_sha256'],argv=cmd,scope='sm89 SASS exact function bodies including encodings/control words; bijective names only; test-hook build changes select signatures/control only. No GPU execution.')
    (A/'SASS_DIFF.json').write_text(json.dumps(r,indent=2)+'\n')
    (A/'CHANGED_KERNELS.txt').write_text('\n'.join(r['demangled'] for r in changed)+'\n')
    print({k:r[k] for k in ('status','kernels','outside_select','outside_select_differences','changed_kernels')})
    raise SystemExit(0 if valid else 1)
