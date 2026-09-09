"""SP4G create-only provenance. Prior art: SP3 a4 (2026), SHA-256/NIST2001,
Make/Feldman1979 per-kind dependency invalidation. New experiment closures only.
"""
import hashlib,json,os,sys,time
from contextvars import ContextVar
from pathlib import Path
R=Path(__file__).resolve().parents[1]; A=R/'artifacts/apa_sp4g'; BUILD=A/'build'
ROOT=R; ART=A
REG_SHA='099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e'
VALIDATION=ContextVar('sp4g_validation',default=None)
class Red(RuntimeError):pass
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def publish(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.partial')
    with tmp.open('x') as f:json.dump(j,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    try:os.link(tmp,p)
    finally:tmp.unlink()
def save_array(p,x):
    import numpy as np
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.save(f,x,allow_pickle=False)
    digest=sha(p);s=p.stat()
    return {'path':str(p.relative_to(R)),'sha256':digest,'stat':{k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}}
def registration():
    if sha(A/'registration.json')!=REG_SHA:raise Red('REGISTRATION_CHANGED')
    return read(A/'registration.json')
def verify_sources():
    r=registration()
    if sha(R/r['order']['path'])!=r['order']['sha256']:raise Red('ORDER_CHANGED')
    for p,h in {**r['source_sha256'],**r['input_sha256']}.items():
        if sha(R/p)!=h:raise Red('SOURCE_CHANGED: '+p)
    return r
def tokens():
    import numpy as np
    p=registration()['protocol']
    if sha(A/'tokens.npy')!=p['tokens_file_sha256']:raise Red('TOKEN_FILE_CHANGED')
    ids=np.load(A/'tokens.npy',allow_pickle=False)
    if ids.dtype.str!='<i8' or ids.ndim!=1 or len(ids)!=p['token_count'] or len(ids)<32800 or (ids<0).any() or (ids>=262144).any() or hashlib.sha256(ids.tobytes()).hexdigest()!=p['token_sha256']:raise Red('TOKEN_STREAM_CHANGED')
    return ids
def verify_weight():
    w=registration()['weight'];s=Path(w['path']).stat()
    if {k:getattr(s,k) for k in w['stat']}!=w['stat']:raise Red('WEIGHT_STAT_CHANGED_SINCE_FULL_SHA')
def build_check():
    m=read(BUILD/'manifest.json')
    if m['registration_sha256']!=REG_SHA:raise Red('BUILD_REGISTRATION')
    for p,h in m['files'].items():
        if sha(R/p)!=h:raise Red('BUILD_CHANGED: '+p)
    return m
def load_runtime():
    build_check();sys.dont_write_bytecode=True
    sys.path[:0]=[str(BUILD),str(R/'tensor_cuda')]
    import tensor_cuda as tc
    if Path(tc.__file__).resolve().parent!=R/'tensor_cuda/tensor_cuda' or Path(tc._C.__file__).resolve().parent!=BUILD:raise Red('WRONG_ENGINE_CHECKOUT')
    return tc
def job_path(name):return A/'jobs'/f'{name}.json'
# Per-kind import closure: report-only edits never invalidate model receipts.
CLOSURES={
 'kernel':['gpu'], 'capture':['gpu','model'], 'trial':['gpu','model'],
 'freeze':['gpu'], 'margin':['gpu','metrics'], 'margin_summary':['gpu','metrics'],
 'eq':['gpu','metrics'], 'ppl':['gpu','model'], 'ppl_summary':['gpu'], 'exactness':['gpu'],
 'ceiling':['gpu','model'], 'decode':['gpu','model']}
def fingerprint(cell):
    names=['common','registry']+CLOSURES[cell['kind']]
    paths=[R/f'scripts/apa_sp4g_{n}.py' for n in sorted(set(names))]
    paths += [R/'scripts/apa_sp4g_lead_gpu.sh',A/'registration.json',BUILD/'manifest.json']
    return {str(p.relative_to(R)):sha(p) for p in paths}
def require_pass(name,cache=None):
    from apa_sp4g_registry import by_id
    cache=(VALIDATION.get() if VALIDATION.get() is not None else {}) if cache is None else cache
    if name in cache:return cache[name]
    j=read(job_path(name));c=by_id()[name]
    if j.get('status')!='PASS' or j.get('cell')!=c or j.get('registration_sha256')!=REG_SHA or False:raise Red('STALE_OR_RED_RECEIPT: '+name)
    expected={}
    for d in c['depends']:require_pass(d,cache);expected[d]=sha(job_path(d))
    if j.get('dependencies')!=expected:raise Red('DEPENDENCY_CHANGED: '+name)
    for f in j.get('result',{}).get('files',[]):
        # SP3 a4: rehash at creation; exact inode/size/mtime/ctime identity
        # thereafter avoids rereading >70GB for every DAG traversal.
        if 'stat' in f:
            s=(R/f['path']).stat()
            if {k:getattr(s,k) for k in f['stat']}!=f['stat']:raise Red('PAYLOAD_STAT_CHANGED: '+f['path'])
        elif sha(R/f['path'])!=f['sha256']:raise Red('PAYLOAD_CHANGED: '+f['path'])
    cache[name]=j;return j
