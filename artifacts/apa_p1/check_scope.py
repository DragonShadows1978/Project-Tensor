# Prior art: SHA-256 (NIST FIPS 180-4, 2015) byte identity; Git diff (2005).
# Taken: exact checks. Ours: order-specific census; no new algorithm.
from pathlib import Path
import subprocess,hashlib,json,difflib
A=Path('artifacts/apa_p1')
def git(*args): return subprocess.check_output(['git',*args])
headers=json.loads((A/'header_identity_seed.json').read_text())
for row in headers:
    p=row['path']; ref=git('show','aaa9b4ea:'+p); got=Path(p).read_bytes()
    row.update(reference_sha256=hashlib.sha256(ref).hexdigest(),working_sha256=hashlib.sha256(got).hexdigest(),identical=ref==got)
hunks=json.loads((A/'hunk_map.json').read_text())
for row in hunks:
    p=row['path']; ref=git('show','0e1dea3:'+p); got=Path(p).read_bytes()
    row['production_bytes_unchanged']=got.startswith(ref)
    row['removed_lines']=sum(x.startswith('-') and not x.startswith('---') for x in difflib.unified_diff(ref.decode().splitlines(),got.decode().splitlines()))
tracked=git('diff','--name-only','0e1dea3','--','tensor_cuda/').decode().splitlines()
new_actual=sorted(str(p) for p in Path('tensor_cuda/src/apa_spo').iterdir())
stat=git('diff','--stat','0e1dea3','--','tensor_cuda/').decode()
diff=git('diff','0e1dea3','--','tensor_cuda/').decode()
(A/'g1_tracked.diff').write_text(diff); (A/'g1_diff_stat.txt').write_text(stat)
result=dict(status='PASS' if all(x['identical'] for x in headers) and all(x['production_bytes_unchanged'] and x['removed_lines']==0 for x in hunks) and sorted(tracked)==sorted(x['path'] for x in hunks) and new_actual==sorted(x['path'] for x in headers) else 'FAIL',headers=headers,hunks=hunks,tracked_engine_paths=tracked,untracked_headers=new_actual,removed_lines=sum(x['removed_lines'] for x in hunks),note='Committed order is the only pre-existing non-engine delta from production. Evidence/build outputs are authorized and outside engine source census.')
(A/'g1.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'],len(headers),'identical headers;',len(hunks),'additive seams;',result['removed_lines'],'removed lines')
print(stat)
assert result['status']=='PASS'
