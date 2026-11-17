"""Enumerate every written file/symlink; retain all scratch, delete nothing.
Prior art: P1/DS1 file manifests (2026), SHA-256 (NIST, 2015).
Taken: per-file hashes; ours: this authorized-write census, no new algorithm.
"""
from pathlib import Path
import json,hashlib,os,datetime
A=Path('artifacts/apa_p1_a1')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
sources=[Path('tensor_cuda/src/apa_spo')/(n+'.cuh') for n in ('eb2b','eb3','eb4','launcher','spq4','spq6')]+[Path('artifacts/apa_p1/gpu_driver.py')]
roots=[A,Path('/tmp/apa_p1_a1_build'),Path('/tmp/apa_p1_a1_g3'),Path('/tmp/apa_p1_a1_lead'),Path('/tmp/apa_p1_a1_rescore')]
paths=set(sources+[Path('logs/apa_p1_a1_r1.log')])
directories=[]
for root in roots:
 if not root.exists():continue
 directories.append(str(root))
 for base,dirs,names in os.walk(root):
  for d in dirs:
   p=Path(base)/d
   if p.is_symlink():paths.add(p)
   else:directories.append(str(p))
  dirs[:]=[d for d in dirs if not (Path(base)/d).is_symlink()]
  paths.update(Path(base)/name for name in names)
paths.update([A/'FILES_WRITTEN.json',A/'FILES_WRITTEN.md'])
rows=[]
for p in sorted(paths):
 if p.name in ('FILES_WRITTEN.json','FILES_WRITTEN.md') and p.parent==A:
  rows.append(dict(path=str(p),kind='manifest_self',note='This manifest enumerates itself; no self-referential content hash.'));continue
 if p.is_symlink():rows.append(dict(path=str(p),kind='symlink',target=os.readlink(p)));continue
 if not p.exists():rows.append(dict(path=str(p),kind='absent_at_census'));continue
 s=p.stat();row=dict(path=str(p),kind='file',bytes=s.st_size,sha256=sha(p))
 if p.parts[0]=='logs':row['note']='Live authorized transcript; size/hash are a census snapshot and it continues to append.'
 rows.append(row)
result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=len(rows),total_regular_bytes=sum(r.get('bytes',0) for r in rows),directories=sorted(set(directories)),rows=rows,moves=[],deletes=[],scope='Seven edited source files, every A1 output, authorized live log and all retained authorized scratch. All file paths enumerated; symlinks are not followed.')
(A/'FILES_WRITTEN.json').write_text(json.dumps(result,indent=1)+'\n')
text=['# Every file written / retained','',f"{result['files']} file/symlink entries; {result['total_regular_bytes']} regular-file bytes at census. Nothing moved or deleted. JSON contains per-file hashes. Live transcript continues to append; manifests cannot hash themselves.",'','## Files and symlinks','']
for r in rows:
 suffix=(' -> '+r['target']) if r['kind']=='symlink' else (' ('+str(r.get('bytes',0))+' B)') if r['kind']=='file' else ' ('+r['kind']+')'
 text.append('- `'+r['path']+'`'+suffix)
text+=['','## Retained directories','']+['- `'+p+'`' for p in result['directories']]
(A/'FILES_WRITTEN.md').write_text('\n'.join(text)+'\n')
print(dict(files=result['files'],regular_bytes=result['total_regular_bytes'],directories=len(result['directories'])))
