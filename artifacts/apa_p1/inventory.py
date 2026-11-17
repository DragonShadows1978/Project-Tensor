# Prior art: SHA-256 file census (NIST 2015); no new algorithm.
from pathlib import Path
import json,hashlib,os
A=Path('artifacts/apa_p1');root=Path.cwd()
headers=[Path(r['path']) for r in json.loads((A/'g1.json').read_text())['headers']]
seams=[Path(r['path']) for r in json.loads((A/'g1.json').read_text())['hunks']]
module=Path(json.loads((A/'build.json').read_text())['module'])
entries={}
def add(p,group):
    if p.is_symlink():
        kind='symlink';size=p.lstat().st_size;digest=None;target=os.readlink(p)
    elif p.is_file():
        kind='file';size=p.stat().st_size;target=None
        if p.name in ('FILES_WRITTEN.md','FILES_WRITTEN.json'):digest='SELF_INVENTORY';size=None
        else:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            digest=h.hexdigest()
    else:return
    name=str(p.relative_to(root)) if p.is_absolute() and p.is_relative_to(root) else str(p)
    entries[name]=dict(path=name,group=group,kind=kind,bytes=size,sha256=digest,target=target)
for p in headers:add(p,'D1 header')
for p in seams:add(p,'D2 seam')
add(module,'package module')
for directory,group in [(Path('tensor_cuda/build-p1'),'build output'),(A,'artifact'),(Path('/tmp/apa_p1_seat'),'retained scratch')]:
    for p in sorted(directory.rglob('*')):add(p,group)
for name in ('FILES_WRITTEN.md','FILES_WRITTEN.json'):
    p=A/name
    if not p.exists():p.write_text('inventory pending\n')
    add(p,'artifact')
rows=sorted(entries.values(),key=lambda r:r['path'])
(A/'FILES_WRITTEN.json').write_text(json.dumps(dict(note='Complete seat-created/changed file and symlink inventory; directories implicit. No deletes/moves. Inventory files list themselves without recursive hashes. Excludes pre-existing dispatch-managed logs, which the seat never directly edits.',files=rows),indent=2)+'\n')
with (A/'FILES_WRITTEN.md').open('w') as f:
    f.write('# Every file written or retained by APA-P1\n\nNo moves or deletions. All scratch/build outputs are left in place. Machine-readable sizes/hashes: FILES_WRITTEN.json. Inventory files self-list without recursive hashes. Symlinks are listed without following them. No direct writes to logs/.\n\n')
    for group in sorted({r['group'] for r in rows}):
        sub=[r for r in rows if r['group']==group];f.write(f'## {group} ({len(sub)})\n\n')
        for r in sub:f.write(f"- `{r['path']}`"+(f" -> `{r['target']}`" if r['kind']=='symlink' else '')+'\n')
print('Inventoried',len(rows),'files/symlinks:',{g:sum(r['group']==g for r in rows) for g in sorted({r['group'] for r in rows})})
