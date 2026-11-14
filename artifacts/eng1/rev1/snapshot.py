# Prior art: SHA-256, NIST FIPS 180-4 (2015); Python os.walk. Taken: byte identity inventory; ours: scoped exclusion.
import os, hashlib, json, sys
from pathlib import Path
root=Path.cwd()
out={}
for base, dirs, names in os.walk(root, followlinks=False):
    b=Path(base)
    dirs[:]=sorted(d for d in dirs if (b/d).relative_to(root).as_posix() != 'artifacts/eng1/rev1')
    for name in sorted(names):
        p=b/name; rel=p.relative_to(root).as_posix()
        if p.is_symlink():
            out[rel]={'symlink':os.readlink(p)}
        elif p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
            out[rel]=h.hexdigest()
Path(sys.argv[1]).write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print(json.dumps({'files':len(out),'map':sys.argv[1]}))
