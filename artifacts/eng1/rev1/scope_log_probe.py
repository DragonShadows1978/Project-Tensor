# Prior art: SHA-256 prefix identity (NIST FIPS180-4, 2015). Taken: byte-level append diagnosis.
# Ours: distinguish the one changing run log from every source/test/receipt byte. No exclusion weakens L46.
import json,hashlib,datetime
from pathlib import Path
root=Path.cwd();rev=root/'artifacts/eng1/rev1'; rel='logs/eng1_rev1.log'
before=json.loads((rev/'worktree_before.json').read_text())
after=json.loads((rev/'worktree_after.json').read_text())
b=(root/rel).read_bytes();h=hashlib.sha256();matches=[]
if h.hexdigest()==before[rel]:matches.append(0)
offset=0
for line in b.splitlines(keepends=True):
    h.update(line);offset+=len(line)
    if h.hexdigest()==before[rel]:matches.append(offset)
# Do not expose the transcript; classify using only benign record names.
markers={name:(name.encode() in b) for name in ('tool_call','functions.exec','commentary','ENG1-REV1','worktree_before.json','scope_log_probe')}
r=dict(file=rel,initial_sha256=before[rel],final_snapshot_sha256=after[rel],
       current_sha256=hashlib.sha256(b).hexdigest(),current_bytes=len(b),
       initial_hash_prefix_bytes=matches,append_only_proved=bool(matches),
       added_bytes=len(b)-matches[0] if matches else None,
       transcript_markers=markers,
       all_other_worktree_paths_unchanged=all(before.get(p)==after.get(p) for p in set(before)|set(after) if p!=rel),
       status='RED: immutable whole-worktree identity gate failed; no exclusions or restoration applied',
       attribution='No review script/command opens this path for writing; it contains the active run transcript. An enclosing session logger is inferred, not directly controlled or independently attested.')
(rev/'scope_log_probe.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
