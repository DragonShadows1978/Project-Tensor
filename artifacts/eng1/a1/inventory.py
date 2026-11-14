"""All A1 writes and tool dispositions. Prior art: SHA256/NIST, ENG1 before/after maps (2026).
Taken: content identity; ours: authorized-file inventory including restored receipt paths and retained scratch.
"""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[3];A=R/'artifacts/eng1/a1';T=R/'artifacts/eng1/tool'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((A/'tool_before.json').read_text());changes=[]
reasons={'arm_origin.py':'new inherited PYTHONPATH binding and subprocess pin check; prior art PSF, NIST',
 'run_gpu_tests.py':'bind environment/preflight before pytest; separate source-reader class; audit omits CUDA API queries',
 'evidence.py':'new seal selection; validate origin receipt; A1 census, mutants and source-reader exclusion',
 'run_gpu_line.py':'use new seal identity; after-exit parent logic retained',
 'start_gpu_run.py':'bind START to new seal; order/arm checks retained',
 'score_gpu.py':'report source-reader class/outcomes separately from numerical comparison',
 'stage.py':'refresh authorized test copies; fresh M15/M16 arms; A1 ID/collection/prediction names; own seal and24 lines',
 'gpu_lines.json':'24 lines including M15/M16; candidate retains source-reader IDs; C_COPY covers new mutants',
 'STAGED_GPU_COMMANDS.sh':'regenerated24 explicit expected-exit/bounded commands',
 'source_reading_tests.json':'16 exact source readers and their paths, candidate-only comparison exclusion',
 'baseline_expected_failures_a1.json':'59 pre-GPU expected baseline failed IDs, derived arithmetic prediction'}
for p in sorted(T.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts:continue
 rel=str(p.relative_to(R));now=sha(p)
 if old.get(rel)==now:continue
 local=p.relative_to(T);reason=reasons.get(p.name)
 if local.parts[0]=='arms':
  reason='refreshed authorized ENG1 test copy' if p.name=='test_eng1.py' else 'fresh M15/M16 arm copy; candidate Python/terrain bytes and built mutant module pin'
 elif local.parts[0].startswith('audit_a1'):reason='CPU stand-in audit generated receipt (including preserved publication hard-link temporary)'
 elif p.name.startswith('ids_a1_'):reason='A1 exact test-ID census; same IDs as round1 retained where unchanged'
 elif p.name.startswith('collect_a1_'):reason='A1 collection receipt or preserved unchanged round1 collection metadata copy; authoritative ID lists named by gpu_lines.json'
 elif p.name.startswith('collect_'):reason='A1 preliminary collection receipt retained from before naming migration'
 elif p.name.startswith('staging_seal_a1'):reason='new A1 seal; preliminary seal preserved separately; original seal unchanged'
 changes.append(dict(path=rel,before_sha256=old.get(rel),after_sha256=now,disposition=reason or 'generated staging evidence; see ledger/diff'))
(A/'tool_changes.json').write_text(json.dumps(changes,indent=2)+'\n')
# Stage initially rewrote these old receipts then restored their exact original content; include in write history.
history=set()
for arm,suite,tag,name in [('baseline','eng1','','eng1'),('candidate','eng1','','eng1_candidate'),('candidate','eng1','_cpu','audit'),('candidate','eng1','_noe6','eng1_noe6'),('baseline','regression','','regression'),('candidate','regression','','regression_candidate'),('baseline','terrain','','terrain'),('candidate','terrain','','terrain_candidate')]:
 history.add('artifacts/eng1/tool/ids_'+name+'.txt');history.add('artifacts/eng1/tool/collect_'+arm+'_'+suite+tag+'.json')
paths={str(p.relative_to(R)) for p in A.rglob('*') if p.is_file()}
paths|={r['path'] for r in changes};paths|=history;paths.add('tensor_cuda/tests/test_eng1.py')
for mid in ('M15','M16'):
 paths|={str(p.relative_to(R)) for p in (R/'artifacts/eng1/mutants'/mid).rglob('*') if p.is_file()}
manifest_names={'artifacts/eng1/a1/FILES_WRITTEN.json','artifacts/eng1/a1/FILES_WRITTEN.md'}
paths|=manifest_names
files=[]
for rel in sorted(paths):
 p=R/rel
 if rel in manifest_names:
  files.append(dict(path=rel,sha256=None,note='inventory itself; self-hash omitted'))
 elif p.is_file():files.append(dict(path=rel,sha256=sha(p),bytes=p.stat().st_size,note='original receipt rewritten then restored/preserved' if rel in history else 'A1 deliverable'))
 else:raise AssertionError('inventory missing '+rel)
scratch=[]
for base in (Path('/tmp/eng1_a1_eng1'),Path('/tmp/eng1_a1_M15'),Path('/tmp/eng1_a1_M16')):
 for p in sorted(base.rglob('*')):
  if p.is_file():scratch.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
data=dict(worktree_files=files,scratch_files=scratch,tool_changes='tool_changes.json',deletions=[],retained_scratch_roots=['/tmp/eng1_a1_eng1','/tmp/eng1_a1_M15','/tmp/eng1_a1_M16'],notes=['Original round1 collection paths include write history followed by exact-byte restoration; all round1 receipt bytes retained.','Generated manifest files omit their own hash to avoid a self-reference.','Scratch is retained; /tmp is exempt from no-deletes, but this seat deleted nothing.'])
(A/'FILES_WRITTEN.json').write_text(json.dumps(data,indent=2)+'\n')
out=['# ENG1-A1 written files','','Complete content-addressed inventory: FILES_WRITTEN.json. Includes final changes, new deliverables, old receipt write/restoration history, and every retained scratch file. No deletions. Inventory hashes omit the two inventory files themselves.','','## Worktree files','','| Path | SHA256 | Note |','|---|---|---|']
for row in files:out.append('| `'+row['path']+'` | `'+str(row['sha256'])+'` | '+row['note']+' |')
out+=['','## Retained scratch files','','All three roots are under /tmp. These can be binned by the lead; no worktree artifact needs removal.','','| Path | SHA256 |','|---|---|']
for row in scratch:out.append('| `'+row['path']+'` | `'+row['sha256']+'` |')
(A/'FILES_WRITTEN.md').write_text('\n'.join(out)+'\n')
print('worktree written paths',len(files),'scratch files',len(scratch),'tool changes',len(changes))
