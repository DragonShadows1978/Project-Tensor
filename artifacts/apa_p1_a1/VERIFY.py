"""Final exact source, staging and scope checks, independent of GPU execution.
Prior art: P1/DS1 source snapshots and schema/dispatch closure (2026),
SHA-256 (NIST, 2015); no new selection or allocation algorithm.
"""
from pathlib import Path
import os,re,ast,json,hashlib,difflib,collections,shlex,importlib.util
A=Path('artifacts/apa_p1_a1');D=Path('../tensor_fork_apa_wt_apa2p');root=Path.cwd()
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=load(A/'REGISTRATION.json');digest=sha(A/'REGISTRATION.json');E=load(A/'EXPECTED.json')
def body(text,name):
 start=text.index(name+'(');op=text.index('{',start);level=0
 for m in re.finditer(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27|[{}]',text[op:]):
  if m.group()=='{':level+=1
  elif m.group()=='}':
   level-=1
   if level==0:return start,op+m.end()
 raise ValueError(name)
def region(s,fixed=False):
 b=s.index('  // APA-DS1 amendment A2') if fixed else s.index('  std::vector<int64_t> sizes=')
 e=s.index('\n',s.index('throw std::runtime_error(',s.index('if(cursor',b)))
 return b,e
def normal(s,fixed=False):
 b,e=region(s,fixed)
 return (s[:b]+'ARENA'+s[e:]).replace('Every slice is 8-byte aligned for the wider FP32-output accumulator.','Every slice is 16-byte aligned, also covering the FP64 accumulator.')
def differences(a,b):return [s for s in difflib.ndiff(a.splitlines(),b.splitlines()) if s[:2] in ('- ','+ ')]
funcs={'eb2b':'apa_eb2b_run','eb3':'apa_eb3_run','eb4':'apa_eb4_run','launcher':'apa_spq1_core','spq4':'apa_spq4_tc','spq6':'apa_spq6_tq4_tc'}
rows=[];diff=[]
for n,fn in funcs.items():
 p=Path('tensor_cuda/src/apa_spo')/(n+'.cuh');before=(A/'source_before'/p).read_text();after=p.read_text();lab=(D/p).read_text();oldlab=(D/'artifacts/apa_ds1_a2/source_before'/p).read_text()
 bi,be=body(before,fn);ai,ae=body(after,fn);li,le=body(lab,fn);oi,oe=body(oldlab,fn)
 x,y=region(after,True);lx,ly=region(lab,True);ox,oy=region(oldlab);bx,by=region(before)
 pre=before[bx:by]==oldlab[ox:oy];fixed=after[x:y]==lab[lx:ly]
 unchanged=before[:bi]==after[:ai] and before[be:]==after[ae:]
 residual_same=differences(normal(oldlab[oi:oe]),normal(before[bi:be]))==differences(normal(lab[li:le],True),normal(after[ai:ae],True))
 full=lab[li:le]==after[ai:ae]
 (A/(n+'_function_A3.diff')).write_text(''.join(difflib.unified_diff(lab[li:le].splitlines(True),after[ai:ae].splitlines(True),fromfile='A3/'+fn,tofile='A1/'+fn)))
 diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='pre_A1/'+str(p),tofile='A1/'+str(p)))
 rows.append(dict(function=fn,file=str(p),pre_arena_matches_pre_A2_lab=pre,fixed_arena_matches_A3=fixed,full_function_equal=full,device_and_outside_function_unchanged=unchanged,residual_identical_to_preexisting=residual_same,residual='none' if full else 'DS1 logical-length/pitch ABI, shape guards, prefetch/launch stride arguments; exact residual predates this amendment.'))
oldp=A/'source_before/artifacts/apa_p1/gpu_driver.py';newp=Path('artifacts/apa_p1/gpu_driver.py')
oldtree=ast.parse(oldp.read_text());newtree=ast.parse(newp.read_text())
def statements(tree):
 out={}
 for i,x in enumerate(tree.body):
  if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):key=x.name
  elif isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BITWISE_CASES' for t in x.targets):key='BITWISE_CASES'
  else:key=ast.dump(x,include_attributes=False)
  out[key]=ast.dump(x,include_attributes=False)
 return out
old,new=statements(oldtree),statements(newtree);changed=[k for k in set(old)|set(new) if old.get(k)!=new.get(k)]
driver_scope=set(changed)=={'regression','BITWISE_CASES'}
cases=next(x.value for x in newtree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BITWISE_CASES' for t in x.targets));cases=ast.literal_eval(cases)
case_match=[(c['id'],c['B'],c['H'],c['KV'],c['S']) for c in cases]==[(c['id'],c['B'],c['H'],c['KV'],c['S']) for c in R['cases']]
diff.extend(difflib.unified_diff(oldp.read_text().splitlines(True),newp.read_text().splitlines(True),fromfile='pre_A1/'+str(newp),tofile='A1/'+str(newp)))
(A/'TRACKED.diff').write_text(''.join(diff))
g1=dict(status='PASS' if driver_scope and case_match and all(r['pre_arena_matches_pre_A2_lab'] and r['fixed_arena_matches_A3'] and r['device_and_outside_function_unchanged'] and r['residual_identical_to_preexisting'] for r in rows) else 'FAIL',sites=rows,driver_changed_AST_nodes=sorted(changed),driver_scope=driver_scope,case_match=case_match,method='Pre-edit snapshots, no git invoked; complete source diff and full-function comparisons.')
save(A/'FUNCTION_COMPARISON.json',g1)

# Static dispatch closure and exact as-written CPU receipt matching.
import sys
sys.path.insert(0,str(A))
from RUN import parse
from schema import fields,payload_bytes
commands=load(A/'STAGED_LINES.json')['commands'];legs=load(A/'LEGS.json')['legs'];labels=[c['label'] for c in commands]
coverage=collections.Counter(l for leg in legs for l in leg['labels'])
stage=(A/'STAGED_GPU_LINES.sh').read_text();stage_lines=[s for s in stage.splitlines() if s.startswith('run_line ')]
assert stage_lines==[c['line'] for c in commands]
runtime=ast.parse((A/'RUN.py').read_text());worker=ast.parse((A/'gpu_worker.py').read_text())
runtime_functions={n.name for n in runtime.body if isinstance(n,ast.FunctionDef)}
assert {'bitwise','byte_check','original_driver','cpu_rehearsal','compare_bitwise'}<=runtime_functions
assert any(isinstance(n,ast.Name) and n.id=='numerical' for n in ast.walk(worker)) and any(isinstance(n,ast.FunctionDef) and n.name=='allocated' for n in worker.body)
plan_rows=[]
for c in commands:
 args=parse(c['args']+['--out','artifacts/apa_p1_a1/unused_static_output'])
 receipt=load(A/'cpu_staged_fixed'/(c['label']+'.exit.json'))
 tokens=shlex.split(c['line']);actual=tokens[2:-3] # remove run_line/label and || exit 2
 actual=[s.replace('$RUN','artifacts/apa_p1_a1/cpu_staged_fixed') for s in actual]
 assert receipt['argv']==actual,(c['label'],receipt['argv'],actual)
 cpu=load(A/'cpu_staged_fixed'/c['label']/'CPU_REHEARSAL.json')
 assert receipt['exit']==0 and cpu['status']=='PASS_CPU_PLAN_ONLY' and cpu['device_calls']==0
 assert cpu['registration_sha256']==digest and c['header'].startswith('# registration_sha256='+digest)
 row=dict(label=c['label'],mode=args.mode,exact_line_executed_on_CPU=True,exit=receipt['exit'],writes=c['writes'])
 if args.mode=='bitwise':
  case=next(x for x in R['cases'] if x['id']==args.case)
  row.update(shape=[case[x] for x in ('B','H','KV','S')],retained_fields=len(fields(case)),raw_payload_bytes_two_arms=2*payload_bytes(case))
 plan_rows.append(row)
regmtime=(A/'REGISTRATION.json').stat().st_mtime_ns
mtime_ok=all((A/p).stat().st_mtime_ns>regmtime for p in ['STAGED_GPU_LINES.sh','STAGED_LINES.json','LEGS.json'])
pins_ok=all(sha(root/p)==h for p,h in E['code_sha256'].items())
assert E['registration_sha256']==digest
g4=dict(status='PASS' if mtime_ok and pins_ok and coverage==collections.Counter(labels) and all(l['sum_max_seconds']<=1240 and l['cooldown_before_seconds']==(600 if l['leg']>1 else 0) for l in legs) and load(A/'L48_READBACK.json')['status']=='PASS' else 'FAIL',lines=len(labels),bitwise=13,bytes=1,regression=44,legs=len(legs),max_leg_sum_seconds=max(l['sum_max_seconds'] for l in legs),cooldown_seconds=600,registration_mtime_precedes_staging=mtime_ok,registration_sha256=digest,source_pins_match=pins_ok,static_plan_handler_closure=True,as_written_CPU_rows=plan_rows,L48=load(A/'L48_READBACK.json'))
save(A/'STAGING_VERIFICATION.json',g4)

# Filesystem census covers all existing files, including frozen GPU1 receipts.
before=load(A/'SCOPE_BEFORE.json');now={}
for base,dirs,names in os.walk('.'):
 dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','.pytest_cache') and not (Path(base)/d).is_symlink() and str(Path(base)/d)!='artifacts/apa_p1_a1']
 for name in names:
  p=Path(base)/name
  if p.is_symlink():now[str(p)]={'symlink':os.readlink(p)}
  else:s=p.stat();now[str(p)]={'size':s.st_size,'mtime_ns':s.st_mtime_ns}
changes=sorted(p for p in set(before)|set(now) if before.get(p)!=now.get(p))
allowed={str(Path('tensor_cuda/src/apa_spo')/(n+'.cuh')) for n in funcs}|{'artifacts/apa_p1/gpu_driver.py'}
outside=[p for p in changes if p not in allowed and not p.startswith('logs/')]
baselines_held=all(sha(root/p)==h for p,h in R['reference']['A3_source_sha256'].items()) and sha(root/R['reference']['lab_module'])==R['reference']['lab_sha256'] and sha(root/R['reference']['pre_A1_module'])==R['reference']['pre_A1_sha256']
g5=dict(status='PASS' if not outside and baselines_held else 'FAIL',changed_existing_or_new_paths_outside_A1=changes,unauthorized_paths=outside,readonly_A3_sources_and_modules_held=baselines_held,existing_files_censused=len(before),gpu1_receipts_stat_unchanged=not any(p.startswith('artifacts/apa_p1/gpu_20261103T062254Z/') for p in changes),execution='CUDA_VISIBLE_DEVICES empty in all seat commands; CPU shim never contacted real forge; constructed readback asserts no CUDA synchronization.',moves=[],deletes=[],leave_and_list=['artifacts/apa_p1_a1/','/tmp/apa_p1_a1_build/','/tmp/apa_p1_a1_g3/','/tmp/apa_p1_a1_lead/','/tmp/apa_p1_a1_rescore/'])
save(A/'SCOPE.json',g5)
sass=load(A/'SASS_COMPARISON.json');build=load(A/'BUILD.json');g3cpu=load(A/'G3_COMPARISON.json');g2=load(A/'SCORER_TEST.json')
gates={'G1':g1['status'],'G2':g2['status'],'G3':'PASS' if sass['status']==build['status']==g3cpu['status']=='PASS' else 'FAIL','G4':g4['status'],'G5':g5['status']}
save(A/'GATES.json',dict(verdict='ACCEPT_CPU' if all(v=='PASS' for v in gates.values()) else 'RED',gates=gates,predictions='P149-P151 unscored pending lead GPU run'))
print(json.dumps(dict(gates=gates,source_sites=len(rows),scope_changes=changes,unauthorized=outside,staged_lines=len(labels),legs=len(legs)),indent=2))
raise SystemExit(0 if all(v=='PASS' for v in gates.values()) else 1)
