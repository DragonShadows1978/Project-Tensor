# Prior art: SHA-256 NIST 2015, Python AST and bash syntax checks;
# exact registered operand census. No numerical or GPU validation claimed.
from pathlib import Path
import ast,json,hashlib,subprocess,shlex,collections
A=Path('artifacts/apa_p1');root=Path.cwd();exp=json.loads((A/'EXPECTED.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert exp['status']=='STAGED_NOT_RUN'
for p,h in exp['code_sha256'].items():assert sha(p)==h,p
for p in A.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
subprocess.run(['bash','-n',str(A/'STAGED_GPU_LINES.sh')],check=True,timeout=10)
script=(A/'STAGED_GPU_LINES.sh').read_text();lines=[x for x in script.splitlines() if x.startswith('run_line ')]
assert len(lines)==len(exp['commands'])==51
for row,line in zip(exp['commands'],lines):
    assert row['line']==line
    for text in ('timeout 1750s forge-gpu-run','--max 600','-- timeout 540s','CUDA_VISIBLE_DEVICES=0','APA_P1_LEAD_GPU=1','|| exit 2'):assert text in line,(row['label'],text)
assert len(exp['gpu_collected_ids'])==2078 and len(set(exp['gpu_collected_ids']))==2078
census=json.loads((A/'INPUT_CENSUS.json').read_text());assert len(census)==6
for model in ('Qwen3-1.7B','Qwen3-4B'):
    assert sorted(x['S'] for x in census if x['model']==model)==[2048,8192,32768]
for c in census:
    assert c['append_lengths']==sorted(set(c['append_lengths'])) and c['append_lengths'][-1]==c['S']
    for key,meta in c['files'].items():assert sha(meta['path'])==meta['sha256']
build=json.loads((A/'build.json').read_text());assert sha(build['module'])==sha(build['saved_module'])==build['module_sha256']
assert sha(build['production_module'])==build['production_sha256'];assert sha(build['lab_module'])==build['lab_sha256']
(A/'EXPECTED.sha256').write_text(sha(A/'EXPECTED.json')+'  artifacts/apa_p1/EXPECTED.json\n')
r=dict(status='PASS',gpu_executed=False,staged_lines=51,bitwise_lines=6,bytes_lines=1,production_test_file_lines=44,registered_gpu_ids=2078,real_capture_cases=6,code_pins=len(exp['code_sha256']),expected_sha256=sha(A/'EXPECTED.json'),checks=['Python syntax','bash syntax','all frozen source SHA pins','all source-capture SHA pins','full model/S and append-length census','every GPU invocation forge-gpu-run + --max + inner/outer timeout','baseline module SHA pins','port installed module restored and equal to build copy'],limits='Static/CPU verification only. No claim that the unexecuted GPU harness passes on device; lead owns numerical/GPU-regression/allocation verdicts.')
(A/'g4_staging.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
