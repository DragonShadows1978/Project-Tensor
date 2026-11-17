"""APA-P1-F2 S5 mutants: registration, staging and the CPU (S4) runs.

  write              : write artifacts/apa_p1f2/mutants/MUTANTS.json once (refuses if it exists), before any mutant
  stage <name>       : copy this worktree's tensor_cuda/{src,include,tensor_cuda,CMakeLists.txt} with timestamps
                       (cp -a) into mutants/<name>/tree and apply the one registered change (the old text must
                       occur exactly once); MUTANT.json records the file and both sha256s
  cpu <name>         : run tensor_cuda/tests/test_apa_p1f2_cpu.py with APA_P1F2_SRC = the mutant arm's package
                       (Build F2's module + the mutant's Python), per-ID receipt; killed iff the named test fails
Device and binding mutants are rebuilt by scripts/apa_p1f2_build.py mutant <name> (incremental, in a copy of
Build F2's tree) and run on GPU by scripts/apa_p1f2_gpu.py mut. Expected failures come from the order's rules
(L47), written here before any mutant runs (L41).
Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); F1's apa_p1f1_mutants.py staging pattern
(Project-Tensor 2026). Plumbing; no new technique. CPU only.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_p1f2'
MUT = ART / 'mutants'
S4 = 'tensor_cuda/tests/test_apa_p1f2_cpu.py'

MUTANTS = [
    dict(name='m01_margin_pin_unchecked', kind='python', file='tensor_cuda/tensor_cuda/apa_packed.py',
         rule='R1', change='lookup and packed_attention accept any PackedRegisteredMargins: check_pinned_margins returns at once',
         old='    if type(margins) is not PackedRegisteredMargins:\n        raise ValueError("apa_sp2: margins must be the pinned',
         new='    return\n    if type(margins) is not PackedRegisteredMargins:\n        raise ValueError("apa_sp2: margins must be the pinned',
         cpu=dict(line='S4', expected_failure=S4 + '::test_r1_hand_built_table_refused_at_lookup_and_at_attention_before_any_device_call[wrong_sha_pinned_entries]',
                  assertion='pytest.raises(ValueError, match="pinned"): DID NOT RAISE at lookup')),
    dict(name='m02_tq4_inf_norm_kept', kind='device', file='tensor_cuda/src/apa_spo/packed.cuh',
         rule='R2', change="the public route's TQ4 quantizer (spq1::prepare) stores the infinite norm unchanged",
         old='norms[row]=nonfinite?__int_as_float(0x7fc00000):norm;', new='norms[row]=norm;',
         gpu=dict(line='LF2', cases=['k_posinf_p1', 'k_neginf_p1', 'r01_k_posinf', 'r01_k_neginf'],
                  expected_failure='G2b containment (never hides): r2_tq4 bfloat16 k_posinf_p1 vs dense_entry',
                  assertion='LF2 G2b: the tq4 entry hides dense row 6 (+Inf at k[0,1,700,64]; F1 LF2 k_inf_more)')),
    dict(name='m03_capacity_unbounded', kind='python', file='tensor_cuda/tensor_cuda/apa_packed.py',
         rule='R3', change="R3's bound removed from PackedSpKeys (the constructor's check_native_range call)",
         old='        check_native_range(B, KV, S, D, capacity, S)\n', new='        pass\n',
         cpu=dict(line='S4', expected_failure=S4 + '::test_r3_packed_capacity_at_the_bound_is_accepted_one_past_is_refused_before_allocation',
                  assertion='pytest.raises(ValueError, match="capacity") for capacity 2**31: DID NOT RAISE (the stub records the allocation)')),
    dict(name='m04_raw_extent_unchecked', kind='binding', file='tensor_cuda/src/apa_spo/packed_launcher.cuh',
         rule='R4', change="R4's positive-extent check removed from packed_rotate (apa_spq1_rotate)",
         old='   apa_spo_require_positive_extents("apa_spq1_rotate (packed_rotate)","q",q);\n', new='',
         gpu=dict(line='LF3', rows=['rotate_q_B0'],
                  expected_failure='LF3 raw helper zero extent raises ValueError, no launch: rotate_q_B0',
                  assertion='LF3: B=0 packed_rotate reaches a zero-grid launch (RuntimeError CUDA invalid configuration, 1 launch) instead of ValueError with 0 launches')),
    dict(name='m05_eb3_timeline_g_late', kind='device', file='tensor_cuda/src/apa_spo/eb3.cuh',
         rule='R4', change="R4's EB3 timeline/G pre-check removed",
         old='  if(timeline && G!=2 && G!=4)\n    throw std::invalid_argument(', new='  if(false)\n    throw std::invalid_argument(',
         gpu=dict(line='LF3', rows=['eb3_timeline_G3', 'eb3_timeline_G5', 'eb3_timeline_G6', 'eb3_timeline_G7', 'eb3_timeline_G8'],
                  expected_failure='LF3 EB3 timeline=True G=3 raises before any launch',
                  assertion='LF3 launch count: rotation and bulk launch (2) before the refusal, which is a RuntimeError, not the pre-launch ValueError')),
    dict(name='m06_factory_capacity_dropped', kind='python', file='tensor_cuda/tensor_cuda/apa_sp2.py',
         rule='R8', change='quantize_sp2_keys ignores capacity (always PackedSpKeys(k))',
         old='packed = PackedSpKeys(k) if capacity is None else PackedSpKeys(k, capacity=capacity)',
         new='packed = PackedSpKeys(k)',
         cpu=dict(line='S4', expected_failure=S4 + '::test_r8_capacity_is_forwarded_to_packed_keys',
                  assertion='assert p.capacity == 164 (got 100: the reservation is dropped)'),
         gpu=dict(line='LF4', expected_failure='LF4 D64: factory capacity forwarded',
                  assertion='LF4: obj.capacity == N (256) fails; every later append is refused')),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write():
    MUT.mkdir(parents=True, exist_ok=True)
    path = MUT / 'MUTANTS.json'
    data = dict(order='orders/APA_P1F2_REV2_FIXES.md S5', written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                note='Written before any mutant is staged, built or run (L41). Each mutant is one change in a patched copy. '
                     'Python-only mutants run against Build F2\'s module (its arm with the mutant Python); device and '
                     'binding mutants are rebuilt incrementally in a copy of Build F2\'s tree.',
                mutants=[dict(m, runs_on='gpu' if 'gpu' in m else 'cpu') for m in MUTANTS])
    with path.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    print(path, sha(path))


def stage(name):
    spec = {m['name']: m for m in json.loads((MUT / 'MUTANTS.json').read_text())['mutants']}[name]
    mdir = MUT / name
    tree = mdir / 'tree/tensor_cuda'
    tree.mkdir(parents=True)
    for part in ('src', 'include', 'tensor_cuda', 'CMakeLists.txt'):
        subprocess.run(['timeout', '300s', 'cp', '-a', str(ROOT / 'tensor_cuda' / part), str(tree / part)], check=True)
    target = mdir / 'tree' / spec['file']
    before = sha(target)
    text = target.read_text()
    assert text.count(spec['old']) == 1, (name, 'old text must occur exactly once')
    target.write_text(text.replace(spec['old'], spec['new']))
    rec = dict(name=name, kind=spec['kind'], file=spec['file'], sha256_before=before, sha256_after=sha(target),
               source_sha256_in_worktree=sha(ROOT / spec['file']), staged_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    assert rec['sha256_before'] == rec['source_sha256_in_worktree'] != rec['sha256_after']
    (mdir / 'MUTANT.json').write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps(rec, indent=2))


def cpu(name):
    spec = {m['name']: m for m in json.loads((MUT / 'MUTANTS.json').read_text())['mutants']}[name]
    arm = json.loads((MUT / name / 'arm/ARM.json').read_text())
    package = ROOT / arm['package']
    out = MUT / name / 'cpu'
    out.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               APA_P1F2_SRC=str(package), TMPDIR='/tmp/apap1f2_mut',
               PYTHONPATH=str(package.parent) + ':' + str(ROOT / 'tensor_cuda/tests') + ':' + str(ROOT / 'artifacts/apa_p1'))
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    cmd = [sys.executable, str(ROOT / 'artifacts/apa_p1/pytest_receipt.py'), str(out / 'S4.json'), '-q', '--tb=line',
           '-p', 'no:cacheprovider', '--basetemp', f'/tmp/apap1f2_mut/bt_{name}', S4]
    with (out / 'S4.log').open('x') as f:
        code = subprocess.run(['timeout', '900s', *cmd], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    rec = json.loads((out / 'S4.json').read_text())
    failed = sorted(k for k, v in rec['reports'].items() if any(r['outcome'] == 'failed' for r in v))
    result = dict(name=name, line='S4', exit=code, package=str(package), module_sha256=arm['module_sha256'],
                  expected_failure=spec['cpu']['expected_failure'], killed=spec['cpu']['expected_failure'] in failed,
                  failed_ids=failed)
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    {'write': lambda: write(), 'stage': lambda: stage(sys.argv[2]), 'cpu': lambda: cpu(sys.argv[2])}[sys.argv[1]]()
