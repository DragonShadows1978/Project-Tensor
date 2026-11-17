"""APA-P1-F3 S5 mutants: registration, staging and the CPU (S4) runs.

  write          : write artifacts/apa_p1f3/mutants/MUTANTS.json once (refuses if it exists), before any mutant
  stage <name>   : F2's stage() (scripts/apa_p1f2_mutants.py, imported, its MUT/ART pointed at this round): cp -a of
                   this worktree's tensor_cuda/{src,include,tensor_cuda,CMakeLists.txt} into mutants/<name>/tree, the
                   one registered change applied (old text must occur exactly once); MUTANT.json records both sha256s
  cpu <name>     : run tensor_cuda/tests/test_apa_p1f3_cpu.py with the mutant's Python (APA_P1F3_SRC = the mutant
                   arm's package) or the mutant's engine tree (APA_P1F3_ENGINE_ROOT = mutants/<name>/tree), per-ID
                   receipt; killed iff the named test fails
Device/host mutants are rebuilt by scripts/apa_p1f3_build.py mutant <name> (F2's incremental build_mutant, in a
copy of Build F3's tree) and run on GPU by scripts/apa_p1f3_gpu.py mut. Expected failures come from the order's
rules and REV3's receipts (L47), written here before any mutant runs (L41).
Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); F1/F2 mutant staging (Project-Tensor 2026),
imported. Plumbing; no new technique. CPU only.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_p1f2_mutants as f2m  # noqa: E402

ART = ROOT / 'artifacts/apa_p1f3'
MUT = ART / 'mutants'
S4 = 'tensor_cuda/tests/test_apa_p1f3_cpu.py'
f2m.ART, f2m.MUT = ART, MUT

MUTANTS = [
    dict(name='m01_lookup_rereads_mapping', kind='python', file='tensor_cuda/tensor_cuda/apa_packed.py', rule="R1'",
         change='lookup returns self.entries[...] again (a second read of the caller\'s mapping after the check)',
         old='        return PACKED_PINNED_ENTRIES[f"{bulk_bits}:{dim}"]\n',
         new='        return self.entries[f"{bulk_bits}:{dim}"]\n',
         cpu=dict(line='S4', env='APA_P1F3_SRC',
                  expected_failure=S4 + '::test_r1p_stateful_mapping_at_lookup_returns_the_pinned_value[64]',
                  assertion=('type(got) is float and bits(got) == bits(pin): lookup returns the third read of the '
                             'mapping, 0.0, not the pinned 1.010646104812622 (the message shows mapping item '
                             'reads = 3, the executed re-read)'))),
    dict(name='m02_overflow_norm_kept', kind='device', file='tensor_cuda/src/apa_spo/packed.cuh', rule="R2'",
         change='prepare keeps the Inf norm for an overflowing finite row (F2\'s store: NaN only for a non-finite element)',
         old='norms[row]=(nonfinite || !isfinite(norm))?__int_as_float(0x7fc00000):norm;',
         new='norms[row]=nonfinite?__int_as_float(0x7fc00000):norm;',
         gpu=dict(line='LF2', cases=['ovf_D128_qpos', 'ovf_D128_qneg', 'ovf_D64_qpos', 'ovf_D64_qneg', 'r01_ovf_qneg',
                                     'r01_ovf_qpos'],
                  expected_failure='G2b containment (never hides): public_default float32 r01_ovf_qneg vs dense_opt_out',
                  assertion=('LF2 G2b on REV3 r01 (D64, q=-ones, k[0,0,1,0]=1e20): the packed public row is finite '
                             'while the dense opt-out row 0 is non-finite, as the lead measured on Build F2 '
                             '(artifacts/apa_p1_rev3/lead_p1rev3_gpu_repro.json; F2\'s prepare store is this mutant\'s); '
                             'executed-line evidence: the stored norm of the overflow key is inf, not nan'))),
    dict(name='m03_eb2b_product_int32', kind='host', file='tensor_cuda/src/apa_spo/eb2b.cuh', rule="R3'",
         change="EB2b's B*H back to a signed int product of the narrowed extents (checked by the same named check)",
         old=('  apa_spo_require_at_most("apa_eb2b","B*H (q batch * q heads; the int row count, times 4, must fit int)",\n'
              '    apa_spo_product({q.shape[0],q.shape[1]}),kApaSpoIntMax/4);\n'
              '  int B=q.shape[0],H=q.shape[1],KVH=k.shape[1],S=k.shape[2];\n'),
         new=('  int B=q.shape[0],H=q.shape[1],KVH=k.shape[1],S=k.shape[2];\n'
              '  apa_spo_require_at_most("apa_eb2b","B*H (q batch * q heads; the int row count, times 4, must fit int)",\n'
              '    B*H,kApaSpoIntMax/4);\n'),
         kill_rule='killed iff the S4 or the LF3 named assertion fails (order S5: "S4 or LF3"); both are run and reported',
         cpu=dict(line='S4', env='APA_P1F3_ENGINE_ROOT',
                  expected_failure=S4 + '::test_r3p_eb2b_product_that_wraps_a_signed_int_raises_value_error_naming_it',
                  assertion=('host-level fake (g++ -O0 -fwrapv): 65536*65536 wraps to 0 in int, the named check '
                             'passes and the prologue returns ("none") instead of ValueError naming apa_eb2b and B*H')),
         gpu=dict(line='LF3', rows=['eb2b_BH_past_bound', 'eb2b_BH_2p32'],
                  expected_failure="LF3 R3' row eb2b_BH_2p32",
                  assertion=('LF3: B=H=65536 passes EB2b\'s check with a wrapped product; the call is refused later by '
                             'packed_rotate\'s own B*H check, so the ValueError names apa_spq1_rotate, not apa_eb2b '
                             '(executed-line evidence: that later message)'))),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write():
    MUT.mkdir(parents=True, exist_ok=True)
    path = MUT / 'MUTANTS.json'
    data = dict(order='orders/APA_P1F3_REV3_FIXES.md S5', written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                note=('Written before any mutant is staged, built or run (L41). Each mutant is one change in a patched '
                      'copy. The Python mutant runs on Build F3\'s module with the mutant Python; the device and host '
                      'mutants are rebuilt incrementally in a copy of Build F3\'s tree. Expected failures come from the '
                      'order\'s rules and REV3\'s receipts, never from this round\'s runs (L47).'),
                mutants=[dict(m, runs_on=[x for x in ('cpu', 'gpu') if x in m]) for m in MUTANTS])
    with path.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    print(path, sha(path))


def cpu(name):
    spec = {m['name']: m for m in json.loads((MUT / 'MUTANTS.json').read_text())['mutants']}[name]
    c = spec['cpu']
    f3 = json.loads((ART / 'arms/F3/ARM.json').read_text())
    if c['env'] == 'APA_P1F3_SRC':
        arm = json.loads((MUT / name / 'arm/ARM.json').read_text())
        package = ROOT / arm['package']
        extra = dict(APA_P1F3_SRC=str(package))
    else:
        package = ROOT / f3['package']
        extra = dict(APA_P1F3_ENGINE_ROOT=str(MUT / name / 'tree'))
    out = MUT / name / 'cpu'
    out.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               TMPDIR='/tmp/apap1f3_mut',
               PYTHONPATH=str(package.parent) + ':' + str(ROOT / 'tensor_cuda/tests') + ':' + str(ROOT / 'artifacts/apa_p1'),
               **extra)
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    cmd = [sys.executable, str(ROOT / 'artifacts/apa_p1/pytest_receipt.py'), str(out / 'S4.json'), '-q', '--tb=short',
           '-p', 'no:cacheprovider', '--basetemp', f'/tmp/apap1f3_mut/bt_{name}', S4]
    with (out / 'S4.log').open('x') as f:
        code = subprocess.run(['timeout', '900s', *cmd], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    rec = json.loads((out / 'S4.json').read_text())
    failed = sorted(k for k, v in rec['reports'].items() if any(r['outcome'] == 'failed' for r in v))
    messages = {k: [r['message'] for r in rec['reports'][k] if r.get('message')] for k in failed}
    result = dict(name=name, line='S4', exit=code, package=str(package), env=extra,
                  expected_failure=c['expected_failure'], killed=c['expected_failure'] in failed, failed_ids=failed,
                  executed_line_evidence=messages.get(c['expected_failure']))
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2)[:3000])


if __name__ == '__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    {'write': lambda: write(), 'stage': lambda: f2m.stage(sys.argv[2]), 'cpu': lambda: cpu(sys.argv[2])}[sys.argv[1]]()
