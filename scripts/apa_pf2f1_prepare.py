"""Freeze PF2-F1 registration before gates. Prior art: PF2 registration (2026), inherited.
No new algorithm; source/stat census is ordinary filesystem bookkeeping.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_pf2f1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def allowed(p):
    s = str(p)
    return (s in ('tensor_cuda/src/apa_spo/packed_prefill.cuh', 'scripts/apa_p1pf2_g45.py',
                  'tensor_cuda/tests/test_apa_pf2_nonfinite.py') or
            s.startswith(('scripts/apa_pf2f1_', 'artifacts/apa_pf2f1/', 'logs/apa_pf2f1_')))


def census():
    rows = {}
    for parent, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.pytest_cache') and
                   not allowed((Path(parent) / d / '_').relative_to(ROOT))]
        for name in files:
            p = Path(parent) / name
            rel = str(p.relative_to(ROOT))
            if allowed(rel) or p.suffix == '.pyc':
                continue
            s = p.lstat()
            rows[rel] = [s.st_size, s.st_mtime_ns, s.st_ino]
    return rows


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ['TMPDIR'].startswith('/tmp/pf2f1_')
    ART.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / 'scripts'))
    import apa_p1pf2_gpu as base
    assert len(base.g1_cases()) == 4848
    reg = dict(order='orders/APA_PF2_F1_REV1_FIXES.md', registered_unix=time.time(),
               order_sha256=sha(ROOT / 'orders/APA_PF2_F1_REV1_FIXES.md'),
               mechanism='PF2R1 F01/F02/F04/F05; docs/APA.md section 8 clauses 2/3',
               G1=dict(cases=4848, grid='apa_p1pf2_gpu.g1_cases, same indices/seeds/call shapes',
                       comparison='shape/dtype then same-width unsigned views, outputs and masks; zero differences',
                       delta='pinned packed delta at epsilon .01', reference_sha256='c3c27f8651d82390aac89d62ef704e20dade16d38e9ac8f6cf0e3fb456f4f831',
                       split='batches by bounded estimated kernel work; single-case lines for largest shapes; full calls, no substituted L=1 rows'),
               G2=dict(sink=16, norm=16, fixtures='REV1 repro/sink_nonfinite.py and norm_nonfinite.py; same constructed operands',
                       oracle='packed decode row carry; all affected prefill rows non-finite; NaN norm control all NaN'),
               G3='Compile and call the actual inline index helper on CPU for all 32 lanes at S=INT_MAX and INT_MAX-31, including successor additions, UBSan; also enumerate small tails',
               G4=dict(combinations=28, cases='dtype x diagnostics x {CAP64 SEQ, CAP128/256/512 SEQ and lane}',
                       comparison='all mask bytes equal packed decode row oracle; every output row within inherited exact per-dtype Amendment-1 tau'),
               G5=dict(mutants=['pf1m01_sink_guard_positive', 'pf1m02_norm_finite_only', 'pf1m03_index_int32'],
                       expected=['G2 sink assertion failure with clean F1 control', 'G2 norm assertion failure with clean F1 control', 'G3 UBSan or arithmetic assertion failure with clean F1 control'],
                       incremental=True, cpu='index mutant only; sink/norm pending GPU'),
               G6=dict(sass='PF2 mapping patterns only; changed kernels must all be spq_pf::prefill; no additions/removals',
                       refusal='all PF2 G4 rows, corrected visible-rows row and separate exact-K row',
                       memory='all 24 PF2 G5 rows, split by shape; pooling off/on, diagnostics off/on'),
               G7='stat census outside write set plus product hashes, git metadata census, all commands CPU-hidden, no deletes, no lease',
               environment=dict(CUDA_VISIBLE_DEVICES='', TMPDIR='/tmp/pf2f1_*', PYTHONDONTWRITEBYTECODE='1',
                                threads=1, torch_threads=1, torch_interop_threads=1),
               rails=dict(mem_available_GB=12, disk_free_GB=40, build_timeout_s=3600, jobs=3, nice=19,
                          gpu_min_s=30, gpu_max_s=540, gpu_inner_timeout_s=500, cooldown_s=600),
               L56=dict(own_script_repairs_max=3, tool_slips='reported under Deviations'),
               acceptance='CPU G3/G5-index/G7 pass plus lead GPU G1/G2/G4/G5/G6 pass; no promotion',
               prior_art='IEEE 754-2008; P1-F1/F2 zero-only guard, P1-F3 NaN norm sentinel (Project-Tensor 2026); ISO C++17 arithmetic widening / CERT INT32-C (unverified - lead to check); existing PF2 harness and SASS mappings. No new numerical algorithm.')
    with (ART / 'REGISTRATION.json').open('x') as f:
        json.dump(reg, f, indent=2); f.write('\n')
    (ART / 'REGISTRATION.sha256').write_text(sha(ART / 'REGISTRATION.json') + '\n')
    with (ART / 'SCOPE_BEFORE.json').open('x') as f:
        json.dump(census(), f, indent=1); f.write('\n')
    originals = ART / 'originals'
    originals.mkdir()
    for p in ('tensor_cuda/src/apa_spo/packed_prefill.cuh', 'scripts/apa_p1pf2_g45.py'):
        (originals / Path(p).name).write_bytes((ROOT / p).read_bytes())
    (ART / 'LEDGER.md').write_text('# APA-PF2-F1 ledger\n\n'
        '- Rules and mechanism read before changes; findings SHA ' + sha(ROOT.parent / 'tensor_fork_apa_wt_pf2rev1/artifacts/pf2_rev1/findings.json') + '.\n'
        '- L01: cmake 3.28.3; default nvcc 12.0.140; pytest 9.0.2; Python 3.12.3; forge dry-run exit 78, no lease. Build uses PF compiler /usr/local/cuda-12.6/bin/nvcc, version checked before build.\n'
        '- Registration sealed before gates. No git commands, no subagents, no GPU.\n'
        '- Prior art: IEEE 754-2008 non-finite propagation; existing Project-Tensor packed decode denominator guard (2026) and P1-F3 norm sentinel (2026), reused at both access sites. C++17/CERT INT32-C widening, ordinary checked index arithmetic; unverified - lead to check those terms. No new numerical algorithm. PF2/P1 build, spy, memory, oracle and SASS harnesses reused; CMake (Kitware 2000), make (Feldman 1979), xUnit precondition isolation (Beck 2002, unverified - lead to check).\n')
    print('REGISTERED', sha(ART / 'REGISTRATION.json'), 'grid', len(base.g1_cases()))


if __name__ == '__main__':
    main()
