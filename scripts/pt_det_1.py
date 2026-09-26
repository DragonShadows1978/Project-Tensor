#!/usr/bin/env python3
"""PT-DET-1 author gates; CPU import never imports a CUDA engine.

Prior art: CC46-B/C (GRAPA 2026) identical-checkpoint replay, cruise receipt
replay, first-step gradient digests and full-precision norm records, reused.
SHA256 (NIST 2001) content receipts and POSIX flock/process deadlines, reused.
Ours: four shared-family arms, strict non-vacuous diff and required lane.
Reduction prior art: CUB/NVIDIA (2024), PyTorch (2021), Demmel/Nguyen (2013),
with the precise taken/ours boundary in tc/deterministic_embed.h.
"""
import argparse
import copy
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import re
import signal
import statistics
import subprocess
import sys
import time

import numpy as np
import pt_tf32_3_storage as storage

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/pt_det_1'
REG = ART / 'REGISTRATION.json'
REG_SHA = '0b1af9f61695c5cf14ae6b0ecedac255dfebc4f18ac9821ed250751d79d48033'
MANIFEST_NAME = 'SOURCE_MANIFEST_003.json'
CC = Path('/mnt/ForgeRealm/wt/grapa-cc46')
ARMS = ('v3_on', 'bf16_on', 'v3_off', 'bf16_off')
ROW_FIELDS = ('step', 'loss', 'gnorm', 'clip_coef', 'sample_i', 'lr', 'refine', 'clip')
# CC46's status grammar, preserving number TEXT, not rounded numeric equality.
STATUS_RE = re.compile(
    r'\| step\s+(\d+) \| loss\s+(\S+) \| refine (\S+) \| lr (\S+) \|\s+\S+ tok/s '
    r'\|\s+(\S+)s/step \| sample_i (\d+) \|.*\| gnorm (\S+) \| clip_coef (\S+) '
    r'\| clip (\S+)$')


class Blocked(RuntimeError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def create_json(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def registration():
    if sha(REG) != REG_SHA:
        raise ValueError('PT_DET_1 registration drift')
    r = json.loads(REG.read_text())
    if sha(r['order']['path']) != r['order']['sha256']:
        raise ValueError('PT_DET_1 immutable order drift')
    from pt_det_2 import registration as registration_2
    registration_2()  # Additive PT-DET-2 order, never rewrite PT-DET-1 registration.
    return r


def seal():
    registration()
    builds = sorted(ART.glob('build_*/receipt.json'))
    b = json.loads(builds[-1].read_text())
    if b['rc'] or not b['sources_unchanged_during_build']:
        raise ValueError('build did not pass')
    for p, h in b['source_pins'].items():
        if sha(ROOT / p) != h:
            raise ValueError('build source drift: ' + p)
    binary = b['binaries'][0]
    if sha(ROOT / binary['path']) != binary['sha256']:
        raise ValueError('build binary drift')
    paths = list((ROOT / 'scripts').glob('pt_det_*.py'))
    paths += list((ROOT / 'scripts').glob('pt_tf32*.py'))
    paths += list((ROOT / 'tests').glob('*pt_det_*'))
    paths += list(ART.glob('AMENDMENT_*.json'))
    art2 = ROOT / 'artifacts/pt_det_2'
    paths += list(art2.glob('AMENDMENT_*.json'))
    paths += [art2/'REGISTRATION.json', art2/'REGISTRATION.sha256',
              ROOT/'orders/PT_DET_2_GATHER_BWD.md', ART/'pt_det_1_host_contract', art2/'pt_det_2_host_contract']
    paths += [p for p in (art2/'baseline').rglob('*') if p.is_file()]
    paths += list((ROOT / 'tensor_cuda/tensor_cuda').glob('*.py'))
    paths += [ROOT / p for p in b['source_pins']]
    old_manifest = json.loads((ROOT / 'artifacts/pt_tf32_4/SOURCE_MANIFEST.json').read_text())
    paths += [ROOT / p for p in old_manifest['pins']]
    cr = json.loads((CC / 'artifacts/cc46/REGISTRATION.json').read_text())
    dependencies = {}
    for rel, expected in cr['trainer_sources_sha256'].items():
        if sha(CC / rel) != expected:
            raise ValueError('CC46 trainer source drift: ' + rel)
        dependencies[str(CC / rel)] = expected
    for p in [CC / 'scripts/cc46_trace.py', CC / 'scripts/cc46_determinism.py',
              CC / 'artifacts/cc46/REGISTRATION.json',
              CC / 'corpus/audit_reports/tokenizer_8k_char_alnum_compat.json']:
        dependencies[str(p)] = sha(p)
    for field in ('config', 'cruise_receipt'):
        item = cr['live'][field]
        if sha(item['path']) != item['sha256']:
            raise ValueError('CC46 ' + field + ' drift')
        dependencies[item['path']] = item['sha256']
    m = dict(registration_sha256=REG_SHA, binary=binary,
             pt_det_2_registration_sha256=sha(art2/'REGISTRATION.json'),
             build_receipt=str(builds[-1].relative_to(ROOT)), build_receipt_sha256=sha(builds[-1]),
             pins={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths)) if p.is_file()},
             dependencies=dependencies,
             prior_pt_det_1_manifest_sha256=sha(ART/'SOURCE_MANIFEST.json'),
             prior_pt_det_1_manifest_002_sha256=sha(ART/'SOURCE_MANIFEST_002.json'),
             prior_tf32_manifest_sha256=sha(ROOT / 'artifacts/pt_tf32_4/SOURCE_MANIFEST.json'))
    create_json(ART / MANIFEST_NAME, m)
    with (ART / MANIFEST_NAME).with_suffix('.sha256').open('x') as f:
        f.write(sha(ART / MANIFEST_NAME) + '\n')
    print('PT_DET_1 SEALED', sha(ART / MANIFEST_NAME))


def verify_manifest():
    registration()
    p = ART / MANIFEST_NAME
    if sha(p) != p.with_suffix('.sha256').read_text().strip():
        raise ValueError('PT_DET_1 manifest drift')
    m = json.loads(p.read_text())
    if m['registration_sha256'] != REG_SHA:
        raise ValueError('manifest registration drift')
    if m['pt_det_2_registration_sha256'] != sha(ROOT/'artifacts/pt_det_2/REGISTRATION.json'):
        raise ValueError('manifest PT_DET_2 registration drift')
    pins = {str(ROOT / p): h for p, h in m['pins'].items()}
    pins.update(m['dependencies'])
    pins[str(ROOT / m['binary']['path'])] = m['binary']['sha256']
    pins[str(ROOT / m['build_receipt'])] = m['build_receipt_sha256']
    pins[str(ART / 'SOURCE_MANIFEST.json')] = m['prior_pt_det_1_manifest_sha256']
    pins[str(ART / 'SOURCE_MANIFEST_002.json')] = m['prior_pt_det_1_manifest_002_sha256']
    pins[str(ROOT / 'artifacts/pt_tf32_4/SOURCE_MANIFEST.json')] = m['prior_tf32_manifest_sha256']
    for p, h in pins.items():
        if sha(p) != h:
            raise ValueError('PT_DET_1 source/binary/dependency drift: ' + p)
    return m


def parse_rows(text, start=32055, steps=30):
    rows = []
    for line in text.splitlines():
        m = STATUS_RE.search(line)
        if m:
            row = dict(step=int(m[1]), loss=m[2], refine=m[3], lr=m[4],
                       sample_i=int(m[6]), gnorm=m[7], clip_coef=m[8], clip=m[9])
            if not all(math.isfinite(float(row[k])) for k in ('loss', 'gnorm', 'clip_coef', 'lr', 'refine', 'clip')):
                raise ValueError('nonfinite logged value')
            rows.append(row)
        elif '| step ' in line:
            raise ValueError('malformed training step row')
    if [r['step'] for r in rows] != list(range(start + 1, start + steps + 1)):
        raise ValueError('missing, duplicate, reordered or unexpected training steps')
    return rows


def array_digest(a):
    a = np.asarray(a)
    if a.dtype.kind not in 'fiu' or not a.size or not np.isfinite(a).all():
        raise ValueError('empty, nonnumeric or nonfinite saved tensor')
    return dict(dtype=a.dtype.str, shape=list(a.shape),
                sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def loss_record(value, n):
    a = np.asarray(value)
    if a.size != 1 or a.dtype != np.dtype('float32'):
        raise ValueError('loss probe requires one FP32 scalar')
    return dict(n=n, value_hex=float(a.reshape(-1)[0]).hex(), **array_digest(a))


def read_losses(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if [r['n'] for r in records] != list(range(1, 31)):
        raise ValueError('missing, duplicate or reordered full-precision losses')
    for r in records:
        # Reconstruct the scalar bytes independently from its exact hex value;
        # reject forged hashes, nonfinite values, dtype or scalar-shape drift.
        value = np.asarray(float.fromhex(r['value_hex']), dtype=r['dtype']).reshape(r['shape'])
        if loss_record(value, r['n']) != r:
            raise ValueError('full-precision loss scalar digest mismatch')
    return records


def install_loss_probe(module, record_path):
    # Prior art: CC46 (2026) full-precision norm probes / NIST SHA256 (2001)
    # receipts. Ours: record the FP32 loss bytes before backward, identically
    # in all arms. The extra synchronization is registered in PT-DET-2 A001.
    path = Path(record_path)
    with path.open('x'): pass
    original = module.nll_loss
    count = 0
    def observed(*args, **kwargs):
        nonlocal count
        value = original(*args, **kwargs)
        count += 1
        record = loss_record(value.numpy(), count)
        with path.open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        return value
    module.nll_loss = observed


def checkpoint_digest(path):
    # Trusted, source-hashed CC46 snapshot / outputs of our own trainer only.
    with Path(path).open('rb') as f:
        b = pickle.load(f)
    if not b['model'] or not b['adam_m'] or len(b['adam_m']) != len(b['adam_v']):
        raise ValueError('empty/incomplete checkpoint')
    extra = b['extra']; loader = extra.get('loader') or {}
    return dict(model={k: array_digest(v) for k, v in sorted(b['model'].items())},
                adam_m=[array_digest(a) for a in b['adam_m']], adam_v=[array_digest(a) for a in b['adam_v']],
                meta=dict(step=extra['step'], adam_t=b['adam_t'], sample_i=loader.get('sample_i'),
                          data_index=b.get('data_index'),
                          blocks=(extra.get('fwd_fp32_blocks') or {}).get('spec'),
                          kernels=(extra.get('fwd_precision_kernels') or {}).get('kernels'),
                          grad_clip=(extra.get('grad_clip') or {}).get('max_norm')))


def tensor_schema(digest):
    def shape(a): return (a['dtype'], a['shape'])
    return dict(model={k: shape(v) for k, v in digest['model'].items()},
                adam_m=[shape(a) for a in digest['adam_m']], adam_v=[shape(a) for a in digest['adam_v']])


def compare_pair(a, b):
    if a.get('status') != 'GREEN' or b.get('status') != 'GREEN':
        status = 'RED' if any(x.get('status') == 'RED' for x in (a, b)) else 'BLOCKED'
        return dict(measured=False, bitwise=None, status=status, reason='a run did not pass')
    want = list(range(32056, 32086))
    if any([r['step'] for r in x.get('rows', [])] != want for x in (a, b)):
        return dict(measured=False, bitwise=None, status='RED', reason='incomplete row coverage')
    if not a['checkpoint']['model'] or not b['checkpoint']['model']:
        return dict(measured=False, bitwise=None, status='RED', reason='empty model')
    if any([r['n'] for r in x.get('loss_steps', [])] != list(range(1, 31)) for x in (a, b)):
        return dict(measured=False, bitwise=None, status='RED', reason='incomplete full-precision losses')
    fields = [s['step'] for s, t in zip(a['rows'], b['rows']) if s != t]
    weights = a['checkpoint']['model'] == b['checkpoint']['model']
    adam = all(a['checkpoint'][k] == b['checkpoint'][k] for k in ('adam_m', 'adam_v', 'meta'))
    probes = a.get('probe_steps') == b.get('probe_steps') and len(a.get('probe_steps', [])) == 30
    # Only numerical probe fields are recorded, no timestamps or allocator IDs.
    losses = a['loss_steps'] == b['loss_steps']
    equal = not fields and weights and adam and probes and losses
    return dict(status='GREEN', measured=True, bitwise=equal, differing_log_steps=fields,
                weights_bitwise=weights, adam_and_meta_bitwise=adam, probe_bitwise=probes,
                losses_bitwise=losses)


def assess_pairs(pairs):
    if any(p.get('status') == 'RED' for p in pairs.values()):
        return 'RED'
    if set(pairs) != set(ARMS) or any(not p.get('measured') for p in pairs.values()):
        return 'BLOCKED'
    if any(pairs[a]['bitwise'] is not True for a in ARMS[:2]):
        return 'RED'
    if any(pairs[a]['bitwise'] is not False for a in ARMS[2:]):
        return 'NOT_RECURRED'
    return 'GREEN'


def batches():
    r = registration()['embedding_gate']; path = Path(r['batches_npz'])
    if path.exists():
        if sha(path) != r['batches_sha256']:
            raise ValueError('real token archive drift (no synthetic fallback on corruption)')
        with np.load(path, allow_pickle=False) as z:
            keys = sorted((k for k in z.files if re.fullmatch(r'x_\d+', k)), key=lambda k: int(k[2:]))
            if len(keys) < 3:
                raise ValueError('real token archive has fewer than three batches')
            picked = [keys[0], keys[len(keys) // 2], keys[-1]]
            arrays = [(k, np.ascontiguousarray(z[k][:-1], dtype=np.int64)) for k in picked]
        source = dict(kind='real_CC46', path=str(path), sha256=r['batches_sha256'])
    else:
        rng = np.random.default_rng(260926); arrays = []
        for i in range(3):
            ids = rng.integers(0, 8192, 4096, dtype=np.int64)
            ids[:1639] = rng.integers(0, 8, 1639)
            rng.shuffle(ids); arrays.append((f'synthetic_{i}', ids))
        source = dict(kind='synthetic', reason='registered real archive absent', seed=260926)
    for _, ids in arrays:
        if ids.shape != (4096,) or (ids < 0).any() or (ids >= 8192).any():
            raise ValueError('invalid model token batch')
        if source['kind'] == 'synthetic' and 1 - len(np.unique(ids)) / len(ids) < .3:
            raise ValueError('synthetic repeat fraction below 30 percent')
    return arrays, source


def reference(ids, grad, vocab=8192):
    # Prior art: NumPy indexed scatter-add; independent FP64 reference, not the
    # sorted implementation. Same per-token input sequence, FP64 output retained.
    out = np.zeros((vocab, grad.shape[1]), np.float64)
    np.add.at(out, ids, grad.astype(np.float64))
    return out


def rel_l2(got, ref):
    if not np.isfinite(got).all() or not np.isfinite(ref).all():
        raise ValueError('nonfinite embedding result')
    diff = np.linalg.norm(np.asarray(got, np.float64) - ref)
    denom = np.linalg.norm(ref)
    if not denom:
        return 0. if diff == 0 else float('inf')
    return float(diff / denom)


def cpu_embedding(out):
    scratch = out/'compiler_tmp'; scratch.mkdir()
    bridge = out / 'cpu_bridge.so'
    command = ['g++', '-std=c++17', '-O2', '-shared', '-fPIC', '-I' + str(ROOT / 'tensor_cuda/include'),
               str(ROOT / 'tests/pt_det_1_cpu_bridge.cpp'), '-o', str(bridge)]
    subprocess.run(command, check=True, timeout=60, env=dict(os.environ,TMPDIR=str(scratch)))
    fn = ctypes.CDLL(str(bridge)).pt_det_cpu
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3
    fn.restype = ctypes.c_int
    arrays, source = batches(); rows = []
    for i, (name, ids) in enumerate(arrays):
        grad = np.random.default_rng(260926 + i).standard_normal((4096, 1024), dtype=np.float32)
        ref = reference(ids, grad); hashes = []; times = []
        for repeat in range(5):
            got = np.empty((8192, 1024), np.float32); start = time.perf_counter()
            rc = fn(grad.ctypes.data, ids.ctypes.data, got.ctypes.data, 4096, 8192, 1024)
            times.append((time.perf_counter() - start) * 1000)
            if rc: raise ValueError('CPU segmented reduction failed')
            hashes.append(array_digest(got)['sha256'])
        err = rel_l2(got, ref)
        rows.append(dict(batch=name, repeated_fraction=1-len(np.unique(ids))/len(ids),
                         hashes=hashes, bitwise=len(set(hashes)) == 1, relative_L2=err,
                         cpu_median_ms=statistics.median(times)))
    ok = all(r['bitwise'] and r['relative_L2'] <= 1e-6 for r in rows)
    result = dict(verdict='GREEN' if ok else 'RED', evidence_class='CPU shared arithmetic only; no GPU timing',
                  source=source, rows=rows)
    create_json(out / 'summary.json', result)
    print('PT_DET_1 CPU_EMBED', result['verdict'], 'batches', len(rows),
          'max_rel_L2', max(r['relative_L2'] for r in rows))
    return 0 if ok else 1


def require_lead(lead, lock_fd):
    # Never open/acquire/unlock the production lock. Verify the inherited
    # descriptor's own Linux fdinfo shows an already-held exclusive flock.
    if not lead or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise Blocked("BLOCKED: lead GPU slot required; CUDA_VISIBLE_DEVICES is empty or --lead-gpu absent")
    if ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise Blocked('BLOCKED: exactly one visible device required')
    if lock_fd is None or lock_fd < 3:
        raise Blocked('BLOCKED: inherited locked descriptor required (--lock-fd)')
    try:
        target = os.readlink(f'/proc/self/fd/{lock_fd}')
        info = Path(f'/proc/self/fdinfo/{lock_fd}').read_text()
    except OSError as e:
        raise Blocked('BLOCKED: inherited lock descriptor unavailable') from e
    if target != '/tmp/forge-gpu.lock' or not re.search(r'lock:.*FLOCK\s+ADVISORY\s+WRITE\s', info):
        raise Blocked('BLOCKED: descriptor does not carry the exclusive production flock')
    return dict(fd=lock_fd, target=target, exclusive_flock=True)


def load_fork():
    m = verify_manifest()
    sys.path.insert(0, str(ROOT / 'tensor_cuda'))
    import tensor_cuda as tc
    if Path(tc._C.__file__).resolve() != (ROOT / m['binary']['path']).resolve():
        raise ValueError('wrong engine import')
    return tc, m


def embedding_gpu(out):
    tc, m = load_fork(); C = tc._C
    tc.set_alloc_pooling(True)
    arrays, source = batches(); rows = []
    for i, (name, ids) in enumerate(arrays):
        grad = np.random.default_rng(260926 + i).standard_normal((4096, 1024), dtype=np.float32)
        ix = tc.tensor(ids, dtype='int64'); g = tc.tensor(grad)
        ref = reference(ids, grad); hashes = []; errors = []
        C.set_deterministic_embed_bwd(True)
        for _ in range(5):
            a = C._embedding_backward(g, ix, [8192, 1024]).numpy()
            hashes.append(array_digest(a)['sha256']); errors.append(rel_l2(a, ref))
        def call(mode):
            C.set_deterministic_embed_bwd(mode)
            a = C._embedding_backward(g, ix, [8192, 1024])
            tc.synchronize()
            del a
        for _ in range(3):
            call(False); call(True)
        timing = {False: [], True: []}
        for repeat in range(10):
            for mode in ((False, True) if repeat % 2 == 0 else (True, False)):
                tc.synchronize(); start = time.perf_counter(); call(mode)
                timing[mode].append((time.perf_counter() - start) * 1000)
        atomic_ms = statistics.median(timing[False]); det_ms = statistics.median(timing[True])
        rows.append(dict(batch=name, repeated_fraction=1-len(np.unique(ids))/len(ids),
                         hashes=hashes, bitwise=len(set(hashes)) == 1, relative_L2=max(errors),
                         atomic_ms=atomic_ms, deterministic_ms=det_ms, ratio=det_ms/atomic_ms,
                         timing_samples_ms={'off': timing[False], 'on': timing[True]}))
    C.set_deterministic_embed_bwd(False)
    ok = all(r['bitwise'] and r['relative_L2'] <= 1e-6 and r['ratio'] <= 2 for r in rows)
    result = dict(verdict='GREEN' if ok else 'RED', registration_sha256=REG_SHA,
                  binary=m['binary'], source=source, rows=rows,
                  evidence_class='GPU embedding backward microbenchmark; synthetic dY on real tokens')
    create_json(out / 'summary.json', result)
    print('PT_DET_1 EMBEDDING', result['verdict'], json.dumps(rows))
    return 0 if ok else 1


def trainer_argv(arm, run_dir):
    r = registration()['repro']; argv = list(r['live_argv'])
    for flag, value in (('--steps', str(r['target'])), ('--ckpt', r['checkpoint']['path']),
                        ('--save-ckpt', str(run_dir / 'replay.ckpt')),
                        ('--log-file', str(run_dir / 'logs/train.log'))):
        if argv.count(flag) != 1: raise ValueError('ambiguous argv: ' + flag)
        argv[argv.index(flag) + 1] = value
    if arm.startswith('bf16'):
        for flag in ('--fwd-fp32-blocks', '--fwd-precision-kernels'):
            i = argv.index(flag); del argv[i:i + 2]
    if arm not in ARMS: raise ValueError('unregistered arm')
    return argv


def historical_precision_record(actual, historical, engine):
    # Prior art: CC46 (2026) replay of write-ahead controller receipts, taken.
    # Ours: substitute ONLY verified engine provenance while preserving every
    # policy field. A changed precision policy must not be hidden by replay.
    expected = copy.deepcopy(historical)
    def rebind(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == 'engine': node[key] = dict(engine)
                else: rebind(value)
        elif isinstance(node, list):
            for value in node: rebind(value)
    rebind(expected)
    if actual != expected:
        raise ValueError('PT_DET_1 cruise precision differs beyond verified engine identity')
    return copy.deepcopy(historical)


def install_engine_replay_adapter(cruise, historical, engine):
    inner = cruise.run_boundary
    def run_boundary(*args, **kwargs):
        for field in ('fwd_fp32_blocks', 'fwd_precision_kernels'):
            kwargs[field] = historical_precision_record(kwargs.get(field), historical.get(field), engine)
        return inner(*args, **kwargs)
    cruise.run_boundary = run_boundary


PREAMBLE = '''import json, os, runpy, sys
sys.path.insert(0, os.environ['PT_DET_HARNESS_PATH'])
from grapa.fwd_precision import load_engine
tc, ident = load_engine(os.environ['GRAPA_ENGINE_PATH'], os.environ['GRAPA_ENGINE_SHA256'])
from pathlib import Path
assert Path(tc._C.__file__).resolve() == Path(os.environ['PT_DET_EXPECT_SO']).resolve()
tc._C.bp_kernel_2_set_variant('g1')
tc._C.bp_kernel_4_set_variant('h')
mode = os.environ['TC_DETERMINISTIC'] == '1'
assert tc._C.get_deterministic() == mode, 'family environment opt-in failed'
assert tc._C.get_deterministic_embed_bwd() == mode, 'environment opt-in failed'
tc._C.set_deterministic(mode)
print('ENGINE SO', ident['so_path'], 'SHA256', ident['so_sha256'], flush=True)
print('PT_DET_1 MODE', int(mode), flush=True)
print('PT_DETERMINISTIC MODE', int(mode), 'SITES embedding,gather_topk', flush=True)
import grapa.loss as loss_module
from pt_det_1 import install_loss_probe
install_loss_probe(loss_module, os.environ['PT_DET_LOSS_RECORD'])
from scripts.cc46_determinism import install_probe, finish_probe
install_probe(os.environ['CC46_REGISTRATION'], os.environ['CC46_TRACE_CKPT'],
              os.environ['CC46B_ARM'], os.environ['CC46B_RECORD'])
if os.environ['CC46B_ARM'] == 'a':
    import grapa.cruise as cruise
    from pt_det_1 import install_engine_replay_adapter
    reg = json.loads(Path(os.environ['CC46_REGISTRATION']).read_text())
    historical = json.loads(Path(reg['live']['cruise_receipt']['path']).read_text())
    install_engine_replay_adapter(cruise, historical, ident)
    print('PT_DET_1 RECORD_ENGINE_REBIND', ident['so_sha256'], flush=True)
sys.argv = sys.argv[1:]
runpy.run_path('grapa/train.py', run_name='__main__')
finish_probe()
assert tc._C.get_deterministic() == mode, 'family mode changed during training'
assert tc._C.get_deterministic_embed_bwd() == mode, 'alias mode changed during training'
print('PT_DET_1 MODE_END', int(mode), flush=True)
print('PT_DETERMINISTIC MODE_END', int(mode), 'SITES embedding,gather_topk', flush=True)
'''


def run_process(argv, cwd, env, log_path, seconds, stdin=None, pass_fds=(), new_process_group=True):
    if seconds <= 0: return dict(status='BLOCKED_TIMEOUT', returncode=None, seconds=0.)
    storage.require_space(log_path.parent)
    start = time.monotonic()
    with log_path.open('x') as log:
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                             text=True, start_new_session=new_process_group, pass_fds=pass_fds)
        try:
            p.communicate(stdin, timeout=seconds)
            status = 'GREEN' if p.returncode == 0 else 'RED'
        except subprocess.TimeoutExpired:
            # Prior art: POSIX owned child groups / PT-TF32 deadlines. PT-DET-2
            # nested trainers join the outer slot group so its timeout cannot
            # orphan them; this inner owner signals only its own child PID.
            if new_process_group: os.killpg(p.pid, signal.SIGTERM)
            else: p.terminate()
            try: p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if new_process_group: os.killpg(p.pid, signal.SIGKILL)
                else: p.kill()
                p.wait(timeout=2)
            status = 'BLOCKED_TIMEOUT'
    return dict(status=status, returncode=p.returncode, seconds=time.monotonic()-start,
                argv=argv, log=str(log_path))


def run_receipt(run_dir, result, arm, source_schema, m):
    if result['status'] != 'GREEN': return result
    r = registration()['repro']; cr = json.loads(Path(r['cc46_registration']['path']).read_text())
    rows = parse_rows((run_dir / 'logs/train.log').read_text())
    text = (run_dir / 'leg.stdout').read_text()
    digest = checkpoint_digest(run_dir / 'replay.ckpt')
    losses = read_losses(run_dir / 'loss.jsonl')
    records = [json.loads(ln) for ln in (run_dir / 'probe.jsonl').read_text().splitlines()]
    steps = [s for s in records if s['kind'] == 'step']; ends = [s for s in records if s['kind'] == 'end']
    mode = int(arm.endswith('_on')); cc_arm = 'd' if arm.startswith('bf16') else 'a'
    live_rows = {x[0]: dict(zip(ROW_FIELDS, x)) for x in cr['live']['train_log']['rows']}
    end = ends[0] if len(ends) == 1 else {}
    checks = dict(
        source_schema=tensor_schema(digest) == source_schema,
        batches=all(all(row[k] == live_rows[row['step']][k] for k in ('sample_i', 'lr', 'refine', 'clip')) for row in rows),
        checkpoint_step=digest['meta']['step'] == r['target'],
        adam_t=digest['meta']['adam_t'] == r['checkpoint']['adam_t'] + 30,
        cursor=digest['meta']['sample_i'] == live_rows[r['target']]['sample_i'],
        data_index=digest['meta']['data_index'] == live_rows[r['target']]['sample_i'],
        precision=(digest['meta']['blocks'], digest['meta']['kernels']) == ((None, None) if cc_arm == 'd' else ('0-10', 'tf32')),
        grad_clip=digest['meta']['grad_clip'] == 1.,
        modes=f'PT_DET_1 MODE {mode}\n' in text and f'PT_DET_1 MODE_END {mode}\n' in text,
        family_modes=all(text.count(f'PT_DETERMINISTIC {tag} {mode} SITES embedding,gather_topk\n') == 1
                         for tag in ('MODE', 'MODE_END')),
        engine=f"ENGINE SO {ROOT / m['binary']['path']} SHA256 {m['binary']['sha256']}" in text,
        resumed=len(re.findall(r'\| RESUMED from .*: step 32055, sample_i 11584,', text)) == 1,
        cruise=len(re.findall(r'^CRUISE REPLAY step=32055 rule=rail_binding_cap ', text, re.M)) == 1,
        no_sensors=not re.search(r'^CRUISE step=', text, re.M),
        probes=[s['n'] for s in steps] == list(range(1, 31)) and end.get('steps') == 30 and end.get('arm') == cc_arm,
        variants=end.get('variants_end') == ['g1', 'h'],
        tf32=(end.get('lt_info_end') is not None and (end['lt_info_end'][0] != -1) == (cc_arm == 'a')),
        gradients=bool(steps) and len(steps[0].get('grad_sha256', [])) == len(source_schema['adam_m'])
                  and all(steps[0]['grad_sha256']))
    normalized = []
    for s in steps:
        if not all(math.isfinite(float.fromhex(s[k])) for k in ('gnorm_hex', 'coef_hex')):
            raise ValueError('nonfinite full-precision norm')
        normalized.append({k: s[k] for k in ('n', 'gnorm_hex', 'coef_hex', 'grad_sha256') if k in s})
    return dict(result, status='GREEN' if all(checks.values()) else 'RED', checks=checks, rows=rows,
                checkpoint=digest, probe_steps=normalized, loss_steps=losses,
                files={str(p.relative_to(run_dir)): sha(p) for p in
                       (run_dir/'logs/train.log', run_dir/'leg.stdout', run_dir/'probe.jsonl', run_dir/'loss.jsonl')})


def family_environment(arm):
    if arm not in ARMS: raise ValueError('unregistered arm')
    mode = str(int(arm.endswith('_on')))
    return dict(TC_DETERMINISTIC=mode, TC_DET_EMBED_BWD=mode)


def repro(out, lock_fd, seconds=1400, keep_ckpts=False):
    deadline = time.monotonic() + min(seconds, 1400)
    slot_deadline = os.environ.get('PT_DET_SLOT_LANE_DEADLINE')
    if slot_deadline is not None:
        inherited = float(slot_deadline)
        if not math.isfinite(inherited): raise ValueError('invalid slot deadline')
        deadline = min(deadline, inherited - 5)
    m = verify_manifest(); r = registration()['repro']; source = Path(r['checkpoint']['path'])
    if sha(source) != r['checkpoint']['sha256']: raise ValueError('source checkpoint drift')
    initial_source_stat = source.stat()
    source_schema = tensor_schema(checkpoint_digest(source))
    pairs = {}; runs = {}; start = time.monotonic()
    for arm in ARMS:
        pair = []
        for repeat in (1, 2):
            run_dir = out / f'{arm}_{repeat}'; run_dir.mkdir(); (run_dir / 'logs').mkdir()
            env = {k: v for k, v in os.environ.items() if k not in
                   ('PYTHONPATH', 'TC_TF32_GEMM', 'NVIDIA_TF32_OVERRIDE', 'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG')}
            env.update(PYTHONPATH=str(ROOT/'tensor_cuda'), PYTHONDONTWRITEBYTECODE='1',
                       GRAPA_ENGINE_PATH=str(ROOT/'tensor_cuda'), GRAPA_ENGINE_SHA256=m['binary']['sha256'],
                       PT_DET_EXPECT_SO=str(ROOT/m['binary']['path']), **family_environment(arm),
                       PT_DET_HARNESS_PATH=str(ROOT/'scripts'),
                       PT_DET_LOSS_RECORD=str(run_dir/'loss.jsonl'),
                       CC46_REGISTRATION=r['cc46_registration']['path'], CC46_TRACE_CKPT=str(run_dir/'replay.ckpt'),
                       CC46B_ARM='d' if arm.startswith('bf16') else 'a', CC46B_RECORD=str(run_dir/'probe.jsonl'),
                       TMPDIR=str(out/'tmp'), CUDA_CACHE_PATH=str(out/'cuda_cache'),
                       OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
            (out/'tmp').mkdir(exist_ok=True)
            argv = [sys.executable, '-B', '-'] + trainer_argv(arm, run_dir)
            remaining = deadline-time.monotonic()-10
            print('PT_DET_1 RUN', arm, repeat, 'remaining_seconds', round(remaining, 1), flush=True)
            try:
                st = source.stat()
                if (st.st_ino, st.st_size, st.st_mtime_ns) != (initial_source_stat.st_ino, initial_source_stat.st_size, initial_source_stat.st_mtime_ns):
                    raise ValueError('source checkpoint changed between replay runs')
                result = run_process(argv, CC, env, run_dir/'leg.stdout', remaining, PREAMBLE, (lock_fd,),
                                     new_process_group=slot_deadline is None)
                receipt = run_receipt(run_dir, result, arm, source_schema, m)
            except Exception as exc:
                receipt = dict(status='BLOCKED' if isinstance(exc, (Blocked, storage.StorageBlocked)) else 'RED', error=repr(exc))
            create_json(run_dir/'receipt.json', receipt); pair.append(receipt); runs[f'{arm}_{repeat}'] = receipt['status']
            # Digest all tensors before removing only the checkpoint we created.
            if not keep_ckpts and (run_dir/'replay.ckpt').exists(): (run_dir/'replay.ckpt').unlink()
        pairs[arm] = compare_pair(*pair)
    verdict = assess_pairs(pairs)
    verify_manifest()  # Recheck trainer/config/engine bytes after all runs.
    result = dict(verdict=verdict, registration_sha256=REG_SHA, binary=m['binary'], steps=30,
                  pt_det_2_registration_sha256=m['pt_det_2_registration_sha256'],
                  deterministic_sites=['embedding', 'gather_topk'],
                  manifest_sha256=sha(ART/MANIFEST_NAME), source_checkpoint_sha256=r['checkpoint']['sha256'],
                  pairs=pairs, runs=runs, seconds=time.monotonic()-start, keep_ckpts=keep_ckpts,
                  evidence_class='two fresh-process 30-step runs per arm, saved tensor bytes and exact text',
                  receipt_sha256={str(p.relative_to(out)): sha(p) for p in out.glob('*/receipt.json')})
    create_json(out/'summary.json', result)
    print('PT_DET_1 REPRO', verdict, ' '.join(f'{a}={pairs[a].get("bitwise")}' for a in ARMS))
    return 0 if verdict == 'GREEN' else 1


def verify_repro(path):
    m = verify_manifest(); p = Path(path).resolve()
    if not p.is_relative_to(ART): raise ValueError('repro receipt must be under PT-DET-1 artifacts')
    result = json.loads(p.read_text())
    if (result.get('registration_sha256') != REG_SHA or result.get('binary') != m['binary'] or
        result.get('pt_det_2_registration_sha256') != m['pt_det_2_registration_sha256'] or
        result.get('deterministic_sites') != ['embedding', 'gather_topk'] or
        result.get('manifest_sha256') != sha(ART/MANIFEST_NAME) or result.get('steps') != 30 or
        result.get('source_checkpoint_sha256') != registration()['repro']['checkpoint']['sha256']):
        raise ValueError('repro provenance mismatch')
    pairs = {}
    for arm in ARMS:
        rr = []
        for i in (1, 2):
            rel = f'{arm}_{i}/receipt.json'; file = p.parent / rel
            if sha(file) != result['receipt_sha256'].get(rel): raise ValueError('repro run receipt drift')
            rec = json.loads(file.read_text())
            if not rec.get('checks') or not all(rec['checks'].values()): raise ValueError('repro structural gate failed')
            for name, digest in rec['files'].items():
                if sha(file.parent / name) != digest: raise ValueError('repro log drift')
            if parse_rows((file.parent/'logs/train.log').read_text()) != rec['rows']:
                raise ValueError('repro rows differ from logged text')
            if read_losses(file.parent/'loss.jsonl') != rec.get('loss_steps'):
                raise ValueError('repro losses differ from full-precision record')
            rr.append(rec)
        pairs[arm] = compare_pair(*rr)
    if pairs != result.get('pairs') or assess_pairs(pairs) != 'GREEN' or result.get('verdict') != 'GREEN':
        raise ValueError('required PT_DET_1 replay did not pass')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('seal', 'cpu', 'embedding', 'repro', 'plan', 'verify-repro'))
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--out', type=Path)
    p.add_argument('--receipt', type=Path)
    p.add_argument('--lead-gpu', action='store_true')
    p.add_argument('--lock-fd', type=int)
    p.add_argument('--seconds', type=float, default=1400)
    p.add_argument('--keep-ckpts', action='store_true')
    args = p.parse_args()
    if args.steps != 30: p.error('registered repro requires exactly 30 steps')
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 1400:
        p.error('--seconds must be in (0, 1400]; never shortens the 30-step gate')
    registration()
    if args.command == 'seal': seal(); return 0
    if args.command == 'verify-repro':
        if args.receipt is None: p.error('required PT_DET_1 repro receipt missing')
        verify_repro(args.receipt); print('PT_DET_1 REQUIRED_REPRO GREEN'); return 0
    if args.command == 'plan':
        from pt_det_2 import registration as registration_2
        base = (args.out or ART/'lead_slot/repro').resolve()
        print(json.dumps(dict(arms={a: [trainer_argv(a, base/f'{a}_{i}') for i in (1, 2)] for a in ARMS},
                              family_environments={a: family_environment(a) for a in ARMS},
                              predictions=registration()['repro']['arms'], slot=registration_2()['slot']), indent=2)); return 0
    out = (args.out or ART / ('cpu' if args.command == 'cpu' else args.command)).resolve()
    if not out.is_relative_to(ART) or out == ART: p.error('out must be a fresh PT-DET-1 artifact subdirectory')
    out.mkdir(parents=True, exist_ok=False)
    try:
        if args.command == 'cpu': return cpu_embedding(out)
        lock = require_lead(args.lead_gpu, args.lock_fd)
        create_json(out/'lock_receipt.json', lock)
        if args.command == 'embedding': return embedding_gpu(out)
        return repro(out, args.lock_fd, args.seconds, args.keep_ckpts)
    except Exception as exc:
        status = 'BLOCKED' if isinstance(exc, (Blocked, storage.StorageBlocked)) else 'RED'
        if not (out/'summary.json').exists():
            create_json(out/'summary.json', dict(verdict=status, reason=str(exc), registration_sha256=REG_SHA,
                                               gpu_executed=False if isinstance(exc, Blocked) else 'unknown'))
        print('PT_DET_1', status, str(exc), file=sys.stderr)
        return 2 if status == 'BLOCKED' else 1


if __name__ == '__main__':
    sys.exit(main())
