"""Lead GPU payloads. Prior art: P1-F3 and PF2-F1 (Project-Tensor, 2026).
Preserve registered operands and unsigned comparisons. Interleave the two
persistent module workers as in PF2-F1. One-load mutation testing follows
DeMillo et al. (1978), with a clean arm and an alternate-route control.
"""
import argparse
import gc
import itertools
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apa_pf2f2_build as B
import apa_pf2f2_cpu as C
import apa_pf2f2_identity_reference as I
import apa_pf2f2_nonfinite as N
import apa_pf2f2_p1_regression as P

PYTHON = [sys.executable, '-B', '-P', str(Path(__file__).resolve())]
RAW_CASES = (
    ('bfloat16', 8, ('r2_tq4', 'r2_int4')),
    ('bfloat16', 2, ('r2_tq4', 'r2_int4')),
    ('float32', 8, ('r2_tq4', 'r2_int4')),
    ('float32', 2, ('r2_tq4',)),
)


def raw_cases():
    return [(dtype, H, name, carry, delta, sink, diag)
            for dtype, H, names in RAW_CASES
            for name, carry, delta, sink, diag in
            itertools.product(names, ('partition', 'row'), (2., 0.), (False, True), (False, True))]


def extra_cases():
    return list(itertools.product((64, 128), (64, 128, 256, 512), (1, 2, 4, 8),
                                  ('float32', 'bfloat16'), (False, True)))


def finite(tc, torch, suite, index):
    if suite == 'prefill':
        c = I.g1_cases()[index]
        engine = I.Env(tc, torch)
        q, k, v, sink = I.case_tensors(torch, c, index)
        ws, packed = engine.store(k, c['cap'])
        scale = float(np.float32(1 / np.sqrt(c['D'])))
        delta = tc.apa_packed.derive_delta(.01, tc.apa_packed.PACKED_PINNED_ENTRIES[f"4:{c['D']}"])
        out, mask = engine.prefill(q, k, v, packed, scale, delta, c['causal'], sink)
        assert tuple(out.shape) == (c['B'], 2*c['G'], c['L'], c['VD'])
        assert tuple(mask.shape) == (c['B'], 2*c['G'], c['L'], c['S']) and mask.dtype == torch.uint8
        return {'out': out, 'mask': mask}

    if suite == 'public':
        c = I.l1_cases()[index]
        q, k, v, sink = I.case_inputs(c, index)
        q, k, v = (tc.tensor(x) for x in (q, k, v))
        sink = tc.tensor(sink) if sink is not None else None
        scale = float(np.float32(1 / np.sqrt(c['D'])))
        margins = tc.PackedRegisteredMargins.load(P.PACKED_TABLE)
        results = {}
        for name, kw in (('default', {}), ('explicit', {'storage': 'packed'})):
            ws = tc.quantize_sp2_keys(k, 4, **kw)
            assert isinstance(ws, tc.PackedSpKeys)
            if name == 'default':
                assert tc.apa_packed.last_route('quantize')['source'] == 'default'
            args = dict(bulk_bits=4, margins=margins, is_causal=c['causal'], sinks=sink)
            out, mask = tc.apa_selective_attention_sp(q, k, ws, v, scale, .01, diagnostics=True, **args)
            plain = tc.apa_selective_attention_sp(q, k, ws, v, scale, .01, **args)
            assert tc.apa_packed.last_route('attention')['backend'] == tc.apa_packed.PACKED_BACKEND
            results.update({f'{name}/out':out, f'{name}/mask':mask, f'{name}/plain':plain})
        return results

    if suite == 'raw':
        dt, H, name, carry, delta, sink, diag = raw_cases()[index]
        c = P.Case(tc, P.R, 20261115, 1, H, 2, 4096, 128, dt, plant=True, dup=True, sinks=True)
        results = P.run_raw(tc, c, name, delta=delta, carry=carry, sinks=c.sinks if sink else None,
                            diag=diag, build='identity')
        return dict(zip(('out', 'mask', 'bulk', 'prefix'), results))

    D, VD, G, dt, diag = extra_cases()[index]
    rng = np.random.default_rng(np.random.SeedSequence([20261118, index]))
    q, k, v = (tc.tensor(rng.standard_normal(shape, dtype=np.float32), dtype=dt) for shape in
               ((1, 2*G, 1, D), (1, 2, 257, D), (1, 2, 257, VD)))
    ws = tc.ApaSpq1Workspace(k, 257, fmt='tq4')
    out = tc._spo2.packed_run(q, k, v, ws.packed, float(np.float32(1/np.sqrt(D))), 0.,
                              False, None, diag, fmt='tq4', carry='partition', gqa_path='r2')
    return dict(zip(('out', 'mask'), out)) if diag else {'out':out}


def wire_data(torch, values):
    dtype_pairs = {torch.float32:(torch.int32, np.uint32), torch.bfloat16:(torch.int16, np.uint16),
                   torch.uint8:(torch.uint8, np.uint8)}
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            value = torch.from_dlpack(value)
        if value.dtype != torch.uint8:
            assert bool(torch.isfinite(value).all()), name
        signed, unsigned = dtype_pairs[value.dtype]
        array = value.contiguous().view(signed).cpu().numpy().view(unsigned)
        flat = array.reshape(-1)
        assert flat.size > 0
        assert any((flat[pos:pos+2**20] != flat[0]).any() for pos in range(0, flat.size, 2**20)), name
        yield dict(name=name, dtype=str(value.dtype), shape=list(value.shape), bits=array.dtype.str,
                   source_width=value.element_size(), nbytes=array.nbytes), array


def worker(label, suite):
    wire = sys.stdout.buffer
    sys.stdout = sys.stderr
    tc = C.load_arm(label)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    while chunk := sys.stdin.buffer.read(4):
        assert len(chunk) == 4
        index, = struct.unpack('<I', chunk)
        with tc.no_grad():
            data = list(wire_data(torch, finite(tc, torch, suite, index)))
        meta = json.dumps(dict(index=index, arrays=[m for m, _ in data])).encode()
        wire.write(struct.pack('<Q', len(meta)))
        wire.write(meta)
        for _, a in data:
            wire.write(memoryview(a).cast('B'))
        wire.flush()
        del data, a
        gc.collect()
        tc.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        wire.write(b'R')
        wire.flush()


def read_exact(stream, size):
    out = bytearray(size)
    view = memoryview(out)
    count = 0
    while count < size:
        got = stream.readinto(view[count:])
        assert got, f'truncated identity stream after {count}/{size} bytes'
        count += got
    return out


def metadata(process, index):
    process.stdin.write(struct.pack('<I', index))
    process.stdin.flush()
    size, = struct.unpack('<Q', read_exact(process.stdout, 8))
    assert 0 < size <= 8192
    result = json.loads(read_exact(process.stdout, size))
    assert result['index'] == index and result['arrays']
    return result['arrays']


def identity(line, spec, dest):
    processes = []
    try:
        for label in ('F1', 'F2'):
            with (dest / f'{label}.log').open('x') as log:
                processes.append(subprocess.Popen(['timeout', '460s', *PYTHON, 'worker', label, spec['suite']],
                                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log))
        ref, current = processes
        rows = []
        for index in spec['indices']:
            B.disk_free()
            before = metadata(ref, index)
            baseline = [read_exact(ref.stdout, m['nbytes']) for m in before]
            assert read_exact(ref.stdout, 1) == b'R'
            after = metadata(current, index)
            assert before == after, (line, index, before, after)
            counts = {}
            for meta, expected in zip(before, baseline):
                dtype = np.dtype(meta['bits'])
                assert dtype.kind == 'u' and dtype.itemsize == meta['source_width']
                assert len(expected) == dtype.itemsize * np.prod(meta['shape'], dtype=np.int64) > 0
                reference = np.frombuffer(expected, dtype=dtype)
                unequal = 0
                for pos in range(0, len(reference), 2**20):
                    count = min(2**20, len(reference)-pos)
                    observed = np.frombuffer(read_exact(current.stdout,count*dtype.itemsize), dtype=dtype)
                    unequal += int(np.count_nonzero(reference[pos:pos+count] != observed))
                counts[meta['name']] = dict(elements=len(reference), unequal=unequal,
                                            shape=meta['shape'], dtype=meta['dtype'])
            assert read_exact(current.stdout, 1) == b'R'
            row = dict(case=index, counts=counts, ok=all(x['unequal'] == 0 for x in counts.values()))
            B.save(dest / f'case_{index:04d}.json', row)
            rows.append(row)
            assert row['ok'], row
            print('G1_CASE', line, index, 'PASS', flush=True)
            del baseline, reference
        for process in processes:
            process.stdin.close()
            assert not process.stdout.read(1) and process.wait(timeout=20) == 0
        B.save(dest / 'RESULT.json', dict(status='PASS', line=line, cases=rows, modules=spec['modules']))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=20)


def nonfinite(label, suite, group=None):
    run, cases = {'sink':(N.run_sink,N.sink_cases), 'norm':(N.run_norm,N.norm_cases),
                  'partition':(N.run_partition_norm,N.partition_norm_cases)}[suite]
    tc = C.load_arm(label)
    rows = []
    with tc.no_grad():
        for case in cases():
            if group is not None and case['G'] != group:
                continue
            row = run(tc, case)
            rows.append(row)
            print('G2_CASE', json.dumps(row), flush=True)
    return rows


def mutation(spec, dest):
    label, group = spec['mutant'], spec['group']
    records = []
    for name, arm, selected, want_exit in (('F2', 'F2', None, 0), ('reverted', label, group, 1),
                                          ('alternate', label, 1 if group == 4 else 4, 0)):
        argv = ['timeout','120s', *PYTHON, 'norm-worker', arm]
        if selected is not None:
            argv.append(str(selected))
        log = dest / f'{name}.log'
        with log.open('x') as stream:
            code = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT).returncode
        rows = [json.loads(s.removeprefix('G2_CASE ')) for s in log.read_text().splitlines()
                if s.startswith('G2_CASE ')]
        rec = dict(arm=arm, group=selected, exit=code, log_sha256=B.sha(log), rows=rows)
        B.save(dest / f'{name}.json', rec)
        assert code == want_exit and len(rows) == (16 if selected is None else 8), rec
        failed = ['raw_inf_decode_row_sets'] if want_exit == 1 else []
        assert all(row['failed'] == failed and row['finite_control'] and row['nan_decode_all_nan']
                   and row['inf_prefill_nan_elements'] > 0 and (selected is None or row['case']['G'] == selected)
                   and (row['inf_decode_nan_elements'] == 0 if want_exit else row['inf_decode_nan_elements'] > 0)
                   for row in rows), rec
        records.append(rec)
    B.save(dest / 'RESULT.json', dict(status='KILLED', mutant=label, kernel=B.MUTANTS[label], runs=records))
    print('G3 KILLED', label, group)


def regression(dest):
    tc = C.load_arm('F2')
    inherited, checks = P.lf2_run(tc, P.R, 'F2')
    overflow, extra_checks = P.overflow_run(tc, P.R, 'F2')
    checks += extra_checks
    gated = [r for r in checks if not r.get('report')]
    failed = [r for r in gated if not r['ok']]
    B.save(dest / 'RESULT.json', dict(status='PASS' if gated and not failed else 'FAIL',
                                     gated=len(gated), failed=failed, checks=checks,
                                     inherited_cases=inherited, overflow_cases=overflow))
    print('G4',len(gated),'checks',len(failed),'failures')
    return int(not gated or bool(failed))


def main():
    assert os.environ.get('APA_PF2F2_LEASE') == '1' and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert os.environ.get('FORGE_GPU_DRY_RUN') == '0'
    if sys.argv[1] == 'worker':
        worker(*sys.argv[2:])
        return 0
    if sys.argv[1] == 'norm-worker':
        rows = nonfinite(sys.argv[2], 'norm', int(sys.argv[3]) if len(sys.argv) == 4 else None)
        assert rows
        return int(any(row['failed'] for row in rows))
    assert len(sys.argv) == 2
    line = sys.argv[1]
    spec = B.read(B.ART / 'LINES.json')[line]
    dest = B.ART / 'results' / line
    dest.mkdir(parents=True)
    if spec['mode'] == 'identity':
        identity(line, spec, dest)
    elif spec['mode'] == 'mutation':
        mutation(spec, dest)
    elif spec['mode'] == 'regression':
        return regression(dest)
    else:
        rows = nonfinite('F2', spec['suite'])
        assert len(rows) == spec['cases']
        failed = sum(bool(row['failed']) for row in rows)
        B.save(dest / 'RESULT.json', dict(status='FAIL' if failed else 'PASS', failed=failed, rows=rows))
        return int(failed > 0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
