"""Lead GPU payloads. Prior art: P1-F3 and PF2-F1 (Project-Tensor, 2026).
Preserve registered operands and unsigned comparisons. Interleave the two
persistent module workers as in PF2-F1/F2. Mutation analysis follows
DeMillo et al. (1978), with clean F3 and unaffected rows as controls.
"""
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
import apa_pf2f3_build as B
import apa_pf2f3_cpu as C
import apa_pf2f3_identity_reference as I
import apa_pf2f3_nonfinite as N
import apa_pf2f3_p1_regression as P

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

    if suite == 'tc':
        return finite_tc(tc, index)

    assert suite == 'extra', suite
    D, VD, G, dt, diag = extra_cases()[index]
    rng = np.random.default_rng(np.random.SeedSequence([20261118, index]))
    q, k, v = (tc.tensor(rng.standard_normal(shape, dtype=np.float32), dtype=dt) for shape in
               ((1, 2*G, 1, D), (1, 2, 257, D), (1, 2, 257, VD)))
    ws = tc.ApaSpq1Workspace(k, 257, fmt='tq4')
    out = tc._spo2.packed_run(q, k, v, ws.packed, float(np.float32(1/np.sqrt(D))), 0.,
                              False, None, diag, fmt='tq4', carry='partition', gqa_path='r2')
    return dict(zip(('out', 'mask'), out)) if diag else {'out':out}


def tc_finite_cases():
    # The F2/F3 release dispatch exposes variant 2 only (Project-Tensor, 2026).
    return list(itertools.product(('int4', 'tq4'), (64, 128), (64, 128),
                                  (False, True), ('partition', 'row'), range(2, 9)))


def finite_tc(tc, index):
    fmt, D, VD, diag, carry, G = tc_finite_cases()[index]
    rng = np.random.default_rng(np.random.SeedSequence([20261118, 3, index]))
    q, k, v = (tc.tensor(rng.standard_normal(shape, dtype=np.float32), dtype='bfloat16')
               for shape in ((1, 2 * G, 1, D), (1, 2, 257, D), (1, 2, 257, VD)))
    ws = tc.ApaSpq1Workspace(k, 257, fmt=fmt)
    out = tc._spo2.packed_run(q, k, v, ws.packed, float(np.float32(1 / np.sqrt(D))), 0.,
                              False, None, diag, fmt=fmt, carry=carry, dump=diag, gqa_path='tc')
    arrays = dict(zip(('out', 'mask', 'bulk', 'prefix'), out, strict=True)) if diag else {'out': out}
    assert tuple(arrays['out'].shape) == (1, 2 * G, 1, VD) and arrays['out'].dtype == 'bfloat16'
    if diag:
        assert all(tuple(arrays[name].shape) == (1, 2 * G, 1, 257) for name in ('mask', 'bulk', 'prefix'))
        assert arrays['mask'].dtype == 'uint8' and arrays['bulk'].dtype == arrays['prefix'].dtype == 'float32'
    return arrays


def tc_norm_cases():
    return [dict(fmt=fmt, D=D, diag=diag)
            for fmt, D, diag in itertools.product(('int4', 'tq4'), (64, 128), (False, True))]


def run_tc_norm(tc, case):
    fmt, D, diag = (case[name] for name in ('fmt', 'D', 'diag'))
    G = 4
    qn, kn, vn, codes, rot, table, norm = N.norm_fixture(tc, D)
    qn = np.repeat(qn[:, :, :1], 2 * G, axis=1)
    kn, vn, codes = (np.repeat(array, 2, axis=1) for array in (kn, vn, codes))
    rot = np.repeat(rot, 2, axis=0)
    if fmt == 'int4':
        # spq4 uses code-7 (Project-Tensor SPQ4, 2026). Reuse REV2's
        # signed dot witness, with +7 for key 0 and -7 for key 1.
        codes[:, :, 0] = 0xee
        codes[:, :, 1] = 0x00

    def tensor(array, dtype='bfloat16'):
        return tc.tensor(np.ascontiguousarray(array), device='cuda', dtype=dtype)

    q, k, v = (tensor(a) for a in (qn, kn, vn))
    codes, rot, table = tensor(codes, 'uint8'), tensor(rot, 'float32'), tensor(table, 'float32')
    scale = float(np.float32(1 / np.sqrt(D)))
    wanted = np.zeros((1, 2 * G, 1), dtype=bool)
    wanted[:, :G] = True
    expected_bulk = np.stack((np.zeros_like(wanted), wanted), axis=-1)
    results, failed = {}, []
    for kind in ('finite', *N.NORM_KINDS):
        raw = np.array([[[norm, norm if kind == 'finite' else float(kind)], [norm, norm]]], np.float32)
        packed = (codes, tensor(raw, 'float32'), rot, table)
        result = tc._spo2.packed_run(q, k, v, packed, scale, 0., False, None, diag,
                                     fmt=fmt, carry='partition', dump=diag, gqa_path='tc')
        out = result[0] if diag else result
        assert tuple(out.shape) == (1, 2 * G, 1, D) and out.dtype == 'bfloat16'
        array = out.float().numpy()
        bulk = result[2].numpy() if diag else None
        if bulk is not None:
            assert bulk.shape == (1, 2 * G, 1, 2) and bulk.dtype == np.float32
        if kind == 'finite':
            N.assert_finite_control(array)
            if bulk is not None:
                N.assert_finite_control(bulk)
                assert np.all(bulk[..., 0] > 0) and np.all(bulk[..., 1] < 0)
            continue
        counted = N.check_nan_rows(array, wanted)
        counted['unaffected_finite'] = bool(np.isfinite(array[:, G:]).all())
        if not counted['ok']:
            failed.append(f'{kind}:output')
        if bulk is not None:
            counted['bulk_all_nan_at_bad_key'] = bool(np.array_equal(np.isnan(bulk), expected_bulk))
            counted['bulk_nan_elements'] = int(np.isnan(bulk).sum())
            if not counted['bulk_all_nan_at_bad_key']:
                failed.append(f'{kind}:bulk')
        results[kind] = counted
    return dict(case=case, failed=failed, finite_control=True, expected_nonfinite_rows=G, results=results)


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
        for label in ('F2', 'F3'):
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


def nonfinite(label, suite):
    run, cases = {
        'sink': (N.run_sink, N.sink_cases),
        'norm': (N.run_norm, N.norm_cases),
        'partition': (run_partition, N.partition_norm_cases),
        'tc': (run_tc_norm, tc_norm_cases),
    }[suite]
    tc = C.load_arm(label)
    rows = []
    with tc.no_grad():
        for case in cases():
            row = run(tc, case)
            rows.append(row)
            print('G2_CASE', json.dumps(row, allow_nan=False), flush=True)
    return rows


def run_partition(tc, case):
    return N.run_partition_norm(tc, case, check_prefill=True)


def mutation(spec, dest):
    label, suite = spec['mutant'], spec['suite']
    records = []
    for arm, want_exit in (('F3', 0), (label, 1)):
        argv = ['timeout', '135s', *PYTHON, 'nonfinite-worker', arm, suite]
        log = dest / f'{arm}.log'
        with log.open('x') as stream:
            code = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT).returncode
        lines = log.read_text().splitlines()
        rows = [json.loads(s.removeprefix('G2_CASE ')) for s in lines if s.startswith('G2_CASE ')]
        markers = [json.loads(s.removeprefix('G3_ASSERTION ')) for s in lines if s.startswith('G3_ASSERTION ')]
        rec = dict(arm=arm, exit=code, log_sha256=B.sha(log), rows=rows, assertions=markers)
        B.save(dest / f'{arm}.json', rec)
        assert code == want_exit and len(rows) == spec['cases'], rec
        expected = spec['expected_failures'] if want_exit else []
        for row in rows:
            wanted = expected
            if suite == 'tc' and row['case']['diag']:
                wanted = [failure for output in expected for failure in (output, output.replace(':output', ':bulk'))]
            assert row['finite_control'] and row['failed'] == wanted, rec
        assert markers == ([dict(failed_cases=spec['cases'], suite=suite)] if want_exit else []), rec
        for row in rows:
            if suite == 'tc':
                for result in row['results'].values():
                    assert result['unaffected_finite'], rec
            else:
                assert all(result['decode']['ok'] for result in row['results'].values()), rec
        records.append(rec)
    B.save(dest / 'RESULT.json', dict(status='KILLED', mutant=label, suite=suite, runs=records))
    print('G3 KILLED', label, suite, spec['expected_failures'])


def regression(dest):
    tc = C.load_arm('F3')
    inherited, checks = P.lf2_run(tc, P.R, 'F3')
    overflow, extra_checks = P.overflow_run(tc, P.R, 'F3')
    checks += extra_checks
    gated = [r for r in checks if not r.get('report')]
    failed = [r for r in gated if not r['ok']]
    B.save(dest / 'RESULT.json', dict(status='PASS' if len(gated) == 643 and not failed else 'FAIL',
                                     gated=len(gated), failed=failed, checks=checks,
                                     inherited_cases=inherited, overflow_cases=overflow))
    print('G5',len(gated),'checks',len(failed),'failures')
    return int(len(gated) != 643 or bool(failed))


def main():
    assert os.environ.get('APA_PF2F3_LEASE') == '1' and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert os.environ.get('FORGE_GPU_DRY_RUN') == '0'
    if sys.argv[1] == 'worker':
        worker(*sys.argv[2:])
        return 0
    if sys.argv[1] == 'nonfinite-worker':
        arm, suite = sys.argv[2:]
        rows = nonfinite(arm, suite)
        assert rows
        failures = [row['failed'] for row in rows if row['failed']]
        try:
            assert not failures, failures
        except AssertionError:
            print('G3_ASSERTION', json.dumps(dict(failed_cases=len(failures), suite=suite)), flush=True)
            return 1
        return 0
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
        assert spec['mode'] == 'nonfinite'
        rows = nonfinite('F3', spec['suite'])
        assert len(rows) == spec['cases']
        failed = sum(bool(row['failed']) for row in rows)
        B.save(dest / 'RESULT.json', dict(status='FAIL' if failed else 'PASS', failed=failed, rows=rows))
        return int(failed > 0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
