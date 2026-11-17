"""APA-P1-F3 BOUNDARY.json (L60, G3): for every refusal or use-site change this round adds, the input, Build F3's
behaviour and Build F2's behaviour on the same input.

Sources: LF3's RESULT.json (Build F3, GPU, under the launch counter) and BND_F2.json (Build F2, one child process
per input); R1' on CPU, run here on both sides' Python (Build F3's arm package and Build F2's arm package) through
F2's recording-fake package (tensor_cuda/tests/test_apa_p1f2_cpu.py make_package, imported); the S4 host-level rows
from artifacts/apa_p1f3/cpu/S4.json. Read-only assembly; no device work. CPU only.
Prior art: test doubles (Meszaros, 2007); time-of-check to time-of-use (MITRE CWE-367). No new technique.
"""
from pathlib import Path
import json
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_p1f3'
sys.path.insert(0, str(ROOT / 'tensor_cuda/tests'))
import test_apa_p1f2_cpu as f2t  # noqa: E402
from _pytest.monkeypatch import MonkeyPatch  # noqa: E402

PACKAGES = {'F3': ART / 'arms/F3/tensor_cuda', 'F2': ROOT / 'artifacts/apa_p1f2/arms/F2/tensor_cuda'}


class StatefulEntries(dict):
    def __init__(self, pins, pinned_reads):
        super().__init__(pins)
        self.pinned_reads, self.reads = pinned_reads, 0

    def __getitem__(self, key):
        self.reads += 1
        return dict.__getitem__(self, key) if self.reads <= self.pinned_reads else 0.0


def r1p_cpu(side):
    out = {}
    for what, n in (('lookup', 2), ('attention', 4)):
        mp = MonkeyPatch()
        try:
            mp.setattr(f2t, 'SRC', PACKAGES[side])
            mp.delenv('TC_APA_SP_STORAGE', raising=False)
            mp.setenv('TC_APA_SP', '1')
            env = f2t.make_package(mp)
            P = env.packed
            m = P.PackedRegisteredMargins(P.PACKED_TABLE_SHA256, StatefulEntries(dict(P.PACKED_PINNED_ENTRIES), n))
            try:
                if what == 'lookup':
                    got = m.lookup(4, 64, 'float32', 1 / math.sqrt(64))
                    out[what] = dict(result='returned', e_q=got, mapping_reads=m.entries.reads)
                else:
                    f2t.attend(env, m, D=128)
                    (_, args, _), = f2t.kinds(env, 'packed_run')
                    out[what] = dict(result='launched (recording backend)', delta=args[5],
                                     e_q=P.last_route('attention')['e_q'], mapping_reads=m.entries.reads)
            except ValueError as e:
                out[what] = dict(result='ValueError: ' + str(e), device_calls=len(f2t.device_calls(env)))
        finally:
            mp.undo()
    return out


def main():
    lf3 = json.loads((ART / 'gpu/LF3/RESULT.json').read_text())
    bnd = {r['row']: r for r in json.loads((ART / 'gpu/LF3/BND_F2.json').read_text())['rows']}
    exp = json.loads((ART / 'EXPECTED.json').read_text())['gpu_lines']['LF3']['rows']
    rows = []
    for c in lf3['checks']:
        if not c['id'].startswith("LF3 R3' row "):
            continue
        name = c['id'][len("LF3 R3' row "):]
        f2 = bnd[name]
        rows.append(dict(ruling="R3'", input=name, source='LF3 (GPU, DLPack re-import with large extents)',
                         registered=c.get('expected'), F3=dict(raised=c.get('raised'), launches=c.get('launches'),
                                                               ok=c['ok']),
                         F2=dict(result=f2.get('result'), launches=f2.get('launches'), context_after=f2.get('context_after'),
                                 child_exit=f2.get('child_exit')),
                         changed=(c.get('raised') or '').split(':')[0] != (f2.get('result') or '').split(':')[0] or
                         (c.get('raised') or '') != (f2.get('result') or '')))
    r1 = [c for c in lf3['checks'] if c['id'].startswith("LF3 R1'")][0]
    f2r1 = bnd['r1p_stateful_mapping']
    rows.append(dict(ruling="R1'", input='public route, hand-built PackedRegisteredMargins with a dict-subclass mapping '
                     'that returns the pinned values to the first 4 item reads and 0.0 after (LF3, GPU)',
                     F3=dict(result='launched with the pinned e_q', e_q=r1.get('e_q'), mapping_reads=r1.get('mapping_reads'),
                             output_bitwise_equal_to_pinned=r1.get('equal'), launches=r1.get('launches')),
                     F2=dict(result=f2r1.get('result'), e_q=f2r1.get('e_q_at_use'), mapping_reads=f2r1.get('mapping_reads'),
                             output_bitwise_equal_to_pinned=f2r1.get('output_bitwise_equal_to_pinned'),
                             launches=f2r1.get('launches')),
                     changed=True))
    cpu = {side: r1p_cpu(side) for side in ('F3', 'F2')}
    for what, n in (('lookup', 2), ('attention', 4)):
        rows.append(dict(ruling="R1'", input=f'CPU recording fakes: stateful mapping pinned to {n} reads, at {what} '
                         '(D64 lookup / D128 attention, epsilon 0.01)', F3=cpu['F3'][what], F2=cpu['F2'][what],
                         changed=cpu['F3'][what] != cpu['F2'][what]))
    s4 = json.loads((ART / 'cpu/S4.json').read_text())
    host = sorted(k for k in s4['reports'] if '::test_r3p_' in k)
    rows.append(dict(ruling="R3'", input='S4 host-level fake (real guard and EB2b prologue text, g++ -O0 -fwrapv, stub '
                     'NDArray): B*H at bound, one past, 65536^2; extents 2^31 on k axis 2 and q axis 0; q B = 0',
                     F3=dict(tests=host, outcomes=sorted({r['outcome'] for k in host for r in s4['reports'][k]})),
                     F2='the same inputs on the real Build F2 entries are the LF3 rows above (eb2b_BH_*, extent_*, zero_*)',
                     changed=None))
    data = dict(order='APA-P1-F3', rows=rows, n_rows=len(rows),
                note=('Every R3\' row refuses before any launch on Build F3 (launches 0). Build F2 also refused every '
                      'large-extent input before any launch, but with RuntimeError or with another entry\'s ValueError '
                      '(the capacity check, or packed_rotate\'s B*H after a wrapped EB2b product). Zero extents keep '
                      'the RuntimeError on both builds. R1\': Build F2 used the 0.0 read at use (e_q 0.0); Build F3 '
                      'uses the pinned value.'))
    with (ART / 'BOUNDARY.json').open('x') as f:
        json.dump(data, f, indent=2, default=str)
        f.write('\n')
    print(json.dumps([dict(input=r['input'][:60], changed=r['changed']) for r in rows], indent=1))
    print('R1 cpu', json.dumps(cpu, default=str))
    return 0


if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    sys.exit(main())
