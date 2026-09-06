"""Amendment-4 cell overlay; the base registration and base registry stay intact.

Prior art: dependency DAGs and checkpoint/restart (standard systems practice;
no prior art known to me for this specific range layout). New: SP3 job wiring.
"""
from apa_sp3_common import ART, ROOT, REG_SHA, Red, read, sha

ORDER_SHA = 'e2b9dd3c074688d9ac457183f9219ea9ad08a29f006232942e6001cf19c0b236'
MANIFEST = 'amendment_005_long_context.json'
CORRECTION = 'amendment_007_execution_details.json'
KINDS = ('capture_range', 'capture_aggregate', 'torch_reference', 'ceiling', 'ppl_long')


def planned(base):
    out = []
    def add(id, kind, depends, estimate, **kw):
        c = dict(id=id, kind=kind, depends=depends, estimate_s=estimate,
                 worker_timeout_s=290 if kind in ('capture_range', 'ceiling') else 480,
                 job_ceiling_s=590, optional_secondary=kw.get('bits') == 8,
                 estimate_scope='planning only; 60s setup/guard + 8s/layer at 8192; quadratic x16 at 32768', **kw)
        out.append(c)
        return id
    for S in (1024, 8192, 32768):
        add(f'ppl_T_{S}', 'torch_reference', ['g0'], [30, 480], arm='T', S=S, bits=16)
    for arm in 'ABC':
        deps = ['g0'] + (['ppl_b4_D_1024', 'freeze_b4'] if arm == 'C' else [])
        for S in (4096, 8192, 16384, 24576, 32768):
            add(f'ceiling_b4_{arm}_{S}', 'ceiling', deps, [30, 280], arm=arm, bits=4, S=S)
    for arm in 'BCD':
        deps = ['g0', 'kernel96', 'ppl_b4_D_1024', 'ppl_T_32768']
        if arm in 'BC':
            deps += [f'ceiling_b4_{arm}_32768']
        if arm == 'C':
            deps += ['freeze_b4']
        add(f'ppl_b4_{arm}_32768', 'ppl_long', deps, [60, 480], arm=arm, bits=4, S=32768)
    captures = [c for c in base if c['kind'] == 'capture' and c['S'] == 8192]
    captures += [dict(id=f'capture_b4_{arm}_32768', arm=arm, bits=4, S=32768,
                      depends=[f'ppl_b4_{arm}_32768']) for arm in 'BC']
    for c in captures:
        previous = None
        ranges = []
        width = 16 if c['S'] == 8192 else 1
        for lo in range(0, 62, width):
            hi = min(62, lo + width)
            id = f"{c['id']}_r{lo:02d}_{hi:02d}"
            estimate = 60 + 8 * (c['S'] // 8192)**2 * (hi - lo)
            ranges.append(add(id, 'capture_range', c['depends'] + ([previous] if previous else []),
                              [30, estimate], arm=c['arm'], bits=c['bits'], S=c['S'],
                              layer_start=lo, layer_stop=hi, predecessor=previous,
                              capture_id=c['id']))
            previous = id
        add(c['id'], 'capture_aggregate', ranges, [5, 280],
            arm=c['arm'], bits=c['bits'], S=c['S'], layers=62)
    return out


def manifest():
    m = read(ART / MANIFEST)
    if (m['registration_sha256'] != REG_SHA or m['order_sha256'] != ORDER_SHA
            or sha(ROOT/'orders/APA_SP3_AMENDMENT_4.md') != ORDER_SHA):
        raise Red('A4 registration/order binding changed')
    # Independent checksum is sealed before gates, beside this immutable JSON.
    if sha(ART/MANIFEST) != (ART/(MANIFEST+'.sha256')).read_text().strip():
        raise Red('A4 cell amendment changed')
    if (ART/CORRECTION).exists():
        change = read(ART/CORRECTION)
        if (change['registration_sha256'] != REG_SHA or change['order_sha256'] != ORDER_SHA
                or change['parent_sha256'] != sha(ART/MANIFEST)
                or sha(ART/CORRECTION) != (ART/(CORRECTION+'.sha256')).read_text().strip()):
            raise Red('A4 execution correction changed')
        for c in m['cells']:
            if c['id'] in change['dependency_overrides']:
                c['depends'] = change['dependency_overrides'][c['id']]
    return m


def overlay(base):
    m = manifest()
    expected = planned(base)
    # The separate pre-gate correction removes only the unintended T->B/C
    # 32K dependency; T remains mandatory for D, per the lead order.
    for c in expected:
        if c['id'] in ('ppl_b4_B_32768', 'ppl_b4_C_32768'):
            c['depends'] = [d for d in c['depends'] if d != 'ppl_T_32768']
    if expected != m['cells']:
        raise Red('A4 implementation differs from registered cells')
    updates = {c['id']: c for c in m['cells']}
    pending = [c for c in base if c['id'] not in updates] + m['cells']
    # Prior art: Kahn (1962) topological scheduling. Reused DAG ordering;
    # new: deterministic stable order of SP3 additions, no novel algorithm.
    done, out = set(), []
    while pending:
        ready = [c for c in pending if set(c['depends']) <= done]
        if not ready:
            raise Red('A4 registry has cycle or unknown dependency')
        for c in ready:
            if c['id'] in done:
                raise Red('duplicate cell')
            out.append(c)
            done.add(c['id'])
            pending.remove(c)
    return out
