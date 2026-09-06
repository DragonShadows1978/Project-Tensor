"""A5 pool timing/P5 and lead-selected command list.

Prior art: conventional controlled benchmark ratios and dependency-ordered
experiment tables; no new metric. P5 threshold is the lead's original 2x.
"""
import json
import math
from apa_sp3_a5_registry import default_cells


def prediction(jobs):
    rows = [jobs.get(f'decode_pool_b4_{arm}_32768', {}) for arm in 'BC']
    if any(j.get('status') != 'PASS' or j.get('result', {}).get('fit') is not True
           or j['result'].get('alloc_pooling') is not True or j['result'].get('steps', 0) < 32
           for j in rows):
        return dict(status='UNASSESSED', ratio=None, source_kind='decode_pool', threshold=2.)
    values = [j['result'].get('tokens_s') for j in rows]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
        return dict(status='UNASSESSED', ratio=None, source_kind='decode_pool', threshold=2.)
    ratio = values[1] / values[0]
    return dict(status='HIT' if ratio >= 2. else 'MISSED', ratio=ratio,
                source_kind='decode_pool', threshold=2., bits=4, S=32768)


def render(art, jobs, cells, include_32k_captures=False):
    p5 = prediction(jobs)
    text = ['## A5 production-pool decode — kernel sweep / in-model timing', '',
            'Pool ON after raw persistent weight loading, including in-process controls and all decode forwards. '
            '32 teacher-forced tokens, per-token CUDA-synchronized wall time. Prefill excluded from tokens/s; '
            'setup, controls and prefill included in worker TERM 290s (+5s grace). '
            '32K plans use same-arm/bit pool-on 8192: setup + guard + 16*prefill + 4*decode_work + 15 seconds. '
            'Missing measurement blocks; estimate >=290s is terminal non-fit before lease.', '',
            '| Bits | S | Arm | Status/outcome | tokens/s | ms/token | Pool reserved peak MiB | Planning seconds |',
            '|---:|---:|---|---|---:|---:|---:|---|']
    rows = []
    for c in cells:
        if c['kind'] != 'decode_pool':
            continue
        j = jobs.get(c['id'], {})
        r = j.get('result', {}) if j.get('status') == 'PASS' else {}
        row = dict(bits=c['bits'], S=c['S'], arm=c['arm'],
                   status=r.get('outcome', j.get('status', 'UNRUN')),
                   tokens_s=r.get('tokens_s'), ms_token=r.get('ms_token'),
                   pool_reserved_peak_mib=r.get('pool_reserved_peak_mib'),
                   estimate_s=r.get('planning', {}).get('estimate_s', r.get('estimate_s', c['estimate_s'])))
        rows.append(row)
        text.append('| '+' | '.join('—' if v is None else str(v) for v in row.values())+' |')
    text += ['', 'Peak source: CUDA default-pool ReservedMemHigh and UsedMemHigh, reset before measured prefill. '
             '`peak_resident_mib` is a compatibility alias for reserved pool high water, **not whole-device resident peak**; '
             'raw weights and driver/context allocations are excluded. Legacy pool-off decode receipts retain exact intercepted-allocation evidence and are never scheduled by default.',
             'P5 (bulk4, C/B at starting S=32768): `'+json.dumps(p5, sort_keys=True)+'`. '
             'Only valid `decode_pool` receipts qualify. No pool-off fallback. Contexts 32769–32800 exceed the trained window. '
             '**G2/G3 rows establish nothing about model quality by themselves.**', '',
             '32K captures remain registered but are OFF by default. Explicit `run CELL` is lead cell-list inclusion; '
             '`APA_SP3_INCLUDE_32K_CAPTURES=1` opts into resume/command generation. '
             'The separately generated `lead_commands_32k_captures.txt` is a lead opt-in list. '
             'Existing a4 disk preflight is unchanged; its conservative 490,783,899,647-byte first-range requirement '
             'exceeds the lead-reported free disk. This amendment does not relax that rail.', '']
    commands = ['# A5: after lead merge; one foreground command at a time; never execute this file as a batch.',
                '# Existing receipts retained. Pool-off decode excluded. 32K captures require explicit opt-in.',
                '# Decode pool worker TERM 290s +5s grace; all prior kinds retain registered rails. Outer bound 590s.',
                '# Estimates are planning only. Add <=20s lease, 30s cooldown and setup/receipt overhead.',
                'cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3',
                'timeout 30s bash scripts/apa_sp3_lead_gpu.sh list',
                'timeout 30s python3 scripts/apa_sp3_a5_audit.py']
    for c in default_cells(cells, include_32k_captures):
        st = jobs.get(c['id'], {}).get('status', 'UNRUN')
        estimate = c['estimate_s']
        if c['kind'] == 'decode_pool' and c['S'] == 32768:
            estimate = 'pending '+c['estimate_from']+'; setup+guard+16*prefill+4*decode_work+15; >=290s NON_FIT'
        commands.append(f"# {c['id']} | kind={c['kind']} | deps={','.join(c['depends']) or '-'} | worker estimate={estimate}s | status={st}")
        if st == 'PASS':
            commands.append('# Existing valid receipt retained; no rerun (including terminal non-fit).')
        elif st in ('RED', 'STALE'):
            commands.append('# Existing RED/stale retained; lead resolves branch, no automatic retry.')
        else:
            commands.append(f"timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run {c['id']}")
    commands.append('timeout 30s bash scripts/apa_sp3_lead_gpu.sh summary')
    (art/'lead_commands.txt').write_text('\n'.join(commands)+'\n')
    opted = ['# OFF by default. Lead-only explicit cell-list inclusion; not an automatic batch.',
             '# Keep a4 disk rail and per-layer predecessors; lead decides whether to run.',
             'cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3']
    for c in cells:
        if c['kind'] in ('capture_range', 'capture_aggregate') and c.get('S') == 32768:
            opted += [f"# deps={','.join(c['depends'])}; estimate={c['estimate_s']}s",
                      f"timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run {c['id']}"]
    (art/'lead_commands_32k_captures.txt').write_text('\n'.join(opted)+'\n')
    data = dict(decode_pool=rows, P5=p5, default_32k_captures=include_32k_captures)
    (art/'a5_results.json').write_text(json.dumps(data, indent=2)+'\n')
    (art/'A5_TABLES.md').write_text('\n'.join(text)+'\n')
    return text, data
