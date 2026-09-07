"""A6 clean timing and P5. Prior art: controlled throughput ratios and A5
reporting (2026) reused; registered 2x threshold, no new metric/algorithm.
"""
import json
import math
from apa_sp3_common import ART, require_pass, receipt_validation, job_path, read
from apa_sp3_a6_registry import manifest, KINDS


def valid_measurement(j,cell):
    r=j.get('result',{})
    return (j.get('status')=='PASS' and j.get('cell')==cell
            and r.get('fit') is True and r.get('steps')==32
            and r.get('config')==cell['config'] and r.get('feeding')==cell['feeding']
            and all(r.get(k)==cell[k] for k in ('bits','arm','S'))
            and all(isinstance(r.get(k),(int,float)) and math.isfinite(r[k]) and r[k]>0
                    for k in ('tokens_s','ms_token')))


def prediction(jobs):
    by={c['id']:c for c in manifest()['cells']}
    def ratio(S):
        ids=[f'decode_clean_b4_{a}_{S}' for a in 'BC']
        if not all(valid_measurement(jobs.get(n,{}),by[n]) for n in ids):return None
        return jobs[ids[1]]['result']['tokens_s']/jobs[ids[0]]['result']['tokens_s']
    r32,r8=ratio(32768),ratio(8192)
    return dict(status='UNASSESSABLE' if r32 is None else 'HIT' if r32>=2 else 'MISSED',
                ratio=r32,ratio_8192=r8,source_kind='decode_clean',bits=4,S=32768,threshold=2.,
                reason='32K clean measurement absent/non-fit/invalid; 8192 ratio does not evaluate P5' if r32 is None else 'measured clean C/B at 32K')


def reproduction(jobs):
    c=manifest()['cells'][0];j=jobs.get(c['id'],{})
    if not valid_measurement(j,c):
        return dict(status='BLOCKED',ms_token=None,comparison='UNASSESSED')
    ms=j['result']['ms_token']
    return dict(status='MEASURED',ms_token=ms,ratio_to_june=ms/21.6,
                comparison='WITHIN_2X_HARNESS_FINDING' if ms<=43.2 else 'OUTSIDE_2X_ENGINE_ON_CARD_FINDING',
                scope='lead decision fork; June S~360/8501a5c vs current pinned engine/S2048; not an isolated hardware comparison')


def render(art,jobs,cells):
    reg=manifest();p5=prediction(jobs);repro=reproduction(jobs)
    rows=[]
    text=['## A6 clean decode — authoritative decode/P5 column', '',
          'Supersedes the historical raw/pool decode P5 columns. GPU timings are absent until the lead runs each cell. '
          '32 synchronized token steps, scalar device argmax copied to host; forward-only times also retained. '
          'One discarded warmup preserves prefill cache and timed contexts. Pool enabled before weights; '
          'pool reserved/used high water is not a whole-device resident peak. A uses absorbed MLA; '
          'B/C retain expanded selective attention with fast projections/norms. No per-layer diagnostic hook.', '',
          '| Cell | Status/outcome | ms/token | Forward ms/token | tokens/s | Pool peak MiB |',
          '|---|---|---:|---:|---:|---:|']
    for c in reg['cells']:
        j=jobs.get(c['id'],{});r=j.get('result',{}) if j.get('status')=='PASS' else {}
        row=dict(cell=c['id'],status=r.get('outcome',j.get('status','GPU_BLOCKED')),
                 ms_token=r.get('ms_token'),forward_ms_token=r.get('forward_ms_token'),
                 tokens_s=r.get('tokens_s'),pool_reserved_peak_mib=r.get('pool_reserved_peak_mib'))
        if not j and c.get('nonfit'):row['status']=c['nonfit']+' (registered, no worker)'
        rows.append(row)
        text.append('| '+' | '.join('—' if v is None else str(v) for v in row.values())+' |')
    text += ['', 'Reproduction: `'+json.dumps(repro,sort_keys=True)+'`.',
             'P5: `'+json.dumps(p5,sort_keys=True)+'`.',
             'Rungs 01–10 each compare with 00 and change exactly one configuration field. Rung 11 is '
             'a combined legacy-flag endpoint, not a single-factor attribution. The old A5 loop also had '
             'in-process PPL controls and lacked argmax/warmup, so 11 is not a byte-for-byte A5 replay.',
             '32K B/C plans require same-bit/arm clean8192 receipt: setup + 16×prefill + '
             '4×(warmup+decode work) + 15 seconds. At or above 290s, record non-fit before lease. '
             'Dense A8192/32768 remains registered non-fit. Missing measurements block planning.', '']
    commands=['# A6 only: run ONE foreground command at a time, never execute this file as a batch.',
              '# Retain the lead original build and all prior receipts. Do not merge this seat build artifacts.',
              '# Worker 290s +5s grace; outer 585s +3s grace (<10min); lease wait <=20s; cooldown 30s.',
              '# No automatic retry. Independent ladder rungs do not depend on a previous rung passing.',
              'cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3',
              'timeout 30s python3 scripts/apa_sp3_a6_audit.py',
              'timeout 30s bash scripts/apa_sp3_lead_gpu.sh list']
    for c,row in zip(reg['cells'],rows):
        commands.append(f"# {c['id']} | deps={','.join(c['depends'])} | estimate={c['estimate_s']}s | {c['estimate_scope']} | {row['status']}")
        if job_path(c['id']).exists():
            commands.append('# Existing immutable receipt: retain; RED/stale needs lead decision, never retry automatically.')
        else:commands.append(f"timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run {c['id']}")
    commands.append('timeout 30s python3 scripts/apa_sp3_a6_report.py')
    (art/'lead_commands.txt').write_text('\n'.join(commands)+'\n')
    data=dict(evidence_class='registered plan / validated receipt reporting',reproduction=repro,P5=p5,cells=rows)
    (art/'a6_results.json').write_text(json.dumps(data,indent=2)+'\n')
    (art/'A6_TABLES.md').write_text('\n'.join(text)+'\n')
    return text,data


def main():
    jobs={}
    for c in manifest()['cells']:
        if job_path(c['id']).exists():
            try:jobs[c['id']]=require_pass(c['id'])
            except Exception as e:jobs[c['id']]=dict(status='RED_OR_STALE',error=str(e))
    from apa_sp3_gpu import cells
    _,r=render(ART,jobs,cells())
    print(json.dumps(dict(reproduction=r['reproduction'],P5=r['P5'],cells=len(r['cells']))))


if __name__=='__main__':
    with receipt_validation():main()
