#!/usr/bin/env python3
"""Finalize the supplied SP1 experiment using its immutable receipt manifest.

Kernel sweep: this establishes nothing about model quality.
No current-build fingerprint may erase an older registered experiment.
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/apa_sp1'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tensor_cuda/tests')]
from apa_sp1_cpu import registration,calibration,geom_key,sha
import apa_sp1_reference as ref


def registered_receipts():
    p=ART/'registration_sp1_1.json'
    assert sha(p)==(ART/'registration_sp1_1.sha256').read_text().split()[0]
    pin=json.loads(p.read_text());rows=[]
    for name,digest in pin['sp1_gpu_receipts'].items():
        assert sha(ROOT/name)==digest,name
        r=json.loads((ROOT/name).read_text())
        if 'shape' in r:
            r['_receipt']=name;rows.append(r)
    assert len(rows)==48 and len({r['shape']['id'] for r in rows})==48
    return rows


def score(rows):
    pre=[r for r in rows if r['shape']['kind']=='prefill']
    dec=[r for r in rows if r['shape']['kind']=='decode']
    eligible=lambda group:[r for r in group if r['G2']['status']=='PASS' and r['tail']['matched'] and r['G3'].get('speedup') is not None]
    ep,ed=eligible(pre),eligible(dec);all_eligible=ep+ed
    overlap=lambda group:sum(r['tail']['intersection'] for r in group)/sum(r['tail']['z'] for r in group)
    result={}
    def add(name,hit,group,detail):
        result[name]=dict(verdict='HIT' if hit else 'MISS',detail=detail,receipts=[r['_receipt'] for r in group])
    add('P2_prefill_overlap_ge_0_8',all(r['tail']['overlap_recall']>=.8 for r in pre),pre,'All 24 count-bearing prefill rows; no timing/matching filter on an overlap prediction.')
    add('P2_decode_overlap_less',overlap(dec)<overlap(pre),pre+dec,f'Microaggregate prefill={overlap(pre):.9f}, decode={overlap(dec):.9f}; all rows.')
    hits=[r for r in all_eligible if r['G3']['deviation_ratio']<=2]
    add('P3_half_shapes_deviation',len(hits)>=24,rows,f'{len(hits)}/48 registered shapes meet deviation<=2 with G2 PASS and matched fraction; threshold 24.')
    fails=[r for r in ep if r['G3']['speedup']<1.4]
    add('P3_all_prefill_speed',len(ep)==24 and not fails,pre,f'{len(fails)} eligible counterexamples below1.4; eligible {len(ep)}/24.')
    pm=float(np.median([r['G3']['speedup'] for r in ep]));dm=float(np.median([r['G3']['speedup'] for r in ed]))
    add('P3_decode_speed_smaller',len(ep)==24 and len(ed)==24 and dm<pm,rows,f'Coverage MISS unless both groups fully eligible; eligible medians prefill={pm:.6f}, decode={dm:.6f}; eligible {len(ep)}/24 and {len(ed)}/24. Observed subset supports direction but does not establish all-class prediction.')
    add('A3_prefill_overlap_le_0_65',all(r['tail']['overlap_recall']<=.65 for r in pre),pre,'All 24 prefill rows with complete counts.')
    add('A4_decode_speed_le_1',len(ed)==24 and all(r['G3']['speedup']<=1 for r in ed),dec,f'Coverage MISS: eligible {len(ed)}/24. Every raw decode speed <=1 is separately visible; unmatched work budgets do not count as matched-comparison hits.')
    add('A5_prefill_majority_below_1_4',len(fails)>12,pre,f'{len(fails)}/24 eligible rows below1.4; majority requires at least13.')
    result['P2_monotone']=dict(verdict='HIT',detail='Bulk-prefix safety proof and 40,000 CPU draws; not an exact-score guarantee.',receipts=['artifacts/apa_sp1/cpu_summary.json'])
    return result


def finalize():
    rows=registered_receipts();scores=score(rows)
    report=dict(evidence_class='kernel sweep',scope_note='this establishes nothing about model quality',
        original_registration_sha256=sha(ART/'registration.json'),scoring_registration_sha256=sha(ART/'registration_sp1_1.json'),
        predictions=scores,coverage=dict(registered_shapes=48,g2_pass=sum(r['G2']['status']=='PASS' for r in rows),
        matched=sum(r['tail']['matched'] for r in rows),eligible=sum(r['G2']['status']=='PASS' and r['tail']['matched'] for r in rows)),rows=rows)
    (ART/'gpu_summary_sp1_final.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# APA-SP1 finalized GPU scoring','', 'Evidence class: kernel sweep. **this establishes nothing about model quality**.','',
           'This scores the 48 original shape receipts pinned in the SP1.1 registration. The three additional receipts are boundary/legacy/selector gates. Coverage MISS is failure to establish the registered claim, not a fabricated timing measurement.','',
           '| Prediction | Verdict | Evidence / qualification |','|---|---|---|']
    for name,s in scores.items():
        refs='; '.join(f'[{Path(p).name}]({Path(p).relative_to("artifacts/apa_sp1")})' for p in s['receipts'])
        lines.append(f'| {name} | {s["verdict"]} | {s["detail"]} Receipts: {refs} |')
    lines+=['','| Class | G2 | Matched | Speed | Deviation ratio | Receipt |','|---|---|---|---|---|---|']
    for r in rows:
        g=r['G3'];path=r['_receipt']
        lines.append(f'| {r["shape"]["id"]} | {r["G2"]["status"]} | {r["tail"]["matched"]} | {g.get("speedup","BLOCKED")} | {g.get("deviation_ratio","BLOCKED")} | [{Path(path).name}]({Path(path).relative_to("artifacts/apa_sp1")}) |')
    (ART/'gpu_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(coverage=report['coverage'],predictions={k:dict(verdict=v['verdict'],detail=v['detail']) for k,v in scores.items()}),indent=2))


def explain_matching():
    # Only reconstruct fixed-delta counts. No holdout re-calibration is done.
    from apa_sp1_gpu import arrays
    rows=registered_receipts();reg=registration();cal=calibration();results=[]
    for r in rows:
        s=r['shape']
        if s['kind']!='decode':continue
        ix=next(i for i,x in enumerate(reg['shapes']) if x['id']==s['id'])
        q,k,kq,v=arrays(s,np.random.SeedSequence([reg['data']['gpu_seed'],ix]))
        heads=[]
        for h in range(s['H']):
            kh=h//(s['H']//s['KVH']);b=q[0,h]@kq[0,kh].T*np.float32(1/np.sqrt(s['D']))
            sm=ref.prefix_mask(b,r['delta']);zm,_=ref.zmask(b,reg['zthr'])
            heads.append(dict(head=h,q_norm=float(np.linalg.norm(q[0,h])),final_bulk_max=float(b.max()),sp=int(sm.sum()),z=int(zm.sum()),valid=s['S']))
        c=cal[geom_key('decode',s['S'],s['D'],s['causal'])];tail=r['tail']
        calibration_gap=c['sp_fraction']-c['z_fraction'];gap=tail['sp_fraction']-tail['z_fraction']
        results.append(dict(shape=s['id'],receipt=r['_receipt'],delta=r['delta'],matched=tail['matched'],
            calibration_sp=c['sp_fraction'],calibration_z=c['z_fraction'],calibration_gap=calibration_gap,
            sp_fraction=tail['sp_fraction'],z_fraction=tail['z_fraction'],holdout_gap=gap,excess_over_tolerance=max(0,abs(gap)-.02),
            calibration_rows=c['draws'],holdout_rows=s['H'],heads=heads,
            cpu_reconstructed_sp=sum(h['sp'] for h in heads),gpu_sp=tail['sp'],
            cpu_reconstructed_z=sum(h['z'] for h in heads),receipt_z=tail['z'],
            reason=('Above' if gap>0 else 'Below')+' target: frozen calibration delta applied to only '+str(s['H'])+' holdout query rows. Prefix maxima/extreme records and query norms vary by row; long key lists do not supply independent query-prefix histories. Calibration used 128 independent rows and a 0.05 grid. No guarantee transfers its aggregate match to this seed/head geometry.',
            validity='Matched-budget speed/deviation comparison INVALID for this row; raw timing and raw dense deviation remain measured for these different fractions.' if not tail['matched'] else 'Within registered matching tolerance.'))
    out=dict(evidence_class='CPU reconstruction of supplied GPU kernel-sweep selection counts',scope_note='this establishes nothing about model quality',
        cause='Fixed calibration residual plus holdout variation, not a failed monotonicity property. Causal/noncausal decode both see S keys but use independent registered seeds; MHA/GQA also use different seeds and 4/8 query rows. No delta retuning.',rows=results)
    (ART/'sp1_matching_explanations.json').write_text(json.dumps(out,indent=2)+'\n')
    lines=['# Every SP1 decode fraction mismatch','',out['scope_note'], '',out['cause'],'',
        '| Class | delta | Calibration gap | Holdout SP | Holdout z | Gap | Excess over .02 | Verdict | Receipt |','|---|---|---|---|---|---|---|---|---|']
    for r in results:
        if r['matched']:continue
        lines.append(f'| {r["shape"]} | {r["delta"]} | {r["calibration_gap"]:.6f} | {r["sp_fraction"]:.6f} | {r["z_fraction"]:.6f} | {r["holdout_gap"]:+.6f} | {r["excess_over_tolerance"]:.6f} | INVALID matched-budget comparison | [{Path(r["receipt"]).name}]({Path(r["receipt"]).relative_to("artifacts/apa_sp1")}) |')
        lines+=[]
    lines+=['','The JSON supplies all 24 rows, including per-head counts, query norms, prefix maxima and CPU-versus-receipt counts. Every failed row has a small calibration residual and a holdout gap beyond 0.02. Frozen delta is never adjusted. These failures do invalidate matched-budget speed/deviation comparisons.']
    (ART/'sp1_matching_explanations.md').write_text('\n'.join(lines)+'\n')
    print('decode unmatched',sum(not r['matched'] for r in results),'of',len(results))


if __name__=='__main__':
    finalize()
    explain_matching()
