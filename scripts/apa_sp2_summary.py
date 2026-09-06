"""Complete-grid curve and explicit prediction scoring; no epsilon selection."""
import json
from apa_sp2_common import ART, ROOT, NOTE, sha, registration, latest, targets, fingerprint


def crossing(rows, predicate):
    for row in rows:
        if row is None or row.get('status')!='PASS':
            return 'BLOCKED_INCOMPLETE'
        if predicate(row):
            return row['epsilon']
    return 'NOT_OBSERVED_ON_GRID'


def score(reg, curves, table):
    result={}
    if table:
        value=table['entries']['2:128']['e_q']
        result['P1']=dict(verdict='HIT' if value>=.05 else 'MISS',observed=value,receipt='artifacts/apa_sp2/e_q_table.json')
    else:result['P1']=dict(verdict='BLOCKED_G2',observed=None)
    def universal(name, rows, predicate, expected):
        ready=[r for r in rows if r and r['status']=='PASS']
        failures=[r['_receipt'] for r in ready if not predicate(r)]
        result[name]=dict(verdict='MISS' if failures else ('HIT' if len(ready)==expected else 'BLOCKED_INCOMPLETE'),
                          measured=len(ready),expected=expected,failures=failures,
                          receipts=[r['_receipt'] for r in ready])
    peak=[];short=[];long=[]
    for c in curves:
        s=c['shape']
        islong=s['S']==(8192 if s['kind']=='prefill' else 32768)
        if s['causal'] and islong:peak.append(c['rows'][4])
        if not s['causal'] and not islong:short.append(c['rows'][4])
        if islong:long.extend(c['rows'][:5])
    universal('P2_peaked_label',peak,lambda r:r['fractions']['sp']<r['fractions']['baseline'],16)
    universal('P2_short_noncausal',short,lambda r:r['fractions']['sp']>r['fractions']['baseline'],16)
    parts=[result[k]['verdict'] for k in ('P2_peaked_label','P2_short_noncausal')]
    result['P2']=dict(verdict='MISS' if 'MISS' in parts else ('HIT' if parts==['HIT','HIT'] else 'BLOCKED_G3'),
                      caveat='causal long-S is the registered group label; no peaked distribution was assumed')
    p3=[]
    for ei in range(2,7):
        for metric in ('relative_frobenius','max_abs'):
            rows=[c['rows'][ei] for c in curves];ready=[r for r in rows if r and r['status']=='PASS']
            wins=sum(r['sp_vs_dense_fp32'][metric]<r['baseline_vs_dense_fp32'][metric] for r in ready)
            verdict=('HIT' if wins>=48 else 'MISS') if len(ready)==64 else 'BLOCKED_INCOMPLETE'
            key=f'P3_e{ei}_{metric}'
            result[key]=dict(verdict=verdict,epsilon=reg['epsilon_grid'][ei],wins=wins,expected=64,required=48,
                              measured=len(ready),receipts=[r['_receipt'] for r in ready])
            if metric=='relative_frobenius':p3.append(verdict)
    result['P3']=dict(verdict='MISS' if 'MISS' in p3 else ('HIT' if all(v=='HIT' for v in p3) else 'BLOCKED_G3'))
    universal('P4',long,lambda r:r['timing'].get('speedup',0)>=1.,160)
    return result


def summary():
    reg=registration();curves=[]
    for bits in reg['bits']:
        for s in reg['shapes']:
            rows=[latest(f'sweep_b{bits}_{s["id"]}_e{i}') for i in range(7)]
            # A changed frozen table never silently reuses older curve rows.
            pin=sha(ART/'e_q_table.json') if (ART/'e_q_table.json').exists() else None
            rows=[r if r and r.get('margin_table_sha256')==pin else None for r in rows]
            c=dict(bits=bits,shape=s,rows=rows)
            c['first_deviation_below_baseline']=crossing(rows,lambda r:r['sp_vs_dense_fp32']['relative_frobenius']<r['baseline_vs_dense_fp32']['relative_frobenius'])
            c['first_max_abs_below_baseline']=crossing(rows,lambda r:r['sp_vs_dense_fp32']['max_abs']<r['baseline_vs_dense_fp32']['max_abs'])
            c['first_speed_below_baseline']=crossing(rows,lambda r:r['timing'].get('speedup',0)<1.)
            curves.append(c)
    table=json.loads((ART/'e_q_table.json').read_text()) if (ART/'e_q_table.json').exists() else None
    if table:
        if table.get('fingerprint')!=fingerprint():
            raise RuntimeError('BLOCKED: stale frozen margin table; do not publish a current curve')
        assert sha(ART/'e_q_table.json')==(ART/'e_q_table.sha256').read_text().split()[0]
        for entry in table['entries'].values():
            for path,digest in entry['receipts'].items():
                assert sha(ROOT/path)==digest
    predictions=score(reg,curves,table)
    ready=[r for c in curves for r in c['rows'] if r and r['status']=='PASS']
    payload=dict(evidence_class=NOTE,registration_sha256=sha(ART/'registration.json'),
                 status='COMPLETE' if len(ready)==448 else 'BLOCKED_INCOMPLETE',
                 completed_epsilon_rows=len(ready),expected_epsilon_rows=448,
                 predictions=predictions,curves=curves,epsilon_recommendation=None)
    (ART/'epsilon_curve.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    lines=['# APA-SP2 epsilon curve for David','',NOTE+'.','',
           f'Status: **{payload["status"]}**; {len(ready)}/448 epsilon rows. No epsilon is selected or recommended.',
           '', 'The table uses independent standard-normal queries and keys, TurboQuant 2/4-bit reconstructed fp32 keys, and the unchanged two-pass kernel at percentile 0.15 (z=1.0364333894937898). Timings exclude quantization and diagnostics for both paths. Each epsilon has a same-invocation interleaved baseline. Dense fp32 deviations cover every output element. Causal decode sees all keys; a causal label does not itself establish peaked attention.', '',
           'Read each class from larger to smaller epsilon: the derived margin grows, and for fixed inputs and margins the refine set can only grow. Neither output deviation nor measured speed has a monotonicity theorem. Crossings are the first strict improvement/regression in that grid order, separately for relative Frobenius and max abs; missing earlier rows block a crossing claim. Margin exceedances are reported without fitting the table to the holdout. With no GPU receipts there is no empirical reading of the curve yet. The eventual choice belongs to David.', '',
           '## Margin table','', '| Bits | D | Count | Max | p99.9 | Mean | Registered e_q |','|---|---|---|---|---|---|---|']
    for bits in (2,4):
        for D in (64,128):
            e=table['entries'][f'{bits}:{D}'] if table else None
            vals=' | '.join(str(e[k]) for k in ('count','max','p99_9','mean','e_q')) if e else 'BLOCKED G2 | — | — | — | —'
            lines.append(f'| {bits} | {D} | {vals} |')
    lines+=['','Rule: maximum over all registered calibration logits, rounded upward to fp32, pooled by (bits,D). This is finite calibration evidence, not a universal bound. p99.9 is descriptive only.','',
            'RED reasoning bound: skipped probability mass <= min(1,N*epsilon*w_star); perturbation L1 <= min(2,2*N*epsilon*w_star*(1-exp(-e_q))). At N=32768 and epsilon=1e-3, N*epsilon=32.768; at epsilon=1e-4 it is 3.2768. Without an observed w_star or tighter information the worst-case mass bound is vacuous even at the smallest grid epsilon. These are arithmetic bounds, not measured errors.','',
            '## Predictions','', '| Prediction | Verdict |','|---|---|']
    for key in ('P1','P2','P3','P4'):lines.append(f'| {key} | {predictions[key]["verdict"]} |')
    for c in curves:
        lines+=['',f'## {c["shape"]["id"]}, {c["bits"]} bits','',
                '| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |',
                '|---|---|---|---|---|---|---|']
        first=next((r for r in c['rows'] if r and r['status']=='PASS'),None)
        if first:
            m=first['baseline_vs_dense_fp32'];lines.append(f'| Two-pass p=0.15 | {first["fractions"]["baseline"]:.8g} | {m["relative_frobenius"]:.8g} | {m["max_abs"]:.8g} | 1 | reference | {first["_receipt"]} |')
        else:lines.append('| Two-pass p=0.15 | BLOCKED | — | — | — | unmeasured | — |')
        for eps,r in zip(reg['epsilon_grid'],c['rows']):
            if r and r.get('status')=='PASS':
                m=r['sp_vs_dense_fp32'];lines.append(f'| {eps:g} | {r["fractions"]["sp"]:.8g} | {m["relative_frobenius"]:.8g} | {m["max_abs"]:.8g} | {r["timing"]["speedup"]:.6g} | {r["guarantee"]["status"]} | {r["_receipt"]} |')
            else:lines.append(f'| {eps:g} | {r["status"] if r else "BLOCKED G3"} | — | — | — | unmeasured | {r["_receipt"] if r else "—"} |')
        lines+=['',f'First relative-Frobenius improvement: {c["first_deviation_below_baseline"]}; first max-abs improvement: {c["first_max_abs_below_baseline"]}; first speed below 1: {c["first_speed_below_baseline"]}.']
    (ART/'EPSILON_CURVE.md').write_text('\n'.join(lines)+'\n')
    print(payload['status'],f'{len(ready)}/448',json.dumps({k:predictions[k]['verdict'] for k in ('P1','P2','P3','P4')}))


if __name__=='__main__':summary()
