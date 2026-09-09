"""Reference/ceiling tables and dependency-ordered, single-cell lead commands.

Prior art: standard experimental tables and fixed-grid summaries; no new
numerical method. A successful largest grid point is not an unbounded ceiling.
"""
import json
from pathlib import Path
from apa_sp3_common import REG_SHA


def render(art, jobs, cells):
    def result(id):
        j=jobs.get(id,{})
        return j.get('result',{}) if j.get('status')=='PASS' else {}
    def status(id):return jobs.get(id,{}).get('status','UNRUN')
    def val(x):return '—' if x is None else str(round(x,6)) if isinstance(x,float) else str(x)
    text=['## A4 full-context reference — model perplexity', '',
          'T: pinned HF MiniCPM3 snapshot, bf16 weights, full SDPA attention. Flash is preferred only when eligible on actual Q/K=96, V=64 tensors; otherwise efficient is forced. Math fallback is disabled. Actual backend is UNRUN until a T receipt exists.',
          '1024 uses six independent windows / 3072 total targets. Long rows use prefix 0 / 512 targets. Only required LM-head rows are projected in new T and 32K engine cells. Attention is never chunked in these PPL/reference cells.', '',
          '| S | Arm | Status | PPL | Engine minus T | SDPA backend |',
          '|---:|---|---|---:|---:|---|']
    reference_rows=[]
    for S in (1024,8192,32768):
        t=result(f'ppl_T_{S}')
        for arm in 'TABCD':
            id=f'ppl_T_{S}' if arm=='T' else 'g0' if S==1024 and arm in 'AB' else f'ppl_b4_{arm}_{S}'
            r=result(id)
            if id=='g0':r=r.get(arm,{})
            gap=(r['ppl']-t['ppl']) if arm!='T' and 'ppl' in r and 'ppl' in t and r.get('target_sha256')==t.get('target_sha256') else None
            row=dict(S=S,arm=arm,status=status(id),ppl=r.get('ppl'),engine_minus_T=gap,
                     sdpa_backend=t.get('sdpa_backends') if arm=='T' else None)
            reference_rows.append(row)
            text.append('| '+' | '.join(val(v) for v in row.values())+' |')
    text += ['', 'INT4 engine versus bf16 T gaps are observations, not RED parity failures. D@32768 requires T with identical targets; the 0.005 D/A gate remains confined to existing engine controls. D@32768 adds a layer-0 first-128-query refine-all check; no full 62-layer 32K diagnostic claim.', '',
             '## A4 ceiling grid — kernel sweep / memory shape', '',
             '| Arm | S | Status | Fit | Outcome | Peak resident MiB estimate |',
             '|---|---:|---|---|---|---:|']
    ceiling=[]
    for arm in 'ABC':
        for S in (4096,8192,16384,24576,32768):
            id=f'ceiling_b4_{arm}_{S}';r=result(id)
            row=dict(arm=arm,S=S,status=status(id),fit=r.get('fit'),outcome=r.get('outcome'),peak_resident_mib=r.get('peak_resident_mib'))
            ceiling.append(row)
            text.append('| '+' | '.join(val(v) for v in row.values())+' |')
    maxima={a:dict(max_successful_grid_S=max([r['S'] for r in ceiling if r['arm']==a and r['fit'] is True],default=None),
                   grid_complete=all(r['status']=='PASS' for r in ceiling if r['arm']==a)) for a in 'ABC'}
    text += ['', 'Measured grid summary: `'+json.dumps(maxima,sort_keys=True)+'`.',
             'A timeout is unknown fit, never an OOM or successful prefill. Every grid point is independent. Largest successful S is only a grid result, not an extrapolated capacity or model-quality finding.', '',
             '## A4 capture split and immutable receipt handling', '',
             '8192: [0,16), [16,32), [32,48), [48,62), at most 188s planning estimate each. 32768: 62 single-layer ranges, 188s each under the registered quadratic extrapolation. These are unmeasured estimates; timeout rails remain authoritative.',
             'Ranges restore predecessor hidden activations; each layer sees the full token prefix. Row-block diagnostic replay is checked bitwise against native output. The unchanged margin ids depend on the original capture id, now an aggregation cell. New/changed receipts are in jobs_a4; legacy RED/PASS receipts are preserved.',
             'Aggregation rehashes each layer manifest and checks exact stat identity of every array since its completed range SHA256. It pins all 62 manifests as one set. Margin workers rehash their input arrays. 32768 captures are B/C bulk4; no new 32768 margin/E calibration cells were authorized.', '',
             'Fingerprint compatibility is governed by amendment_006_fingerprint.json: per-kind import closures and exact reviewed source transitions. Unknown closure changes reject reuse. Source eligibility and current runtime/build prerequisites are reported separately in a4_receipt_audit.json.', '']
    data=dict(reference=reference_rows,ceiling=ceiling,maxima=maxima,registration_sha256=REG_SHA)
    (art/'a4_results.json').write_text(json.dumps(data,indent=2)+'\n')
    (art/'A4_TABLES.md').write_text('\n'.join(text)+'\n')
    commands=['# A4, after the lead merges this isolated worktree. One foreground command at a time.',
              '# No automatic loop, rebuild, retry, receipt replacement, or process intervention.',
              '# Source-compatible existing PASS cells are omitted at execution time by preflight; do not rerun them.',
              '# Worker estimates below; add 30s foreground cooldown + <=20s lease and <=45s setup/receipt.',
              '# Capture ranges and ceiling: worker TERM 290s +5s own-child grace; other workers 480s+5s. Outer rail 590s.',
              'cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3',
              'timeout 30s bash scripts/apa_sp3_lead_gpu.sh list',
              'timeout 30s python3 scripts/apa_sp3_a4_audit.py',
              '# If first-range preflight reports CAPTURE_DEST_EXISTS, the lead must archive the prior partial directory; never overwrite it.',
              '# First 32768 range needs 490783899648 free bytes under the disk rail; free space is checked before GPU access.',
              '# The lead retains original immutable RED capture receipts; jobs_a4 removes receipt-name collision.',
              '# Dependency order follows the full DAG. Re-render summary after each completed cell to refresh statuses.']
    for c in cells:
        id=c['id'];st=status(id)
        commands.append(f"# {id} | kind={c['kind']} | deps={','.join(c['depends']) or '-'} | worker estimate={c['estimate_s']}s | status={st}")
        if st=='PASS':
            commands.append('# Existing valid PASS; retained, no rerun.')
        elif st in ('RED','STALE'):
            commands.append('# Existing RED/stale: retain receipt; resolve this branch with the lead. Independent branches remain runnable.')
        else:
            if c['kind']=='ppl_long' and c['arm'] in 'BC':
                commands.append('# ONLY if own ceiling_b4_*_32768 receipt has fit=true; preflight enforces this before GPU access.')
            commands.append(f'timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run {id}')
    commands += ['timeout 30s bash scripts/apa_sp3_lead_gpu.sh summary']
    (art/'lead_commands.txt').write_text('\n'.join(commands)+'\n')
    return text,data
