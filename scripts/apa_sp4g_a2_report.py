"""A2 evidence report. Prior art: SP3 (2026) receipt-scoped tabulation and
lead commands. No inference from missing data; no new algorithm.
"""
from apa_sp4g_a2_common import *
from apa_sp4g_a2_registry import cells
from apa_sp4g_registry import cells as historical_cells

HYPOTHESES=[
 ('a: K/V source','diag_a2_source_2048','Source inspection argues against storage cost: QUANT_V/QUANT_KV4 off, standard and APA prefill share exact k/v tuples; reconstructed kq is separate. Runtime same-state Q/K/V identity, value comparison, and standard-over-APA-input check pending.'),
 ('b: scale/softcap','diag_a2_scale_2048','Both global prefill routes scale1.0; no attention-logit softcap in either branch. Final-logit processing is outside this fork. Runtime scale and same-tensor outputs pending.'),
 ('c: shared K=V projection','diag_a2_value_2048','Shared kraw feeds K norm and scale-free V RMSNorm; V is not roped. Same V variable at both forks. Per-layer same-state comparison pending.'),
 ('d: p-RoPE/qk-norm order','diag_a2_rope_2048','Q/K RMSNorm then p-RoPE precede both branches (gemma4_tc.py:523-544); repeat same x/cos/sin/offset/cache, compare post-fork Q/K bitwise. Runtime pending.'),
 ('e: accumulation/intermediates','diag_a2_fp32_2048_w0','Concrete numerical-path difference: A materializes bf16 QK logits and probabilities; D keeps fused FP32 dot/softmax/value accumulation. A32 precursor plus D32 one-window treatment and per-call comparisons quantify it; no full-model FP32 claim. Cause of PPL gap remains unnamed until card treatment.')]

def rail_rows():
    rows=[]
    for c in historical_cells():
        if (c['kind']=='ppl' and c['S']==16384) or (c['kind']=='ceiling' and c['S']>=16384):
            p=job_path(c['id']);j=read(p) if p.exists() else None
            measured=bool(j and j.get('result',{}).get('outcome')=='RAIL')
            rows.append(dict(cell=c['id'],outcome='RAIL',fit=False,capacity_fit=None,
                evidence_class='lead card termination at285s' if measured else 'lead A2 administrative rail non-fit; unrun',
                receipt=str(p.relative_to(R)) if measured else None,
                error=j.get('error') if measured else None,retry=False))
    return rows

def render():
    preserved();reg=registration();valid={};red=[]
    for c in historical_cells():
        p=job_path(c['id'])
        if not p.exists():continue
        if read(p)['status']=='PASS':valid[c['id']]=require_pass(c['id'])
        else:red.append(dict(cell=c['id'],path=str(p.relative_to(R)),error=read(p)['error']))
    current={};failed=[]
    for c in cells():
        if not path_a2(c['id']).exists():continue
        try:current[c['id']]=require_a2(c['id'])
        except Exception as e:failed.append(dict(cell=c['id'],error=str(e)))
    def result(n):return valid.get(n,{}).get('result',{})
    gate=read(A/'CPU_GATES_A2.json') if (A/'CPU_GATES_A2.json').exists() else {'status':'PENDING'}
    lines=['# APA-SP4G A2 results','',
        '**RED: D/A exactness and old G2 replay remain unresolved on the card. A2 CPU '+gate['status']+'.** This seat has no CUDA device; new diagnostics are prepared, not claimed executed or fixed. Historical A1 measurements below retain their original fingerprints.','',
        '## Model perplexity — QAT INT4 Gemma-4-12B-it; only eight global layers change','',
        '| S / scored targets | A standard | B two-pass4 r=.15 | D refine-all | D−A |','|---|---:|---:|---:|---:|']
    for S in (2048,8192):
        a,b,d=[result(f'ppl_{arm}_{S}').get('ppl') for arm in 'ABD']
        lines.append(f'| {S} / {4096 if S==2048 else 512} | {a:.6f} | {b:.6f} | {d:.6f} | {d-a:+.6f} ({100*(d-a)/a:.3f}%) |')
    lines+=['','Evidence class: model perplexity, `jobs_a1/ppl_{A,B,D}_{2048,8192}.json`. Short is four2048-token windows with **1024 targets each,4096 total**, as the immutable protocol and receipts specify (the lead card header said2048×4 targets). Window PPLs:','', '| Arm | w0 | w1 | w2 | w3 |','|---|---:|---:|---:|---:|']
    for arm in 'ABD':lines.append('| '+arm+' | '+' | '.join(f'{result(f"ppl_{arm}_2048_w{w}")["ppl"]:.4f}' for w in range(4))+' |')
    lines+=['','Raw untemplated wikitext PPL near165 is the known -it regime, not evidence by itself of a port failure. The June refine sweep recorded standard121.74 and attributed high raw-text PPL to the model’s template binding (`/mnt/ForgeRealm/GraftRepository/docs/GEMMA4_PORT_LEDGER.md:287-295`, historical source evidence; its HF control was pending). The52→538 window spread is context supplied by the lead as template-boundness; this amendment does not run a new templating experiment.','',
        f'PROTOCOL-G unchanged: {reg["protocol"]["token_count"]} canonical int64 tokens, SHA `{reg["protocol"]["token_sha256"]}`; raw test newline join, -it default tokenizer, no chat template; cached64-query blocks, fp64 NLL, aggregate exp(total NLL/targets). `registration.json` SHA `{REG_SHA}`. B fused at threshold0, fast_max_seq0, scale1, bulk4; clean decode incremental kq_count retained. No torch/BF16-weight reference.','',
        '## Hypotheses and registered card cells','',
        '| Hypothesis | Diagnostic cell | Current finding / decision |','|---|---|---|']
    for h,c,f in HYPOTHESES:
        status='MEASURED: '+str(current[c]['result'].get('focus')) if c in current else 'UNRUN'
        lines.append(f'| {h} | `{c}` ({status}) | {f} |')
    lines+=['','Registration before code/gates: `amendment_006_a2_hypotheses.json` and `amendment_007_a2_execution.json`. The four source/scale/value/RoPE jobs each run standard window0 and repeat its first global call per layer from the same x, cos, sin, offset and immutable tuple cache. They compare Q/K/V and post-output projection, both bf16 and FP32 attention outputs. A completed diagnostic PASS means a measurement exists, not that a parity hypothesis passed. Full receipt probes are authoritative.','',
        '**A′ decision:** standard over the exact K/V at the APA entry is implemented only as a registered diagnostic comparator (`standard` in a2_model.py); storage quantization is explicitly off. No independent AP reference arm is adopted or storage cost claimed without runtime evidence. The same-state test returns original A output to preserve its propagation. Kq is never substituted for exact K. If storage evidence contradicts source inspection, lead may activate A′ with separate PPL receipts and report A′−A.','',
        '**Tolerance proposal to lead:** preserve abs(D−A)≤0.005 PPL and both existing REDs. There is no cause-confirmed Gemma-specific replacement rule yet. Conditional on confirmed storage difference, use abs(D−A′)≤0.005 and report storage cost separately. Do not widen0.005. Four PPL points exceed MiniCPM3’s0.0005 refine-all evidence by orders of magnitude; this seat does not dismiss the gap as rounding. BF16 intermediate rounding is a specific candidate requiring the FP32 treatment. A32/D32 retain QAT/bf16 everywhere except global attention; they cannot establish full-model FP32 parity.','',
        '## Replay finding and change','',
        'All eight historical8192 B margins are RED `NATIVE_REPLAY_NOT_BITWISE`. Source inspection found no atomic reduction in the active D512 B kernel and no cuBLAS blend in the forced-fused route. The old harness reconstructs128-query bands and uses final-prefix Kq for early calls. It did not preserve original-call Kq or masks. Quantizer FP32 matmul dimensions change with prefix length, so row independence in real arithmetic does not prove bitwise Kq identity. This is a concrete replay-context defect in the evidence chain; its responsibility for the observed error versus native nondeterminism is **not yet determined**.','',
        '`diag_a2_replay_B_8192_l05` repeats the first captured512-query call with final Kq and with regenerated original-prefix Kq, then compares the old128-band shape. It reports repeat variation, Kq/context treatment effect, max-abs and relative error (denominator max(abs(reference),1e-30)), without accepting a tolerance. Old capture regeneration is diagnostic only; it cannot certify the original mask.','',
        'The correction records Q/K/Kq/V/output and packed native mask at every call during the **actual PPL run**, using lossless bf16 bit payloads. G2 replays the entire original call and requires bitwise output **and mask** before tiling FP64 statistics. No selection is recomputed on statistical bands. The PPL output remains the original native output and the diagnostic copy must match it bitwise. New cells `ppl_capture_{B,C}_{2048,8192}_w0` and `margin_a2_{B,C}_{2048,8192}_l{05,11,17,23,29,35,41,47}` share dependency hashes and exact selected/pair counts. C remains blocked by original exactness/calibration; no bypass.','',
        'Population amendment: short G2 uses actual window0 PPL queries0..2046, long0..8190 (2047/8191 rows), every eligible pair across16 heads. Last input token is a scored target, not an executed query. This is explicitly distinct from old independent prefix captures of2048/8192 rows. All-population nearest-rank percentiles are retained; no sampling or averaging band percentiles. Extra capture/I/O cost may hit285s and must remain RED. No end-to-end margin fix claimed before card replay.','',
        '## Rail non-fits and clean decode','',
        '`RAIL_NONFITS_A2.json` records every `ppl_*_16384` and16K/24K/32K ceiling as **RAIL non-fit under285s**. This is a time-budget non-fit; memory capacity is unknown, not OOM. Five observed returncode124 terminations are distinguished from administratively deferred unrun rows. No retries scheduled. All additional≥16K work waits for a lead long lease.','',
        '| Arm | 4096 ceiling | 8192 ceiling | decode2048 ms/token | decode8192 ms/token |','|---|---|---|---:|---:|']
    for arm in 'ABD':
        values=[result(f'decode_{arm}_{S}').get('ms_token') for S in (2048,8192)]
        lines.append(f'| {arm} | FIT | FIT | '+ ' | '.join('—' if x is None else f'{x:.4f}' for x in values)+' |')
    lines+=['','Evidence: corresponding `jobs_a1/ceiling_*` and `decode_*` receipts. G2/G3 rows establish nothing about model quality by themselves.','',
        '## Fingerprints, CPU gates and lead commands','',
        f'{len(valid)} historical PASS receipts remain valid, including kernel512, all A/B/D PPL, measured decode/ceiling and capture_B_8192. All historical RED bytes remain unchanged. All A1 scripts, tests, runner and build are byte-identical; A2 renders the refreshed report through its isolated runner. Old captures remain historical evidence but lack actual-PPL per-call masks required by new G2. No old RED is reused as PASS. New jobs write create-only `jobs_a2/` with a separate complete closure. Unknown transitions reject. `amendment_008_a2_fingerprint.json`, `CPU_GATES_A2.json`, `GPU_BLOCKED_A2.json`, and `receipt_audit_A2.json` provide exact inventories.','',
        'CPU gate: `'+json.dumps({k:v for k,v in gate.items() if k!='execution_sha256'})+'`. Author baseline only; blind verification remains lead-owned UNRUN.','',
        'Use refreshed `lead_commands.txt` and `cells.json`. Each command is one foreground lease; no batch. A2 runner rejects historical/long cell IDs. Original runner and report generator are retained to preserve measured closures; use the A2 summary and do not use the old long schedule.','',
        '```bash','cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g','bash scripts/apa_sp4g_a2_lead_gpu.sh list','bash scripts/apa_sp4g_a2_lead_gpu.sh run diag_a2_source_2048','bash scripts/apa_sp4g_a2_lead_gpu.sh resume','bash scripts/apa_sp4g_a2_lead_gpu.sh summary','```','',
        'Per-cell wall estimates are registered and unmeasured for A2: parity/FP32/PPL-capture60–285s; replay5–90s; margin2048 10–90s,8192 60–270s. A1 measured QAT loading mostly75–76s (first A84.7s); original30–120s planning range retained. Extra paired attention and disk writes may hit the rail. Worker285s TERM/290s hard, outer588s, lease wait20s, foreground cooldown30s. No A2 GPU worker or model load was started in this seat.','',
        '## Prior art','',
        'A2 reuses the June Gemma port/floor (2026) MQA standard branch and quantizer; SP3 (2026) same-tensor references, native-mask replay and exact dependency provenance; IEEE754 (2019) float bit representation and NumPy packbits (system year unverified — lead to check: NumPy packbits bitorder release). Controlled same-state and precision ablations are standard experimental methods; no prior art known to me for any distinct novel method introduced here, and no novelty claimed. New work is diagnostic/capture wiring. Code sites and ledger carry the same annotations.','']
    for title,detail in reg['prior_art'].items():lines.append(f'- {title}: {detail}.')
    lines+=['','Prior literature annotations are retained from the immutable registration; no new external experiment reproduced. BLASST running-max comparison, ThriftAttention weight-sensitive precision, FA2 online softmax and TurboQuant reconstructed-key quantization are inherited, not newly implemented.','',
        '## RED, process safety and identity','',
        'Not claimed fixed: D/A PPL exactness; actual GPU replay; whether old error is context or nondeterminism; FP32 treatment; C/E calibration and quality; new margin rail clearance; device memory ceilings above8192. No tolerance was widened. No card result is fabricated from CPU tests. No GPU device (cudaGetDeviceCount100, count0, `no CUDA-capable device is detected`).','',
        'No git, subagents, background jobs/waits, process kills/signals, live-service changes, model writes, product/kernel edits or SP3 edits. All executed commands foreground and under10min. Driver retains bounded owned-child timeout semantics for lead use.','',
        'Seat: gpt-6-astra, reasoning xhigh, recorded in `logs/apa_sp4g_a2_r1.log`. Model under test: Gemma-4-12B-it QAT q4_0 symmetric-8 group32, bf16 engine compute.','',
        'A2 measured diagnostic results: `'+json.dumps({k:{x:y for x,y in j['result'].items() if x not in ('files','probes')} for k,j in current.items()})+'`. A2 failures: `'+json.dumps(failed)+'`.']
    (A/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    rails=rail_rows();(A/'RAIL_NONFITS_A2.json').write_text(json.dumps(rails,indent=2)+'\n')
    audit=dict(valid_historical=[dict(cell=k,path=str(job_path(k).relative_to(R)),sha256=sha(job_path(k))) for k in valid],historical_red=red,a2_valid=list(current),a2_red_or_stale=failed)
    (A/'receipt_audit_A2.json').write_text(json.dumps(audit,indent=2)+'\n')
    manifest=dict(schema='apa_sp4g_cells_a2',registration_sha256=REG_SHA,historical_manifest='a2_baseline/artifacts/apa_sp4g/cells.json',amendments=[dict(path=n,sha256=sha(A/n)) for n in ('amendment_006_a2_hypotheses.json','amendment_007_a2_execution.json')],cells=cells(),rail_nonfits=rails)
    (A/'cells.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cmd=['# APA-SP4G A2: execute each reviewed foreground command separately; stop on RED.', '# New driver; original measured execution closure preserved. No long rows/retries.', '# Worker285s / hard290s; outer588s; lease<=20s and cooldown30s.', 'cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g','bash scripts/apa_sp4g_a2_lead_gpu.sh list']
    for c in cells():
        cmd += [f'# {c["kind"]}: worker estimate{c["estimate_s"]}s; deps='+','.join(c['depends'])+('; C BLOCKED until lead resolves exactness and original calibration' if c['arm']=='C' else ''),f'bash scripts/apa_sp4g_a2_lead_gpu.sh run {c["id"]}']
    cmd+=['bash scripts/apa_sp4g_a2_lead_gpu.sh summary']
    (A/'lead_commands.txt').write_text('\n'.join(cmd)+'\n')
    (A/'lead_commands_fallback.txt').write_text('# A2: no band fallback registered for actual-PPL original-call replay. Stop on rail; request lead decision. Historical A1 band recipe is not PPL-selection evidence.\n')
    print(json.dumps(dict(report=str(A/'RESULTS.md'),historical_valid=len(valid),historical_red=len(red),a2_cells=len(cells()),a2_valid=len(current))))

if __name__=='__main__':render()
