"""A4 receipt-driven synthesis. Prior art: SP3 (2026) evidence classes,
immutable provenance and registered-reference tables. No new statistic.
"""
import math
from apa_sp4g_a4_common import *
from apa_sp4g_a4_registry import cells,GATE

def rows():
    out=[]
    for c in cells():
        status='UNRUN';error=None
        if path_a4(c['id']).exists():
            j=read(path_a4(c['id']))
            try:require_a4(c['id']);status='PASS'
            except Exception as e:status='RED';error=str(e)
        out.append(dict(cell=c,status=status,error=error))
    return out

def value(name):
    try:return require_a4(name)['result']['ppl']
    except (OSError,KeyError,Red):return None

def fmt(x):return 'UNRUN / blocked' if x is None else f'{x:.9g}'

def main():
    preserved();data=rows();gate_ok=False
    try:gate_ok=exactness(require_a4(GATE)['result'])
    except (OSError,KeyError,Red):pass
    txt=['# APA-SP4G amendment 4 — reference ruling and numerical path', '',
      ('D32 exactness PASS; C/E may execute in registered dependency order.' if gate_ok else
       'RED / GPU handoff: D32 exactness has not passed. C/E remain blocked; no dtype bug is claimed fixed.'),'',
      'Evidence classes: source inspection, historical card measurements, author CPU tests; new GPU cells only where a current PASS receipt exists.', '',
      '## The suspect A2 cell', '',
      '`diag_a2_fp32_2048_w0` = 53.472390467473765, exactly the saved D bf16 PPL. '
      'The sealed A2 worker explicitly ran `.float()` on Q/K/Kq/V, called native SP on those values, '
      'and returned its output cast to bf16. A32 used the same conversions and final cast, with standard QK/softmax/PV. '
      'Both receipts contain 144 comparisons; the native binding forwards tensor data and the launcher dispatches on q.dtype. '
      '**The proposed missed-cast cause is not supported by source evidence.** A2 omitted observed dtype fields, so its actual runtime dtype cannot be retroactively asserted.', '',
      'SP already uses float dots, online softmax and accumulation for bf16 inputs, then rounds once on output. '
      'Consequently an unchanged D output after fp32 attention and a bf16 return is plausible. A4 uses explicit '
      '`astype("float32")` plus assertions immediately before the native call and on its output; every call is receipted. '
      'This fixes the missing verification, with no identified arithmetic correction. Extending fp32 through o_proj would change the registered A32 comparator and is not implemented.', '',
      'Source receipts: `scripts/apa_sp4g_a2_model.py:85-93`, `tensor_cuda/src/bindings.cpp:222,664-680`, '
      '`tensor_cuda/src/ops.cpp:1013`, `tensor_cuda/src/kernels.cu:7357-7505`; hashes in `a4_before.json`, original registration/build manifest.', '',
      'Additional historical-payload measurement: `a4_suspect_cell_audit.json` verifies that on BOTH A3 layer5 calls, '
      'saved SP fp32 cast to bf16 equals saved SP bf16 **bitwise**, and their projected outputs also equal **bitwise** '
      '(max abs and rel-F both0). This directly supports precision insensitivity on those calls; it does not prove '
      'all-layer equality or retrospectively supply A2 runtime dtype pins.', '',
      '## Registered reference', '',
      'Lead prediction: D32 within **0.005** PPL of A32 **49.92117893813879** on window0. '
      'Gate additionally requires complete fp32 native input/output pins and the same 144-call schedule as A2 A32. '
      'Seat prediction: no source-backed missed cast was found; unchanged arithmetic may repeat D32=53.47239 and fail. '
      'If the new gate fails, STOP; do not run propagation/calibration/C/E or adjust precision scope. '
      'A32 here means global attention in fp32 with bf16 activations/projections elsewhere, not a whole-model fp32 reference.', '',
      'The lead changes the active comparator in amendment013 only. Historical bf16 exactness RED receipts remain byte-identical; '
      'the bf16 differences are reported as numerical-path sensitivity. Local agreement to a dense fp32 calculation is finite numerical evidence, not exact real arithmetic.', '',
      '## PPL — QAT weights; APA touches only the 8 global layers', '',
      '| Population | A bf16 | A32 global attention | B bf16 | D bf16 | C bf16 | E bf16 | C minus A | C minus A32 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    all_a32=[]
    for S,w in [(2048,w) for w in range(4)]+[(8192,0)]:
        suffix=f'{S}_w{w}' if S==2048 else str(S)
        a,b,d=[value(f'ppl_{arm}_{suffix}') for arm in 'ABD']
        ref='diag_a2_fp32_A_2048_w0' if (S,w)==(2048,0) else f'ppl_a4_A32_{suffix}'
        a32=value(ref);cv=value('ppl_a4_C_'+suffix);ev=value('ppl_a4_E_'+suffix)
        if S==2048:all_a32.append((ref,a32))
        nums=[a,a32,b,d,cv,ev,None if cv is None or a is None else cv-a,None if cv is None or a32 is None else cv-a32]
        txt.append('| '+suffix+' | '+' | '.join(fmt(n) for n in nums)+' |')
    a32short=None
    if all(v is not None for _,v in all_a32):
        refs=[require_a4(n)['result'] for n,_ in all_a32]
        a32short=math.exp(sum(r['total_nll'] for r in refs)/sum(r['targets'] for r in refs))
    a,b,d=[value(f'ppl_{arm}_2048') for arm in 'ABD'];cv=value('ppl_a4_C_2048');ev=value('ppl_a4_E_2048')
    nums=[a,a32short,b,d,cv,ev,None if cv is None else cv-a,None if cv is None or a32short is None else cv-a32short]
    txt.append('| 2048 pooled (4096 scored targets) | '+' | '.join(fmt(n) for n in nums)+' |')
    if path_a4(GATE).exists():
        j=read(path_a4(GATE));txt+=['',f"D32 gate receipt status **{j['status']}**: `{json.dumps({k:v for k,v in j['result'].items() if k in ('ppl','A32_ppl','absolute_difference','tolerance','native_call_count','dtype_pin_complete')})}`; error `{j.get('error')}`."]
    txt+=['','Short rows score last1024 of each2048 window; long8192 scores last512; fp64 NLL and original adaptive512-prefix/64-scoring feed unchanged. '
      'Raw WikiText with the instruction-tuned model has a known large per-window spread; the approximate165 pooled baseline is not by itself evidence of a port failure.', '',
      '## A3 per-call card evidence (layer5)', '',
      '| Comparison | c0 max abs / rel-F | cached c1 max abs / rel-F |', '|---|---:|---:|']
    from apa_sp4g_a3_common import require_a3
    calls=[require_a3(f'diag_a3_call_l05_c{i}')['result']['comparisons'] for i in (0,1)]
    for key in calls[0]:
        if key in calls[1] and 'max_abs' in calls[0][key]:
            vals=[f"{c[key]['max_abs']:.7g} / {c[key]['relative_frobenius']:.7g}" for c in calls]
            txt.append('| '+key+' | '+' | '.join(vals)+' |')
    txt+=['','Same Q/K/V bitwise; c0 L512/S512, c1 L511/S1023/offset512. '
           'Receipts `jobs_a3/diag_a3_call_l05_c{0,1}.json`; local FP32 agreement and bf16 errors are descriptive per-call measurements.','',
           '## Residual propagation','',
           'Two separate bf16 workers save all2047 executed rows after layers5,11,17,23,29,35,41,47 (complete block including layer_scalar), '
           'plus final norm after compute cast and before logit row slicing. Both use the same weights/ids/schedule. '
           'The aggregate computes ||D-A||F/||A||F over all rows, with per-chunk rows also retained; it does not average chunk ratios.']
    try:
        prop=require_a4('diag_a4_propagation_2048_w0')['result']
        txt+=['','| Layer | rel-F | max abs |','|---|---:|---:|']
        for r in prop['per_layer']:txt.append(f"| {r['layer']} | {r['relative_frobenius']:.9g} | {r['max_abs']:.9g} |")
        txt+=['',f"Capture PPL: A={prop['A_ppl']}, D={prop['D_ppl']}; relative difference={prop['relative_ppl_difference']}. {prop['caveat']}"]
    except (OSError,KeyError,Red):txt+=['','UNRUN / blocked by D32. No propagation numbers inferred from the per-call table.']
    txt+=['','## C/E execution and G2/G3','',
      'Calibration uses B actual-PPL window0 selection fraction and up to12 C trials on that same population (queries0..2046), '
      'classical bounded bisection and one frozen delta matched within0.01. Completed matching trials carry the existing receipt and execute no model. '
      'C window0/8192 PPL workers also capture their exact per-call tensors/output/native mask. The16 C margin cells reuse A2 bitwise full-call replay '
      'before tiling statistics; error/mass/selected populations are all actually executed PPL pairs. '
      'E reuses SP2 delta=upward(log(100)+2*upward(empirical max eq)); B and C8 layers×2 lengths are calibration evidence only. '
      'This is a conditional finite-key bound, not a universal certification. Clean C decode uses the original unwrapped June protocol with32 synced steps.', '',
      'G2/G3 rows establish nothing about model quality by themselves.', '',
      '| Cell | Status | Worker estimate (s) | Error |','|---|---|---:|---|']
    measured=[]
    for r in data:
        c=r['cell'];txt.append(f"| {c['id']} | {r['status']} | {c['estimate_s'][0]}–{c['estimate_s'][1]} | {r['error'] or ''} |")
        if r['status']=='PASS' and c['kind'] in ('margin','decode','freeze','eq'):
            result=require_a4(c['id'])['result'];small={k:v for k,v in result.items() if k in ('error','unrefined_mass_mean','unrefined_mass_max','max_skipped_relative_weight','fraction','eq_sp','delta','eq','match_abs','ms_token','steps','fit','outcome')}
            measured.append(c['id']+': `'+json.dumps(small)+'`')
    if measured:txt+=['','Measured G2/G3 and calibration receipts:','']+['- '+m for m in measured]
    gate_path=A/'CPU_GATES_A4_V2.json'
    if gate_path.exists():
        cpu=read(gate_path)
        txt+=['',f"Author CPU gates: **{cpu['passed']} passed, {cpu['failed']} failed, {cpu['skipped']} skipped**; "
              f"mutations **{cpu['mutations']['killed']}/{cpu['mutations']['nonerror']}** (threshold0.80). "
              f"Fingerprint amendment016 SHA `{cpu['fingerprint_amendment_sha256']}`."]
    txt+=['','## Gates, fingerprints, commands and RED','',
      f'Immutable registration013 SHA `{REGISTRATION_SHA}`. Original registration SHA `{REG_SHA}`. '
      'Fingerprint amendment016 pins the final A4 closure; original execution files and all historical receipts preserved. '
      'Amendment015 corrects the new calibration lookup to the real B PPL receipt global_fraction.fraction, with no algorithm/threshold change. '
      'The new regression failed with KeyError before correction; initial100-pass CPU gate/fingerprint014 remain as superseded receipts. '
      'Existing A2 captures/error arrays retained. `receipt_audit_A4.json` and `DELIVERY_CHECKS_A4_V2.json` contain the delivery inventory.', '',
      'CPU dtype regression: `test_a4_fp32_native_dispatch_dtype_pin_and_returned_arm`. '
      'Full author baseline, negative cases and copied-source mutation results: `CPU_GATES_A4_V2.json`. '
      'These are not blind verification; lead-owned blind review UNRUN.', '',
      'Each command in `lead_commands.txt` is a separate foreground call. Worker285s/hard290s, outer585s/hard588s, '
      'lease wait20s, cooldown30s. Model load planning75–130s within cell estimate90–275s; margin10–90s short/60–270s long, '
      'CPU aggregates1–30s. New timings unmeasured. Disk floor12GiB. No >=16384 retries: prior16K PPL/16K–32K ceilings remain RAIL/non-fit within lease, not OOM or a hardware capacity conclusion.', '',
      'GPU visibility: `a4_device_visibility.json`. This sandbox has no CUDA device; no new model loads/card results claimed. '
      'Not claimed fixed: D/A model exactness, a missed-cast root cause, complete numerical propagation cause, C/E quality or long-context capacity. '
      'A3 local agreement does not itself unblock C. No git, subagents, background work/waits, process kills/signals, services, model/product/kernel/SP3 edits. '
      'Seat gpt-6-astra / reasoning xhigh (`logs/apa_sp4g_a4_r1.log`).', '',
      '## Prior art','',
      'June Gemma port/floor and SP3/SP4G A2/A3 (2026): seam, feeding, replay, provenance, capture and clean decode reused; '
      'new work is explicit dtype assertions, residual instrumentation, registered-reference routing and tables. '
      '[BLASST, Yuan et al. 2025/2026](https://arxiv.org/abs/2512.12087): inherited running-maximum criterion; '
      '[ThriftAttention, Sharratt 2026](https://arxiv.org/abs/2605.23081): precision/softmax-weight motivation; '
      '[FlashAttention-2, Dao 2023](https://arxiv.org/abs/2307.08691): inherited online softmax; '
      '[TurboQuant, Zandieh et al. 2025](https://arxiv.org/abs/2504.19874): inherited reconstructed Kq. '
      'Primary arXiv records checked in this turn; no external benchmark reproduced. '
      'SP2 (2026) conditional error-bound delta reused. Standard Frobenius norm, FP precision ablation, bisection, '
      'SHA256/NIST (2001), Make/Feldman (1979), DeMillo/Lipton/Sayward (1978) mutation testing '
      '(unverified — lead to check: Hints on Test Data Selection). No new attention or optimization algorithm; '
      'no prior art known to me for a distinct novel method introduced here.']
    (A/'RESULTS.md').write_text('\n'.join(txt)+'\n')
    commands=['# A4: foreground only, one command per call, STOP on any RED. No automatic retry.',
      '# Calibration/C/E execute only after current D32 PASS. No >=16384 rows.',
      '# Model load75-130s included in90-275s worker estimate; +30s GPU cooldown; lease<=20s; outer<10min.']
    for r in data:
        c=r['cell'];commands += [f"# {r['status']}; worker estimate {c['estimate_s']} seconds",f"bash scripts/apa_sp4g_a4_lead_gpu.sh run {c['id']}"]
    commands+=['bash scripts/apa_sp4g_a4_lead_gpu.sh summary']
    (A/'lead_commands.txt').write_text('\n'.join(commands)+'\n')
    (A/'cells.json').write_text(json.dumps(dict(amendment='013',registration_sha256=REGISTRATION_SHA,cells=cells()),indent=2)+'\n')
    print('RESULTS.md / lead_commands.txt / cells.json refreshed; D32='+('PASS' if gate_ok else 'BLOCKED/RED'))

if __name__=='__main__':
    token=VALIDATION.set({})
    try:main()
    finally:VALIDATION.reset(token)
