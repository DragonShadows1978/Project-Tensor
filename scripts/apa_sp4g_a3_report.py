"""Receipt-based A3 report. Prior art: SP3 (2026) evidence-class tabulation.
Diagnostic PASS means completed measurement, never a quality-gate bypass.
"""
from apa_sp4g_a3_common import *
from apa_sp4g_a3_registry import cells

def render():
    before = preserved()
    reg = read(REGISTRATION)
    current, failures = {}, []
    for c in cells():
        if not path_a3(c['id']).exists(): continue
        try: current[c['id']] = require_a3(c['id'])
        except Exception as error:
            raw = read(path_a3(c['id']))
            failures.append(dict(cell=c['id'],error=str(error),worker_error=raw.get('error'),traceback=raw.get('traceback')))
    gate = read(A/'CPU_GATES_A3.json') if (A/'CPU_GATES_A3.json').exists() else dict(status='PENDING')
    lines = ['# APA-SP4G amendment 3', '',
        '**RED: D/A model exactness remains unresolved. A3 diagnostics are prepared; no harness fix, kernel change, or A′ comparator is claimed validated. C/E remain blocked.**', '',
        'The fork/merge audit is in `docs/APA_SP4G_LEDGER.md` and its hash-pinned copy `A3_FORK_MERGE_AUDIT.md`. Source inspection finds identical scale1, no sinks, bottom-right causal mask, H16/KV1 mapping, exact K/V, and shared output projection/block scalar. Prediction: no single argument correction; inspect the cast/o_proj boundary and downstream propagation if nominal FP32 outputs agree. This is reasoning, not a measured cause.', '',
        '## Existing card evidence', '',
        '| Measurement | Value / finding | Evidence |','|---|---|---|']
    jdir = A/'jobs_a2'
    for name in ('diag_a2_source_2048','diag_a2_scale_2048','diag_a2_value_2048','diag_a2_rope_2048',
                 'diag_a2_fp32_A_2048_w0','diag_a2_fp32_2048_w0'):
        j = read(jdir/(name+'.json'))
        r = j['result']
        detail = f"PPL {r['ppl']:.9f}; {j['status']} diagnostic"
        if r.get('probes') and 'fp32_D_vs_standard' in r['probes'][0]:
            detail += '; max per-call FP32 difference '+str(max(p['fp32_D_vs_standard']['max_abs'] for p in r['probes']))
        lines.append(f'| `{name}` | {detail} | `jobs_a2/{name}.json` |')
    b = [p for p in jdir.glob('margin_a2_B_*.json') if read(p)['status']=='PASS']
    lines += ['', f'{len(b)}/16 B margin receipts PASS. The corrected actual-PPL original-call capture establishes bitwise replay of output and native selection; lead confirms the old context bug. Old RED receipts are preserved. C capture/margins reported `FileNotFoundError` by the lead because calibration never ran behind the RED exactness gate; no C result is inferred.', '',
        'A32=49.9211789381 improves on A bf16=52.48049 by about2.56 PPL; D32=53.4723904675 is essentially unchanged from D bf16. D32−A32=3.5512115293. All144 per-call FP32 A/SP comparisons in each precision receipt have max_abs≤4.38690185546875e-5. This tension requires direct reference and merge measurements; it does not identify an A-only semantic operation or justify widening0.005.', '',
        'Historical model PPL: four2048-token windows,1024 scored targets each: A165.644, B167.649, D169.789;8192 last512: A38.864, B38.561, D39.382 (rounded lead card values; exact `jobs_a1/ppl_*` receipts authoritative). Historical full report preserved at `a3_baseline/RESULTS.md`; its A2 pending-state narrative is superseded here. High raw-wikitext PPL and52→538 window spread are the lead’s known -it template-bound regime, not a new template experiment. Model input and feeding remain PROTOCOL-G. All≥16384 RAIL non-fits remain deferred under A2; no retries or OOM reinterpretation.', '',
        '## Registered diagnostics and interpretations', '',
        '| Cell | Input / work | Outcome meaning | Status |','|---|---|---|---|']
    meanings = [
        ('first actual prefix call L512/S512/offset0', 'Tests nominal full-prefix mask, scale, sinks and head/V mapping; saves native QK/probabilities and fork/merge outputs.'),
        ('second actual prefix call L511/S1023/offset512', 'Adds cached-prefix causal offset; discrepancy appearing only here directs attention to cache/offset handling.'),
        ('both saved calls;24 FP32 combinations each', 'Scale1 or512^-0.5 × sink None/zero × causal on/off × native/absolute-rowwise/zero-rowwise offsets; classify single-change matches across BOTH calls.')]
    for c,(what,meaning) in zip(cells(),meanings):
        state = 'MEASURED (diagnostic completion)' if c['id'] in current else ('RED' if path_a3(c['id']).exists() else 'UNRUN / GPU blocked in seat')
        lines.append(f"| `{c['id']}` | {what} | {meaning} | {state} |")
    lines += ['', 'The real PPL prefix is1023 tokens, so c1 is511 queries. Neither forcing flag enters the adapter chunk-size formula. Prior calls propagate original A. Each diagnostic stops by a caught local exception after the selected attention finishes; it measures no PPL. Saved arrays include q/k/kq/v, native scores/probabilities/attention, A/SP bf16 and FP32 outputs, dense NumPy FP32, staged-bf16 NumPy, and outputs after FP32→bf16 and o_proj. Exact source tuple-cache/QKV/replay checks reject changed inputs or missing operations.', '',
        'Per-call max-abs and relative Frobenius `||candidate-reference||F/max(||reference||F,1e-30)` use float64 reductions. Numerical FP32 agreement requires BOTH max-abs≤1e-4 and relative-Frobenius≤1e-5. BF16 metrics are descriptive; no BF16 acceptance tolerance or PPL widening. NumPy dot/softmax/value arithmetic is FP32. Staged-bf16 emulation is descriptive and does not claim bitwise cuBLAS summation. Native A/isolation/replacement projection and SP diagnostic/native output must replay bitwise; D’s mask must select every eligible key.', '',
        'None maps to nullptr in the SP binding. Zero sink means an extra zero-logit/zero-value denominator term. The SP ABI has no offset argument: explicit absolute and lost-offset probes invoke existing L1 split-K on each allowed key prefix. This changes geometry and is labelled as such; it is not a product fix. Causal-off offset labels reuse their literally identical native calls. BF16 nominal controls live in the call cells; the suspect sweep is FP32 to separate large semantic differences from materialization effects.', '',
        '## Fix or A′ decision path', '']
    for name,meaning in reg['decision_forks'].items(): lines.append(f'- **{name}:** {meaning}')
    lines += ['', 'Reserved successor IDs: `ppl_a3_D_fixed_2048_w0`, `ppl_a3_AP_2048_w0`, `exactness_a3_D_AP_2048_w0`. They are conditional registration paths, not executable cells: the runner rejects them until a separate immutable amendment names the measured culprit, exact harness treatment or removed operation, fingerprints, reference and dependencies. There is no known op to remove honestly before these diagnostics. A named harness fix must first clear original D-vs-A window0 at0.005; original full-short and8192 exactness then protect existing C calibration/freeze/PPL/margins. A′ requires lead adoption of the like-for-like comparator and A′−A cost reporting. Existing exactness REDs are never rewritten.', '',
        '## Measured A3 comparisons', '']
    if not current:
        lines += ['No A3 card receipts yet. No fabricated output-error numbers or culprit.']
    for name,j in current.items():
        result=j['result']; lines += ['',f'### {name}','']
        if 'comparisons' in result:
            lines += ['| Pair | max-abs | relative-Frobenius | bitwise |','|---|---:|---:|---|']
            for pair,metric in result['comparisons'].items():
                lines.append(f"| {pair} | {metric['max_abs']:.10g} | {metric['relative_frobenius']:.10g} | {metric['bitwise']} |")
        if 'decision' in result:
            lines += ['Decision: `'+json.dumps(result['decision'])+'`','',
                      '| Call | Variant | SP–A max / relF | SP–own NumPy max / relF |','|---|---|---|---|']
            for call in result['calls']:
                for row in call['rows']:
                    a,d=row['vs_standard'],row['vs_own_dense']
                    lines.append(f"| {call['source_cell']} | `{json.dumps(row['variant'])}` | {a['max_abs']:.9g} / {a['relative_frobenius']:.9g} | {d['max_abs']:.9g} / {d['relative_frobenius']:.9g} |")
    if failures: lines += ['', 'Failures, verbatim:','```json',json.dumps(failures,indent=2),'```']
    lines += ['', '## Fingerprints, CPU and lead handoff', '',
        f"Registration010 SHA `{REGISTRATION_SHA}`. Source/payload/receipt identity is fail-closed in new `jobs_a3/`. All{len(before['source_sha256'])} preexisting execution files and{len(before['receipt_sha256'])} historical receipt files retain exact bytes; no old receipt is upgraded. Original registration, compiled build, product adapter and kernels stay pinned. Source closure and validation identities: `amendment_011_a3_fingerprint.json`, `CPU_GATES_A3.json`, `GPU_BLOCKED_A3.json`, `DELIVERY_CHECKS_A3.json`.", '',
        'CPU gate: `'+json.dumps({k:v for k,v in gate.items() if k!='execution_sha256'})+'`. Author baseline and copied-source mutants only; blind review remains lead-owned UNRUN.', '',
        'Use `lead_commands.txt` (three dependency-ordered foreground calls) and `cells.json`. Estimates: each capture90–250s including model load; sweep30–270s with no model load. Historical QAT load≈75–85s, planning30–120s; A3 estimates unmeasured. Worker285s/hard290s, outer588s, lease≤20s, cooldown30s; ≥2GiB disk headroom. Every cell has its own receipt and log. Stop on RED; no background chain.', '',
        '```bash','cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g',
        'bash scripts/apa_sp4g_a3_lead_gpu.sh run diag_a3_call_l05_c0',
        'bash scripts/apa_sp4g_a3_lead_gpu.sh run diag_a3_call_l05_c1',
        'bash scripts/apa_sp4g_a3_lead_gpu.sh run diag_a3_sweep_l05',
        'bash scripts/apa_sp4g_a3_lead_gpu.sh summary','```','',
        '## Prior art','',
        'June Gemma port/floor (2026) supplies the standard MQA branch, exact shared KV, chunking and PPL protocol. SP3 (2026) supplies same-tensor comparison/native replay and registered receipt dependencies; Make/Feldman (1979) and SHA256/NIST (2001) are provenance precedents. Dense attention is the established [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) operation, with ordinary stable softmax and Frobenius error norms. IEEE754 (2019) informs ties-to-even BF16 bit rounding. New work is capture, comparison and bounded sweep wiring; no novel attention/selector/optimization claim. No prior art known to me for a distinct novel method introduced here.', '',
        'Inherited SP uses [BLASST, Yuan et al. (2025/2026)](https://arxiv.org/abs/2512.12087) running-max comparison, [ThriftAttention, Sharratt (2026)](https://arxiv.org/abs/2605.23081) weight-sensitive precision motivation, [FlashAttention-2, Dao (2023)](https://arxiv.org/abs/2307.08691) online softmax/work partition, and [TurboQuant, Zandieh et al. (2025)](https://arxiv.org/abs/2504.19874) reconstructed-key quantization. No kernels or selection rules changed. Author mutation gates follow DeMillo/Lipton/Sayward (1978), unverified — lead to check: Hints on Test Data Selection. A3 externally checked arXiv titles/authors/years for Vaswani, FA2, BLASST and TurboQuant; other literature annotations are inherited from original registration, not a new reproduction.', '',
        '## RED, safety and identity','',
        'Not claimed fixed: D/A exactness, a named argument or A-only-op cause, C/E calibration/quality, new GPU numerical comparisons, long-context memory capacity. G2/G3 rows establish nothing about model quality by themselves. GPU unavailable to this seat; exact visibility receipt is `a3_device_visibility.json`. Zero GPU workers and zero model loads started here. No git, subagents, background jobs/waits, process kills/signals, services, model writes or product/kernel edits. All executed calls foreground and under10min. Runner preserves existing bounded owned-worker timeouts for the lead.', '',
        'Seat: **gpt-6-astra / reasoning xhigh**, `logs/apa_sp4g_a3_r1.log`. Model under test: **Gemma-4-12B-it QAT q4_0 symmetric-8 group32**, bf16 engine compute; original -it token stream unchanged.']
    (A/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (A/'cells.json').write_text(json.dumps(dict(schema='apa_sp4g_cells_a3',registration_sha256=REGISTRATION_SHA,
        cells=cells(),conditional_successors=reg['conditional_successor_cells'],C_unblocked=False),indent=2)+'\n')
    commands=['# A3: run each foreground command separately; stop on RED.',
              '# Capture90-250s incl load30-120s; sweep30-270s without model load.',
              '# Worker285s/hard290s, outer588s, lease20s, cooldown30s; no C bypass.',
              'cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g',
              'bash scripts/apa_sp4g_a3_lead_gpu.sh list']
    for c in cells():
        commands += ['# depends: '+','.join(c['depends']),f"bash scripts/apa_sp4g_a3_lead_gpu.sh run {c['id']}"]
    commands += ['bash scripts/apa_sp4g_a3_lead_gpu.sh summary']
    (A/'lead_commands.txt').write_text('\n'.join(commands)+'\n')
    (A/'lead_commands_fallback.txt').write_text('# A3: no automatic fallback. Named harness treatment or A-prime removed-op registration requires diagnostic receipts and a separate immutable amendment. Existing C exactness gate remains RED.\n')
    print(json.dumps(dict(report=str(A/'RESULTS.md'),a3_completed=len(current),a3_failed=len(failures),cpu=gate['status'])))

if __name__=='__main__':render()
