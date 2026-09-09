"""A6 registration before gates. Prior art: SP4G A4/A5 (2026),
Make/Feldman (1979), NIST SHA256 (2001), classical bisection calibration.
New work is lead-ruling and diagnostic dependency wiring, no new algorithm.
"""
from apa_sp4g_common import *


def main():
    sources = sorted(p for p in R.glob('scripts/apa_sp4g_*') if p.is_file() and '_a6_' not in p.name)
    sources += sorted(p for p in R.glob('tensor_cuda/tests/test_apa_sp4g*.py') if '_a6' not in p.name)
    before = dict(source_sha256={str(p.relative_to(R)): sha(p) for p in sources},
                  receipt_sha256={str(p.relative_to(R)): sha(p) for p in sorted(A.glob('jobs*/*.json'))},
                  amendment_sha256={str(p.relative_to(R)): sha(p) for p in sorted(A.glob('amendment*.json'))})
    publish(A/'a6_before.json', before)
    for name in ('RESULTS.md', 'lead_commands.txt', 'cells.json'):
        p = A/'a6_baseline'/name
        p.parent.mkdir(exist_ok=True)
        with p.open('xb') as f:
            f.write((A/name).read_bytes())
    call_paths = [A/'jobs_a3'/f'diag_a3_call_l05_c{i}.json' for i in (0, 1)]
    call_paths += [A/'jobs_a5'/f'diag_a5_call_l{layer:02d}_b{block:02d}.json'
                   for layer, block in ((5, 0), (5, 15), (47, 15))]
    per_call = []
    for p in call_paths:
        j = read(p); c = j['cell']; r = j['result']['comparisons']['SP_vs_A_fp32']
        per_call.append(dict(cell=c['id'], path=str(p.relative_to(R)), sha256=sha(p),
                             layer=c['layer'], L=c['L'], S_all=c['S_all'],
                             max_abs=r['max_abs'], relative_frobenius=r['relative_frobenius'],
                             literal_bounds_pass=r['max_abs'] <= 3.5e-5 and r['relative_frobenius'] <= 7e-7))
    cs = []
    def add(name, kind, deps=(), **kw):
        S = kw.pop('S', 2048)
        est = {'aggregate': [1, 30], 'freeze': [1, 15], 'eq': [1, 30],
               'margin': [10, 90] if S == 2048 else [60, 270]}.get(kind, [90, 275])
        cs.append(dict(id=name, kind=kind, depends=list(deps), S=S,
                       arm=kw.pop('arm', 'A'), window=kw.pop('window', 0), bits=4,
                       apa_min_context=0, worker_s=285, estimate_s=est, **kw))
        return name
    diag = add('diag_a6_propagation_A32_vs_A_2048_w0', 'propagation_check',
               ['diag_a4_propagation_A_2048_w0', 'diag_a2_fp32_A_2048_w0',
                'a4_completed:ppl_a4_D32_2048_w0'], arm='A32')
    trials = []
    for i in range(12):
        trials.append(add(f'trial_a6_{i:02d}', 'trial',
                          [diag, 'ppl_capture_B_2048_w0'] + trials, arm='C', index=i))
    freeze = add('freeze_a6', 'freeze', [diag, 'ppl_capture_B_2048_w0'] + trials, arm='C')
    cp = [add(f'ppl_a6_C_2048_w{w}', 'ppl_capture' if w == 0 else 'ppl',
              [diag, freeze], arm='C', window=w) for w in range(4)]
    add('ppl_a6_C_2048', 'aggregate', cp, arm='C', operation='ppl')
    cap8 = add('ppl_a6_C_8192', 'ppl_capture', [diag, freeze], arm='C', S=8192)
    margins = []
    for S, cap in ((2048, cp[0]), (8192, cap8)):
        for layer in range(5, 48, 6):
            margins.append(add(f'margin_a6_C_{S}_l{layer:02d}', 'margin', [diag, cap],
                               arm='C', S=S, layer=layer, population_rows=S-1, capture_cell=cap))
    bm = [f'margin_a2_B_{S}_l{layer:02d}' for S in (2048, 8192) for layer in range(5, 48, 6)]
    eq = add('eq_a6', 'eq', [diag] + bm + margins, arm='E')
    ep = [add(f'ppl_a6_E_2048_w{w}', 'ppl', [diag, eq], arm='E', window=w) for w in range(4)]
    add('ppl_a6_E_2048', 'aggregate', ep, arm='E', operation='ppl')
    add('ppl_a6_E_8192', 'ppl', [diag, eq], arm='E', S=8192)
    for S in (2048, 8192):
        add(f'decode_a6_C_{S}', 'decode', [diag, freeze], arm='C', S=S)
    ap = A/'jobs_a1/ppl_A_2048_w0.json'
    a32p = A/'jobs_a2/diag_a2_fp32_A_2048_w0.json'
    av, a32v = (read(p)['result']['ppl'] for p in (ap, a32p))
    order = R/'orders/APA_SP4G_AMENDMENT_6.md'
    publish(A/'amendment_019_a6_ruling.json', dict(
        order=dict(path=str(order.relative_to(R)), sha256=sha(order)),
        original_registration_sha256=REG_SHA, before_sha256=sha(A/'a6_before.json'),
        ruling_verbatim=order.read_text(), cells=cs, receipt_namespace='jobs_a6',
        exactness=dict(scope='PER-CALL identical inputs only; finite captured calls',
                       lead_verdict='PASS', literal_bounds=dict(max_abs=3.5e-5, relative_frobenius=7e-7),
                       per_call=per_call, literal_bounds_verdict='RED',
                       discrepancy='Receipt c1 max_abs=3.528594970703125e-5; b15 layer5 max_abs=3.910064697265625e-5 / relF=9.043649367824469e-7 exceed the stated bounds. Lead PASS is recorded as a ruling, never as literal threshold compliance. Bounds are not relaxed.',
                       C_E_authority='Explicit lead amendment 6; no old gate or receipt is rewritten. New propagation stop remains binding.'),
        noise_floor=dict(absolute_ppl=2.56, measured_absolute_ppl=abs(av-a32v), A_bf16=av, A32=a32v,
                         evidence=[dict(path=str(p.relative_to(R)), sha256=sha(p)) for p in (ap, a32p)],
                         measured_scope='2048 window0; one deterministic standard-attention precision ablation',
                         reporting_scope='Lead directs application to every model-level arm PPL difference, including other windows/8192; extrapolated reporting convention there, not separately measured confidence interval',
                         inside='not resolvable on this model', rule='abs(left_ppl-right_ppl) <= 2.56',
                         historical_gate=dict(tolerance=.005, verdict='RED', applicable=False,
                                              reason='Standard branch own rounding path shifts PPL by 2.5593081470667265; retain historical failure and annotate inapplicable under lead ruling'),
                         P2='C - B inside the floor or outside the floor; no within-floor wins/losses'),
        predictions=dict(lead='Same amplification profile, relF growing to ~0.1 by layer29; if flat stop, the lead was wrong.',
                         seat='Expect amplification too: layer29 relF roughly 0.05 to 0.2 and at least 5 times layer5. Exact A/D magnitudes need not repeat; this is a prediction, not a measurement.'),
        propagation_fork=dict(cell=diag, reference='saved A bf16 all2047 rows in jobs_a5',
                              amplified_min_layer29=.05, amplified_min_growth=5., flat_max_growth=2.,
                              rule='AMPLIFIED if L29 relF >= .05 AND >= 5*L5; FLAT if L29 <= 2*L5; otherwise INCONCLUSIVE. Flat or inconclusive => RED STOP all C/E, no retry or widened fork.',
                              note='Seat operationalization registered before the new diagnostic; not a theorem, causal proof, or test of chaos.'),
        protocol=dict(calibration_target=read(A/'jobs_a2/ppl_capture_B_2048_w0.json')['result']['global_fraction']['fraction'],
                      match_abs=.01, max_trials=12, calibration='Inherited bounded bisection, actual global-layer window0 PPL pair population; freeze one delta',
                      ppl='C/E bf16 production path; four consecutive2048 windows last1024, 8192 prefix0 last512; unchanged PROTOCOL-G',
                      margins='All8 global layers, all executed PPL query rows at2048/8192; actual-call bitwise native output and selection replay; finite empirical eq over32 B/C rows',
                      E='Inherited upward(log(100)+2*upward(max empirical eq_sp)); conditional finite-population bound only',
                      decode='Inherited Model.decode, June flags, pooling before load,32 synced steps, no trace/wrapper/interposer/counters, one argmax host copy'),
        safety=dict(foreground_only=True, no_git=True, no_subagents=True, no_signals=True,
                    worker_cooperative_s=285, lease_wait_s=20, cooldown_s=30, call_budget_s=599,
                    load_estimate_s=[75, 130], no_rows_ge=16384, disk_gib=12,
                    limitation='Cooperative Python-boundary checks cannot force-bound a hung native call without signals. Clean decode checks before and after measurement only to preserve timing. No process kill authorized.'),
        cpu_gates=['All existing SP4G tests and A6 tests, zero skips', 'Exact receipt table including literal bound failures',
                   'Floor inclusive boundary and nonfinite rejection', 'A32 native fp32 and returned bf16 pins, complete residual coverage',
                   'Flat/inconclusive stop enforced transitively for every C/E kind', 'Stale fingerprint and payload rejection',
                   'B fraction matched within .01; four short windows and8192;16 margins;2 clean decode cells',
                   'Mutation kill rate >= .80 of non-error mutants; all registered mutants run; no error lanes'],
        mutation_plan=['floor_relaxed', 'floor_nan', 'flat_accepted', 'fingerprint_ignored',
                       'coverage_dropped', 'precision_wrong_arm', 'calibration_relaxed', 'decode_wrapped'],
        prior_art=[
            'SP4G A2/A4/A5 and June Gemma port/floor (2026): reuse precision seam, residual capture, population replay, frozen delta and clean decode.',
            'Haber and Ruthotto (2017), Stable Architectures for Deep Neural Networks, https://arxiv.org/abs/1705.03341: verified abstract for perturbation/stability motivation only; does not prove QAT rounding chaos or a2.56 PPL floor.',
            'Higham and Mary (2022), Mixed precision algorithms in numerical linear algebra, https://doi.org/10.1017/S0962492922000022: mixed-precision error context; no Gemma-specific conclusion.',
            'BLASST/Yuan (2025/2026), ThriftAttention/Sharratt (2026), FlashAttention2/Dao (2023), TurboQuant/Zandieh (2025): inherited selector, precision, online softmax and quantizer; unverified this seat, lead to check arXiv2512.12087,2605.23081,2307.08691,2504.19874.',
            'NIST SHA256 (2001), Make/Feldman (1979), classical bisection, Frobenius norms and DeMillo/Lipton/Sayward (1978) mutation tests; inherited methods. No prior art known to me for a distinct new method introduced here; no novelty claim.'],
        seat=dict(model='gpt-6-astra', reasoning_effort='xhigh', evidence='logs/apa_sp4g_a6_r1.log session header')))
    p = A/'amendment_019_a6_ruling.json'
    with p.with_suffix('.json.sha256').open('x') as f:
        f.write(sha(p)+'  '+p.name+'\n')
    print(sha(p))


if __name__ == '__main__':
    main()
