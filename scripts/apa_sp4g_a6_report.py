"""Prior art: SP4G A4/A5 (2026) receipt-driven reports, descriptive
pairwise differences; lead A6 floor. No novel statistic or algorithm.
"""
from itertools import combinations
from apa_sp4g_a6_common import *
from apa_sp4g_a6_registry import cells


def result_or_pending(name):
    try:
        return require_a6(name)['result'], None
    except (OSError, Red) as e:
        return {}, str(e)


def table_rows():
    rows = []
    for suffix in ['2048_w'+str(w) for w in range(4)]+['2048', '8192']:
        results = {}; receipts = {}; errors = {}
        for arm in 'ABCDE':
            name = f'ppl_{"a6_" if arm in "CE" else ""}{arm}_{suffix}'
            r, error = result_or_pending(name)
            if 'ppl' in r:
                results[arm] = r['ppl']
                receipts[arm] = dict(path=str(path_a6(name).relative_to(R)), sha256=sha(path_a6(name)))
            else:
                errors[arm] = error
        pairs = {a+'-'+b: floor_comparison(results[a], results[b])
                 for a, b in combinations('ABCDE', 2) if a in results and b in results}
        # P2 sign is C minus B, explicitly, regardless of pair iteration order.
        p2 = floor_comparison(results['C'], results['B']) if {'B', 'C'} <= results.keys() else None
        rows.append(dict(row=suffix, ppl=results, receipts=receipts, unavailable=errors,
                         pairwise=pairs, P2=p2))
    return rows


def cell_states():
    from apa_sp4g_a6_gpu import preflight
    rows = []
    for c in cells():
        error = None
        try:
            state = preflight(c)
            state = 'PASS' if state == 'DONE' else 'READY_'+state
        except (OSError, Red) as e:
            state = 'RED' if path_a6(c['id']).exists() else 'BLOCKED_DEPENDENCY'
            error = str(e)
        rows.append(dict(cell=c, state=state, error=error,
                         command=f"bash scripts/apa_sp4g_a6_lead_gpu.sh run {c['id']}"))
    return rows


def main():
    preserved()
    from apa_sp4g_a3_common import require_a3
    reg = read(REGISTRATION)
    for row in reg['exactness']['per_call']:
        j = require_a3(row['cell']) if '_a3_' in row['cell'] else a5.require_a5(row['cell'])
        r = j['result']['comparisons']['SP_vs_A_fp32']
        if sha(R/row['path']) != row['sha256'] or any(r[k] != row[k] for k in ('max_abs', 'relative_frobenius')):
            raise Red('A6_PER_CALL_TABLE_STALE')
    old = a5.completed_d32()['result']
    ad = a5.require_a5('diag_a4_propagation_2048_w0')['result']
    new, new_error = result_or_pending(DIAG)
    if not new and path_a6(DIAG).exists():
        c = by_id()[DIAG]
        try:
            for d in c['depends']:
                require_a6(d)
            j = validate_receipt(read(path_a6(DIAG)), c, fingerprint(c),
                                 {d: sha(path_a6(d)) for d in c['depends']}, allow_completed_red=True)
            new = j['result']; new_error = j['error']
        except (OSError, Red):
            pass  # stale/partial metrics are never displayed
    measured = table_rows(); states = cell_states()
    report = ['# APA-SP4G amendment 6', '',
        '**RED residuals. D/A per-call exactness: PASS by lead ruling; literal stated bounds: RED.** '
        'C/E are released from the historical0.005 PPL dependency by amendment6 and scheduled behind the new propagation stop. '
        'A6 GPU measurements are '+('completed for the diagnostic.' if new else 'UNRUN or blocked; no A32-vs-A amplification finding yet.'), '',
        '**Evidence discrepancy:** '+reg['exactness']['discrepancy'], '',
        'The recorded PASS is a lead interpretation of finite same-input agreement. It does not certify every call, '
        'bitwise equality, or compliance with the stated scalar bounds. No threshold or old receipt was edited.', '',
        '## Per-call D/A exactness', '',
        'Evidence class: prior card per-call diagnostics, validated receipt and payload provenance; '
        'fp32 SP versus standard on identical input values. Max-abs limit3.5e-5; relF limit7e-7.', '',
        '| Cell | Layer | L | S_all | fp32 max-abs | fp32 relF | Literal bounds |',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in reg['exactness']['per_call']:
        report.append(f"| [{row['cell']}]({row['path'].replace('artifacts/apa_sp4g/', '')}) | {row['layer']} | {row['L']} | {row['S_all']} | {row['max_abs']:.15g} | {row['relative_frobenius']:.15g} | {'PASS' if row['literal_bounds_pass'] else 'RED'} |")
    f = reg['noise_floor']
    report += ['', '## Model perplexity — registered numerical-path floor ±2.56 PPL', '',
        f"Measured window0: A bf16={f['A_bf16']:.15g}, A32={f['A32']:.15g}; |A32−A|={f['measured_absolute_ppl']:.15g}, rounded floor±2.56. "
        'This is a deterministic numerical-path sensitivity measurement, not a statistical confidence interval. '
        'The lead applies this floor to every model-level arm difference here. It is **extrapolated, not independently measured**, '
        'for windows1–3, their pooled NLL, and8192. An outside-floor gap alone does not establish a robust gain.', '',
        '**Gemma4 QAT INT4; bf16 production arms; APA on the8 global layers only;40 sliding layers unchanged. '
        'Every PPL comparison uses the ±2.56 floor.** Short rows score1024 targets each; pooled2048 uses4096 total targets;8192 scores512.', '',
        '| Row | A bf16 | B | C | D | E | P2: C − B against ±2.56 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in measured:
        values = [f"{row['ppl'][arm]:.8f}" if arm in row['ppl'] else 'UNRUN/RED' for arm in 'ABCDE']
        p2 = row['P2']
        text = f"{p2['difference']:+.8f}; C − B {'inside' if p2['inside_floor'] else 'outside'} the floor; {p2['verdict']}" if p2 else 'UNRUN'
        report.append('| '+row['row']+' | '+' | '.join(values)+' | '+text+' |')
    report += ['', 'Validated PPL receipt paths/hashes and **every available arm pair** with its floor classification '
        'are in `PPL_TABLE_A6.json`. No within-floor difference is labelled a win or loss.', '',
        '| Row | B − A | D − A | Interpretation against ±2.56 |', '|---|---:|---:|---|']
    for row in measured:
        p = row['ppl']
        if {'A', 'B', 'D'} <= p.keys():
            b, d = floor_comparison(p['B'], p['A']), floor_comparison(p['D'], p['A'])
            report.append(f"| {row['row']} | {b['difference']:+.8f} | {d['difference']:+.8f} | B−A: {b['verdict']}; D−A: {d['verdict']} |")
    od = floor_comparison(old['ppl'], old['A32_ppl'])
    report += ['', f"Historical D32={old['ppl']:.15g}, A32={old['A32_ppl']:.15g}; D32−A32={od['difference']:+.15g}: {od['verdict']}. "
        'All144 fp32 pins complete. D32 PPL is bit-identical to D bf16 PPL on window0 (difference0, not resolvable on this model); '
        'this does not certify bit-identical logits. Historical gate≤0.005 stays **RED**, annotated **inapplicable under the lead numerical-path ruling**. '
        'Error retained verbatim: `A4_D32_A32_EXACTNESS_FAILED; STOP, lead investigates`. Not claimed fixed: historical model-PPL exactness.', '',
        '## Propagation receipt and predictions', '',
        f"Cell: `{DIAG}`; state: "+((('PASS, ' if new['decision']['C_E_unblocked'] else 'RED STOP, ')+new['decision']['outcome']+'; '+str(new_error)) if new else 'UNRUN/RED: '+str(new_error)), '',
        '**Lead prediction:** '+reg['predictions']['lead'], '',
        '**Seat prediction:** '+reg['predictions']['seat'], '',
        reg['propagation_fork']['rule'], '',
        'One new A32 load; compare against saved A bf16 residuals from `jobs_a5/diag_a4_propagation_A_2048_w0.json`. '
        'Native fp32 Q/K/Kq/V/output pins, bf16 cast before o_proj, complete144-call schedule, and A32 PPL reproduction are mandatory. '
        'All2047 query rows after each complete global block and final norm; sum squared errors and reference norms across blocks.', '',
        '| Layer | D−A bf16 relF | D−A max-abs | A32−A relF | A32−A max-abs |', '|---|---:|---:|---:|---:|']
    by_layer = {r['layer']: r for r in new.get('per_layer', [])}
    for r in ad['per_layer']:
        nr = by_layer.get(r['layer'])
        extra = f"{nr['relative_frobenius']:.9g} | {nr['max_abs']:.9g}" if nr else 'UNRUN | UNRUN'
        report.append(f"| {r['layer']} | {r['relative_frobenius']:.9g} | {r['max_abs']:.9g} | {extra} |")
    report += ['', 'Evidence class: prior card residual diagnostic, `jobs_a5/diag_a4_propagation_2048_w0.json`. '
        'D/A L29-to-L5 relF ratio='+f"{ad['per_layer'][4]['relative_frobenius']/ad['per_layer'][0]['relative_frobenius']:.5g}"+
        '. The approximately0.3% figure describes attention-output differences; post-block L5 residual relF is0.4821%. '
        'Observed amplification is a mechanism lead, not proof of chaos or a complete causal explanation of PPL.', '',
        '## C/E cells, estimates and commands', '',
        f"Calibration target is B's measured{reg['protocol']['calibration_target']:.15g}, rounded0.152, on the actual window0 global-layer pair population; tolerance±0.01. "
        'Twelve registered bisection trials; after a match, later trials carry its immutable result without loading the model. '
        'C uses one frozen delta. E uses the inherited conditional empirical bound from all32 B/C layer-margin receipts. '
        'Neither E’s finite e_q sample nor G2/G3 rows establish model quality by themselves.', '',
        'Run **each command separately in the foreground**, in `lead_commands.txt` order. `resume` runs exactly one next cell. '
        'Historical0.005 does not block this DAG; missing/flat/inconclusive A6 propagation does. '
        'Model-cell estimates include75–130s load, based on earlier75–126s receipts; estimates are not new measurements. '
        'GPU calls add up to20s lease wait and30s cooldown. Worker cooperative rail285s, call budget599s.', '',
        '| Cell | Worker estimate (s) | State |', '|---|---:|---|']
    for row in states:
        c = row['cell']
        report.append(f"| `{c['id']}` | {c['estimate_s'][0]}–{c['estimate_s'][1]} | {row['state']} |")
    report += ['', 'Exact command per cell and dependency reason: `GPU_BLOCKED_A6.json`; complete command list: `lead_commands.txt`.', '',
        '## CPU gates, fingerprints and RED residuals', '',
        f"Ruling019 SHA256 `{REGISTRATION_SHA}`. Original registration unchanged: `{REG_SHA}`."]
    for p in (SEAL, A/'CPU_GATES_A6.json'):
        if p.exists():
            report.append(f"`{p.name}` SHA256 `{sha(p)}`.")
    if (A/'CPU_GATES_A6.json').exists():
        cpu = read(A/'CPU_GATES_A6.json')
        report += ['', f"Author CPU baseline: {cpu['passed']} passed, {cpu['failed']} failed, {cpu['skipped']} skipped; "
            f"mutations {cpu['mutations']['killed']}/{cpu['mutations']['nonerror']}, threshold0.80. "
            'Blind review is lead-owned and UNRUN. CPU doubles validate harness wiring, never actual GPU numerics.']
    report += ['', 'Not claimed fixed: original0.005 PPL gate, literal3.5e-5/7e-7 bound discrepancy, '
        'unmeasured A6 propagation and C/E quality while pending; no extension of the floor’s measured scope. '
        'Product, kernel, adapter, model, order, historical execution files and receipts preserved byte-for-byte.', '',
        'Process safety: no git, subagents, background jobs/waits, signals/kills, services or model writes. '
        'No A6 GPU worker or model load in this dispatched seat. CUDA visibility is recorded in `a6_device_visibility.json`. '
        'Cooperative deadlines cannot forcibly bound a hung native call without violating no-kill; clean decode checks '
        'before/after measurement only, preserving the unwrapped timed path. Each lead command is planned below10minutes; no hard no-kill wall-time guarantee is claimed.', '',
        'Seat: **gpt-6-astra, reasoning xhigh**, live session header `logs/apa_sp4g_a6_r1.log`. '
        'Model under test: **Gemma-4-12B-it QAT q4_0 exact (symmetric-8 g32)**; bf16 engine, fp32 global-attention diagnostic only.', '',
        '## Prior art', '',
        'SP4G A2/A4/A5 and June Gemma port/floor (2026) provide the precision seam, residual capture, native-mask replay, '
        'calibration and clean decode. A6 adds ruling/floor reporting, combines A32 with that capture, and replaces successor dependencies. '
        'No new attention algorithm or novelty claim.', '',
        '[Haber and Ruthotto (2017), Stable Architectures for Deep Neural Networks](https://arxiv.org/abs/1705.03341) '
        'provides dynamical-system stability context; [Higham and Mary (2022), Mixed precision algorithms in numerical linear algebra]'
        '(https://doi.org/10.1017/S0962492922000022) provides mixed-precision error context. Primary abstracts/metadata verified this seat. '
        'Neither establishes Gemma QAT rounding chaos or this PPL floor; that connection is an experimental inference to test.', '',
        'Inherited prior art: BLASST/Yuan (2025/2026) running-max selection, ThriftAttention/Sharratt (2026) weight-sensitive precision, '
        'FlashAttention2/Dao (2023) online softmax, TurboQuant/Zandieh (2025) quantization; '
        'unverified this seat — lead to check arXiv2512.12087,2605.23081,2307.08691,2504.19874. '
        'Classical bisection, Frobenius norms, NIST SHA256 (2001), Make/Feldman (1979) dependency DAGs; '
        'DeMillo/Lipton/Sayward (1978) mutation testing, unverified — lead to check Hints on Test Data Selection. '
        'No prior art known to me for a distinct new method introduced here.', '']
    output = '\n'.join(report)
    for name in ('RESULTS.md', 'A6_REPORT.md'):
        (A/name).write_text(output)
    commands = ['# A6: EACH command is a separate foreground call; never launch this list as one batch.',
                '# First diagnostic must be AMPLIFIED. Flat/inconclusive => STOP; no retry or threshold change.',
                '# Estimates include load; GPU adds up to20s lease wait plus30s cooldown. No signals/kills.']
    for row in states:
        c = row['cell']
        commands += [f"# {row['state']}; worker estimate{c['estimate_s'][0]}–{c['estimate_s'][1]}s",
                     row['command']]
    commands += ['bash scripts/apa_sp4g_a6_lead_gpu.sh summary']
    (A/'lead_commands.txt').write_text('\n'.join(commands)+'\n')
    for name, obj in [('cells.json', cells()), ('PPL_TABLE_A6.json', dict(floor=reg['noise_floor'], rows=measured)),
                      ('GPU_BLOCKED_A6.json', dict(status='RED_GPU_UNRUN_OR_PENDING',
                       a6_registration_sha256=REGISTRATION_SHA, floor=2.56, diagnostic=DIAG, cells=states))]:
        (A/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='RED', report='artifacts/apa_sp4g/A6_REPORT.md',
                          cells=len(states), states={s: sum(r['state'] == s for r in states) for s in set(r['state'] for r in states)}), indent=2))


if __name__ == '__main__':
    token = VALIDATION.set({})
    try:
        main()
    finally:
        VALIDATION.reset(token)
