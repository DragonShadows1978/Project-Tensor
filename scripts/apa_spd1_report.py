"""Receipt-only SPD1 table and prediction reader; no CUDA imports.

Prior art: E1/SP1 (Project-Tensor, 2026) receipt tables and fixed-universe
prediction scoring; classical ratios and IQR summaries, not a new estimator.
Our addition is one explicit cross-contender table with missing-data accounting.
"""
from __future__ import annotations

import json
from pathlib import Path

from apa_spd1_common import (ART, ROOT, SCOPE, fingerprint, receipt_valid,
                            registration, registry, summarize_samples, write_json)


def assemble(reg, attempts, pins):
    by_cell, rejected = {}, []
    for path, r in sorted(attempts, key=lambda pair: pair[0]):
        if not receipt_valid(r, reg, pins):
            rejected.append(dict(path=str(path), reason='stale fingerprint or invalid shape/row/sample contract'))
            continue
        r = dict(r, receipt=str(path))
        # Last explicitly attempted receipt wins, all attempts stay on disk.
        by_cell[r['shape']['id']] = r
    table = []
    for shape in reg['shapes']:
        receipt = by_cell.get(shape['id'])
        rows = receipt['rows'] if receipt else [dict(s, status='BLOCKED_NO_RECEIPT') for s in registry()]
        for row in rows:
            item = dict(row, shape=shape, receipt=receipt['receipt'] if receipt else None,
                        cell_status=receipt['status'] if receipt else 'BLOCKED_NO_RECEIPT')
            if row['status'] == 'OK':
                # Derive displayed medians from raw calls, never trust cached
                # report fields or make an incomplete timing row look measured.
                item['timing'] = summarize_samples(row['timing']['samples_ms'])
                if 'wall' in row:
                    item['wall'] = summarize_samples(row['wall']['samples_ms'])
            table.append(item)
    return dict(table=table, by_cell=by_cell, rejected=rejected)


def score_predictions(reg, table):
    lookup = {(r['shape']['id'], r['id']): r for r in table}
    scores = {}

    def row(s, name):
        result = lookup.get((s['id'], name))
        return result if result and result['status'] == 'OK' else None

    def time(s, name):
        r = row(s, name)
        return r['timing']['median_ms'] if r else None

    def compare(s, names, rule):
        values = [time(s, n) for n in names]
        return None if any(v is None for v in values) else bool(rule(*values))

    def record(name, cases, *, any_hit=False):
        known = [v for _, v in cases if v is not None]
        complete = len(known) == len(cases) and len(cases) > 0
        hit = any(known) if any_hit else all(known)
        scores[name] = dict(verdict=('HIT' if hit else 'MISS') if complete else 'BLOCKED',
            expected=len(cases), evaluated=len(known), hits=sum(known),
            failures=[k for k, v in cases if v is False], blocked=[k for k, v in cases if v is None])

    shapes = reg['shapes']
    pre = [s for s in shapes if s['kind'] == 'prefill']
    dec = [s for s in shapes if s['kind'] == 'decode']
    parity = [(s['id'], (row(s, 'torch_math_fp32') or {}).get('fp32_parity')) for s in shapes]
    record('P1_fp32_parity', parity); record('A1_fp32_parity', parity)
    for name in ['torch_flash_bf16', 'torch_efficient_bf16']:
        record('P1_' + name + '_ge2x_vs_dense_fp32_mixed_dtype',
               [(s['id'], compare(s, ['engine_dense_fp32', name], lambda d, f: d / f >= 2)) for s in pre if s['S'] >= 2048])
    record('P2_two_pass_slower', [(s['id'], compare(s, ['apa_two_pass_fp32', 'engine_dense_fp32'], lambda a, d: a > d))
                                for s in shapes if s['S'] <= 8192])
    gap = []
    for s in pre:
        if s['S'] != 8192:
            continue
        sp = row(s, 'apa_sp_fp32')
        eligible = sp and sp.get('fraction', {}).get('matched_budget_eligible')
        old, dense = time(s, 'apa_two_pass_fp32'), time(s, 'engine_dense_fp32')
        positive_gap = old is not None and dense is not None and old > dense
        result = compare(s, ['apa_two_pass_fp32', 'apa_sp_fp32', 'engine_dense_fp32'],
                         lambda old, new, dense: (old-new)/(old-dense) >= .5) if eligible and positive_gap else None
        gap.append((s['id'], result))
    record('P3_half_gap_prefill_s8192_matched_only', gap)
    record('P3_decode_s32768_within2x', [(s['id'], compare(s, ['apa_sp_fp32', 'engine_dense_fp32'], lambda sp, d: sp <= 2*d)) for s in dec if s['S'] == 32768])
    record('P3_SP_never_beats_flash_mixed_dtype', [(s['id'], compare(s, ['apa_sp_fp32', 'torch_flash_bf16'], lambda sp, flash: sp >= flash)) for s in shapes])
    record('A2_flash_faster_prefill_mixed_dtype', [(s['id'], compare(s, ['apa_sp_fp32', 'torch_flash_bf16'], lambda sp, flash: flash < sp)) for s in pre if s['S'] >= 2048])
    for name in ['apa_two_pass_fp32', 'apa_sp_fp32', 'apa_two_pass_bf16', 'apa_sp_bf16']:
        cases = []
        for s in pre:
            if s['S'] != 8192:
                continue
            dense = row(s, 'engine_dense_' + name.rsplit('_', 1)[1]); apa = row(s, name)
            result = (apa['peak_allocated_delta_bytes'] <= dense['peak_allocated_delta_bytes'] / 50
                      if dense and apa and dense['peak_allocated_delta_bytes'] > 0 else None)
            cases.append((s['id'], result))
        record('P4_' + name + '_same_dtype', cases)
        if name.endswith('fp32'):
            record('A4_' + name, [(k, v) for k, v in cases if k.startswith('prefill_')])
    record('A3_at_least_one_decode_fraction_mismatch',
           [(s['id'], (row(s, 'apa_sp_fp32')['fraction']['absolute_difference'] > .02
                       if row(s, 'apa_sp_fp32') and 'fraction' in row(s, 'apa_sp_fp32') else None)) for s in dec], any_hit=True)
    scores['A5_paper_shape_mismatch'] = dict(verdict='HIT', evidence_class='registered shape/source comparison',
        reason='all SPD1 B=1; paper section 4.5 B=2; no exact-shape paper row')
    return scores


def number(value):
    return '—' if value is None else f'{value:.6g}'


def gate_verdict(reg, by_cell):
    receipts = list(by_cell.values())
    if any(r['status'] not in ('COMPLETE', 'BLOCKED') or r.get('numeric_status') == 'RED'
           for r in receipts):
        return 'RED'
    full = len(receipts) == len(reg['shapes'])
    all_good = all(r['status'] == 'COMPLETE' and r.get('numeric_status') == 'PASS'
                   and all(x['status'] in ('OK', 'UNAVAILABLE') for x in r['rows']) for r in receipts)
    return 'PASS (kernel table only)' if full and all_good else 'BLOCKED / INCOMPLETE'


def render(reg, assembled):
    table = assembled['table']
    scores = score_predictions(reg, table)
    complete = sum(r.get('status') == 'COMPLETE' for r in assembled['by_cell'].values())
    text = ['# APA-SPD1 speed chain', '',
        f'G2: {gate_verdict(reg, assembled["by_cell"])}; {complete}/{len(reg["shapes"])} complete cell receipts.',
        '', SCOPE, '',
        'Primary engine/APA rows are FP32; fused SDPA rows are BF16. Additional BF16 dense, math, two-pass and SP rows provide same-dtype comparisons. Accuracy always uses the full engine FP32 output from the original shared inputs; BF16 errors include input rounding.', '',
        'Latency: CUDA-event median and IQR, nine calls after three warmups, rotating interleaving, legacy stream 0. Wall median is separate. Peak: max of three call-local allocation high-water deltas, including output and measured GQA expansion; engine CUDA pool and torch native allocator are separate accounting systems. Resident q/k/v/kq, quantizer setup, diagnostics and transfers are excluded. kq remains floating; this does not measure compressed KV-cache residency.', '',
        'SP1 frozen δ values are reused on TurboQuant4 keys with r=0.15. E1 extras use symmetric INT4 and r=0.10; their transferred δ values are UNCALIBRATED. Fraction matching is a bounded CPU z-score estimate against the corresponding GPU SP mask sample, not exact CUDA baseline selection.', '',
        '| Cell (B/H/KV/L/S/D/causal in registration) | Contender | dtype | Status | CUDA median ms | IQR ms | Wall ms | Peak MiB | rel Frobenius | max abs | SP / two-pass fraction estimate |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in table:
        timing, acc, frac = r.get('timing', {}), r.get('accuracy', {}), r.get('fraction', {})
        fraction = (f"{frac['gpu_sp_sample_fraction']:.5f} / {frac['cpu_two_pass_sample_fraction']:.5f}; "
                    + ('estimate matched' if frac.get('matched_budget_eligible') else 'UNMATCHED/UNCALIBRATED')) if frac else '—'
        peak = r.get('peak_allocated_delta_bytes')
        text.append('| ' + ' | '.join([r['shape']['id'], r['id'], r['dtype'], r['status'],
            number(timing.get('median_ms')), number(timing.get('iqr_ms')), number(r.get('wall', {}).get('median_ms')),
            number(peak / 2**20 if peak is not None else None), number(acc.get('relative_frobenius')),
            number(acc.get('max_abs')), fraction]) + ' |')
    text += ['', '## Reading the measurements', '']
    lookup = {(r['shape']['id'], r['id']): r for r in table}
    measured = [r for r in table if r['status'] == 'OK']
    if not measured:
        text += ['No contender has a current GPU timing receipt. Wins, losses, speed crossover, sm_89 backend support, numerical parity and peak-memory predictions remain unmeasured.']
    for spec in registry():
        for family in ('prefill', 'decode'):
            group = [r for r in measured if r['id'] == spec['id'] and r['shape']['kind'] == family]
            ratios, wins, losses, overlaps = [], 0, 0, 0
            for r in group:
                suffix = 'fp32' if r['dtype'] == 'float32' else 'bf16'
                d = lookup[(r['shape']['id'], 'engine_dense_' + suffix)]
                if d['status'] != 'OK':
                    continue
                ratios.append(d['timing']['median_ms'] / r['timing']['median_ms'])
                wins += r['timing']['median_ms'] < d['timing']['median_ms']
                losses += r['timing']['median_ms'] > d['timing']['median_ms']
                overlaps += max(r['timing']['q1_ms'], d['timing']['q1_ms']) <= min(r['timing']['q3_ms'], d['timing']['q3_ms'])
            if ratios:
                text.append(f'- {spec["id"]}, {family}: {wins} faster / {losses} slower median cells versus same-dtype engine dense; dense/contender speed ratio {min(ratios):.3g}–{max(ratios):.3g}× over {len(ratios)} cells; {overlaps} overlapping IQRs (timing separation uncertain).')
    text += ['', '## Registered predictions', '', '| Prediction | Verdict | Evaluated / expected | Hits |', '|---|---|---:|---:|']
    for name, s in scores.items():
        text.append(f'| {name} | {s["verdict"]} | {s.get("evaluated", "—")} / {s.get("expected", "—")} | {s.get("hits", "—")} |')
    text += ['', 'P1/P3 flash speed comparisons above are explicitly FP32 engine/APA versus BF16 flash. Same-dtype ratios appear in the reading. P3 half-gap requires a positive two-pass latency gap and estimated matched budget; uncalibrated E1 transfers cannot satisfy that premise. Missing or unsupported rows block universal predictions, never count as wins. Per-cell prediction failures and blocked IDs are in speed_chain.json.', '',
        '## Reconciliation with earlier receipts', '',
        'The two E1 extras match B=1, H=16, KV=4, D=128, causal L=512 at S=8192/32768, with BF16 rows and r=0.10 symmetric INT4 bulk reconstruction. E1 RESULTS.md measured one warmup and three synchronized wall calls; SPD1 uses three warmups and nine CUDA-event calls plus separate wall medians. Seeds and rounding preparation differ, so timing differences require those qualifications. SP1’s 48 geometries match, but its bulk K+0.1-noise data and FP32-only timing are not the TurboQuant4 data here; its delta values are retained without assuming budget equality. The paper §4.5 used B=2,H=4,D=64,r=0.15 and an older kernel generation: its 2048-token 27.77 ms SDPA / 13.03 ms APA (2.1×) result has no identical SPD1 shape and cannot be chained as a flash speedup. Existing receipts are context, never filled into missing SPD1 rows.', '',
        '## Availability, errors, and receipts', '']
    for cell, r in assembled['by_cell'].items():
        text.append(f'- {cell}: {r["status"]}; `{r["receipt"]}`')
        if r.get('error'):
            text.append('```text\n' + r['error'].strip() + '\n```')
        for row in r['rows']:
            if row.get('reason'):
                text.append(f'  {row["id"]}: {row["status"]}: {row["reason"]}')
    if assembled['rejected']:
        text.append('Rejected stale/invalid receipts: ' + json.dumps(assembled['rejected']))
    text += ['', '## Prior art', '',
        'Reused: Vaswani et al. (2017) dense attention; [PyTorch SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.attention.sdpa_kernel.html) (contributors, 2023–2026); [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao, 2023); [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025; repository MSE reconstruction); APA/E1/SP1/SP1.1 (David and Project-Tensor seats, 2026); [BLASST](https://arxiv.org/abs/2512.12087) (Yuan et al., 2025/2026) running-max prior art; [online normalizer](https://arxiv.org/abs/1805.02867) (Milakov and Gimelshein, 2018); Flash-Decoding (Dao et al., 2023). [ThriftAttention](https://arxiv.org/abs/2605.23081) (Sharratt, 2026) is related selective mixed precision, not a contender. No new attention algorithm is introduced. Full attribution and verification limits: PRIOR_ART.md.', '']
    return '\n'.join(text), scores


def summary():
    reg, pins = registration(), fingerprint()
    attempts = []
    malformed = []
    for path in (ART / 'gpu').glob('*.receipt.json'):
        try:
            attempts.append((str(path.relative_to(ROOT)), json.loads(path.read_text())))
        except (ValueError, OSError) as exc:
            malformed.append(dict(path=str(path), reason=str(exc)))
    assembled = assemble(reg, attempts, pins)
    assembled['rejected'] += malformed
    text, scores = render(reg, assembled)
    exit_receipts = []
    for path in sorted((ART / 'gpu').glob('*.exit.json')):
        exit_receipts.append(dict(path=str(path.relative_to(ROOT)), **json.loads(path.read_text())))
    # Exits/timeouts are retained even if no final worker JSON exists.
    text += '\n## Process exits\n\n' + ('No GPU worker exits recorded.\n' if not exit_receipts else '\n'.join(
        f'- {r["cell"]}: exit {r["exit_code"]}; `{r["path"]}`' for r in exit_receipts) + '\n')
    (ART / 'SPEED_CHAIN.md').write_text(text)
    write_json(ART / 'speed_chain.json', dict(evidence_class='kernel sweep receipt assembly',
        scope_note=SCOPE, fingerprint=pins, predictions=scores, exit_receipts=exit_receipts, **assembled))
    print(json.dumps(dict(report=str(ART / 'SPEED_CHAIN.md'), cells=len(assembled['by_cell']),
                          expected=len(reg['shapes']), rejected=len(assembled['rejected']))))
