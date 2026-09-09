"""A5 registration BEFORE gates. Prior art: SP3/SP4G (2026) immutable
experiment records; Make/Feldman (1979) dependencies, NIST SHA256 (2001).
Only diagnostic scope/dependency wiring is new; no attention algorithm.
"""
from apa_sp4g_common import *


def main():
    paths = sorted(R.glob('scripts/apa_sp4g_*'))
    paths += sorted(R.glob('tensor_cuda/tests/test_apa_sp4g*.py'))
    paths = [p for p in paths if p.is_file() and '_a5_' not in p.name]
    before = dict(source_sha256={str(p.relative_to(R)): sha(p) for p in paths},
                  receipt_sha256={str(p.relative_to(R)): sha(p) for p in sorted(A.glob('jobs*/*.json'))})
    publish(A/'a5_before.json', before)
    for name in ('RESULTS.md', 'lead_commands.txt', 'cells.json'):
        p = A/'a5_baseline'/name
        p.parent.mkdir(exist_ok=True)
        with p.open('xb') as f:
            f.write((A/name).read_bytes())
    cs = []
    def add(name, kind, depends=(), **kw):
        cs.append(dict(id=name, kind=kind, depends=list(depends), S=2048,
                       arm=kw.pop('arm', 'A'), window=0, bits=4, apa_min_context=0,
                       worker_s=285, estimate_s=[1,30] if kind=='aggregate' else [90,275], **kw))
        return name
    refs = ['diag_a2_fp32_A_2048_w0', 'a4_completed:ppl_a4_D32_2048_w0']
    names = []
    for layer, block in ((5,0),(5,15),(47,15)):
        names.append(add(f'diag_a5_call_l{layer:02d}_b{block:02d}', 'call', refs,
                         layer=layer, block=block, call_index=2+block, L=64,
                         S_all=1087+64*block, position_offset=1023+64*block,
                         order_stated_S_all=1088+64*block))
    props = [add(f'diag_a4_propagation_{arm}_2048_w0', 'propagation', refs, arm=arm) for arm in 'AD']
    add('diag_a4_propagation_2048_w0', 'aggregate', props, operation='propagation')
    add('ppl_a4_D32_2048_w0', 'precision', names+refs, arm='D32',
        conditional='all three calls completed; at least one SP vs standard fp32 relF > 0.001; no C/E unblock')
    order = R/'orders/APA_SP4G_AMENDMENT_5.md'
    publish(A/'amendment_017_a5_registration.json', dict(
        order=dict(path=str(order.relative_to(R)), sha256=sha(order)),
        before_sha256=sha(A/'a5_before.json'), original_registration_sha256=REG_SHA,
        cells=cs, receipt_namespace='jobs_a5; same propagation/D32 ids, original receipts immutable',
        prediction=dict(text='These DISAGREE beyond 1e-3 rel-Frobenius in fp32; SP handling of L << S_all with a cache (offset / causal bound / dispatch), not standard.',
                        threshold=0.001, strict_greater=True, scope='each of the three registered calls'),
        shape_correction=dict(reason='Actual PPL source and A4 144 dtype pins: prefill ids[:1023], scored input positions 1023..2046 predict targets 1024..2047. Order numbers are one key larger.',
                              evidence='artifacts/apa_sp4g/jobs_a4/ppl_a4_D32_2048_w0.json',
                              preserve_actual_schedule=True, add_fake_token=False),
        source_prediction=dict(dispatch='prefill apa_selective_sp_kernel<T,512,DIAG>, grid 1024 blocks x 32 threads; split-K only L==1',
                               cache='tuple; kq_count null/not applicable; whole-span Kq reconstruction'),
        forks=dict(disagree='Name cause from arguments; harness fix requires named evidence and separate amendment; correct arguments plus SP/dense disagreement and standard/dense agreement imply SP path defect for lead, no kernel edit. Conditional D32 rerun preserves prior RED.',
                   agree='Report prediction falsified at these calls; propagation is remaining lead.',
                   partial='Mixed or failed captures reported individually, never generalize; propagation independent.',
                   propagation='Requires completed, provenance-valid A32 and D32, accepts completed numerical RED; rejects missing/failed/stale D32.',
                   quality='C/E remain behind original <=0.005 PPL gate; no compatibility waiver.'),
        cpu_gates=['actual L64 cached schedule coverage including final layer', 'tuple cache kq_count and source dispatch',
                   'full schedule capture rather than prefill-only', 'completed RED accepted; worker RED and stale provenance rejected',
                   'propagation independent of call outcome and quality gate', 'conditional create-only D32 rerun',
                   'same-input parity and native dtype pins', 'author full CPU baseline, mutations >=0.80 of non-error mutants'],
        mutation_plan=['prefill_only', 'off_by_one', 'splitk_for_cached', 'completed_red_rejected',
                       'worker_red_accepted', 'fingerprint_ignored', 'rerun_without_disagreement', 'capture_wrong_layer'],
        safety=dict(foreground_only=True, no_git=True, no_subagents=True, no_signals=True,
                    worker_cooperative_s=285, lease_wait_s=20, per_call_limit_s=599,
                    caveat='Cooperative deadlines check Python boundaries; no signal timeout can guarantee termination of a hung native call. No kills are authorized.'),
        prior_art='SP4G A3/A4 (2026) same-input replay, dense reference, residual capture, dtype pins; Vaswani et al. (2017) attention; NIST SHA256 (2001), Make/Feldman (1979), DeMillo/Lipton/Sayward (1978) mutation tests. New wiring only; no novelty claim.'))
    print(sha(A/'amendment_017_a5_registration.json'))


if __name__ == '__main__':
    main()
