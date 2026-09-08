"""Prior art: SP4G/June Gemma (2026), SHA256/NIST (2001), Make/Feldman
(1979). Additive registration, dimensional KV accounting and lead-requested
quadratic extrapolation; no new attention algorithm or selection rule.
"""
from apa_sp4g_common import *


def main():
    before_path = A/'a7_before.json'
    old = sorted(p for p in R.glob('scripts/apa_sp4g_*') if p.is_file() and '_a7_' not in p.name)
    old += sorted(p for p in R.glob('tensor_cuda/tests/test_apa_sp4g*.py') if '_a7' not in p.name)
    before = dict(source_sha256={str(p.relative_to(R)): sha(p) for p in old},
                  receipt_sha256={str(p.relative_to(R)): sha(p) for p in sorted(A.glob('jobs*/*.json'))},
                  amendment_sha256={str(p.relative_to(R)): sha(p) for p in sorted(A.glob('amendment*.json'))})
    publish(before_path, before)
    for name in ('RESULTS.md', 'lead_commands.txt', 'cells.json'):
        p = A/'a7_baseline'/name
        p.parent.mkdir(exist_ok=True)
        with p.open('xb') as f:
            f.write((A/name).read_bytes())
    anchors = {}
    for arm, name in zip('ABC', ('jobs_a1/ceiling_A_8192.json', 'jobs_a1/ceiling_B_8192.json',
                               'jobs_a6/decode_a6_C_8192.json')):
        p = A/name; j = read(p); r = j['result']
        if j['status'] != 'PASS' or not r['fit']:
            raise Red('A7_ANCHOR_NOT_COMPLETED')
        anchors[arm] = dict(path=str(p.relative_to(R)), sha256=sha(p), S=8192,
                            load_s=r['load_s'], prefill_s=r['prefill_s'],
                            scope='prefill component only; excludes decode, scoring and capture overhead')
    cs = []; previous = None
    for arm in 'ABC':
        for S in (16384, 24576, 32768, 49152, 65536, 98304, 131072):
            anchor = anchors[arm]; prefill = anchor['prefill_s']*(S/8192)**2
            name = f'ceiling_long_{arm}_{S}'
            cs.append(dict(id=name, kind='ceiling_long', arm=arm, S=S, window=0,
                           depends=[previous] if previous else [], worker_s=1500, outer_s=1560,
                           estimate_prefill_s=prefill, estimate_worker_s=anchor['load_s']+prefill,
                           estimate_outer_s=anchor['load_s']+prefill+50,
                           predicted_exceeds_rail=anchor['load_s']+prefill > 1500))
            previous = name
    reg = dict(schema='apa_sp4g_a7_ceiling_long_v1', immutable=True,
        registered_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        original_registration_sha256=REG_SHA, before_sha256=sha(before_path),
        order=dict(path='orders/APA_SP4G_AMENDMENT_7.md', sha256=sha(R/'orders/APA_SP4G_AMENDMENT_7.md')),
        cells=cs, anchors=anchors,
        protocol=dict(stream='tokens.npy, original pinned prefix, no repetition or retokenization',
            token_count=registration()['protocol']['token_count'], prefill_only=True,
            feeding='Model.prefill(ids[:S]) -> June Gemma4_TC.__call__(last_token_only=True); automatic adaptive chunks unchanged',
            pooling_before_load=True, prefill_chunk=512, minimum_chunk=64, apa_min_context=0,
            bulk_bits=4, refine_percentile=.15, C_delta=3., quant_v=False, quant_kv4=False,
            global_kv_bytes_per_token=8*2*1*512*2, sliding_fixed_bytes=40*2*8*256*1023*2,
            sliding_capacity_upper_bytes=40*2*8*256*1024*2,
            cache_scope='logical bf16 K+V tuple payload at completed prefill; sliding W-1=1023 rows. No persistent Kq in prefill. Allocator/copies/RoPE/Kq scratch separate.',
            freeze=dict(path='artifacts/apa_sp4g/jobs_a6/freeze_a6.json', sha256=sha(A/'jobs_a6/freeze_a6.json'))),
        rails=dict(worker_s=1500, outer_s=1560, lease_wait_s=20, cooldown_s=30,
            implementation='foreground cooperative deadlines, before/after load and each attention/block/chunk boundary; worker clock includes validation. No timeout utility/signals or retries. A hung native call cannot be forcibly bounded under no-kill.',
            on_oom='Stop that arm only. Remaining larger cells become create-only NON_FIT_AFTER_OOM receipts, no lease/model execution.',
            on_rail='RAIL is unknown memory fit, never OOM/non-fit. Record registered S squared wall extrapolation; no retry of that cell. Advance to next registered rung.',
            on_error='Unexpected errors are RED, never OOM. Stop dependent execution; no retries.',
            authority='David via amendment7 only; old runner rails unchanged'),
        predictions=dict(lead=[
            'Standard OOMs between 12K and 16K (16 heads x S squared bf16 scores on global layers = 8 GB at 16K on top of 6.8 GB weights).',
            'Two-pass and single-pass reach 32K with resident under 10 GB; wall is time, not memory, until at least 64K.',
            'Single-pass reaches at least one rung further than two-pass because it has no O(S) bulk/rank/recon transients.'
        ], seat=[
            'All three arms fit 16K; expect 24K to complete too. Standard is not expected to OOM at 16K: June adaptive queries reach a 64-row floor, so its score tensor there is 32 MiB, not 8 GiB.',
            'At 32K all arms would plausibly reside below 10 GiB if completed, but registered quadratic timing predicts RAIL there and above. KV alone is 512 MiB global plus 319.6875 MiB sliding at 32K. Memory may remain feasible at 64K; no measured fit claim.',
            'I do not predict a guaranteed extra rung for C. Both B and C use the same chunked reconstructed Kq, and B already streams attention without full bulk/rank score matrices. C may save kernel scratch but this does not remove shared Kq reconstruction.'
        ], evidence_class='pre-run predictions/reasoning; not measurements'),
        estimates=dict(formula='load_8192 + prefill_8192 * (S/8192)^2',
            caveat='Requested S squared assumption, not an empirically fitted exponent. 4096-to-8192 A prefill ratio is about 2.03, so extrapolations may be conservative. Fresh load and telemetry can differ.',
            C_anchor='clean decode_a6_C_8192 prefill component, frozen delta3; no ceiling_C_8192 receipt exists'),
        telemetry=dict(peak='NVML accounting maxMemoryUsage own PID (read only, when supported/enabled); else null with RED_PEAK_UNAVAILABLE',
            samples='NVML own-PID samples at model/layer/chunk boundaries; sampled maximum is a lower bound, never relabelled exact peak',
            pool='CUDA pool reserved/used high-water separately; never labelled process resident peak',
            failure='OOM/RAIL retains partial peak/samples, wall, theoretical full-S KV and completed-prefix count. Missing telemetry explicit, not zero.'),
        cpu_gates=dict(baseline='full test_apa_sp4g*.py suite, zero failures/skips',
            new=['exact 21 cells/kind/rail and ordering', 'anchor arithmetic and enough pinned tokens',
                 'only first OOM derives non-fits; RAIL advances and never retries', 'create-only receipts and tamper rejection',
                 'June prefill, frozen delta and tuple cache bytes', 'worker/outer boundaries and no-signal foreground shell',
                 'CUDA/NVML ABI, sampled versus exact peak', 'partial OOM/RAIL receipts and unexpected errors'],
            mutations=dict(threshold=.8, lanes=['rail_boundary','oom_classifier','rail_as_oom','fingerprint_ignored','kv_slope','prefill_shortened','delta_wrong','pool_off']),
            blind_review='lead owned, UNRUN; author CPU baseline is not blind verification'),
        june_context=dict(source='/mnt/ForgeRealm/GraftRepository/docs/GEMMA4_PORT_LEDGER.md',
            sha256=sha('/mnt/ForgeRealm/GraftRepository/docs/GEMMA4_PORT_LEDGER.md'),
            evidence_class='external local June port ledger, different 3070 8GB configuration',
            bf16='prefill ~10-11K solid; 12K ragged (7805 MiB survived one run, OOM in ladder); 16K OOM',
            qv='INT8 V: 12K solid at 7802 MiB; 16K OOM',
            terminology='June bf16 means KV/compute with the QAT INT4 resident body, not 12B bf16 weights fitting in 8GB. Order wording bf16 weights is retained as a lead wording discrepancy.'),
        prior_art='June Gemma/SP4G/SP3 (2026): adapter, chunks, pooled prefill, leases and provenance reused. BLASST/Yuan2025/26 running-max; ThriftAttention/Sharratt2026; FA2/Dao2023 online softmax; TurboQuant/Zandieh2025 quantizer inherited unchanged. NVIDIA NVML/CUDA12.6(2024) memory counters. Make/Feldman1979, SHA256/NIST2001, DeMillo/Lipton/Sayward1978 mutation tests. New work is ceiling harness/accounting only; no prior art known to me for a distinct new method; no novelty claim.',
        seat=dict(model='gpt-6-astra', effort='xhigh', evidence='logs/apa_sp4g_a7_r1.log header'),
        safety='No git, subagents, background jobs/waits, signals/kills, service/product/model edits. CPU-only dispatched seat per standing order; lead owns card.')
    path = A/'amendment_021_a7_ceiling_long.json'; publish(path,reg)
    with path.with_suffix('.json.sha256').open('x') as f:
        f.write(sha(path)+'  '+path.name+'\n')
    print(json.dumps(dict(registration=str(path),sha256=sha(path),cells=len(cs),preserved={k:len(v) for k,v in before.items()}),indent=2))


if __name__ == '__main__':
    main()
