"""Prior art: SP4G (2026) receipt tables and dependency commands. A7
reports measured, inferred and unavailable values distinctly; no new statistic.
"""
from apa_sp4g_a7_common import *


def states():
    from apa_sp4g_a7_gpu import preflight, previous_receipt
    rows = []
    for c in cells():
        result = {}; error = None
        try:
            if path_a7(c['id']).exists():
                j = require_a7(c['id']); result = dict(j['result'],worker_wall_s=j['worker_wall_s'])
                state = result['outcome']
            else:
                previous_receipt(c)
                state = 'READY_'+preflight(c)
        except (OSError, Red) as e:
            error = str(e); state = 'RED' if path_a7(c['id']).exists() else 'BLOCKED'
        rows.append(dict(cell=c,state=state,result=result,error=error,
                         command=f"bash scripts/apa_sp4g_a7_lead_gpu.sh run {c['id']}"))
    return rows


def show(value, digits=2):
    return 'UNAVAILABLE' if value is None else f'{value:.{digits}f}'


def main():
    preserved(); reg = registered(); rows = states()
    visibility = read(A/'a7_device_visibility.json') if (A/'a7_device_visibility.json').exists() else {'status':'UNPROBED'}
    gates = read(A/'CPU_GATES_A7.json') if (A/'CPU_GATES_A7.json').exists() else {}
    done = sum(bool(r['result']) for r in rows)
    incomplete = done != 21
    peak_red = any(r['result'].get('peak_status') == 'RED_PEAK_UNAVAILABLE' for r in rows)
    rail = any(r['state'] == 'RAIL' for r in rows)
    verdict = 'RED — ceiling unmeasured/pending' if incomplete else (
        'RED residuals — time-censored or missing exact resident peaks' if peak_red or rail else 'Completed registered ceiling grid')
    out = ['# APA-SP4G amendment 7 — long ceiling', '', '**'+verdict+'.** '+
           f'{done}/21 new cells have validated terminal receipts. CPU-only dispatched seat; lead runs the GPU. '+
           'Old receipts and rails are unchanged. G2/G3 rows establish nothing about model quality by themselves.', '',
           '## Cells, rails and estimates', '',
           'Each cell has a 1,500 s cooperative worker rail and a 1,560 s outer budget, including a ≤20 s foreground flock wait and 30 s cooldown. '+
           'Run each command separately, A ascending first, then B, then C. `resume` executes exactly one cell. '+
           'After the first CUDA OOM in an arm, larger registered cells write inferred NON_FIT_AFTER_OOM receipts without a lease or model load. '+
           'RAIL means unknown fit; it does not stop later rungs or authorize a retry. Unexpected errors block descendants.', '',
           '`estimate = load_8192 + prefill_8192 × (S/8192)^2`. This is the requested quadratic extrapolation anchored to measured times, '+
           'not a measured S² exponent or new timing result. The original A 4096→8192 prefill ratio is about 2.03. '+
           'C uses the prefill component of the clean 8192 decode receipt (delta=3); scoring, captures and decode are excluded. '+
           'Estimates include one fresh load (~75–76 s measured); outer planning adds 50 s. Instrumentation adds unmeasured overhead.', '',
           '| Arm | 8192 anchor | Load s | Prefill s |', '|---|---|---:|---:|']
    for arm,a in reg['anchors'].items():
        out.append(f"| {arm} | [{a['path'].split('/')[-1]}]({a['path'].replace('artifacts/apa_sp4g/','')}) | {a['load_s']:.6f} | {a['prefill_s']:.6f} |")
    out += ['', '| Cell | Extrapolated worker s | Prefill s | KV MiB (global + fixed sliding) | State |',
            '|---|---:|---:|---:|---|']
    for row in rows:
        c = row['cell']; kv = kv_size(c['S'])
        out.append(f"| `{c['id']}` | {c['estimate_worker_s']:.2f} | {c['estimate_prefill_s']:.2f} | {kv['global_bytes']/(1<<20):.0f} + {kv['sliding_fixed_bytes']/(1<<20):.4f} | {row['state']} |")
    out += ['', '## Both prediction sets (registered before CPU gates)', '']
    for who in ('lead','seat'):
        out += [f'**{who.capitalize()} predictions:**', '']+[f'{i}. {p}' for i,p in enumerate(reg['predictions'][who],1)]+['']
    out += ['These are predictions/reasoning. The adapter uses `PREFILL_CHUNK=512` and an adaptive 64-row floor. '+
            'At 16K, the late standard score tensor is 16×64×16384×2 = 32 MiB; a single-shot 8 GiB score tensor is not allocated by this protocol. '+
            'The B fused path already streams attention; B and C both construct Kq through the same June chunked quantizer. '+
            'Source: `/mnt/ForgeRealm/GraftRepository/core/gemma4_tc.py:689–712,724–742,843–889`.', '',
            '## Ceiling measurements — Gemma-4-12B-it QAT q4_0, bf16 KV/compute, global layers only', '',
            'Prefill-only from the pinned prefix; no scoring/decode/capture arrays; `Model.prefill` invokes the same June model entry used by PPL context feeding. '+
            'Pooling ON before load; B bulk4/r=0.15; C frozen delta=3.0; `apa_min_context=0`; KV quantization OFF. '+
            'Logical global cache is 8 layers × K/V × 1 head × 512 × 2 bytes = 16,384 bytes/token. '+
            'Sliding tuple caches retain 1023 rows = 335,216,640 bytes (319.6875 MiB), fixed for this grid; capacity upper bound 320 MiB. '+
            'Kq, cache copies, RoPE, weights and allocator retention are separate from logical KV payload.', '',
            '| Cell | Outcome | Worker wall s | Peak resident MiB | Sampled peak lower bound MiB | Pool reserved high MiB | Completed tokens |',
            '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        r=row['result'];c=row['cell']
        link=f"[{c['id']}](jobs_a7/{c['id']}.json)" if r else f"`{c['id']}`"
        out.append(f"| {link} | {row['state']} | {show(r.get('worker_wall_s'))} | {show(r.get('peak_resident_mib'))} | {show(r.get('sampled_peak_resident_mib'))} | {show(r.get('pool_reserved_high_mib'))} | {r.get('completed_tokens','UNRUN')} |")
    out += ['', 'An exact resident peak is reported only from NVML own-PID accounting `maxMemoryUsage`, when already supported/enabled. '+
            'If unavailable, its value is null and `peak_status=RED_PEAK_UNAVAILABLE`; synchronous boundary samples are labelled a lower bound. '+
            'No accounting mode/device settings are changed. CUDA pool used/reserved high-water counters cover the pool only. '+
            'Subtracting the logical KV payload from a process peak still includes weights, scratch, allocator retention and CUDA overhead; it is not pure transient usage. '+
            'OOM/RAIL receipts retain partial measurements and full-S theoretical KV; partial KV is not relabelled a completed S cache.', '',
            '## June 8 GB RTX 3070 context', '',
            '| June path | Prefill evidence |', '|---|---|',
            '| bf16 KV/compute, QAT INT4 body | ~10–11K solid; 12K ragged (7805 MiB in one run, OOM in ladder); 16K OOM |',
            '| qv / INT8 V | 12K solid at 7802 MiB; 16K OOM |', '',
            'Evidence class: external local June port ledger, `/mnt/ForgeRealm/GraftRepository/docs/GEMMA4_PORT_LEDGER.md:49–55`, '+
            'SHA256 `'+reg['june_context']['sha256']+'`. Different GPU/configuration; not an A7 measurement. '+
            'Order wording “bf16 weights” is a terminology discrepancy: the June resident body was QAT INT4; full 12B bf16 weights do not fit in 8 GB.', '',
            '## Fingerprints, CPU gates and RED', '',
            f'Registration021 SHA256 `{REGISTRATION_SHA}`. Original registration SHA256 `{REG_SHA}`.']
    if SEAL.exists():
        out += [f'Fingerprint022 SHA256 `{sha(SEAL)}`.']
    if gates:
        out += [f"Author CPU suite: **{gates['passed']} passed, {gates['failed']} failed, {gates['skipped']} skipped**; "+
                f"mutation kills {gates['mutations']['killed']}/{gates['mutations']['nonerror']}, threshold0.80. "+
                'CPU doubles check harness semantics, not GPU numerics. Blind verification is lead owned and UNRUN.']
    out += ['', f"Current device probe: `{json.dumps(visibility,sort_keys=True)}`.", '',
            'Exact commands and blocking dependencies: `lead_commands.txt`, `GPU_BLOCKED_A7.json`. '+
            'Not claimed fixed: unmeasured memory ceiling, time censoring, historical exactness RED, or unavailable exact resident peak on devices without accounting. '+
            'Archived A6 results below remain historical; amendment7 makes no quality claim.', '',
            '## Process safety and seat', '',
            'No git, subagents, background jobs/waits, process signals/kills, service edits, product/kernel edits or model writes. '+
            'One foreground cell per flock lease. CUDA OOM is distinguished from host MemoryError, worker exit137/124 and arbitrary errors. '+
            'Deadline checks are cooperative, including before/after load and at block/chunk boundaries. A hung native operation cannot be forcibly bounded under no-kill. '+
            'The outer elapsed receipt includes cooldown and flags any overrun; this is not a hard no-kill wall-time guarantee.', '',
            'Seat: **gpt-6-astra / reasoning xhigh**, live header `logs/apa_sp4g_a7_r1.log`. '+
            'Model under test: **Gemma-4-12B-it QAT q4_0 exact (symmetric-8 g32)**. '+
            'Worktree/head is the lead-provided `apa-sp4g` at `29882ae`; no git command used to verify it.', '',
            '## Prior art', '',
            'June Gemma port/floor and SP3/SP4G (2026) supply adaptive chunks, pooling, attention dispatch and foreground receipts. '+
            'A7 adds long-ceiling registration, dimensional KV reporting and scoped telemetry; no new attention algorithm. '+
            '[BLASST, Yuan et al. (2025/2026)](https://arxiv.org/abs/2512.12087) supplies inherited running-max softmax selection; '+
            '[FlashAttention-2, Dao (2023)](https://arxiv.org/abs/2307.08691) supplies inherited online-softmax implementation context. '+
            'Primary abstracts checked this seat. ThriftAttention/Sharratt (2026), weight-sensitive precision, and TurboQuant/Zandieh (2025), '+
            'key quantization, are inherited unchanged; unverified — lead to check arXiv2605.23081 and2504.19874. '+
            '[NVIDIA NVML accounting](https://docs.nvidia.com/deploy/nvml-api/structnvmlAccountingStats__t.html) '+
            'and local CUDA12.6/NVML headers (2024) supply process accounting and pool-counter ABIs. No new profiler algorithm.', '',
            'Make/Feldman (1979) dependencies, SHA256/NIST (2001) provenance, classical dimensional analysis and quadratic extrapolation, '+
            'DeMillo/Lipton/Sayward (1978) mutation tests are reused. Historical citations unverified this seat — lead to check '+
            'Make a program for maintaining computer programs; FIPS180; Hints on Test Data Selection. '+
            'No prior art known to me for a distinct new method introduced here; no novelty claim.', '']
    text='\n'.join(out)
    (A/'A7_REPORT.md').write_text(text)
    (A/'RESULTS.md').write_text(text+'\n---\n\n## Archived A6 report (unchanged snapshot)\n\n'+(A/'a7_baseline/RESULTS.md').read_text())
    commands=['# A7 only: 1500s worker / 1560s outer, one foreground invocation per cell.',
              '# A ascending, then B, then C. OOM => later same-arm commands write inferred non-fits on CPU.',
              '# RAIL => unknown fit; proceed to next registered rung, never retry. No batches or background waits.']
    for row in rows:
        c=row['cell'];commands += [f"# {row['state']}; worker extrapolation {c['estimate_worker_s']:.2f}s, outer extrapolation {c['estimate_outer_s']:.2f}s; capped by rails",row['command']]
    commands += ['bash scripts/apa_sp4g_a7_lead_gpu.sh summary']
    (A/'lead_commands.txt').write_text('\n'.join(commands)+'\n')
    (A/'cells.json').write_text(json.dumps(cells(),indent=2)+'\n')
    (A/'GPU_BLOCKED_A7.json').write_text(json.dumps(dict(status=verdict,device_visibility=visibility,
        a7_registration_sha256=REGISTRATION_SHA,cells=rows),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=verdict,terminal_receipts=done,cells=len(rows)),indent=2))


if __name__=='__main__':
    token=VALIDATION.set({})
    try:
        main()
    finally:
        VALIDATION.reset(token)
