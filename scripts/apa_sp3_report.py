#!/usr/bin/env python3
"""Render only provenance-valid receipts; pending data remains null/blocked.

Prior art: standard tables and evidence-class separation, House Rules reporting.
No numerical algorithm or scientific novelty claimed by this renderer.
"""
import json
from pathlib import Path
from apa_sp3_common import ART, ROOT, REG_SHA, Red, fingerprint, read, registration, require_pass, sha
from apa_sp3_gpu import cells


def value(x):
    return '—' if x is None else f'{x:.6g}' if isinstance(x,float) else str(x)


def receipts():
    out={}
    for p in sorted((ART/'jobs').glob('*.json')) if (ART/'jobs').exists() else []:
        j=read(p)
        if j.get('fingerprint')!=fingerprint():
            j=dict(j,status='STALE',result={})
        elif j.get('status')=='PASS':
            try:require_pass(j['job'])
            except Exception as e:j=dict(j,status='STALE',result={},error=str(e))
        out[j['job']]=j
    return out


def main():
    r=registration();jobs=receipts()
    def get(name):return jobs.get(name,{}).get('result',{})
    def status(name):return jobs.get(name,{}).get('status','BLOCKED / unrun')
    def arm(bits,a,S):
        if bits==4 and S==1024 and a in 'AB':
            return get('g0').get(a,{}),status('g0')
        name=f'ppl_b{bits}_{a}_{S}'
        return get(name),status(name)
    has_model=(status('g0')=='PASS' and status('ppl_b4_C_1024')=='PASS' and status('ppl_b4_D_1024')=='PASS')
    headline='# APA-SP3 — model comparison measured; remaining RED/blocked rows listed' if has_model else '# APA-SP3 — RED / model-quality result not established'
    text=[headline, '',
          'Evidence class: **model perplexity** for scored rows; **kernel sweep** for activation margins and decode timing.',
          '**G2/G3 rows establish nothing about model quality by themselves.**', '',
          f'Registration SHA256: `{REG_SHA}` (immutable).',
          'Model: openbmb/MiniCPM3-4B; INT4 affine groups of 128; BF16 compute; 62 layers, 40 heads, composite D=96 and zero-padded V=96.',
          'Execution seat: gpt-6-astra, reasoning effort xhigh, confirmed by logs/apa_sp3_r1.log. No GPU model execution by this seat.', '',
          '## G0 protocol and parity', '',
          'Published targets: A=20.065 and B=19.817, absolute tolerance 0.01, same process. These are historical targets, not reproduced values.',
          'Source: `/mnt/ForgeRealm/GraftRepository/docs/MiniCPM3-MLA_Results.md`.',
          'The named `/mnt/ForgeRealm/AI-AtlasForge/workspace/APA-Quant-Rust_LLM_testing/mission_b74b7906/test_graft_e1_mla.py` is a graft-router experiment, with no PPL/corpus prefix calculation.',
          'Dispatch finding: documented `/tmp/minicpm3_engine_bench.py`, `minicpm3_reference.py`, `minicpm3_ceiling.py` were absent. Exact original guide source, concatenation, prefix, special-token policy and scorer were not recovered. Token SHA is **null** in the immutable base registration; any later recovery must be pinned in a separate amendment.',
          'The later `GraftRepository/tests/minicpm3_apa_recovery.py` uses six WikiText-or-local-document windows in cached chunks and scores 511 targets; it does not establish the original protocol. No fallback corpus is substituted.',
          'Every model worker requires a separately pinned recovery amendment and rechecks A/B in that process before an SP arm. G0 miss stops PPL arms; D miss stops SP model arms. No tolerances change.', '',
          '## Perplexity and prefill table — model perplexity', '',
          '| Bits | S | Arm | Status | PPL last-512 | ms wall | Peak resident MiB* | Refined fraction | Layer min..max | δ |',
          '|---|---:|---|---|---:|---:|---:|---:|---|---:|']
    ppl=[]
    for bits in (4,8):
        for S in (1024,8192):
            for a in ('A','B','C','D','E') if bits==4 else ('A','B','C','D'):
                m,st=arm(bits,a,S);f=m.get('refinement',{})
                if a=='B' and not f:
                    f=get(f'capture_b{bits}_B_{S}').get('refinement',{})
                row=dict(bits=bits,S=S,arm=a,status=st,ppl=m.get('ppl'),wall_ms=m.get('wall_ms'),
                         peak_resident_mib=m.get('peak_resident_mib'),fraction=f.get('fraction'),delta=m.get('delta'))
                ppl.append(row)
                text.append('| '+' | '.join(map(value,[bits,S,a+' '+r['arms'][a],st,m.get('ppl'),m.get('wall_ms'),
                    m.get('peak_resident_mib'),f.get('fraction'),
                    f"{value(f.get('per_layer_min'))}..{value(f.get('per_layer_max'))}",m.get('delta')]))+' |')
    text += ['', '*Peak is explicitly an estimate: exact intercepted cudaMalloc allocation high-water plus the pre-call device/context offset. Internal driver transient allocations can be missed. Raw pooling is OFF. No background sampling thread. Diagnostic/capture timings never fill this table.',
             '8192 rows request full prefill plus last-512 logits in one forward. A deadline or OOM is RED, including if logits do not fit. No chunking or precision fallback. Bulk8 is optional secondary; no E8 is registered.', '',
             '## Empirical margin table — kernel sweep on model activations', '',
             '| Bits | Arm | S | Layer | Error mean | p99 | p99.9 | max | Unrefined mass mean | p99 | max | Max skipped w/w* | Fraction |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    margins=[]
    for c in cells():
        if c['kind']!='margin' or status(c['id'])!='PASS':continue
        m=get(c['id']);margins.append(m);e=m['error'];u=m['unrefined_mass']
        text.append('| '+' | '.join(map(value,[c['bits'],c['arm'],c['S'],c['layer'],e['mean'],e['p99'],e['p99_9'],e['max'],
                    u['mean'],u['p99'],u['max'],m['max_skipped_relative_weight'],m['fraction']]))+' |')
    if not margins:
        text.append('| 4 / 8 | B / C | 1024 / 8192 | 0–61 | BLOCKED | — | — | — | — | — | — | — | — |')
    bc=[m for m in margins if m['bits']==4 and m['arm'] in 'BC']
    text += ['', f'{len(bc)}/248 primary B/C layer receipts available. Every row requires all eligible heads, queries and keys; future masked keys are excluded. Bulk scores come from a native SP-order FP32 CUDA score probe; exact scores are float64 Q.K dots at the actual float32 scale (amendment_002_native_bulk.json). B blend error replays its exact native bulk chunks and requires the captured byte SHA; B/C also expose sp_error in JSON for the E margin (amendment_003_B_native_scores.json). Error percentiles are exact nearest ranks after float32 storage; mean/max use float64 errors. Actual selection masks come from native SP diagnostics or literal instrumented B copies, checked bit-identical against B output on the same inputs.', '']
    if bc:
        text.append(f'Observed maximum unrefined exact-softmax mass is {max(m["unrefined_mass"]["max"] for m in bc):.6g}; maximum skipped-key weight relative to the row maximum is {max(m["max_skipped_relative_weight"] for m in bc):.6g}. These quantify attention importance of keys kept at bulk precision. Coverage is limited to the completed rows above; they supply no distributional bound beyond those activations.')
    else:
        text.append('The heuristic tail cannot yet be judged: no real-activation error, unrefined-mass or skipped-relative-weight rows have run. A skipped key remains in the softmax at bulk precision; these measurements will quantify its exact-softmax importance, not mass removed from attention. Synthetic sweep outliers do not supply this model’s margin.')
    text.append('')
    cm,cs=arm(4,'C',1024);bm,bs=arm(4,'B',1024)
    if cs=='PASS' and bs=='PASS':
        text.append(f'C−B PPL is {cm["ppl"]-bm["ppl"]:+.6g} in the registered last-512 window. Interpret it only alongside C’s actual matched fraction. This one-window model-perplexity evidence does not establish cross-corpus or cross-length quality.')
    else:
        text.append('The model comparison is incomplete: reproduced original-token G0 parity and a scored C comparison at matched actual fraction are both required; see their individual statuses above. G2/G3 rows establish nothing about model quality by themselves. The ε=1e−3 margin is conditional on the measured finite error envelope; E’s separate capture checks its transfer, and neither check supplies a universal quantization bound or a CUDA rounding proof.')
    text += ['', '## Decode table — kernel sweep / in-model timing', '',
             '| Bits | Starting S | Arm | Status | tokens/s | Steps | Prefill seconds |',
             '|---|---:|---|---|---:|---:|---:|']
    decode=[]
    for c in cells():
        if c['kind']!='decode':continue
        m=get(c['id']);st=status(c['id'])
        row={**m,'bits':c['bits'],'S':c['S'],'arm':c['arm'],'status':st};decode.append(row)
        text.append('| '+' | '.join(map(value,[c['bits'],c['S'],c['arm'],st,m.get('tokens_s'),m.get('steps'),m.get('prefill_s')]))+' |')
    text += ['', 'Identical teacher-forced continuations, 32 measured steps, CUDA-synchronized wall time per token; expanded MLA for A/B/C, absorbed decode OFF. Cache prefill is excluded from tokens/s but included in the 480s worker ceiling. At starting S=32768, measured attention lengths are 32769–32800: explicitly beyond the trained window, with no quality claim.',
             '**G2/G3 rows establish nothing about model quality by themselves.**', '',
             '## Registry, predictions and gates', '',
             'C uses one global δ per bitwidth, matched over all layers at S=1024 to B’s actual diagnostic fraction ±0.01. Fixed grid on B activations initializes at the smallest tied δ; at most eight actual C fraction-only trials, bounded bisection on [0,32]. First match freezes δ for long prefill/decode; no PPL-based tuning. Report per-layer min/max/std.',
             'D uses finite float32 max δ, requires every eligible pair selected and |D−A|≤0.005 PPL. E uses upward_float32(ln(1000)+2·eq), eq the upward float32 global SP-arithmetic error maximum across B/C, both lengths, all layers. Missing layer blocks E. E’s own captures separately check the finite-envelope transfer.', '',
             '| Owner | ID | Registered prediction | Status |','|---|---|---|---|']
    for owner,key in [('Lead','lead_predictions'),('Seat','seat_predictions')]:
        for name,pred in r[key].items():text.append(f'| {owner} | {name} | {pred} | UNASSESSED; requires applicable model receipts |')
    cpu_path=ART/'CPU_GATES_FINAL.json' if (ART/'CPU_GATES_FINAL.json').exists() else ART/'CPU_GATES.json'
    cpu=read(cpu_path) if cpu_path.exists() else {}
    text += ['', 'CPU gates: '+json.dumps(cpu,sort_keys=True),
             'D=96 tests: `test_d96_prefill_contract_and_dense_pin`, `test_d96_splitk_contract_and_dense_pin`, and compiled flag/device/scalar guards for both geometries. Native GPU numerical pins are delivered in job `kernel96`, unrun here.',
             'Author tests and mutations are baseline evidence only. Independent blind verification under House Rules §8 is lead-owned and unrun; no subagents were launched.', '',
             '## Lead commands and bounds', '', '```bash',
             'cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3',
             'timeout 30s bash scripts/apa_sp3_lead_gpu.sh list',
             'timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run kernel96',
             '# Recover original protocol/token pins in a separate amendment before G0:',
             'timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run g0',
             '# Each resume invocation runs at most ONE cell; stops on RED/stale receipts:',
             'timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh resume 4',
             'timeout 30s bash scripts/apa_sp3_lead_gpu.sh summary','```',
             'Complete commands: `artifacts/apa_sp3/lead_commands.txt`; every cell and per-job estimate: `dry_run.json`. Optional secondary uses `resume 8` after primary prerequisites. No multi-hour automatic batch is launched.',
             'Each leased operation: flock --wait 20, worker timeout 480s plus 5s own-child termination grace, foreground 30s cooldown; outer guard TERM at 585s plus 3s grace. Device/PID inspection fails closed. Never signal a foreign PID. Operator keeps right of way; advisory-lock cooperation is required.',
             'Planning estimates, not timings: G0 35–150s; 1024 PPL 45–180s; 1024 capture/calibration 50–250s; 8192 prefill/capture 120–480s or RED timeout/OOM; per-layer G2 3–60s at 1024, 30–480s at 8192; decode including cache prefill 60–480s or RED. Actual upper-bound compliance is enforced, not predicted.',
             'Memory reasoning: a single BF16 40-head S² score tensor needs 5.0 GiB at 8192 and 80 GiB at 32768, before intermediates and ~2.9 GB model residency. Standard full prefill at 32768 cannot fit on 12 GB; its decode setup will record OOM. 8192 standard fit and B/SP full-prefill deadlines remain unverified. No alternate cache-building scheme is silently substituted.', '',
             '## Prior art', '',
             '| Work | What is reused / what SP3 adds |','|---|---|']
    links={'BLASST':'https://arxiv.org/abs/2512.12087','ThriftAttention':'https://arxiv.org/abs/2605.23081',
           'FlashAttention-2':'https://arxiv.org/abs/2307.08691','TurboQuant':'https://arxiv.org/abs/2504.19874'}
    for name,desc in r['prior_art'].items():
        label=f'[{name}]({links[name]})' if name in links else name
        text.append(f'| {label} | {desc} |')
    text += ['', 'The APA draft is David Perry (2026), `docs/APA_PAPER_DRAFT.md`; treated as local prior art, not independently verified theorem. Arm A uses standard scaled dot-product attention (Vaswani et al. 2017, https://arxiv.org/abs/1706.03762), not an external FA2 package. Standard NLL (Shannon 1948), nearest-rank order statistics, directed rounding, memory maps, ELF allocation interposition, content hashes and leases are not new algorithms. Mutation-testing historical attribution is an unverified lead: DeMillo/Lipton/Sayward 1978, search “Hints on Test Data Selection”. Code sites and ledger contain the same provenance distinctions.',
             'Bulk4/8 here means a TurboQuant codebook reconstructed to BF16 Kq. Existing SP bulk dots execute floating-point instructions; this experiment cannot establish packed FP4 speed or compressed KV residency. No source/kernels outside the authorized new harness/test paths were edited.', '',
             '## Deviations, RED and residuals', '',
             '- Original token SHA and exact guide protocol unavailable: registration honestly records null; no PPL reproduction claimed. Separate amendment is required before model gates. This is a delivered block, not a waived gate.',
             '- Missing local SP2/SPD1 artifacts were read from sibling worktrees, read-only. SP1/SP1.1 registration hashes are corroborated by the local ledger and SP2 parent registration; their original JSONs are absent here.',
             '- Native GPU behavior, diagnostic bit parity, BF16 D≈A, long-context fit/time and actual model quality remain untested. Kernel-body hash equality proves source preservation, not GPU correctness.',
             '- Resident peak is an explicitly qualified estimate. Cold/warm effects and single-call prefill variability remain; decode measures 32 steps and includes expansion/quantization overhead common to arms.',
             '- Full long-context captures use tens of GB of disk and native diagnostic masks add quadratic transient memory. Disk/OOM/time failures remain RED; no sampling reduction.',
             '- No git, subagents, shell background jobs, live-service changes or foreign-process termination. Only explicitly bounded own child processes may be terminated by timeout. Host build/tests complete; lead GPU and blind verification pending.', '',
             'Files: `scripts/apa_sp3_*`, `tensor_cuda/tests/test_apa_sp3.py`, `docs/APA_SP3_LEDGER.md`, and `artifacts/apa_sp3/` (registration, source pins, build, CPU/mutation receipts, exact commands, blocked JSON and this renderer output).']
    (ART/'RESULTS.md').write_text('\n'.join(text)+'\n')
    (ART/'results.json').write_text(json.dumps(dict(status='PARTIAL_MODEL_RESULTS' if has_model else 'RED_UNTIL_REQUIRED_GATES_PASS',registration_sha256=REG_SHA,
           ppl=ppl,margins=margins,decode=decode,jobs=jobs,
           scope='G2/G3 rows establish nothing about model quality by themselves'),indent=2)+'\n')
    (ART/'margins.json').write_text(json.dumps(dict(evidence_class='kernel sweep',primary_completed=len(bc),
         primary_required=248,rows=margins,missing_are_not_zero=True),indent=2)+'\n')
    dry=dict(status='PLANNED_NOT_RUN',registration_sha256=REG_SHA,protocol=r['protocol'],cells=cells())
    (ART/'dry_run.json').write_text(json.dumps(dry,indent=2)+'\n')
    commands=['# One foreground invocation per cell. STOP at each RED. No automatic batch.',
              '# Recover and pin original protocol before model jobs. Never replace registration.',
              'cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3']
    for c in cells():
        commands.append(f"# {c['id']}: estimate {c['estimate_s']} seconds; {'optional bulk8' if c['optional_secondary'] else 'primary'}")
        commands.append(f"timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run {c['id']}")
    (ART/'lead_commands.txt').write_text('\n'.join(commands)+'\n')
    print(json.dumps({'report':str(ART/'RESULTS.md'),'gpu_receipts':len(jobs),'primary_G2_rows':len(bc),'cells':len(cells())}))


if __name__=='__main__':main()
