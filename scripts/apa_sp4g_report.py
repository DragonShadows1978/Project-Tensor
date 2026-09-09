"""Evidence-scoped synthesis/lead commands. Prior art: SP3 (2026) receipt
rendering and dependency-ordered commands; no inference from missing receipts.
No new algorithm: ordinary tabulation and registered comparisons only.
"""
import collections,json,math
from apa_sp4g_common import *
from apa_sp4g_registry import cells,fallback_cells,LAYERS

def render():
    r=registration();cs=cells();cache={};jobs={};audit=[]
    for c in cs:
        p=job_path(c['id'])
        if not p.exists():continue
        try:j=require_pass(c['id'],cache);jobs[c['id']]=j
        except Exception as e:audit.append(dict(cell=c['id'],error=str(e)))
    def result(name):return jobs.get(name,{}).get('result',{})
    def status(name):
        if name in jobs:return result(name).get('outcome','PASS')
        return 'RED / STALE' if job_path(name).exists() else 'UNRUN'
    def fmt(x):return '—' if x is None else f'{x:.6g}' if isinstance(x,float) else str(x)
    def row(*x):return '| '+' | '.join(fmt(v) for v in x)+' |'
    cpu=read(A/'CPU_GATES_A1.json') if (A/'CPU_GATES_A1.json').exists() else {'status':'PENDING'}
    lines=['# APA-SP4G results', '',
      f'**CPU {cpu["status"]}; GPU {"receipts present, inspect tables" if jobs or audit else "BLOCKED: no device in this seat; zero GPU cells executed"}.** No Gemma PPL, margin, ceiling or decode result is claimed without a valid current receipt.',
      '',f'Registration SHA: `{REG_SHA}`. Model reference A is the engine\'s QAT INT4 Gemma-4-12B-it (exact q4_0 symmetric-8 group32 import, BF16 compute). A 12B BF16 weight set is roughly 24GB and does not fit the 12GB card; no torch or full-BF16 model reference was loaded. This tests attention changes inside this engine; it cannot establish parity to a BF16 model.',
      '', '## Amendment 1 status', '', 'Original card evidence: kernel512 PASS; ppl_A_2048_w0 RED with `AttributeError: /usr/local/cuda-12.6/lib64/libcudart.so.12: undefined symbol: cudaGetDeviceDefaultMemPool`. Both original receipts remain byte-identical in jobs/. The CUDA pool typo is repaired and all four APIs resolve on model-module import before model loading. New executions write jobs_a1/. Not claimed fixed: model PPL execution on the card has not been rerun in this seat.', '', 'Kernel512 remains valid through amendment_003_a1_fingerprint.json: raw common/registry/gpu closure files changed, so byte-identical closure is NOT claimed. The exact reviewed bridge pins before/after hashes; kernel512 function bytes, kernel dispatch, runtime loader, compiled build and numerical inputs remain unchanged. Unknown transitions fail closed; no RED receipt is eligible.', '', '## Adapter facts and PROTOCOL-G', '',
      'Source inspection: `gemma4_tc.py:568/658` gates APA by strict `S_all > apa_min_context`; EVERY B/C/D/E cell sets `apa_min_context=0` and `fast_max_seq=0`. Thus the short windows actually exercise fused APA. A stays standard. B is reconstructed-key `apa_selective_attention` at lines 609/712, bulk 4/r=.15, with INT4/GEMM opt-ins off. C/D/E dispatch from that SAME binding with Q[B,16,L,512] and K/Kq/V[B,1,S,512], BF16, scale=1. No V padding or head expansion difference between arms. Hidden activations can differ as attention changes.',
      '', 'Both SP launchers already instantiate 512 (kernels.cu:7496, apa_sp1_1.cuh:179) and map 16 query heads to KV=1. Tests: `test_d512_mqa_prefill_contract_and_dense_pin`, `test_d512_mqa_splitk_contract_and_dense_pin`, `test_compiled_d512_mqa_flag_device_scalar_guards`; native numerical receipt `jobs/kernel512.json` PASSED on the lead card. No production source or kernel-body edits.',
      '', '`KVRing.quantized_keys` at gemma4_tc.py:353-388 quantizes only `[kq_count:count)`, in 512-row chunks at cold start and ONE new row thereafter. CPU behavior pin: `test_live_gemma_kq_cache_only_quantizes_new_rows`. This path retains the June incremental fix. Prefill still quantizes whole prefixes (in 2048-row chunks above 4096); that is distinct from decode. Gemma K/V shared projection does not imply equal stored tensors: K is normalized/roped and V uses scale-free normalization without RoPE.',
      '', f'Offline wikitext-2-raw-v1 test: {r["protocol"]["rows"]} Arrow rows, newline join exactly as June floor, no fallback. Default -it tokenizer, no chat template or per-window BOS. **{r["protocol"]["token_count"]} tokens**, canonical little-endian int64 SHA `{r["protocol"]["token_sha256"]}`; `.npy` SHA `{r["protocol"]["tokens_file_sha256"]}`. `tokens.npy` and all tokenizer/corpus inputs are pinned in registration.',
      '', 'One feeding scheme for all arms: fresh caches per window; prefix of S−scored−1 tokens through June adaptive PREFILL_CHUNK=512, then 64-query cached blocks. Logits at positions S−scored−1 through S−2 predict targets S−scored through S−1. Short: four consecutive 2048-token windows at offsets 0/2048/4096/6144, exactly 1024 targets each (4096 total). Long: prefix 0, last 512 targets within input at 8192/16384/32768. FP64 log-softmax; aggregate exp(total NLL / total targets), never average PPL. June floor starts one query later and actually scores 1023; the mandated1024/fp64 correction is pre-registered.',
      '', 'C calibration is SP3-style prefix 0 at S2048, ALL eligible global-layer pairs, separate from the four-window PPL population. First of at most 12 native-C trials within ±0.01 of B freezes ONE delta for every layer, scored window and length. PPL fractions are reported independently; no hidden rematching. E uses epsilon=.01 and upward-rounded log(100)+2eq, where eq is the maximum actual SP-bulk versus FP64 exact real-key error over B/C and S2048/8192. This is a finite calibration maximum and a conditional bound, not a theorem about unseen keys.',
      '', '## Model perplexity — only the eight global layers change; forty sliding layers see at most 1024 valid keys and never APA', '',
      'A is the reference. Differences originate in the global-layer attention changes and propagate through subsequent layers.', '',
      '| Input S | Arm | Status | Targets | PPL | PPL minus A | Global refine fraction |',
      '|---:|---|---|---:|---:|---:|---:|']
    for S in (2048,8192,16384,32768):
        ar=result(f'ppl_A_{S}');av=ar.get('ppl')
        for arm in 'ABCDE':
            name=f'ppl_{arm}_{S}';j=result(name);p=j.get('ppl')
            lines.append(row(S,arm,status(name),j.get('targets'),p,p-av if p is not None and av is not None else None,j.get('global_fraction',{}).get('fraction')))
    lines+=['','Short in-model D/A gate: '+status('exactness_2048')+'. Tolerance remains 0.005; RED blocks C calibration. Long D/A differences remain independently visible above.','',
      '## G2 global margins — kernel sweep on real model activations', '',
      'All causal pairs, no sampling. One whole-layer cell per (S, arm, layer); internal 128-query tiles preserve absolute causal alignment and must replay the captured output bitwise. Error percentiles are nearest-rank over the entire layer population; no averaging percentiles. Unrefined mass is the mean per-query exact softmax mass on skipped keys; relative weight is max exp(exact_skipped − row_max_exact). These are different quantities.', '',
      '| S | Arm | Layer | Status | Mean error | p99 | p99.9 | Max error | Mean unrefined mass | Max skipped relative weight | Fraction |',
      '|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for S in (2048,8192):
        for arm in 'BC':
            for l in LAYERS:
                name=f'margin_{arm}_{S}_l{l:02d}';j=result(name);e=j.get('error',{})
                lines.append(row(S,arm,l,status(name),e.get('mean'),e.get('p99'),e.get('p99_9'),e.get('max'),j.get('unrefined_mass_mean'),j.get('max_skipped_relative_weight'),j.get('fraction')))
    lines+=['','E calibration: '+json.dumps(result('eq')) if result('eq') else '\nE calibration: UNRUN.', '',
      '## G3 ceilings and clean decode — kernel sweep / in-model timing', '',
      'EVERY arm retains June adaptive prefill from the same pinned prefix. FIT requires completed synchronized prefill; explicit allocation errors are OOM; a timeout is RAIL with unknown fit. Resident is an own-PID snapshot after load/prefill or at a caught failure, NOT a peak. Pool high water is a separate counter, NOT whole-device residency. A single successful grid point is not an extrapolated ceiling.', '',
      '| Arm | S | Status/outcome | Fit | Resident after/failure MiB | Pool reserved high MiB |', '|---|---:|---|---|---:|---:|']
    for arm in 'ABCDE':
        for S in (4096,8192,16384,24576,32768):
            name=f'ceiling_{arm}_{S}';j=result(name)
            lines.append(row(arm,S,status(name),j.get('fit'),j.get('resident_after_mib',j.get('resident_failure_mib')),j.get('pool_reserved_high_mib')))
    lines+=['','Clean decode: June fused GEMV/RMSNorm/softmax, pool ON BEFORE QAT load; no attention-class wrapper, no interposer. C installs only the required native SP binding dispatch. Greedy 32 synchronized steps; sole per-step tensor host copy is one device argmax int64. First step included; no throwaway warmup because Gemma mutates its KVRing. At 32K require same-arm 8192 estimate setup+16*prefill+4*decode+15 <285s; planned rail is unknown fit, not OOM.', '',
      '| Arm | Starting S | Status/outcome | Steps | ms/token including argmax | Resident after MiB |', '|---|---:|---|---:|---:|---:|']
    for arm in 'ABC':
        for S in (2048,8192,32768):
            name=f'decode_{arm}_{S}';j=result(name);lines.append(row(arm,S,status(name),j.get('steps'),j.get('ms_token'),j.get('resident_after_mib')))
    lines+=['','**G2/G3 rows establish nothing about model quality by themselves.**','', '## Registered predictions', '', '| Lead | Prediction | Seat prediction registered alongside |', '|---|---|---|']
    for i in range(1,6):lines.append(row(f'P{i}',r['predictions_lead'][f'P{i}'],r['predictions_seat'][f'S{i}']))
    assessments={}
    assessments['P1']={S:result(f'exactness_{S}') or status(f'exactness_{S}') for S in (2048,8192,16384,32768)}
    b=result('ppl_B_2048').get('ppl');c=result('ppl_C_2048').get('ppl')
    assessments['P2']=dict(passes=c<=b+.02 and b-c<.118,B_minus_C=b-c) if b is not None and c is not None else 'UNASSESSED'
    masses=[(arm,result(f'margin_{arm}_{S}_l{l:02d}').get('unrefined_mass_mean')) for S in (2048,8192) for arm in 'BC' for l in LAYERS]
    assessments['P3']=all(m>=.8 if arm=='B' else m<=.3 for arm,m in masses) if all(m is not None for arm,m in masses) else 'UNASSESSED'
    eq=result('eq').get('eq');ef=result('ppl_E_2048').get('global_fraction',{}).get('fraction')
    assessments['P4']=eq>=.5 and ef>=.95 if eq is not None and ef is not None else 'UNASSESSED'
    ceiling={arm:[(S,result(f'ceiling_{arm}_{S}').get('fit')) for S in (4096,8192,16384,24576,32768)] for arm in 'ABC'}
    if all(v is not None for values in ceiling.values() for S,v in values):
        tops={arm:max([S for S,fit in values if fit],default=0) for arm,values in ceiling.items()}
        assessments['P5']=dict(passes=tops['C']>=tops['B']>=tops['A'] and dict(ceiling['A'])[16384] is False,grid_max_fit=tops)
    else:assessments['P5']='UNASSESSED; incomplete grid or rail is not an OOM'
    lines+=['','Assessments: `'+json.dumps(assessments)+'`. Predictions are not measurements or pass criteria except the explicitly registered D/A exactness gate.', '',
      '## CPU gates, lead commands, and wall estimates', '',
      '`'+json.dumps({k:v for k,v in cpu.items() if k!='execution_sha256'})+'` (complete source pins: `CPU_GATES_A1.json`).', '',
      'Full cell manifest: `cells.json`; exact dependency-ordered commands and PER-JOB estimates: `lead_commands.txt`. Each resume runs at most ONE cell, stops on RED/stale evidence, and never overwrites a receipt. Independent cells can be selected by explicit run commands after a different cell hits a rail. No automatic retries or unbounded batch.', '',
      '```bash','cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g','bash scripts/apa_sp4g_lead_gpu.sh list','bash scripts/apa_sp4g_lead_gpu.sh run kernel512','bash scripts/apa_sp4g_lead_gpu.sh run ppl_A_2048_w0','bash scripts/apa_sp4g_lead_gpu.sh resume','bash scripts/apa_sp4g_lead_gpu.sh summary','```', '',
      'QAT 12B load estimate ALONE: 30–120s, unmeasured, including GGUF read/repack/upload and per-layer garbage collection. Each model job estimate 60–285s including load; long runs may exceed the rail and become RED. Kernel 2–45s; whole-layer margins at2048:10–90s, at8192:60–270s, including replay, CPU exact dots and full-population percentiles, with no model load. Estimates are unmeasured. Explicit fallback bands5–120s and fallback aggregation1–120s. Worker TERM 285s plus5s owned-child grace =290s hard maximum. Outer TERM 585s plus3s grace =588s, under 590. Lease wait 20s; foreground 30s cooldown while retaining lock after GPU jobs. Only owned timeout children may be signalled; no process discovery-to-kill behavior.', '',
      f'There are {len(cs)} default cells, including32 whole-layer G2 cells (16 per length). S8192 cooldown falls from1024 to16 jobs:512 minutes to8 minutes; both lengths together960s=16 minutes. ONE KV head is shared by16 query heads: S8192 has536,936,448 causal query-head/key pairs per layer, and S2048 has33,570,816. All are included. Internal replay tiles remain128 queries; full retained float64 errors require about73GB plus captures. A whole-layer job hitting the unchanged rail stays RED. Only then may the lead explicitly run that layer’s margin_band and margin_summary cells listed in lead_commands_fallback.txt; default resume never dispatches fallback cells. A valid completed fallback summary can satisfy the layer dependency while retaining the original RED. Per-cell12GiB disk headroom covers retained errors plus population scratch.', '',
      '## Prior art', '', 'Amendment1 introduces no new algorithm or prior art. Reuses SP3 a4(2026) exact reviewed fingerprint transitions, existing SP4G/June(2026) replay and population statistics, and NVIDIA CUDA Runtime12.6(2024) ABI verified against installed cuda_runtime_api.h/driver_types.h. New work is symbol spelling, import-time resolution and dispatch coalescing only.', '']
    for title,detail in r['prior_art'].items():lines.append(f'- **{title}:** {detail}.')
    lines+=['',r['literature_status']+'. Code comments and ledger state what is reused. SP4G adds experiment wiring, not a new attention algorithm. Bulk4 is reconstructed BF16 Kq computed by existing floating-point dots; this does not demonstrate packed FP4 arithmetic or compressed KV residency.', '',
      '## Deviations, RED, residual risks, and process safety', '',
      '- No GPU was available in this seat. P1–P5, real-model diagnostic replay/exactness/quality, observed fractions, memory fit, and throughput remain unmeasured until valid lead receipts exist; the original D512/MQA kernel pin passed. Host compilation is not CUDA numerical validation.',
      '- June off-by-one correction/fp64 and forced fused threshold 0 are registered. The floor default fast_max_seq=4096 would use blend at short S; the order explicitly requests fused B. INT4/GEMM opt-ins and K/V-storage quantization are off for tensor parity. Adaptive prefill can change the lead ceiling prediction; no full dense-prefill result is implied.',
      '- E\'s eq is a finite maximum over B/C activations, not a universal bound on E-induced or unseen activations/longer sequences. Transfer is a measured hypothesis. C\'s matched fraction is the prefix calibration population, not a guarantee of ±0.01 on all scored windows or decode partition-local maxima.',
      '- Diagnostic masks and captures add overhead only to diagnostic/PPL cells. Clean decode has no diagnostics. Full clean numerical output parity at real Gemma shapes and long rails is still untested.',
      '- `Not claimed fixed`: the full model cell has not yet been rerun on the card. The reported symbol failure was reproduced and repaired at CPU import/symbol resolution scope. No production source changed. Author tests/mutations are baseline evidence; House Rules blind verification is lead-owned and UNRUN. Bulk8 secondary is deferred; no bulk8 claims.',
      '- Receipts are atomic create-only, dependency/fingerprint checked per kind. Array creation hashes plus exact stat identity protect large artifacts on reuse; metadata JSON is rehashed. Unknown source transitions invalidate reuse and require separate amendments. Timeout/OOM evidence is preserved, never promoted to a successful model-quality result.',
      '- No git commands, subagents, shell background jobs, live-service changes, or foreign-process signals. All execution was local preparation, host build and CPU verification. Read-only SP3 modules/receipts, Graft adapter/docs, and models were preserved.',
      '- Amendment1 seat: gpt-6-astra, reasoning xhigh, confirmed by logs/apa_sp4g_a1_r1.log. Original registration identity fields remain immutable.',
      '', 'Receipt audit failures: `'+json.dumps(audit)+'`.', '',
      'Sources of truth: registration.json, cells.json, CPU_GATES_A1.json, GPU_BLOCKED_A1.json, amendment_002_a1_execution.json, amendment_003_a1_fingerprint.json, build/manifest.json, jobs/*.json and jobs_a1/*.json, logs/apa_sp4g_*, and docs/APA_SP4G_LEDGER.md.']
    (A/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (A/'receipt_audit.json').write_text(json.dumps(dict(valid=len(jobs),red_or_stale=audit,unrun=len(cs)-len(jobs)-len(audit)),indent=2)+'\n')
    command=['# APA-SP4G exact commands, dependency order. No batch is automatically started.', '# Model load ALONE: 30-120s unmeasured. Worker hard max 290s; outer 588s; GPU cooldown 30s plus lease up to 20s.', '# CPU aggregate cells have no GPU lease/cooldown. Original kernel512 is reused through the exact A1 bridge; original model RED stays in jobs/. New jobs go to jobs_a1/.', '# Whole-layer margins:2048 10-90s;8192 60-270s, no model load. Fallback ONLY after corresponding whole-layer RED RAIL; see lead_commands_fallback.txt.', 'cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g','bash scripts/apa_sp4g_lead_gpu.sh list']
    from apa_sp4g_gpu import CPU_KINDS
    for c in cs:
        low,high=c['estimate_s'];extra=0 if c['kind'] in CPU_KINDS else 50
        command += [f'# {c["id"]}: {c["kind"]}; worker {low}-{high}s (planning, not measurement); +0-{extra}s lease/cooldown; outer<=588s; depends: '+','.join(c['depends']),f'bash scripts/apa_sp4g_lead_gpu.sh run {c["id"]}']
    command += ['bash scripts/apa_sp4g_lead_gpu.sh summary']
    (A/'lead_commands.txt').write_text('\n'.join(command)+'\n')
    from apa_sp4g_a1_provenance import BRIDGE
    manifest=dict(schema='apa_sp4g_cells_a1',registration_sha256=REG_SHA,amendments=[dict(order='orders/APA_SP4G_AMENDMENT_1.md',execution='amendment_002_a1_execution.json',execution_sha256=sha(A/'amendment_002_a1_execution.json'),fingerprint=BRIDGE,fingerprint_sha256=sha(A/BRIDGE))],cells=cs,fallback_cells=fallback_cells(),fallback_policy='explicit run after current whole-layer RED RAIL only; excluded from default resume')
    (A/'cells.json').write_text(json.dumps(manifest,indent=2)+'\n')
    fallback=['# OPTIONAL ONLY. Run ONLY the group for a whole-layer cell that hit RED RAIL.', '# Default resume does not dispatch these. Every band preflight verifies the current whole-layer RAIL.', '# Run the group in order; the final summary satisfies the layer dependency while retaining the RED.', 'cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g']
    for c in fallback_cells():
        fallback += [f'# fallback_for={c["fallback_for"]}; kind={c["kind"]}; worker estimate={c["estimate_s"]}s; outer<=588s; GPU cooldown30s/lease<=20s (summary CPU only)',f'bash scripts/apa_sp4g_lead_gpu.sh run {c["id"]}']
    (A/'lead_commands_fallback.txt').write_text('\n'.join(fallback)+'\n')
    print(json.dumps(dict(report=str(A/'RESULTS.md'),cells=len(cs),valid_receipts=len(jobs),audit_failures=len(audit),cpu=cpu['status'])))
if __name__=='__main__':
    t=VALIDATION.set({})
    try:render()
    finally:VALIDATION.reset(t)
