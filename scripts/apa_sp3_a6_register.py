"""Create-only A6 registration. Prior art: SP3 A4/A5 immutable experiment DAGs
(2026) reused; one-factor controlled experiments. No prior art known to me
for this exact ladder or rail formula. No numerical kernel is introduced.
"""
from pathlib import Path
from apa_sp3_common import ART, ROOT, REG_SHA, read, sha, publish


def main():
    cfg = dict(attention_wrapper=False, full_logits_host_copy=False,
               last_token_only=True, use_cache=True, pool_before_load=True,
               interposer=False, fused_decode=True, fused_rms_norm=True,
               absorbed_decode=True, fused_softmax=True)
    def cell(id, kind, arm='A', bits=4, S=2048, feeding='teacher_forced', config=cfg, **extra):
        deps = ['g0', 'kernel96'] + (['ppl_b4_D_1024', f'freeze_b{bits}'] if arm != 'A' else [])
        return dict(id=id, kind=kind, arm=arm, bits=bits, S=S, steps=32,
                    warmup_steps=1, feeding=feeding, config=dict(config), depends=deps,
                    worker_timeout_s=290, job_ceiling_s=590, estimate_s=[20,280],
                    optional_secondary=bits==8,
                    estimate_scope='unmeasured planning: load, prefill, discarded cache-preserving warmup, 32 steps', **extra)
    new = [cell('decode_repro_b4_A_2048', 'decode_repro', feeding='greedy'),
           cell('decode_bisect_00_reference', 'decode_bisect', rung=0, change=None, compare_to=None)]
    variants = [('wrapper','attention_wrapper',True), ('host_logits','full_logits_host_copy',True),
                ('last_token_only','last_token_only',False), ('cache_recompute','use_cache',False),
                ('pool_after','pool_before_load',False), ('interposer','interposer',True),
                ('int4_eager','fused_decode',False), ('norm_eager','fused_rms_norm',False),
                ('expanded_mla','absorbed_decode',False), ('softmax_eager','fused_softmax',False)]
    for i,(name,key,value) in enumerate(variants,1):
        new.append(cell(f'decode_bisect_{i:02d}_{name}', 'decode_bisect',
                        config=dict(cfg, **{key:value}), rung=i, change=key,
                        compare_to='decode_bisect_00_reference'))
    legacy = dict(cfg, attention_wrapper=True, full_logits_host_copy=True,
                  pool_before_load=False, interposer=True, fused_decode=False,
                  fused_rms_norm=False, absorbed_decode=False, fused_softmax=False)
    new.append(cell('decode_bisect_11_legacy_stack','decode_bisect',config=legacy,
                    rung=11,change='combined legacy endpoint; not one-factor evidence',
                    compare_to='decode_bisect_00_reference'))
    for bits in (4,8):
        for S in (2048,8192,32768):
            for arm in 'ABC':
                c = cell(f'decode_clean_b{bits}_{arm}_{S}', 'decode_clean', arm, bits, S,
                         config=dict(cfg,absorbed_decode=arm=='A'))
                if arm=='A' and S>=8192:
                    c.update(nonfit='NON_FIT_REGISTERED_DENSE', estimate_s=None,
                             estimate_scope='unchanged dense full-prefill memory non-fit; no alternate cache setup')
                elif S==32768:
                    source=f'decode_clean_b{bits}_{arm}_8192'
                    c.update(estimate_s=None,estimate_from=source,
                             estimate_scope='setup + 16*prefill + 4*(warmup+decode_work) + 15; >=290s non-fit')
                    c['depends'].append(source)
                new.append(c)
    doc=Path('/mnt/ForgeRealm/GraftRepository/docs/MiniCPM3-MLA_Results.md')
    m=dict(immutable=True,registration_sha256=REG_SHA,
           order_sha256=sha(ROOT/'orders/APA_SP3_AMENDMENT_6.md'),
           parent_a5_effective_sha256=read(ART/'a6_before.json')['effective_a5_sha256'],cells=new,
           june_receipt=dict(path=str(doc),sha256=sha(doc),engine='8501a5c',S_approx=360,
                             ms_token=21.6,device='RTX 3070 8GB',comparison_limit_ms=43.2),
           policies=dict(
               ladder='variants 01-10 each change ONE field from 00; same teacher-forced tokens; 11 combined descriptive endpoint; independent leases; variant RED does not prevent independent rungs',
               timing='32 CUDA-synchronized step walls including scalar argmax transfer; also forward-only series comparable to A5; full-logit variant copies after forward timer inside step timer',
               warmup='one discarded first-position forward retains original prefill cache; lazy absorbed weights excluded from steady timing; measured contexts S+1..S+32',
               finite='one final finite check after timing; no per-step full host copies in clean/repro; scalar argmax bounds every step; no model quality claim',
               pool_peak='default pool reserved/used high water reset before prefill; before-load mode can include pooled persistents; NOT whole-device resident peak',
               hook='clean never installs MLAAttentionTC.__call__ or blend diagnostics; C uses only necessary native SP API dispatch with diagnostics=False',
               guard='validated G0/kernel/freeze dependencies retained; no historical in-process PPL rerun under changed June flags',
               P5='valid measured clean bulk4 C/B at 32768 only; >=2 HIT else MISSED; absent/nonfit -> UNASSESSABLE plus same-kind 8192 ratio',
               reproduction='June documented fast stack, current pinned adapter/engine, requested S2048; not historical engine/S360 bit reproduction',
               build='local missing build compiled for CPU probes; preserve lead build at integration; no runtime-identity waiver'),
           cpu_gates=['registration/DAG/old cells/rails','test_decode_clean_no_attention_hook_installed',
                      'pool before loader and on every forward','cache/offset/greedy/teacher-forced/scalar-copy/32-step pins',
                      'one-factor ladder','C dispatch diagnostics off and B unchanged',
                      'shell no-preload + worker process-map rejection',
                      '32K absent/bad source and >=290 boundary non-fit before lease',
                      'P5 clean-only and 8192 no substitution','old bridge accepts reviewed endpoints rejects unknown',
                      'kernel/source and 736 old receipt byte pins',
                      'passing mutation baseline then >=0.80 valid mutants killed'],
           prior_art=['GraftRepository MiniCPM3/TensorCUDA June 2026 documented fast stack reused; no new kernel',
                      'DeepSeek-AI DeepSeek-V2 2024 absorbed MLA cited by local adapter; unverified - lead to check DeepSeek V2 MLA weight absorption',
                      'NVIDIA CUDA 12.6 default-pool counters / A5 PoolPeak reused',
                      'one-factor controlled experiments; no prior art known to me for exact ladder/planning formula',
                      'A4/A5 hash bridge; Make Feldman 1979 / Nix Dolstra 2004 unverified - lead to check dependency content hashing',
                      'DeMillo Lipton Sayward 1978 mutation testing unverified - lead to check Hints on Test Data Selection'])
    p=ART/'amendment_011_decode_clean.json'
    publish(p,m)
    (ART/(p.name+'.sha256')).write_text(sha(p)+'\n')
    print(f'Registered {len(new)} cells: {sha(p)}')


if __name__=='__main__':main()
