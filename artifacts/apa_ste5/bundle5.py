#!/usr/bin/env python3
"""Stage disclosures or seal recomputed STE5 evidence, never publish.

Prior art: NIST FIPS 180-4 (2015), SHA-256; STE1/4 bundle protocol (2026).
Unverified - lead to check. Ours: explicit variant/evidence boundaries and F4
package hash refresh; no new algorithm or optimization.
"""
import json
from pathlib import Path
import common as c

MANIFEST=c.OLD/'engine/ENGINE_MANIFEST.json'


def write_json(path,value):
    # This round's mutable output summaries, not immutable registrations.
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def seal(manifest,status):
    assert c.digest(c.ARM_PACKAGE/c.SO_NAME)==c.ARM_PIN
    manifest['files']={name:c.digest(c.ARM_PACKAGE/name) for name in manifest['files']}
    assert (c.ARM_PACKAGE/'__init__.py').read_bytes()==(c.ROOT/'tensor_cuda/tensor_cuda/__init__.py').read_bytes()
    manifest['source_file_hashes']['tensor_cuda/tensor_cuda/__init__.py']=c.digest(c.ARM_PACKAGE/'__init__.py')
    manifest['status']=status
    write_json(MANIFEST,manifest)
    bundle={'schema':'apa-ste5-bundle-v1','manifest':'artifacts/apa_ste1/engine/ENGINE_MANIFEST.json',
            'manifest_sha256':c.digest(MANIFEST),'so_sha256':c.digest(c.ARM_PACKAGE/c.SO_NAME),
            'status':status}
    write_json(c.HERE/'BUNDLE.json',bundle)
    return bundle


def stage():
    c.cpu_environment()
    original=json.loads(MANIFEST.read_text())
    c.put(c.HERE/'receipts/ORIGINAL_ENGINE_MANIFEST.json',original)
    old=json.loads((c.HERE.parent/'apa_ste_rev1/oracle.json').read_text())['grapa']['variant_a']
    split=json.loads((c.HERE.parent/'apa_ste_rev1/order_vs_bias.json').read_text())
    original['disclosure']={
        '1_variants':{
            'GRAPA':'backward g1 / forward h, including a_base; scripts/cc78_launch.py:230-234 in grapa_wt_cc80',
            'g1':'STE5 stages g1/h on small1, small2 and B1 H16 KVH16 L=S4096 D128 VD64 causal BF16; 3 fresh processes/config. GPU evidence PENDING lead; no GRAPA-shape g1 claim yet. Receipt after run: artifacts/apa_ste5/lead/VERIFY5.json.',
            'f_pass_a':'STE4 lead/VERIFY4.json ACCEPT: all D-G1..D-G6 PASS on small1, small2 and GRAPA shape, with forward a. Exact byte controls plus conditional elementwise bound; not a g1 proof.',
            'a':'STE3 envelopes and float64 comparisons; REV1 oracle and accumulation-error disclosure below. Not GRAPA arm settings.',
            'others':'STE1 small-shape evidence only; no GRAPA-shape validation or compatibility claim for other variants.'},
        '2_switch':{
            'semantics':'TC_APA_DK_STE unset or 0 = off (shipped default); 1 = on. Read once during extension import, cached process-wide. Change requires a fresh process. Experiment engine.',
            'rule':'On adds dScore*scale*q for visible unselected pairs to dK through identity STE of quantized K. Selected pairs, forward, dQ and dV keep their rules; finite gates test resulting bytes.',
            'path':'ops.cpp:21-31 -> bindings.cpp:221 -> kernels.cu:3157-3159 (g1). Unchanged .so af9ffce4; F4 changes only Python import-error handling.',
            'invalid':'Other strings raise ImportError naming TC_APA_DK_STE and allowed unset, 0, or 1; no fallback masks extension initializer errors.',
            'inherited_environment':'An inherited 1 enables STE even if an arm omits the variable. GRAPA-side scrubbing/assertion is F5 and remains outside this engine change.'},
        '3_variant_a_BF16_accumulation':{
            'summary':'BF16-destination accumulation error, with both common error and run-to-run ordering spread; not merely atomic-order noise. Base dK/dV are not bitwise repeatable. At refine-all the on/off dK computation is the same, subject to accumulation ordering.',
            'grapa_run0_all_elements':split,
            'grapa_on_certain_key_elements':old['on_dk_vs_R_on_certain_keys'],
            'grapa_off_certain_key_elements':old['off_dk_vs_R_off_certain_keys'],
            'delta_certain_key_elements':old['ste_delta_certain'],
            'scope':'REV1 oracle.json certain-key metrics exclude 11,292 keys with selection uncertainty: 6,943,232 elements retained; order_vs_bias.json split uses all elements of two runs. On rel-L2 about 6.56-7.06%, off 3.04-3.19%; per-element on ULP median 7.03, p99 272.34 on certain keys. Do not confuse ULP at max reference magnitude with per-element percentiles.',
            'envelopes':'artifacts/apa_ste4/VARIANT_A_DISCLOSURE.json and .md (5 runs/config, all within/cross pairs); REV1 findings F2 supersede their atomic-order-only wording.'},
        '4_STE4_bound':{
            'formula':'For every element: |decode(dk)-R_rule| <= Epre + 2^-8*(|R_rule|+Epre) + 2^-134, with outward float64 guards. Epre includes coefficient, selection-interval, FP32 reduction, float64 evaluation and FTZ floors; see complete derivation BOUND.md.',
            'conditions':'f_pass_a only: FP32 accumulation/key ownership then one BF16 RN store; decoded BF16 inputs, finite no-overflow IEEE RN arithmetic; stated CUDA exp/log/sqrt contracts and domains. Not a g1 bound. No mask in the STE4 element gate. Finite validation does not prove all-input or library-contract correctness.',
            'receipt':'artifacts/apa_ste4/BOUND.md; lead/VERIFY4.json; REV1 K3 confirms zero element failures and byte controls',
            'BOUND_sha256':c.digest(c.HERE.parent/'apa_ste4/BOUND.md')},
        '5_cost':{
            'g1_h':'PENDING lead timing: one APA forward h + backward g1 pair at GRAPA shape, off/on, 2 samples after one warmup; verify5 reports raw samples, means and on-minus-off difference.',
            'step_time':'GRAPA arm step time comes from GRAPA own smoke. A single APA-call delta is not a step-time projection.',
            'prior_variant_a':'REV1 K5: about +10.2% backward single-call kernel delta (n=2, 14 ms spread), inapplicable to g1/h. No g1 step cost inferred.'},
        '6_provenance':{
            'base':'69b154cd2e9a5a7cbe288e1419d963697e5add57d2b4ee827c4c78a273ff2c5c; source baseline f2c686e plus three restored untracked files, byte-identical to base build inputs.',
            'restored_sources':json.loads((c.OLD/'receipts/restored_source.json').read_text()),
            'arm':'af9ffce4733628caac5b6bd7a345de86c66c1f67d5150a53d7454b3d4bcb3760; no .so or kernel change in STE5. Identical F4 Python file at source and in engine package.',
            'registration':'STE4 registration was committed with data; temporal ordering rests on registered_utc, mtimes and capture-recorded hashes (REV1 F6). STE5 thresholds are the immutable lead order; INPUT_PINS.json is written before CPU gates and any lead capture. This seat registers no new gate; future amendments must be separate files.'}}
    return seal(original,'STE5 staged; g1/h GPU gates and timing PENDING; not accepted or promoted')


def validate():
    bundle=json.loads((c.HERE/'BUNDLE.json').read_text())
    assert bundle['manifest_sha256']==c.digest(MANIFEST)
    assert bundle['so_sha256']==c.digest(c.ARM_PACKAGE/c.SO_NAME)==c.ARM_PIN
    m=json.loads(MANIFEST.read_text())
    assert len(m['disclosure'])==6
    for name,h in m['files'].items(): assert c.digest(c.ARM_PACKAGE/name)==h
    return bundle


def complete(report):
    # Called only by verify5 after recomputing every gate, not from a saved PASS.
    assert report['PASS'] and report['gates']=={f'G-G{i}':True for i in range(1,8)}
    cpu=json.loads((c.HERE/'receipts/S5_CPU.json').read_text())
    assert cpu['PASS'] and cpu['exit_code']==0
    assert cpu['test_source_sha256']==c.digest(c.HERE/'tests/test_verify5.py')
    assert cpu['junit_sha256']==c.digest(c.HERE/'receipts/cpu.xml')
    m=json.loads(MANIFEST.read_text())
    m['disclosure']['1_variants']['g1']='STE5 G-G1..G-G7 PASS under g1/h on small1, small2 and GRAPA shape; three fresh processes per configuration. Finite captured shapes only; no all-input proof. See artifacts/apa_ste5/lead/VERIFY5.json.'
    m['disclosure']['1_variants']['g1_evidence']={'gates':report['gates'],
        'input_pins_sha256':report['input_pins_sha256'],'effective_pins_sha256':report['effective_pins_sha256'],'verifier_sha256':c.digest(c.HERE/'verify5.py'),
        'per_shape':report['correctness']}
    m['disclosure']['5_cost']['g1_h']=report['timing']
    return seal(m,'STE5 S5-G1..S5-G3 PASS; experiment engine, shipped default off; not promoted')


if __name__=='__main__':
    import sys
    assert sys.argv[1:]==['--stage']
    print(json.dumps(stage()))
