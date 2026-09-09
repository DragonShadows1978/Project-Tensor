"""Create-only registered semantic mutants, disposable module copies only.

Prior art: mutation testing, DeMillo/Lipton/Sayward (1978), unverified lead
"Hints on Test Data Selection". Reuse defect probes, new A4 attack sites.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from apa_sp3_common import ART, ROOT, publish, sha

MUTANTS = [
 ('target_shift','apa_sp3_a4_torch','np.asarray(ids)[-513:]','np.asarray(ids)[-514:-1]',
  'test_compact_reference_oracle_six_windows_and_long_rows'),
 ('no_file_integrity','apa_sp3_a4_capture',"if stat_pin(path) != pin['stat'] or (rehash and sha(path) != pin['sha256']):",'if False:',
  'test_checkpoint_continuation_matches_full_layer_sequence'),
 ('mask_bitorder','apa_sp3_a4_capture',"np.packbits(mn, axis=-1, bitorder='little')","np.packbits(mn, axis=-1, bitorder='big')",
  'test_streamed_mask_roundtrip_and_gaps'),
 ('native_parity_bypass','apa_sp3_a4_capture','if not np.array_equal(out.slice(2, lo, n).numpy(), other.numpy()):','if False:',
  'test_native_ranged_replay_matches_and_rejects_output_change'),
 ('future_source_reuse','apa_sp3_a4_provenance',"and digest == transition.get('after_sha256')",'and True',
  'test_per_kind_compatibility_allowlist_and_unknown_changes'),
 ('force_ineligible_flash','apa_sp3_a4_torch','flash = t.backends.cuda.can_use_flash_attention(params, debug=False)','flash = True',
  'test_actual_tensor_fused_choice_never_math'),
 ('ceiling_fit_bypass','apa_sp3_a4_jobs',"if r.get('fit') is not True:",'if False:',
  'test_fit_dependency_blocks_before_model_load'),
 ('range_row_skip','apa_sp3_a4_capture','for i in range(lo, hi):','for i in range(lo, hi-1):',
  'test_checkpoint_continuation_matches_full_layer_sequence'),
]


def main():
    directory=ART/'a4_mutations';directory.mkdir(exist_ok=False)
    test=ROOT/'tensor_cuda/tests/test_apa_sp3_a4.py'
    publish(directory/'registration.json',dict(threshold=.8,no_invalid_mutants=True,
        evidence_class='unit test / author mutation baseline',mutants=MUTANTS,
        source_sha256={m:sha(ROOT/'scripts'/(m+'.py')) for _,m,_,_,_ in MUTANTS},
        prior_art=__doc__))
    baseline=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(test)],
                            capture_output=True,text=True,timeout=60)
    (directory/'baseline.log').write_text(baseline.stdout+baseline.stderr)
    if baseline.returncode:
        raise RuntimeError('mutation baseline RED')
    results=[]
    for name,module,before,after,case in MUTANTS:
        source=(ROOT/'scripts'/(module+'.py')).read_text()
        if source.count(before)!=1:
            raise RuntimeError('mutation site drift: '+name)
        dest=directory/(name+'.py');dest.write_text(source.replace(before,after))
        # Import normal graph first so injected module constants bind to this
        # checkout; replace only the named module for the selected CPU test.
        code=(f'import sys, importlib.util, pytest; sys.path.insert(0,{str(ROOT/"scripts")!r})\n'
              'import apa_sp3_gpu\n'
              f'spec=importlib.util.spec_from_file_location({module!r},{str(dest)!r})\n'
              f'm=importlib.util.module_from_spec(spec);sys.modules[{module!r}]=m;spec.loader.exec_module(m)\n'
              f'raise SystemExit(pytest.main(["-q","-p","no:cacheprovider",{str(test)+"::"+case!r}]))')
        p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
        (directory/(name+'.log')).write_text(p.stdout+p.stderr)
        killed=p.returncode==1 and any(s in p.stdout for s in ('AssertionError','DID NOT RAISE','Failed: must stop'))
        results.append(dict(name=name,exit_code=p.returncode,killed=killed,
                            invalid=p.returncode not in (0,1) or p.returncode==1 and not killed))
    rate=sum(r['killed'] for r in results)/len(results)
    result=dict(status='PASS' if rate>=.8 and not any(r['invalid'] for r in results) else 'RED',
                kill_rate=rate,mutants=results,production_edits=False)
    publish(directory/'results.json',result)
    print(json.dumps(result))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
