# APA-DK-STE5B: ACCEPT crash amendment; re-capture required

B-G1 PASS: **39 passed in 11.03s**, rc0. B-G2 uses the order's explicit re-capture exception. Existing base captures DO NOT COUNT. No G-G1 verdict or full STE5 acceptance claimed.

## Cause

The record missing the key is each of the six lead capture.json records. Pre-fix capture5.py:67 writes input_pins_sha256 but no effective_pins_sha256; verify5.py:72 requires it and :76 dereferences meta[key]. The order's common.py:154 attribution is not the failing call: it reads a saved premise report that already contains the key. STAGING.json is not that operand.

Historical scratch/amend_test.py:28 replaced a literal without spaces after colons. capture5.py uses spaces, so the replacement missed it. INPUT_PINS_AMENDMENT_001.json confirms capture5's before and after hashes are equal. Tests only checked helper comparison logic, a missing-governor early exit, import behavior and historical reference parity; none passed a real capture record through validate_capture/premise CLI.

Before fix, CPU reproduction returned exactly:

```json
{"PASS": false, "verdict": "RED", "gates": {}, "error": "KeyError('effective_pins_sha256')"}
```

Receipt: scratch/ste5b/before-premise.json, .md, .log, .rc. Evidence class: CPU reproduction, not a G-gate verdict.

## Diff

Exact source diff, also saved in STE5B.diff:

```diff
--- before/artifacts/apa_ste5/capture5.py
+++ after/artifacts/apa_ste5/capture5.py
@@ -66,6 +66,9 @@
         'switch_environment': actual, 'all_selected': all_selected, 'variant': 'g1',
         'so_sha256': c.digest((c.BASE_PACKAGE if which == 'base' else c.ARM_PACKAGE)/c.SO_NAME),
         'driver_sha256': c.digest(__file__), 'input_pins_sha256': c.digest(c.HERE/'INPUT_PINS.json'),
+        # Crash amendment STE5B: preserve the verifier's strict amendment binding.
+        # Prior art: no prior art known to me; no new algorithm.
+        'effective_pins_sha256': c.evidence_pins_digest(),
         'pid': os.getpid(), 'host': socket.gethostname(), 'gpu_identity': identity,
         'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
         'TC_DETERMINISTIC': os.environ['TC_DETERMINISTIC'],
--- before/artifacts/apa_ste5/verify5.py
+++ after/artifacts/apa_ste5/verify5.py
@@ -63,6 +63,11 @@
 def validate_capture(config, suite, run):
     path = c.folder(config, suite, run)
     meta = json.loads((path/'capture.json').read_text())
+    # Prior art: no prior art known to me; schema repair, no new algorithm.
+    # Never infer a capture-time amendment binding from today's pin files.
+    assert 'effective_pins_sha256' in meta, (
+        f're-capture required: {path}/capture.json lacks effective_pins_sha256; '
+        'capture-time amendment binding was not recorded')
     which, switch, all_selected = c.CONFIGS[config]
     for key,value in {'config':config, 'suite':suite, 'run':run, 'which':which,
                       'switch_environment':None if switch is None else str(switch),
--- before/artifacts/apa_ste5/tests/test_verify5.py
+++ after/artifacts/apa_ste5/tests/test_verify5.py
@@ -5,6 +5,7 @@
 lead to check. Test cases fixed by STE5; ours adds real small-fixture parity.
 """
 import copy
+import ast
 import importlib.util
 import json
 import os
@@ -183,3 +184,81 @@
 
 def test_ulp_subnormal_spacing():
     assert ref.ulp(np.array([0.,2.**-126,1.])).tolist()==[2.**-133,2.**-133,2.**-7]
+
+
+@pytest.mark.parametrize('defect', [None, 'missing_binding', 'wrong_binding',
+                                  'wrong_input_pin', 'one_bit_tensor'])
+def test_real_capture_shape_premise_cli(tmp_path, defect):
+    # Prior art: no prior art known to me; no new algorithm.
+    # Faithful real exports, with only new-writer provenance supplied for this
+    # CPU fixture. These are NOT re-captures or admissible GPU gate evidence.
+    # Compare the full key set to the actual capture5 writer, so a future writer
+    # schema change cannot silently leave this integration fixture behind.
+    tree=ast.parse((HERE/'capture5.py').read_text())
+    emitted=[node.args[1] for node in ast.walk(tree)
+             if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
+             and isinstance(node.func.value,ast.Name) and node.func.value.id=='c'
+             and node.func.attr=='put']
+    assert len(emitted)==1 and isinstance(emitted[0],ast.Dict)
+    writer_keys={key.value for key in emitted[0].keys}
+    exported=tmp_path/'captures'
+    exported.mkdir()
+    (exported/'governor.tsv').write_bytes((c.LEAD/'governor.tsv').read_bytes())
+    for suite in ('small','grapa'):
+        for run in range(c.N):
+            source=c.folder('base',suite,run)
+            target=exported/c.label('base',suite)/f'run-{run}'
+            target.mkdir(parents=True)
+            meta=json.loads((source/'capture.json').read_text())
+            meta.update(driver_sha256=c.digest(HERE/'capture5.py'),
+                        effective_pins_sha256=c.evidence_pins_digest())
+            assert set(meta)==writer_keys
+            if suite=='small' and run==0:
+                if defect=='missing_binding':
+                    del meta['effective_pins_sha256']
+                elif defect=='wrong_binding':
+                    meta['effective_pins_sha256']='0'*64
+                elif defect=='wrong_input_pin':
+                    meta['input_pins_sha256']='0'*64
+            for case in meta['cases']:
+                name=case['name']
+                if defect=='one_bit_tensor' and suite=='small' and run==0 and name=='small1':
+                    with np.load(source/f'{name}.npz',allow_pickle=False) as f:
+                        words={k:f[k] for k in f.files}
+                    words['dk'].flat[0] ^= np.uint16(1)
+                    with (target/f'{name}.npz').open('xb') as f:
+                        np.savez(f,**words)
+                    case['sha256']=c.digest(target/f'{name}.npz')
+                else:
+                    (target/f'{name}.npz').symlink_to(source/f'{name}.npz')
+            (target/'capture.json').write_text(json.dumps(meta))
+    output=tmp_path/'premise.json'
+    # Use the actual CLI entry point/--premise; redirect only the read root to
+    # the faithful export. All pin, governor and tensor checks stay active.
+    code='''import runpy,sys
+sys.path.insert(0,sys.argv[1])
+import common as c
+c.LEAD=c.Path(sys.argv[2])
+sys.argv=[str(c.HERE/'verify5.py'),'--premise','--output',sys.argv[3]]
+runpy.run_path(sys.argv[0],run_name='__main__')
+'''
+    result=subprocess.run([sys.executable,'-B','-c',code,str(HERE),str(exported),str(output)],
+                          text=True,capture_output=True,timeout=120)
+    report=json.loads(output.read_text())
+    assert 'KeyError' not in report.get('error',''),result
+    if defect is None:
+        assert result.returncode==0 and report['PASS'],report
+        assert report['gates']=={'G-G1':True}
+        assert len(report['base_inventory'])==6 and len(report['identity'])==54
+        assert all(r['PASS'] for r in report['identity'])
+        assert all(r['PASS'] for r in report['governor_base'].values())
+    elif defect=='one_bit_tensor':
+        assert result.returncode==1 and not report['PASS']
+        assert report['gates']=={'G-G1':False}
+        assert any(not r['PASS'] for r in report['identity'])
+    else:
+        assert result.returncode==1 and not report['PASS'] and report['gates']=={}
+        expected={'missing_binding':'re-capture required',
+                  'wrong_binding':'wrong capture metadata',
+                  'wrong_input_pin':'wrong capture metadata'}[defect]
+        assert expected in report['error'],report
```

INPUT_PINS_AMENDMENT_002.json separately records the three new harness/test hashes, before CPU gates. Original INPUT_PINS.json, amendment 001 and both orders remain immutable. No common.py edit. The exact effective pin equality and all other checks remain enabled.

## Test result

```text
39 passed in 11.03s
```

Command from workspace root:

```bash
export TMPDIR="$PWD/artifacts/apa_ste5/scratch"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
timeout --foreground 300s python3 -B -m pytest -p no:cacheprovider --basetemp artifacts/apa_ste5/scratch/ste5b/pytest --junitxml artifacts/apa_ste5/scratch/ste5b/cpu.xml artifacts/apa_ste5/tests/test_verify5.py
```

Five new cases run the actual verify5.py --premise CLI entry point on faithful exports of all six real records and nine real tensor files, using unchanged real governor rows. The export key set is checked against capture5.py's actual writer dictionary. Supplied repaired-writer metadata is explicitly CPU test provenance, not a GPU capture: clean export passes with 54 tensor comparisons; missing binding, wrong effective binding and wrong original pin are rejected; one changed dK bit with a recomputed file hash produces G-G1 FAIL. No checks are monkeypatched; only the capture read-root is redirected in the subprocess. All tensors are read at original widths.

Evidence: scratch/ste5b/cpu.log, cpu.xml, cpu.rc and per-case JSON/MD under scratch/ste5b/pytest/. Refreshed receipts/S5_CPU.json, cpu.xml and cpu.log for bundle5's existing receipt reader. Their superseded originals are retained at scratch/ste5b/previous-*. Author CPU baseline only; no blind review claimed.

## Premise verdict

Existing captures: **RE-CAPTURE REQUIRED**, no admissible G-G1 verdict. The exact CPU rerun summary is:

```json
{"PASS": false, "verdict": "RED", "gates": {}, "error": "AssertionError('re-capture required: /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1/artifacts/apa_ste5/lead/ste5v1-base-small/run-0/capture.json lacks effective_pins_sha256; capture-time amendment binding was not recorded')"}
```

The rc1 record is a schema/provenance rejection, not a tensor gate failure and not ACCEPT of existing captures. Both original governor rows pass ran/rc0 checks. Neither those rows nor the original pin hash supplies the missing capture-time amendment binding. The changed capture driver hash also continues to be enforced. Re-capture is mandatory with the fixed writer; the lead owns GPU execution and preservation/relocation of prior output paths before reuse. The existing lead/PREMISE.json remains the original crash receipt and must not be used to authorize arms.

CPU command actually run:

```bash
CUDA_VISIBLE_DEVICES= timeout --foreground 300s python3 -B artifacts/apa_ste5/verify5.py --premise --output artifacts/apa_ste5/scratch/ste5b/existing-premise.json
```

Same TMPDIR/one-thread environment as tests. Receipts: scratch/ste5b/existing-premise.json, .md, .log, .rc. No GPU use. Stored tensors cannot count for this real gate without admissible capture provenance.

## Restage status

GPU_COMMANDS.sh **unchanged**, SHA-256 `b78e580ac575161f07daca10ff0dfc5a2429b1e76a2127c091243fcfe20afb8c`. STAGING.json unchanged. No restage required. CPU stage5.py --check rc0: `pinned inputs, staged commands and bundle hashes PASS`. Exact logs/rc in scratch/ste5b/stage-check.*.

All 18 files under lead/ match pre-work hashes, including captures, tensor files, governor and old premise receipts. Original pin set plus amendments passes check_pins; .so, engine, kernels, thresholds and variants remain unchanged. See scratch/ste5b/integrity.json and STE5B_OPERAND_CENSUS.md.

## Prior art

no prior art known to me. No new algorithm; writer/schema repair and CPU integration coverage only.

## Every file written

See STE5B_FILES_WRITTEN.md for an exhaustive list of source edits, new registration/amendment/census/report/diff, refreshed CPU receipts, retained baseline/prior receipts, every test scratch file and every tensor symlink. All are under artifacts/apa_ste5/. No writes under lead/, no deletions or DELETE_ME moves. Scratch is retained.
