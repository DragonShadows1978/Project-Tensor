#!/usr/bin/env python3
"""CPU STE5 judge: all verdicts recomputed from fresh full-width captures.

Prior art: lead's STE5 (2026) exact experimental protocol; no prior art
known to me for these particular G-gates. Higham (2002), float64 numerical reference;
Bengio et al. (2013), STE, reused in reference5.py; NIST FIPS 180-4 (2015), hashes.
IEEE 754-2019 byte representations. External sources unverified - lead to check.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

import reference5 as ref
import bundle5
import common as c

GATES = ('G-G1','G-G2','G-G3','G-G4','G-G5','G-G6','G-G7')


def compare(left, right):
    """IEEE full width words: signed zeros and every mantissa bit count."""
    assert left.shape == right.shape and left.dtype == right.dtype
    assert left.dtype in (np.dtype('uint16'), np.dtype('uint32'))
    return {'bitwise': left.tobytes() == right.tobytes(),
            'different_elements': int(np.count_nonzero(left != right)),
            'maxabs': float(np.max(np.abs(c.decode(left)-c.decode(right))))}



def identity_case(groups, base_only=False):
    assert set(groups) == ({'base'} if base_only else set(c.CONFIGS))
    assert all(len(runs) == c.N for runs in groups.values())
    rows = []

    def check(gate, lc, rc, pairs, keys=c.KEYS, different=()):
        for i,j in pairs:
            assert lc != rc or i != j, 'self comparison forbidden'
            for key in keys:
                l, r = groups[lc][i][key], groups[rc][j][key]
                c.healthy(c.decode(l)); c.healthy(c.decode(r))
                stats = compare(l,r)
                rows.append({'gate': gate, 'left': [lc,i], 'right': [rc,j], 'tensor': key,
                             'PASS': not stats['bitwise'] if key in different else stats['bitwise'], **stats})

    check('G-G1','base','base', itertools.combinations(range(c.N),2))
    premise = all(r['PASS'] for r in rows)
    if base_only or not premise:
        return rows, {'G-G1': premise}
    product = lambda: itertools.product(range(c.N), repeat=2)
    check('G-G2','off','base', product())
    check('G-G2','unset','base', product())
    check('G-G3','on-all','off-all', product())
    check('G-G4','on','off', product(), different=('dk',))
    check('G-G6','on','on', itertools.combinations(range(c.N),2))
    return rows, {g: all(r['PASS'] for r in rows if r['gate']==g)
                  for g in ('G-G1','G-G2','G-G3','G-G4','G-G6')}


def validate_capture(config, suite, run):
    path = c.folder(config, suite, run)
    meta = json.loads((path/'capture.json').read_text())
    # Prior art: no prior art known to me; schema repair, no new algorithm.
    # Never infer a capture-time amendment binding from today's pin files.
    assert 'effective_pins_sha256' in meta, (
        f're-capture required: {path}/capture.json lacks effective_pins_sha256; '
        'capture-time amendment binding was not recorded')
    which, switch, all_selected = c.CONFIGS[config]
    for key,value in {'config':config, 'suite':suite, 'run':run, 'which':which,
                      'switch_environment':None if switch is None else str(switch),
                      'all_selected':all_selected, 'variant':'g1',
                      'so_sha256':c.BASE_PIN if which=='base' else c.ARM_PIN,
                      'driver_sha256':c.digest(c.HERE/'capture5.py'),
                      'input_pins_sha256':c.digest(c.HERE/'INPUT_PINS.json'),'effective_pins_sha256':c.evidence_pins_digest(),
                      'CUDA_VISIBLE_DEVICES':'0', 'TC_DETERMINISTIC':'1',
                      'TC_DET_EMBED_BWD':'1'}.items():
        assert meta[key] == value, f'wrong capture metadata {path}/{key}'
    assert isinstance(meta['pid'],int) and meta['pid'] > 0
    assert meta['host'] and meta['gpu_identity']
    assert len(meta['cases']) == len(c.names(suite))
    assert {r['name'] for r in meta['cases']} == set(c.names(suite))
    data = {}
    for row in meta['cases']:
        name = row['name']
        assert row == {'name':name, 'sha256':c.digest(path/f'{name}.npz'),
                       'zthr':-1e6 if all_selected else c.Z, 'shape':list(c.SHAPES[name]),
                       'variant':'g1', 'forward_variant':'h', 'dtype':'bfloat16', 'causal':name!='small1'}
        with np.load(path/f'{name}.npz', allow_pickle=False) as f:
            d = {k:f[k] for k in f.files}
        assert set(d) == set(c.KEYS+c.INPUTS)
        B,H,L,S,D,VD,KH = c.SHAPES[name]
        shapes = {'out':(B,H,L,VD), 'lse':(B,H,L), 'thr':(B,H,L), 'dq':(B,H,L,D),
                  'dk':(B,KH,S,D), 'dv':(B,KH,S,VD)}
        for key in c.KEYS:
            assert d[key].shape == shapes[key]
            assert d[key].dtype == (np.uint32 if key in ('lse','thr') else np.uint16)
            c.healthy(c.decode(d[key]))
        expected = c.input_words(name)
        for key in c.INPUTS:
            assert compare(d[key], expected[key])['bitwise'], f'fixture mismatch: {name}/{key}'
        if all_selected:
            threshold = c.decode(d['thr'])
            if name == 'grapa':
                assert np.all(threshold[:,:,1:] < 0), 'refine-all multi-key rows not all selected'
                q, kq = c.decode(d['q']), c.decode(d['kq'])
                assert np.all(q[:,:,0,:] == .25) and np.all(kq[:,:,0,:] == .25)
                bulk = np.float32(D*.25*.25)*np.float32(D**-.5)
                assert np.all(threshold[:,:,0] <= bulk), 'refine-all singleton not selected'
            else:
                assert np.all(threshold < 0), 'refine-all threshold must select every visible key'
        data[name] = d
    return meta, data


def load_groups(configs):
    groups = {name:{cfg:[] for cfg in configs} for name in c.SHAPES}
    inventory, identities, pids = [], set(), set()
    # Only store
    # paths here. Provenance validation visits every file before any verdict.
    for suite in ('small','grapa'):
        for cfg in configs:
            for run in range(c.N):
                meta, data = validate_capture(cfg,suite,run)
                identities.add((meta['host'],meta['gpu_identity']))
                assert meta['pid'] not in pids, 'fresh capture PID reused; inspect provenance'
                pids.add(meta['pid'])
                inventory.append({'config':cfg,'suite':suite,'run':run,
                                  'capture_sha256':c.digest(c.folder(cfg,suite,run)/'capture.json'),
                                  'pid':meta['pid'], 'cases':meta['cases']})
                for name in c.names(suite):
                    groups[name][cfg].append(c.folder(cfg,suite,run)/f'{name}.npz')
    assert len(identities) == 1, 'GPU/host mismatch across captures'
    return groups, inventory


def materialize(paths):
    groups = {}
    for cfg, files in paths.items():
        groups[cfg] = []
        for p in files:
            with np.load(p, allow_pickle=False) as f:
                groups[cfg].append({k:f[k] for k in c.KEYS})
    return groups


def witnesses(on, off, arrays):
    a,b=c.decode(on),c.decode(off)
    candidates=arrays['certified_unselected'] & np.any(a!=0,axis=-1) & np.all(b==0,axis=-1)
    indices=np.argwhere(candidates)
    index=tuple(indices[0]) if len(indices) else None
    return {'PASS':bool(len(indices)), 'certified_unselected_keys':int(arrays['certified_unselected'].sum()),
            'witness_keys':int(len(indices)), 'index':[int(x) for x in index] if index else None,
            'on_maxabs':float(np.abs(a[index]).max()) if index else None,
            'off_maxabs':float(np.abs(b[index]).max()) if index else None}


def correctness(name, groups, paths):
    # Every G-G5 element remains in the metrics, including near selection ties.
    # Only G-G7 uses a conservative saved-threshold certificate.
    words=c.input_words(name)
    with np.load(paths['off'][0],allow_pickle=False) as f:
        saved_thr=f['thr']
    arrays=ref.oracle(words,name,saved_thr=saved_thr)
    signal_rows=[]; witness_rows=[]; errors=[]
    for i,j in itertools.product(range(c.N),repeat=2):
        on,off=groups['on'][i]['dk'],groups['off'][j]['dk']
        signal_rows.append({'shape':name,'on_run':i,'off_run':j,
            **ref.signal_check(c.decode(on),c.decode(off),arrays['on'],arrays['off'],arrays['extra'])})
        # Certificate valid only for this pair's identical saved thresholds.
        same_thr=compare(groups['on'][i]['thr'],saved_thr)['bitwise'] and compare(groups['off'][j]['thr'],saved_thr)['bitwise']
        w=witnesses(on,off,arrays)
        w['threshold_identity']=same_thr; w['PASS'] &= same_thr
        witness_rows.append({'shape':name,'on_run':i,'off_run':j,**w})
    for cfg in ('on','off'):
        for i,d in enumerate(groups[cfg]):
            errors.append({'shape':name,'config':cfg,'run':i,
                **ref.error_percentiles(c.decode(d['dk']),arrays[cfg])})
    # Mathematical refinement-all control, independent of equality gate G-G3.
    all_arrays=ref.oracle(words,name,z=-1e6)
    assert all_arrays['stats']['all_visible_selected'], 'refine-all reference missing selected pair'
    return {'shape':name,'G-G5':all(r['PASS'] for r in signal_rows),
            'G-G7':all(r['PASS'] for r in witness_rows),'signals':signal_rows,
            'witnesses':witness_rows,'dk_error_BF16_ULP':errors,'oracle_stats':arrays['stats'],
            'oracle_health':'all reference tensors finite, nonzero, nonconstant; no G-G5 mask'}


def write_table(path, report):
    lines = ['| G-gate | Verdict |','|---|---|']
    for gate in GATES:
        v = report['gates'].get(gate)
        lines.append(f'| {gate} | '+('PASS' if v else 'FAIL' if v is False else 'NOT EVALUABLE')+' |')
    lines.append(f"| Overall | {report['verdict']} |")
    with Path(path).open('x') as f:
        f.write('\n'.join(lines)+'\n')


def timing_results():
    governor=c.require_governor(c.LEAD/'governor.tsv',[f'ste5v1-timing-{s}' for s in (0,1)])
    runs={}
    for switch in (0,1):
        path=c.LEAD/f'timing-{switch}.json'
        row=json.loads(path.read_text())
        for key,value in {'switch':switch,'variant':'g1','forward_variant':'h',
                'shape':list(c.SHAPES['grapa']),'dtype':'bfloat16','causal':True,'zthr':c.Z,
                'TC_DETERMINISTIC':'1','TC_DET_EMBED_BWD':'1','warmups':1,
                'so_sha256':c.ARM_PIN,'driver_sha256':c.digest(c.HERE/'timing5.py'),
                'input_pins_sha256':c.digest(c.HERE/'INPUT_PINS.json'),'effective_pins_sha256':c.evidence_pins_digest()}.items():
            assert row[key]==value, f'timing metadata {key}'
        samples=np.asarray(row['samples_ms'],dtype=np.float64)
        assert samples.shape==(2,) and np.isfinite(samples).all() and (samples>0).all()
        runs[str(switch)]={'samples_ms':samples.tolist(),'mean_ms':float(samples.mean()),
                           'receipt_sha256':c.digest(path)}
    return {'runs':runs,'on_minus_off_mean_ms':runs['1']['mean_ms']-runs['0']['mean_ms'],
            'scope':'one forward h + backward g1 APA pair at GRAPA shape, n=2 after warmup',
            'step_time':'GRAPA own smoke supplies arm step time; this is not a step projection',
            'governor':governor}


def verify(output, premise=False):
    c.cpu_environment()
    report = {'PASS':False,'verdict':'RED','gates':{},'premise_only':premise,
              'evidence_class':'stored g1/h GPU tensors; CPU ideal float64 oracle and full-width comparisons'}
    try:
        c.check_pins()
        bundle5.validate()
        report['arm_so_sha256'] = c.digest(c.ARM_PACKAGE/c.SO_NAME)
        report['base_so_sha256'] = c.digest(c.BASE_PACKAGE/c.SO_NAME)
        report['input_pins_sha256'] = c.digest(c.HERE/'INPUT_PINS.json')
        report['effective_pins_sha256']=c.evidence_pins_digest()
        # G-G1 is evaluated globally BEFORE loading/checking other configs.
        report['governor_base'] = c.require_governor(c.LEAD/'governor.tsv',
                                        [c.label('base',s) for s in ('small','grapa')])
        basepaths, baseinventory = load_groups(('base',))
        report['base_inventory'] = baseinventory
        report['identity'] = []
        d1 = True
        for name,paths in basepaths.items():
            rows,gates = identity_case(materialize(paths),base_only=True)
            report['identity'].extend(dict(shape=name,**r) for r in rows)
            d1 &= gates['G-G1']
        report['gates']['G-G1'] = d1
        if not d1:
            raise AssertionError('G-G1 FAIL: determinism premise fails; STOP')
        if premise:
            report.update(PASS=True,verdict='PREMISE PASS; remaining G-gates not evaluated')
        else:
            report['governor'] = c.require_governor(c.LEAD/'governor.tsv',
                            [c.label(cfg,s) for cfg in c.CONFIGS for s in ('small','grapa')])
            allpaths,report['capture_inventory'] = load_groups(tuple(c.CONFIGS))
            totals = {g:True for g in GATES}
            report['correctness'] = []
            for name,paths in allpaths.items():
                groups = materialize(paths)
                rows,gates = identity_case(groups)
                assert gates['G-G1'], 'G-G1 changed after premise'
                report['identity'].extend(dict(shape=name,**r) for r in rows if r['gate']!='G-G1')
                for gate,value in gates.items():
                    totals[gate] &= value
                result = correctness(name,groups,paths)
                report['correctness'].append(result)
                totals['G-G5'] &= result['G-G5']
                totals['G-G7'] &= result['G-G7']
                del groups
            report['timing'] = timing_results()
            report['gates'] = totals
            report['PASS'] = all(totals.values())
            report['verdict'] = 'ACCEPT' if report['PASS'] else 'RED'
            if report['PASS']:
                report['bundle']=bundle5.complete(report)
    except Exception as error:
        report['PASS']=False
        report['verdict']='RED'
        report['error'] = repr(error)
        if 'missing governor ran/rc0' in str(error):
            report['verdict'] = 'PENDING CAPTURES; not accepted'
    report['verifier_sha256'] = c.digest(__file__)
    c.put(output,report)
    write_table(Path(output).with_suffix('.md'),report)
    print(json.dumps({k:report.get(k) for k in ('PASS','verdict','gates','error')}),flush=True)
    return 0 if report['PASS'] else 2 if report['verdict'].startswith('PENDING') else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--premise',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    sys.exit(verify(args.output,args.premise))
