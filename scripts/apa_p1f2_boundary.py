"""APA-P1-F2 L60 boundary record (BOUNDARY.json) and S5 summary (mutants/MUTANT_RESULTS.json).

BOUNDARY.json: for every refusal this round adds, the input, F2's refusal (measured: LF3 RESULT.json on GPU, or the
S4 CPU receipt), and Build F's behaviour on the same input (measured on GPU in its own child process,
LF3/BND_F.json; or, where an input cannot be constructed or would index outside a buffer on Build F, stated from
the source and labelled "source reasoning, not run").
Prior art: none; receipt assembly. CPU only.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_p1f2'


def classify_f(r):
    res = str(r.get('result', ''))
    if 'CUDA error' in res and 'invalid configuration' in res:
        return 'launch error (zero-grid CUDA invalid configuration, after a launch attempt)'
    if res.startswith('returned'):
        return 'silent acceptance'
    if r.get('launches'):
        return f'refusal after {r["launches"]} launch(es)'
    return 'refusal before any launch (other message/type)'


def main():
    lf3 = json.loads((ART / 'gpu/LF3/RESULT.json').read_text())
    bnd = {r['row']: r for r in json.loads((ART / 'gpu/LF3/BND_F.json').read_text())['rows']}
    s4 = json.loads((ART / 'cpu/S4.json').read_text())
    rows = []
    for c in lf3['checks']:
        if 'raised' not in c:
            continue
        key = c['id'].split(': ')[-1] if ': ' in c['id'] else c['id']
        if c['id'].startswith('LF3 EB3 timeline'):
            key = 'eb3_timeline_G' + c['id'].split('G=')[1].split(' ')[0]
        if 'capacity 2**31 refused' in c['id']:
            key = c['id'].split(': ')[-1]
        f = bnd.get(key)
        if f is not None:
            fb = dict(behaviour=classify_f(f), measured=f.get('result'), launches=f.get('launches'),
                      output=f.get('output'), context_after=f.get('context_after'), evidence='gpu/LF3/BND_F.json (child)')
        elif 'native packed_append' in c['id']:
            fb = dict(behaviour='silent acceptance, then out-of-buffer indexing (not run)',
                      evidence='source reasoning, not run: Build F buffer_check has no upper bound; apa_spq1_pack passes '
                               'codes.shape[2]=2^31 to pack_tq int capacity, which narrows to -2147483648, and '
                               'dst=key/N*capacity+start+key%N indexes before the codes buffer (REV2 P1R2-F04)')
        elif 'native packed_run' in c['id']:
            fb = dict(behaviour='silent acceptance, then out-of-buffer indexing (not run)',
                      evidence='source reasoning, not run: int capacity=(int)codes.shape[2] narrows to -2147483648 and '
                               'the bulk kernels index (int64)bh*capacity, before the buffers for bh >= 1 (REV2 F04)')
        else:
            fb = dict(behaviour='not recorded')
        rows.append(dict(input=c['id'], f2_refusal=c.get('raised'), f2_launches=c.get('launches'),
                         f2_device_bytes_delta=c.get('device_bytes_in_use_delta'), f2_ok=c['ok'], build_F=fb,
                         evidence='gpu/LF3/RESULT.json'))
    cpu_rows = [
        ('R1 hand-built PackedRegisteredMargins (9 cases) refused at lookup and at packed_attention',
         'test_r1_hand_built_table_refused_at_lookup_and_at_attention_before_any_device_call',
         'silent acceptance (GPU-measured on 5 equivalent tables in gpu/LF3/BND_F.json margins_*; Build F lookup has no pin check)'),
        ('R1 subclass / entries or sha replaced after load', 'test_r1_subclass_and_entries_replaced_after_load_are_refused',
         'silent acceptance (gpu/LF3/BND_F.json margins_subclass, margins_replaced_after_load)'),
        ('R1 load(expected_sha256 != pin)', 'test_r1_load_cannot_override_the_pin',
         'pinned file with another expected digest: the same kind of refusal (SHA256 mismatch); a synthetic file with its own digest: accepted (REV2 receipt 016; F1 test_apa_p1d_policy packed_table())'),
        ('R3 PackedSpKeys capacity 2**31 / 2**40', 'test_r3_packed_capacity_at_the_bound_is_accepted_one_past_is_refused_before_allocation',
         'silent acceptance, then a 2 TiB host allocation attempt fails (gpu/LF3/BND_F.json PackedSpKeys_capacity: MemoryError)'),
        ('R3 B*KV*S > INT_MAX (prepare grid) in PackedSpKeys', 'test_r3_prepare_grid_product_at_and_past_the_bound',
         'source reasoning, not run: Build F computes B*H*N in int in apa_spq1_prepare (signed overflow) and launches'),
        ('R3 append product past the bound', 'test_r3_append_product_refused_before_the_generation_moves',
         'source reasoning, not run: Build F bumps the generation, then the native prepare grid overflows int'),
        ('R3 ApaSpq1Workspace capacity 2**31 / 2**40 / float', 'test_r3_workspace_capacity_at_the_bound_accepted_one_past_refused_before_allocation',
         'silent acceptance, then a 2 TiB host allocation attempt fails (gpu/LF3/BND_F.json ApaSpq1Workspace_capacity: MemoryError)'),
        ('R3 quantize_sp2_keys(capacity=2**31)', 'test_r3_factory_capacity_past_the_bound_refused_before_allocation',
         'TypeError: no capacity keyword (gpu/LF3/BND_F.json quantize_sp2_keys_capacity)'),
        ('R8 dense storage + capacity (argument, environment, 2-bit fallback)', 'test_r8_dense_storage_refuses_a_capacity_before_device_work',
         'TypeError: no capacity keyword'),
        ('R8 capacity < S, 0, float, bool, str', 'test_r8_capacity_below_rows_or_not_an_int_is_refused',
         'TypeError: no capacity keyword (PackedSpKeys itself already refused these)'),
    ]
    for what, test, fbeh in cpu_rows:
        ids = [k for k in s4['reports'] if test in k]
        rows.append(dict(input=what, f2_refusal='ValueError (S4 CPU test, recording stubs: no allocation, no device call)',
                         f2_tests=ids, f2_ok=all(all(r['outcome'] == 'passed' for r in s4['reports'][i]) for i in ids),
                         build_F=dict(behaviour=fbeh), evidence='cpu/S4.json'))
    source_only = [
        ('R3 S + 2047 > INT_MAX on the attention entries (core, spq4 tc, spq6 tc, eb3)',
         'ValueError naming "S + 2047" before any allocation (apa_spo_require_key_range)',
         'source reasoning, not run (S >= 2^31 - 2047 keys cannot be allocated on this GPU): Build F computes S+2047 '
         'and nc*256 in int (signed overflow), then launches'),
        ('R3 eb2b_values numel/8+127 > INT_MAX', 'ValueError before the launch',
         'source reasoning, not run: Build F narrows total to int and the kernel index overflows'),
        ('R3 packed_append grid block count > INT_MAX', 'ValueError before the launch',
         'source reasoning, not run: Build F passes an int64 block count to a dim3 (narrowing) and launches'),
    ]
    for what, ref, fb in source_only:
        rows.append(dict(input=what, f2_refusal=ref, f2_ok=None, build_F=dict(behaviour=fb),
                         evidence='source (INDEX_CENSUS.json); not constructible on a 12 GB device'))
    out = ART / 'BOUNDARY.json'
    with out.open('x') as f:
        json.dump(dict(rule='L60', rows=rows, count=len(rows)), f, indent=2)
        f.write('\n')
    print('BOUNDARY rows', len(rows), hashlib.sha256(out.read_bytes()).hexdigest())
    # mutants summary
    m = json.loads((ART / 'mutants/MUTANTS.json').read_text())
    gpu = {r['name']: r for r in json.loads((ART / 'gpu/MUT/RESULT.json').read_text())['results']}
    res = []
    for spec in m['mutants']:
        row = dict(name=spec['name'], kind=spec['kind'], rule=spec['rule'], change=spec['change'])
        mdir = ART / 'mutants' / spec['name']
        if spec['kind'] != 'python':
            b = [json.loads(p.read_text()) for p in sorted(mdir.glob('build*/BUILD.json'))]
            row['builds'] = [dict(dir=Path(x['module']).parts[-3] if x.get('module') else None, status=x['status'],
                                  module_sha256=x.get('module_sha256'), rebuilt=x.get('rebuilt_objects'),
                                  attempt=x.get('attempt', 1)) for x in b]
        if 'cpu' in spec:
            c = json.loads((mdir / 'cpu/RESULT.json').read_text())
            row['S4'] = dict(expected_failure=spec['cpu']['expected_failure'], killed=c['killed'],
                             module_sha256=c['module_sha256'], failed_ids=len(c['failed_ids']))
        if 'gpu' in spec:
            g = gpu[spec['name']]
            row[spec['gpu']['line']] = dict(expected_failure=spec['gpu']['expected_failure'], killed=g['killed'],
                                            module_sha256=g['detail'].get('module_sha256'),
                                            failed_assertions=len(g['failed_assertions']))
        row['killed'] = all(v['killed'] for k, v in row.items() if isinstance(v, dict) and 'killed' in v)
        res.append(row)
    out = ART / 'mutants/MUTANT_RESULTS.json'
    with out.open('x') as f:
        json.dump(dict(killed=sum(r['killed'] for r in res), total=len(res), mutants=res), f, indent=2)
        f.write('\n')
    print('MUTANTS killed', sum(r['killed'] for r in res), '/', len(res))


if __name__ == '__main__':
    main()
