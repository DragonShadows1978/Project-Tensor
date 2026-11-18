"""APA-P1-PF2 G7 mutants: registration and staging (CPU only).

  write          : write artifacts/apa_p1pf2/mutants/MUTANTS.json once (refuses if it exists), before any mutant is
                   staged, built or run (L41)
  stage <name>   : F2's stage() (scripts/apa_p1f2_mutants.py, imported, its MUT/ART pointed at this round): cp -a of
                   this worktree's tensor_cuda/{src,include,tensor_cuda,CMakeLists.txt} into mutants/<name>/tree, the
                   one registered change applied (old text must occur exactly once); MUTANT.json records both sha256s

The 16 mutants are the design's (DESIGN.md 5.1, registered as written by the order). Device mutants are rebuilt by
scripts/apa_p1pf2_build.py mutant <name> (F2's incremental build_mutant in a copy of Build PF's tree; m13 in a copy of
Build PFH's tree, because its named kill is the hooks bulk bits) and run by scripts/apa_p1pf2_mut.py on GPU. Python
mutants run on Build PF's module with the mutant Python. Expected failures come from the order and the design, never
from this round's runs (L47).
Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); F1/F2/F3 mutant staging (Project-Tensor 2026),
imported. Plumbing; no new technique.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_p1f2_mutants as f2m  # noqa: E402

ART = ROOT / 'artifacts/apa_p1pf2'
MUT = ART / 'mutants'
f2m.ART, f2m.MUT = ART, MUT
K = 'tensor_cuda/src/apa_spo/packed_prefill.cuh'
P = 'tensor_cuda/tensor_cuda/apa_packed.py'
BULK_LINE = '      bulk = __fmul_rn(__fmul_rn(__shfl_sync(FULL, x, 0), kn[j]), scale);\n'
CALL = ('        result = spo2.packed_prefill(q, k, v, kq._packed_tuple(), float(scale), delta, is_causal,\n'
        '                                     sinks, diagnostics, fmt=PACKED_FORMAT)\n')
G1 = 'g1'      # the G1 identity runner (scripts/apa_p1pf2_mut.py runs the named subset with the named assertion)

MUTANTS = [
    dict(name='m01_partition_restart', kind='device', file=K, design='1: running maximum restarts every 2,048 keys',
         old='    bulk_max = fmaxf(bulk_max, bulk);\n',
         new='    if ((j & 2047) == 0) bulk_max = -INFINITY;\n    bulk_max = fmaxf(bulk_max, bulk);\n',
         run=dict(check=G1, subset=dict(sub='grid', D=[64, 128], G=[1, 4], dt=['float32'], B=[1], S=[2049], L=[33],
                                        causal=[True], sinks=[False], capx=[0])),
         expected_failure='pf_i1_mask_equal',
         executes='every key j of rows with smax > 2048 passes the restart at j = 2048 (S 2049 cases)'),
    dict(name='m02_gt_for_ge', kind='device', file=K, design='2: > in place of >=',
         old='    const bool refine = bulk >= bulk_max - delta;\n',
         new='    const bool refine = bulk > bulk_max - delta;\n',
         run=dict(check='ties'), expected_failure='pf_i1_ties',
         executes='the compare runs for every key; the tie rows put a key exactly at the threshold'),
    dict(name='m03_dense_delta', kind='python', file=P, design='3: delta from the dense table in the packed branch',
         old=CALL,
         new=('        e_q = {64: 1.010645866394043, 128: 0.8716114163398743}[q.shape[-1]]   # dense table e_q\n'
              '        delta = derive_delta(epsilon, e_q)\n' + CALL),
         run=dict(check='g4_accept'), expected_failure='g4_route_record_packed_delta',
         executes='the public L > 1 branch (G4 acceptance row calls apa_selective_attention_sp with L = 33)'),
    dict(name='m04_causal_bound', kind='device', file=K, design='4: causal bound S-L+i',
         old='  const int smax = causal ? S - L + i + 1 : S;\n',
         new='  const int smax = causal ? S - L + i : S;\n',
         run=dict(check=G1, subset=dict(sub='grid', D=[64, 128], G=[1, 2], dt=['float32'], B=[1], S=[257],
                                        L=[33, 257], causal=[True], sinks=[False], capx=[0])),
         expected_failure='pf_i1_mask_equal', executes='every causal row'),
    dict(name='m05_kv_head_mod', kind='device', file=K, design='5: KV head taken as h % KVH',
         old='  const int64_t kvbh = (int64_t)b * KVH + h / group;\n',
         new='  const int64_t kvbh = (int64_t)b * KVH + h % KVH;\n',
         run=dict(check=G1, subset=dict(sub='grid', D=[64, 128], G=[2, 4], dt=['float32'], B=[1], S=[257], L=[33],
                                        causal=[False], sinks=[False], capx=[0])),
         expected_failure='pf_i1_mask_equal', executes='every row; at G > 1 head 1 maps to KV head 1, not 0'),
    dict(name='m06_norm_dropped', kind='device', file=K, design='6: norm multiply dropped (lane order site)',
         old=BULK_LINE, new='      bulk = __fmul_rn(__shfl_sync(FULL, x, 0), scale);\n',
         run=dict(check=G1, subset=dict(sub='grid', D=[128], G=[1, 4], dt=['float32'], B=[1], S=[257], L=[33],
                                        causal=[False], sinks=[False], capx=[0])),
         expected_failure='pf_i1_mask_equal', executes='the lane-order bulk (D128 cases)'),
    dict(name='m07_nibble_swap', kind='device', file=K, design='7: nibble order swapped (lane order site)',
         old=('          const int ix = (c[d / 2] >> (4 * (d % 2))) & 15;\n'
              '          x = __fmaf_rn(qd[t], table[ix], x);\n'),
         new=('          const int ix = (c[d / 2] >> (4 * (1 - d % 2))) & 15;\n'
              '          x = __fmaf_rn(qd[t], table[ix], x);\n'),
         run=dict(check=G1, subset=dict(sub='grid', D=[128], G=[1, 2], dt=['float32'], B=[1], S=[257], L=[33],
                                        causal=[False], sinks=[False], capx=[0])),
         expected_failure='pf_i1_mask_equal', executes='the lane-order bulk (D128 cases)'),
    dict(name='m08_rotation_transposed', kind='device', file=K, design='8: rotation indexed R[i,d]',
         old='rot[((int64_t)(h / group) * D + d) * D + i2]', new='rot[((int64_t)(h / group) * D + i2) * D + d]',
         run=dict(check=G1, subset=dict(sub='grid', D=[64, 128], G=[1, 4], dt=['float32'], B=[1], S=[257], L=[33],
                                        causal=[False], sinks=[False], capx=[0])),
         expected_failure='pf_i1_mask_equal', executes='the in-kernel rotation of every row'),
    dict(name='m09_s_stride', kind='device', file=K, design='9: codes addressed with an S stride when capacity > S',
         old='  const uint8_t* kc = codes + kvbh * (int64_t)capacity * (D / 2);\n',
         new='  const uint8_t* kc = codes + kvbh * (int64_t)S * (D / 2);\n',
         run=dict(check=G1, subset=dict(sub='grid', D=[64, 128], G=[1], dt=['float32'], B=[1, 2], S=[257], L=[33],
                                        causal=[False], sinks=[False], capx=[37])),
         expected_failure='pf_i1_mask_equal', executes='every row of KV head >= 1 (kvbh > 0) at capacity S + 37'),
    dict(name='m10_refined_keeps_bulk', kind='device', file=K, design='10: refined key keeps its bulk score',
         old='      score = __shfl_sync(FULL, ex, 0) * scale;\n', new='      score = bulk;\n',
         run=dict(check=G1, subset=dict(sub='grid', D=[64, 128], G=[1, 4], dt=['float32'], B=[1], S=[257], L=[33],
                                        causal=[True], sinks=[False], capx=[0])),
         expected_failure='pf_i2_equal', executes='every refined key (all keys at delta 1e30)'),
    dict(name='m11_exact_from_codes', kind='device', file=K,
         design='11: exact score read from the reconstruction or the rotated domain, not from k (L111)',
         old='        if (d < D) ex += qr[t] * ld<T>(k, koff + d);\n',
         new=('        if (d < D) ex += qr[t] * (table[(kc[(int64_t)j * (D / 2) + d / 2] >> (4 * (d % 2))) & 15] * '
              'kn[j]);\n'),
         run=dict(check=G1, subset=dict(sub='grid', D=[64, 128], G=[1, 4], dt=['float32'], B=[1], S=[257], L=[33],
                                        causal=[True], sinks=[False], capx=[0])),
         expected_failure='pf_i2_equal', executes='every refined key (all keys at delta 1e30)'),
    dict(name='m12_unrefined_dropped', kind='device', file=K, design='12: unrefined key dropped from the softmax',
         old='    float score = bulk;\n', new='    float score = -INFINITY;\n',
         run=dict(check=G1, subset=dict(sub='grid', D=[64, 128], G=[1, 4], dt=['float32'], B=[1], S=[257], L=[33],
                                        causal=[True], sinks=[False], capx=[0])),
         expected_failure='pf_i3_rows_within_tau', executes='every unrefined key at the registered delta'),
    dict(name='m13_rotation_order', kind='device', base='PFH', file=K,
         design='13: rotated query in a different reduction order (one-ulp drift)',
         old='    for (int i2 = 0; i2 < D; ++i2)\n', new='    for (int i2 = D - 1; i2 >= 0; --i2)\n',
         run=dict(check='g1h', subset=dict(sub='grid', D=[64, 128], G=[1, 4], dt=['float32'], B=[1], S=[257],
                                           L=[33], causal=[False], sinks=[False], capx=[0])),
         expected_failure='pf_i1_bulk_equal', executes='the in-kernel rotation of every row (hooks build: bulk dump)'),
    dict(name='m14_sinks_omitted', kind='device', file=K, design='14: sinks omitted',
         old='  if (sinks) {\n    const float sink = ld<T>(sinks, h);\n',
         new='  if (false && sinks) {\n    const float sink = ld<T>(sinks, h);\n',
         run=dict(check=G1, subset=dict(sub='grid', D=[64, 128], G=[1, 4], dt=['float32'], B=[1], S=[257], L=[33],
                                        causal=[True], sinks=[True], capx=[0])),
         expected_failure='pf_i2_equal', executes='every row of a case with sinks'),
    dict(name='m15_silent_fallback', kind='python', file=P,
         design='15: silent fallback: reconstruct and call the dense kernel',
         old=CALL,
         new=('        result = importlib.import_module(__package__)._C.apa_selective_attention_sp(\n'
              '            q, k, kq._dense_reconstruction(), v, float(scale), delta, is_causal, sinks, diagnostics)\n'),
         run=dict(check='g5'), expected_failure='g5_no_key_shaped_transient',
         executes='the public L > 1 branch (G5 census calls apa_selective_attention_sp with L > 1)'),
    dict(name='m16_nan_norm_as_zero', kind='device', file=K, design='16: NaN norm treated as zero (key skipped)',
         old=BULK_LINE, new=BULK_LINE + '      if (isnan(kn[j])) bulk = -INFINITY;\n',
         run=dict(check='i1n', subset=dict(D=[128], kind=['nan_k', 'inf_k'])), expected_failure='pf_i1n_ok',
         executes='the lane-order bulk of a key with a NaN stored norm (D128 non-finite k cases)'),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write():
    MUT.mkdir(parents=True, exist_ok=True)
    path = MUT / 'MUTANTS.json'
    src = {K: (ROOT / K).read_text(), P: (ROOT / P).read_text()}
    for m in MUTANTS:
        assert src[m['file']].count(m['old']) == 1, (m['name'], 'old text must occur exactly once')
    data = dict(order='orders/APA_P1PF2_PACKED_PREFILL.md (5c9b9584), G7: the design\'s 16 mutants as written',
                written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                note=('Written before any mutant is staged, built or run (L41). Each mutant is one change in a patched '
                      'copy. Python mutants run on Build PF\'s module with the mutant Python; device mutants are '
                      'rebuilt incrementally in a copy of Build PF\'s tree (m13: Build PFH\'s). A mutant is killed iff '
                      'its expected_failure assertion fails. Expected failures come from the order and the design '
                      '(DESIGN.md 5.1 mutant table), never from this round\'s runs (L47).'),
                source_sha256={K: sha(ROOT / K), P: sha(ROOT / P)},
                mutants=MUTANTS)
    with path.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    print(path, sha(path))


if __name__ == '__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    {'write': lambda: write(), 'stage': lambda: f2m.stage(sys.argv[2])}[sys.argv[1]]()
