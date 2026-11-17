"""APA-P1-F2 operand censuses (L99 seat premises): TQ4_CENSUS.json (R2), INDEX_CENSUS.json (R3),
HELPER_CENSUS.json (R4), written to artifacts/apa_p1f2/.

Each census carries the grep it started from (L82: `rg -n 'pack_tq|tq4' tensor_cuda/src/apa_spo`,
`rg -n 'int capacity|int dst_s|int S\\b' tensor_cuda/src/apa_spo`, the spo2 m.def list) with its raw
output, and the seat's row-by-row classification from reading the code at the cited lines. Line numbers
are recomputed from the working tree when the script runs (the `site` strings are searched for).
Prior art: none; a source census (reasoning evidence class). CPU only.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/apa_p1f2'
SPO = 'tensor_cuda/src/apa_spo'


def rg(pattern, *paths):
    p = subprocess.run(['rg', '-n', pattern, *paths], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode in (0, 1), p.stderr
    return [l[:220] for l in p.stdout.splitlines()]


def where(path, needle):
    lines = (ROOT / path).read_text().splitlines()
    hits = [i + 1 for i, l in enumerate(lines) if needle in l]
    assert hits, (path, needle)
    return f'{path}:{hits[0]}'


def tq4():
    rows = [
        dict(site=where(f'{SPO}/packed.cuh', 'template<class T> __global__ void prepare('), kernel='spq1::prepare<T>',
             role='OWNS the TQ4 key quantization of the packed route: per key row norm (sequential add/mul, sqrt_rn), '
                  'unit = k/(norm+1e-12), rotated coordinates; writes norms[row]',
             reached_by='ApaSpq1Workspace.append (tq4) -> _spo2.packed_prepare; PackedSpKeys (public default route) '
                        'owns an ApaSpq1Workspace(fmt="tq4"); raw _spo2.packed_prepare',
             r2_fix='YES: a row with any non-finite element stores a quiet NaN norm; the shared norm used for the '
                    'unit vector is unchanged, so rotated outputs are bit-identical; finite rows unchanged'),
        dict(site=where(f'{SPO}/packed.cuh', '__global__ void pack_tq('), kernel='spq1::pack_tq',
             role='OWNS the code assignment (15 boundaries, NaN > bnd is false -> code 0) and copies the norm '
                  'unchanged: scales[dst]=norms[key]; no norm of its own',
             reached_by='ApaSpq1Workspace.append (tq4) -> _spo2.packed_append(tq=True); raw _spo2.packed_append',
             r2_fix='NO CHANGE NEEDED: it stores prepare\'s norm verbatim, so the NaN carried by prepare reaches '
                    'scales[]. A norm made non-finite elsewhere (caller-supplied norms on the raw helper) is '
                    'stored as given (caller contract)'),
        dict(site=where(f'{SPO}/packed.cuh', 'template<class T> __global__ void pack_int4_rows('),
             kernel='spq1::pack_int4_rows<T>', role='INT4 (not TQ4) key quantizer; F1 already stores a NaN scale for a '
                                                     'row with any non-finite element', r2_fix='out of R2 (INT4); unchanged'),
        dict(site=where(f'{SPO}/eb3.cuh', 'std::vector<NDArray> apa_eb3_run('), kernel='EB3 (rotate_q, bulk_copy, bulk_swap, spq6::bulk_mma_tq)',
             role='READS caller codes + norms (packed tuple from ApaSpq1Workspace); rotates only the query; owns no '
                  'key quantizer', r2_fix='none (reader)'),
        dict(site=where(f'{SPO}/spq6.cuh', 'apa_spq6_tq4_tc('), kernel='SPQ6 tc tq4 (spq6::bulk_mma_tq)',
             role='READS caller codes + norms; rotates the query via apa_spq1_rotate; owns no key quantizer',
             r2_fix='none (reader)'),
        dict(site=where(f'{SPO}/spq4.cuh', 'std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq4_run('),
             kernel='SPQ4 dispatch (tq4 -> SPQ6 / r2)', role='READS caller codes; dispatches; owns no key quantizer',
             r2_fix='none (reader)'),
        dict(site=where(f'{SPO}/packed_launcher.cuh', 'std::pair<NDArray,NDArray> apa_spq1_prepare('),
             kernel='host apa_spq1_prepare / apa_spq1_pack / apa_spq1_run',
             role='LAUNCH the two kernels above (prepare_launcher, pack); apa_spq1_run reads codes; no quantizer of '
                  'their own', r2_fix='none (launchers)'),
        dict(site=where('tensor_cuda/src/apa_spo/eb4.cuh', 'std::vector<NDArray> apa_eb4_run('), kernel='EB4 (eb4::bulk_stats)',
             role='READS caller codes + norms (not in L99\'s five-file list; checked because it is a packed entry)',
             r2_fix='none (reader)'),
        dict(site=where('tensor_cuda/src/apa_spo/eb2b.cuh', 'std::vector<NDArray> apa_eb2b_run('), kernel='EB2b',
             role='READS caller codes + norms; eb2b::quantize is the int8 V quantizer, not a key quantizer',
             r2_fix='none (reader)'),
        dict(site='tensor_cuda/tensor_cuda/quant.py (_quantize_keys)', kernel='Python TurboQuant dense quantizer',
             role='the DENSE route\'s quantizer (fp32 reconstruction); not owned by any packed entry; read-only here',
             r2_fix='out of scope (dense comparator)'),
    ]
    return dict(census='R2 TQ4 key quantizers owned by packed entries', start_grep=dict(
        command="rg -n 'pack_tq|tq4' tensor_cuda/src/apa_spo", output=rg('pack_tq|tq4', SPO)),
        files_with_matches=sorted({l.split(':')[0] for l in rg('pack_tq|tq4', SPO)}),
        owners=['spq1::prepare<T> (norm)', 'spq1::pack_tq (codes; norm copied)'],
        conclusion='One TQ4 key quantizer exists for every packed entry: prepare (norm, rotation) + pack_tq '
                   '(codes). eb3.cuh, spq4.cuh, spq6.cuh (and eb4.cuh, eb2b.cuh) only read codes and norms made by '
                   'it; packed_launcher.cuh only launches it. The census does not widen R2: the one fix in '
                   'prepare covers the public route and every raw packed entry fed by ApaSpq1Workspace.',
        rows=rows)


def index():
    rows = [
        dict(entry='PackedSpKeys(k, capacity) / quantize_sp2_keys(..., capacity=)', site=where(
            'tensor_cuda/tensor_cuda/apa_packed.py', 'check_native_range(B, KV, S, D, capacity, S)'),
             receives='capacity -> ApaSpq1Workspace -> codes.shape[2] -> native int capacity/dst_s; B*KV*S -> int '
                      'prepare grid', guard='Python check_native_range before the workspace allocates (ValueError)'),
        dict(entry='PackedSpKeys.append', site=where('tensor_cuda/tensor_cuda/apa_packed.py',
                                                     'check_native_range(self.batch, self.kv_heads, S'),
             receives='B*KV*(new rows) -> int prepare grid', guard='Python, before the generation moves'),
        dict(entry='ApaSpq1Workspace(k, capacity, fmt) / append', site=where('tensor_cuda/tensor_cuda/__init__.py',
                                                                             'apa_packed.check_native_range(B, H, S, D, int(capacity), S, fmt)'),
             receives='same as PackedSpKeys', guard='Python check_native_range before zeros() and before each native append'),
        dict(entry='apa_spq1_buffer_check (every packed attention + append entry)',
             site=where(f'{SPO}/packed_launcher.cuh', '"capacity (codes.shape[2])"'),
             receives='codes.shape[2] -> int capacity (spq1::bulk/bulk_shared/pack_tq), int dst_s (pack_int4_rows), '
                      'int capacity (spq4::bulk_mma, spq6::bulk_mma_tq, eb3::bulk_copy/bulk_swap, eb4::bulk_stats); '
                      'each launcher does `int capacity=(int)codes.shape[2]` or passes codes.shape[2] to an int '
                      'parameter', guard='native ValueError naming capacity, before any allocation or launch'),
        dict(entry='apa_spq1_rotate (packed_rotate)', site=where(f'{SPO}/packed_launcher.cuh', '"B*H (rotate grid rows)"'),
             receives='grid q.shape[0]*q.shape[1] (dim3 x, kernel int row); H, KVH, D int',
             guard='per-axis extents 1..INT_MAX and B*H <= INT_MAX (ValueError) before the launch'),
        dict(entry='apa_spq1_prepare (packed_prepare)', site=where(f'{SPO}/packed_launcher.cuh', '"B*KV*N (prepare grid rows)"'),
             receives='int B,H,S,D,N from shape; grid B*H*N computed in int; kernel int row',
             guard='per-axis extents 1..INT_MAX before narrowing; B*KV*N <= INT_MAX before allocation'),
        dict(entry='apa_spq1_pack (packed_append)', site=where(f'{SPO}/packed_launcher.cuh', '"B*KV",apa_spo_product'),
             receives='int S,D,BH,N; pack_tq int BH,N,capacity,start and grid (BH*N*D/2+127)/128; pack_int4_rows '
                      'int src_s,dst_s,row0,nrows and grid (BH*N+3)/4',
             guard='extents of k/codes/scales, capacity (buffer_check), B*KV and the grid block count <= INT_MAX'),
        dict(entry='apa_spq1_int4_stats (int4_stats)', site=where(f'{SPO}/packed_launcher.cuh', 'apa_spq1_int4_stats: launch range'),
             receives='all axes and B*H', guard='already complete since F1 (ValueError); unchanged'),
        dict(entry='apa_spq1_core_check (spo2 run / packed r2 run)', site=where(f'{SPO}/launcher.cuh', 'apa_spo_require_key_range(entry,S)'),
             receives='per-axis int (existing RuntimeError, kept), B*H*L (existing), S+2047 in '
                      'apa_int4_fixed_partition_layout(int) and nc*256 (int pk)',
             guard='new: S + 2047 <= INT_MAX (ValueError); capacity via buffer_check on the packed path'),
        dict(entry='apa_spq4_tc (tc int4)', site=where(f'{SPO}/spq4.cuh', 'apa_spo_require_key_range("apa_spq4 tc int4",S)'),
             receives='int capacity, (int)S, nc, pk', guard='existing B*H <= INT_MAX/4; new S + 2047; capacity via buffer_check'),
        dict(entry='apa_spq6_tq4_tc (tc tq4)', site=where(f'{SPO}/spq6.cuh', 'apa_spo_require_key_range("apa_spq6 tc tq4",S)'),
             receives='int capacity, (int)S, nc, pk', guard='existing B*H <= INT_MAX/4; new S + 2047; capacity via buffer_check'),
        dict(entry='apa_eb3_run', site=where(f'{SPO}/eb3.cuh', 'apa_spo_require_key_range("apa_eb3",S)'),
             receives='int capacity, (int)S, nc, pk, rows', guard='existing B*H <= INT_MAX/4; new S + 2047; capacity via buffer_check'),
        dict(entry='apa_eb2b_run', site=where(f'{SPO}/eb2b.cuh', 'S>65536'),
             receives='int B,H,KVH,S; codes.shape[2] as int capacity', guard='existing S <= 65536, B*H <= INT_MAX/4; capacity via buffer_check'),
        dict(entry='apa_eb4_run', site=where(f'{SPO}/eb4.cuh', 'int capacity=(int)codes.shape[2]'),
             receives='int capacity, S <= 65536', guard='existing S and B*H bounds; capacity via buffer_check'),
        dict(entry='apa_eb2b_values', site=where(f'{SPO}/eb2b.cuh', '"numel(v)/8+127 (int value-group index)"'),
             receives='int total=numel/8; kernel int i up to total+127', guard='new: numel/8+127 <= INT_MAX; positive extents'),
        dict(entry='apa_eb2b_quantize', site=where(f'{SPO}/eb2b.cuh', 'v.numel()/128>std::numeric_limits<int>::max()'),
             receives='rows=numel/128 int', guard='existing; positive extents added (R4)'),
    ]
    readers = rg(r'_packed_tuple\(\)|\.packed\b|\.codes\b|\.scales\b|\.norms\b', 'tensor_cuda/tensor_cuda/apa_packed.py',
                 'tensor_cuda/tensor_cuda/apa_sp2.py')
    return dict(census='R3 native index types receiving a capacity, extent or index product',
                start_grep=dict(command="rg -n 'int capacity|int dst_s|int S\\b' tensor_cuda/src/apa_spo",
                                output=rg(r'int capacity|int dst_s|int S\b', SPO)),
                bound='INT_MAX = 2147483647 (the int index); refusal = ValueError naming the operand',
                widening='none: every site is refused, nothing widened to 64 bits',
                rows=rows,
                L32_bounded_buffer=dict(
                    buffer='PackedSpKeys / ApaSpq1Workspace codes [B,KV,capacity,D/2] and norms [B,KV,capacity]',
                    full_behaviour='append beyond capacity raises ValueError before any launch; object unchanged',
                    readers_by_grep=readers,
                    readers='packed_attention passes kq._packed_tuple() (codes/norms at capacity) plus exact K/V with '
                            'visible S rows; every packed kernel indexes rows bh*capacity + j for j < S (the visible '
                            'rows, k.shape[2]); _dense_reconstruction slices [:S]; owned_buffers/codes/norms '
                            'properties return the buffers to the caller (who reads [:, :, :visible_rows], as LF4).'))


def helpers():
    defs = rg(r'm\.def\("', 'tensor_cuda/src/bindings.cpp')
    module = (ROOT / 'tensor_cuda/src/bindings.cpp').read_text()
    module = module[module.index('PYBIND11_MODULE(_tensor_cuda_spo2'):]
    names = re.findall(r'm\.def\("([A-Za-z0-9_]+)"', module)
    rows = {
        'run': ('attention entry (D1 dense-K route)', 'refuses empty extents before any launch (core_check, '
                'RuntimeError "dimensions must be positive and fit int", F1 R11 dense-entry parity); unchanged'),
        'packed_prepare': ('RAW HELPER', 'F1: B=0/KV=0 reached a zero-grid launch (CUDA invalid configuration, lead '
                           'repro p1rev2-03). F2: ValueError (positive extents) before the launch'),
        'packed_rotate': ('RAW HELPER', 'F1: B=0/H=0 reached a zero-grid launch (lead repro p1rev2-03). F2: ValueError'),
        'packed_append': ('RAW HELPER', 'F1: zero B/KV with N>0 reached a zero-grid launch (REV2 F05). F2: ValueError'),
        'packed_run': ('attention entry (r2 / tc)', 'refuses empty extents before any launch (core_check / '
                       'apa_spo_require_extents, RuntimeError, F1 R11); unchanged'),
        'int4_stats': ('RAW HELPER (diagnostic)', 'already ValueError for every non-positive axis since F1 R6; unchanged'),
        'eb2b_quantize': ('RAW HELPER', 'F1: numel()==0 refused as RuntimeError before launch. F2: positive extents '
                          'ValueError first'),
        'eb2b_values': ('RAW HELPER', 'F1: apa_spo_require_extents RuntimeError before launch. F2: positive extents '
                        'ValueError first'),
        'eb2b_run': ('attention entry', 'refuses empty extents before any launch (RuntimeError, F1 R11); unchanged'),
        'eb3_run': ('attention entry', 'refuses empty extents before any launch (RuntimeError, F1 R11); unchanged. '
                    'R4 timeline/G pre-check added (below)'),
        'eb3_resources': ('diagnostic, no operands', 'no extents'),
        'eb4_run': ('attention entry', 'refuses empty extents before any launch (RuntimeError, F1 R11); unchanged'),
        'eb4_resources': ('diagnostic, no operands', 'no extents'),
        '_p1_test_run': ('hooks build only (not in Build F2)', 'not compiled in the default build'),
        '_p1_test_merge_tickets': ('hooks build only (not in Build F2)', 'not compiled in the default build'),
    }
    table = [dict(name=n, kind=rows[n][0], extent_behaviour=rows[n][1]) for n in names]
    assert set(names) == set(rows), (names, rows.keys())
    timeline = dict(
        site=where(f'{SPO}/eb3.cuh', 'if(timeline && G!=2 && G!=4)'),
        templates='byG: a timeline select has ON=true, so REG is false and only the G=2/G=4 cases launch; variants '
                  'allowed a timeline are 0 (registered select options, ON), 1 (occupancy-only) and 6 (baseline '
                  'footprint, sel=2); ablation variants were already limited to G=2,4 before any launch',
        timeline_G=[2, 4], refused_G=[3, 5, 6, 7, 8],
        f1_behaviour='passed every early check, launched rotation and bulk, then byG threw "G not instantiated for '
                     'this variant" (REV2 F05)',
        f2_behaviour='ValueError before the ticket lock, the arena and any launch')
    return dict(census='R4 raw spo2 helpers and EB3 timeline templates', spo2_mdefs=table,
                grep=dict(command='rg -n \'m\\.def\\("\' tensor_cuda/src/bindings.cpp (spo2 module part)',
                          output=[d for d in defs]),
                raw_helpers=[r['name'] for r in table if r['kind'].startswith('RAW HELPER')],
                interpretation='"raw spo2 helper" = a non-attention native entry of the spo2 module (rotate, '
                               'prepare, append, int4_stats, eb2b_quantize, eb2b_values). The five attention entries '
                               'already refuse empty extents on the host before any launch; their exception type '
                               '(RuntimeError, chosen by F1 R11 for dense-entry parity) is left unchanged (deviation '
                               'D2 in REPORT.md asks the lead to rule).',
                eb3_timeline=timeline)


def main():
    OUT.mkdir(exist_ok=True)
    for name, fn in (('TQ4_CENSUS.json', tq4), ('INDEX_CENSUS.json', index), ('HELPER_CENSUS.json', helpers)):
        data = fn()
        data['written_by'] = 'scripts/apa_p1f2_census.py'
        data['source_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(
            str(x.relative_to(ROOT)) for x in (ROOT / SPO).glob('*.cuh'))}
        path = OUT / name
        path.write_text(json.dumps(data, indent=2) + '\n')
        print(name, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    sys.exit(main())
