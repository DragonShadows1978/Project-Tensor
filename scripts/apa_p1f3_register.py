"""APA-P1-F3 S3/S6 registration: write artifacts/apa_p1f3/EXPECTED.json ONCE (refuses if it exists or if any
Build F3 receipt exists), before the first Build F3 and before any GPU line.

Prior art: pre-registration of outcomes before measurement (house rules; Project-Tensor P1 rounds, 2026).
Plumbing only; no prior art known to me beyond that. CPU only.
"""
from pathlib import Path
import datetime
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_p1f3'
F2_SHA = '6f03b2cdd5d7f9bee4456ed34759c6c710cccfab9a50f9ac0fc6a678c076f579'
LF6A = json.loads((ROOT / 'artifacts/apa_p1f2/gpu/LF6a/RESULT.json').read_text())['files']
LF6B = json.loads((ROOT / 'artifacts/apa_p1f2/gpu/LF6b/RESULT.json').read_text())['files']
TQ4_BF16 = ['r2_tq4', 'tc_tq4', 'eb2b', 'eb3', 'eb4']

EXPECTED = dict(
    order='orders/APA_P1F3_REV3_FIXES.md (25f59b9c)',
    note=('Written once, before the first Build F3 and before any GPU line (S3, S6, L35). Never amended; every '
          'F3 build and mutant build is compared against this file.'),
    sass=dict(
        comparison=('Build F3 vs Build F2 (artifacts/apa_p1f2/arms/F2, module sha256 ' + F2_SHA + '), per kernel, '
                    'exact Function-block text of cuobjdump -sass (sm_89), by scripts/apa_p1f1_sass.py compare_builds '
                    '(after F2 R7), imported unchanged'),
        expected_changed={
            "R2' (P1R3-F02) spq1::prepare stores a quiet NaN norm for any key row whose computed norm is non-finite "
            "(non-finite element or fp32 overflow of the sum of squares) (packed.cuh); DISPATCH_FLOAT instantiates "
            "float, __half, __nv_bfloat16": [
                'void tc::spq1::prepare<float>', 'void tc::spq1::prepare<__half>',
                'void tc::spq1::prepare<__nv_bfloat16>']},
        expected_removed={},
        expected_added={},
        not_expected_to_change=("Every other kernel. R1' is Python only (apa_packed.py lookup). R3' is host code "
                                "only: the named-extent guard and the int64 product check (launcher.cuh), their calls "
                                "at the top of the packed attention entries (packed_launcher.cuh apa_spq1_run, "
                                "spq4.cuh apa_spq4_run/apa_spq4_tc, spq6.cuh apa_spq6_tq4_tc, eb2b.cuh apa_eb2b_run, "
                                "eb3.cuh apa_eb3_run, eb4.cuh apa_eb4_run), and N = S - start formed in int64 in the "
                                "host launchers apa_spq1_prepare and apa_spq1_pack. No __global__ function other than "
                                "spq1::prepare is edited."),
        non_code_sources_of_difference=[
            dict(source='source path',
                 rule=('EQUAL, not mapped. Build F2 compiled this worktree (artifacts/apa_p1f2/build_F2/BUILD.json '
                       'source_root = /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/'
                       'tensor_fork_apa_wt_p1f2) and Build F3 compiles the same path, so nvcc path-derived names '
                       '(_GLOBAL__N__<8 hex>_, __nv_static_<n>__<8 hex>_) must be equal; any difference in them is '
                       'an unmapped difference (G4 FAIL).')),
            dict(source='build directory',
                 rule=('MAPPED: the per-build _cu_<8 hex> symbol suffix with an optional _<n> is mapped bijectively '
                       'to _cu_BUILD (apa_p1f1_sass.read_blocks SUFFIX, as F1 and F2 used it). Build F2 built in '
                       'artifacts/apa_p1f2/build_F2, Build F3 builds in artifacts/apa_p1f3/build_F3.')),
            dict(source='TMPDIR of the build (/tmp/apap1f2_* vs /tmp/apap1f3_*) and the cmake_source directory',
                 rule=('NOT MAPPED separately. nvcc temporary names and the CMakeLists location can only reach kernel '
                       'symbols through the per-build _cu_ suffix, which the build-directory rule already maps; any '
                       'other difference they caused would be an unmapped difference.')),
            dict(source='compiler, flags and dump tool',
                 rule=('EQUAL by construction, checked from receipts: the P1-D adapter replacements and configure '
                       'flags are imported (F2 build script -> apa_p1f1_build.p1d_adapter_and_flags) and recorded in '
                       'BUILD.json (p1d_flags must equal Build F2\'s); the same cuobjdump dumps both modules. As a '
                       'check of the dump tool, the fresh dump of Build F2\'s module must have the sha256 of F2\'s '
                       'own dump (artifacts/apa_p1f2/sass/F2.sass, 3b7efd2a...). Not mapped.')),
        ]),
    builds=dict(
        F3=('this worktree, default configuration (hooks off), P1-D adapter and flags imported through '
            'scripts/apa_p1f2_build.py (scripts/apa_p1f3_build.py), into artifacts/apa_p1f3/build_F3[_r<n>]'),
        F2=('artifacts/apa_p1f2/arms/F2, module sha256 ' + F2_SHA + ', checked before every use; refuse on mismatch'),
        mutants=('m02 (packed.cuh) and m03 (eb2b.cuh): incremental rebuilds in a copy of Build F3\'s tree, by '
                 'scripts/apa_p1f2_build.py build_mutant imported (ART and base receipt pointed at F3)'),
        F3H='NOT BUILT: the diff changes no code compiled only with TC_APA_P1_TEST_HOOKS'),
    cpu=dict(
        S4=("tensor_cuda/tests/test_apa_p1f3_cpu.py: R1' (stateful hand-built mapping at lookup and at attention: "
            "the recording backend receives the pinned e_q or the call is refused before any device call; loaded "
            "pinned table unchanged); R3' (host-level fake: the real guard blocks of launcher.cuh and the real "
            "EB2b prologue text of eb2b.cuh (its first statement up to the GQA check) compiled with g++ -O0 -fwrapv "
            "against a stub NDArray (shape only), TC_APA_SP=1: EB2b's B*H at its bound 536870911 not refused by the "
            "B*H check, one past refused with ValueError naming apa_eb2b and B*H, 65536*65536 refused the same way, "
            "an extent 2**31 refused naming entry, operand and axis, a zero extent raising the RuntimeError; plus "
            "source order: each packed attention entry calls the guard first, before any launch). The native "
            "refusals on real entries are LF3 (DLPack re-imports with large extents, GPU)."),
        per_id=('A1 REGISTRATION cpu_g3_files (F2\'s CPU G3 set) + test_apa_p1d_policy.py + test_apa_p1f1_cpu.py + '
                'test_apa_p1f2_cpu.py; before = Build F2 arm (artifacts/apa_p1f2/arms/F2) running /tmp/apap1f3_tree0 '
                '(git archive 25f59b9c, this branch as committed); after = Build F3 arm running this worktree; '
                'CUDA_VISIBLE_DEVICES empty; APA_P1F2_BASE_TREE=/tmp/apap1f3_treeF0 (git archive d01118b4) on both '
                'sides for test_apa_p1f2_cpu.py\'s R6 probe'),
        intended_changes=dict(ruling='none expected', ids=[])),
    gpu_lines=dict(
        common=('Each line: precheck before the lease (Build F2 module == ' + F2_SHA + '; Build F3 module == its '
                'receipt and the working tree engine source map == the receipt\'s; mutant modules == their '
                'receipts), the disk rail (stop under 40 GB free), one FORGE_GPU_DRY_RUN=1 dry run (exit 78), then '
                'forge-gpu-run --min 30 --max <=1800 --label p1f3-<line> -- timeout <=1750s python3 -B '
                'scripts/apa_p1f3_gpu.py <mode> --out artifacts/apa_p1f3/gpu/<line>, from the worktree root. Exit 76: '
                'nothing ran; retry unchanged after >= 5 min, at most 6 times. Exit 124 = timeout, never a pass. '
                'Bitwise = np.array_equal on same-width unsigned views after dtype and shape checks, both sides '
                'nonzero and non-constant (L19). A row = one (batch, head, query) output vector, non-finite if any '
                'element is. "No launch" = F1\'s LD_PRELOAD cudaLaunchKernel counter (F1 SHIM_SOURCE, compiled into '
                '/tmp/apap1f3_shim) reads 0 across the call.'),
        LF1=dict(label='p1f3-LF1', builds='Build F2 vs Build F3, one child process per build',
                 cases=('F2\'s LF1 cases unchanged = F1\'s (scripts/apa_p1f1_gpu.py lf1_arm, imported, with its '
                        'ART/receipt globals pointed at each build): P1-D\'s L1 grid on the public route and every raw '
                        'packed entry on REV1\'s rule shapes'),
                 gate=('G1 PASS iff the key sets are identical, every array is bitwise equal, and every compared '
                       'array is informative on both sides (F2 measured 6,922 of 6,922)')),
        LF2=dict(label='p1f3-LF2', build='F3 only, one process',
                 inherited=('F2\'s LF2 cases unchanged, by scripts/apa_p1f2_gpu.py lf2_run imported (entries, '
                            'comparators, declarations and gates exactly as F2\'s G2)'),
                 overflow_cases=dict(
                     construction=('one key element set to float32(1e20) (finite in fp32 and bf16; its square '
                                   'overflows fp32), all other inputs finite'),
                     ovf_D128_qpos=('F1 Case(seed 11, B1 H8 KV2 S1024 D128, plant=False) with k[0,0,100,5]=1e20, '
                                    'q unchanged; dtypes bfloat16 (entries ' + ', '.join(TQ4_BF16) + ') and float32 '
                                    '(r2_tq4, public_default)'),
                     ovf_D128_qneg='as ovf_D128_qpos with q negated (q = -q)',
                     ovf_D64_qpos=('F1 Case(seed 11, B1 H8 KV2 S1024 D64, plant=False) with k[0,0,100,5]=1e20; '
                                   'bfloat16 (r2_tq4, tc_tq4) and float32 (r2_tq4, public_default); eb2b, eb3 and eb4 '
                                   'are D128-only and do not take D64'),
                     ovf_D64_qneg='as ovf_D64_qpos with q negated',
                     r01_ovf_qneg=('REV3 r01 exactly: fp32 B1 H1 KV1 S3 D64, q=-ones, k zeros with k[0,0,1,0]=1e20, '
                                   'v=arange(192)/256, scale .125, epsilon .5; public default route vs the dense '
                                   'opt-out route and the dense entry'),
                     r01_ovf_qpos='as r01_ovf_qneg with q=+ones'),
                 comparators=('every raw entry and the public default route vs (a) dense_entry: '
                              '_C.apa_selective_attention_sp(q,k,kq=k,v,scale,2.0) (F2\'s comparator) and (b) '
                              'dense_sp_quantized: _C.apa_selective_attention_sp(q,k,kq=quant._quantize_keys(k, '
                              '*quant._tables(D,4,KV,True,\'cuda\')),v,scale,2.0), the dense SP quantizer; the public '
                              'default route also vs (c) dense_opt_out: quantize_sp2_keys(k,4,storage=\'dense\') + '
                              'the dense registered table (F2\'s and REV3\'s comparator)'),
                 declarations=('L19, before the run: for every overflow case the dense_entry side (exact k, '
                               'finite scores) is declared all_finite and only containment and no-zero-row are gated '
                               'against it; for the H8 overflow cases the dense_sp_quantized side must have at least '
                               'one non-finite and one finite row (0 < n < 8), and so must dense_opt_out for the fp32 '
                               'H8 cases; the r01_ovf cases are declared single_row (H1)'),
                 gates=('G2 as F2\'s: G2a NaN row-set equality for NaN in q, k or v (inherited cases only; overflow '
                        'cases carry no NaN input); G2b containment (entry non-finite rows contain the comparator\'s) '
                        'and G2c no all-zero row where the comparator is non-finite, in every case and against every '
                        'comparator; loud rows listed, not gated; every case informative (L19) or declared'),
                 l41_record=('for every overflow case the stored norm of the overflow key is recorded from the '
                             'packed state (expected quiet NaN on Build F3; the m02 mutant stores Inf)')),
        LF3=dict(label='p1f3-LF3', build='F3 in one process under the launch counter; Build F2 boundary rows (no gate) '
                 'each in its own child process',
                 inputs=('large extents by DLPack re-import over small real buffers (F1 reimport(), imported; F2 LF3\'s '
                         'method); every other operand geometry-consistent'),
                 rows=dict(
                     eb2b_BH_at_bound=('eb2b_run with q [1,536870911,1,128] (B*H = INT_MAX/4 = 536870911), KV 1: the '
                                       'B*H check accepts it; the call is refused by the later GQA check with '
                                       'RuntimeError "apa_eb2b: geometry/GQA/range mismatch" (536870911 has no factor '
                                       '2..8, so no valid GQA geometry exists at the bound); 0 launches'),
                     eb2b_BH_past_bound=('q [1,536870912,1,128], KV 134217728 (G 4): ValueError naming "apa_eb2b" and '
                                         '"B*H"; 0 launches'),
                     eb2b_BH_2p32=('q [65536,65536,1,128], KV 8192 (G 8; the int product wraps to 0): ValueError '
                                   'naming "apa_eb2b" and "B*H"; 0 launches'),
                     extent_rows=('S = 2**31 on k and v (codes and scales capacity 2**31): packed_run r2 fp32 (KV2 '
                                  'H8 D128), packed_run tc tq4 and tc int4 bf16 (KV2 H8 D128), eb2b, eb3, eb4 (bf16 '
                                  'KV2 H8 D128); and q B = 2**31 on packed_run r2: ValueError naming the entry '
                                  '("apa_spq4_run (packed_run)", "apa_eb2b", "apa_eb3", "apa_eb4") and the operand '
                                  '("operand \'k\' axis 2" / "operand \'q\' axis 0"); 0 launches'),
                     zero_extent_rows=('q B = 0 on packed_run r2, eb2b, eb3 and eb4: RuntimeError "dimensions must be '
                                       'positive" (F1\'s contract, F2\'s D2), not ValueError; 0 launches'),
                     r1p_stateful_mapping=('public route with a hand-built PackedRegisteredMargins whose dict-subclass '
                                           'mapping returns the pinned values to the first 4 reads and 0.0 after: the '
                                           'call runs with the pinned e_q (route e_q == pinned) and its output is '
                                           'bitwise equal to the loaded pinned table\'s on the same inputs, both '
                                           'informative'),
                     controls=('packed_rotate and the public default route before every refusal and again after: '
                               'bitwise equal, informative, launches > 0')),
                 gate='G3 (LF3 part): every row as stated; Build F2\'s behaviour on each input recorded in BOUNDARY.json'),
        LF6=dict(label='p1f3-LF6a / p1f3-LF6b', builds=('Build F2 (tests from /tmp/apap1f3_tree0 = git archive '
                                                         '25f59b9c) vs Build F3 (this worktree)'),
                 files=dict(LF6a=LF6A, LF6b=LF6B),
                 expected=('per-ID outcomes identical; intended changes: none (none of the 44 files uses a stateful '
                           'hand-built margin mapping, a finite key whose squared norm overflows fp32, an EB2b B*H past '
                           'its bound, or an attention extent above INT_MAX)')),
        MUT=dict(label='p1f3-MUT', mutants=('m02 (device) against LF2\'s overflow cases and m03 (host, rebuilt) '
                                            'against LF3\'s eb2b rows, each with its named assertion in '
                                            'mutants/MUTANTS.json (m01 and m03 also run on S4, CPU)'))),
)


def main():
    path = ART / 'EXPECTED.json'
    assert not path.exists(), 'EXPECTED.json is written once'
    assert not list(ART.glob('build_F3*')), 'a Build F3 exists; EXPECTED.json can no longer be written'
    data = dict(EXPECTED, written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    ART.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    print(path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    sys.exit(main())
