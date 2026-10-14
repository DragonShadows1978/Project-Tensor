"""P1T-M3 G3 mutant builder: copies <worktree>/tc_signal (sources and tests; no build dir, no built module) to
/tmp/p1t_m3_mut_<id>/tc_signal and applies one registered defect per mutant. Every replacement must match exactly once
(assert), and a unified diff is written to artifacts/p1t_m3/mutants/<id>/mutant_<id>.diff as the receipt.

Usage: python3 make_mutants.py <worktree>/tc_signal <worktree>/artifacts/p1t_m3/mutants

Prior art: mutation testing (DeMillo, Lipton & Sayward 1978, "Hints on test data selection"); the five defects are the
order's registered list (P1T-M3 G3 (a)-(e)). The builder mirrors M2's artifacts/p1t_m2/mutants/make_mutants.py.
"""
import difflib
import shutil
import sys
from pathlib import Path

SRC = Path(sys.argv[1])            # <worktree>/tc_signal
OUT = Path(sys.argv[2])            # <worktree>/artifacts/p1t_m3/mutants

LSTM_CU = 'src/lstm.cu'

MUTANTS = {
    'a': ('torch gate order (i, f, g, o) instead of ONNX (i, o, f, c)', [
        (LSTM_CU, 'constexpr int64_t kGateI = 0;\nconstexpr int64_t kGateO = 1;\nconstexpr int64_t kGateF = 2;\n'
                  'constexpr int64_t kGateC = 3;',
         'constexpr int64_t kGateI = 0;  // MUTANT (a): torch.nn.LSTM gate order i, f, g, o\n'
         'constexpr int64_t kGateO = 3;\nconstexpr int64_t kGateF = 1;\nconstexpr int64_t kGateC = 2;'),
    ]),
    'b': ('the reverse direction not reversed (reverse slot walks t = 0..T-1)', [
        (LSTM_CU, 'const int64_t tt = (d >= first_reverse) ? (T - 1 - s) : s;',
         'const int64_t tt = s;  // MUTANT (b): reverse slot not reversed'),
    ]),
    'c': ('TF32 math on the LSTM cuBLAS handle', [
        (LSTM_CU, 'check_cublas(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode(CUBLAS_DEFAULT_MATH)");',
         'check_cublas(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH), "cublasSetMathMode(TF32)");  // MUTANT (c)'),
    ]),
    'd': ('initial_h / initial_c ignored (state always starts at zero)', [
        (LSTM_CU, '  // 1. Initial state into the outputs that carry the recurrence.\n',
         '  // 1. Initial state into the outputs that carry the recurrence.\n'
         '  h0 = nullptr;  // MUTANT (d): initial_h ignored\n  c0 = nullptr;  // MUTANT (d): initial_c ignored\n'),
    ]),
    'e': ('expf -> __expf (fast intrinsic) in the cell sigmoid', [
        (LSTM_CU, 'if (z >= 0.0f) return 1.0f / (1.0f + expf(-z));',
         'if (z >= 0.0f) return 1.0f / (1.0f + __expf(-z));  // MUTANT (e)'),
        (LSTM_CU, 'const float e = expf(z);', 'const float e = __expf(z);  // MUTANT (e)'),
    ]),
}


def _ignore(directory, names):
    skip = {'build', '__pycache__', '.pytest_cache'}
    return [n for n in names if n in skip or (n.startswith('_tc_signal') and n.endswith('.so'))]


def main():
    wanted = sys.argv[3:] or sorted(MUTANTS)
    for mid in wanted:
        desc, edits = MUTANTS[mid]
        root = Path(f'/tmp/p1t_m3_mut_{mid}')
        dst = root / 'tc_signal'
        if dst.exists():
            raise SystemExit(f'{dst} exists; mutants are built once into a fresh directory (no deletes)')
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SRC, dst, ignore=_ignore)
        diffs = []
        for rel, old, new in edits:
            p = dst / rel
            before = p.read_text()
            n = before.count(old)
            assert n == 1, f'mutant {mid}: {rel}: pattern matched {n} times'
            after = before.replace(old, new)
            p.write_text(after)
            diffs.append(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                      fromfile=f'a/tc_signal/{rel}', tofile=f'b/tc_signal/{rel}')))
        od = OUT / mid
        od.mkdir(parents=True, exist_ok=True)
        (od / f'mutant_{mid}.diff').write_text(''.join(diffs))
        (od / 'MUTANT.txt').write_text(f'mutant ({mid}): {desc}\ncopy: {dst}\n')
        print(f'mutant {mid}: {desc} -> {dst}')


if __name__ == '__main__':
    main()
