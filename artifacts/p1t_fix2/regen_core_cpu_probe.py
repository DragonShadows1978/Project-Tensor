"""P1T-FIX2: regenerate the round-2 reviewer's core_cpu.cpp probe from another source tree, exactly as the reviewer built it.

The committed probe (artifacts/p1t_rev2/probes/core_cpu.cpp, read-only here) is the reviewer's fixed text around four
VERBATIM source slices of d116ac0 (artifacts/p1t_rev2/source_slices.json; generator in logs/p1t_rev2_r1.log ~9124):
  kernels.cu   'size_t dtype_size' .. 'namespace {\ntemplate <typename T> __device__'
  autograd.cpp 'Tensor Tensor::make' .. 'Tensor Tensor::from_op'
  bindings.cpp 'std::string shape_str(const int64_t*' .. '// ---- export:'
  bindings.cpp 'Tensor tensor_empty' .. 'const char* const kDlpackDoc'
Because the probe embeds the OLD code, compiling it unmodified can only ever test d116ac0. This script takes the
reviewer's fixed text from the committed probe (everything outside the slices), cuts the same four slices from
<root> with the reviewer's own markers, and joins them in the same order. Self-check: with <root> = the d116ac0 tree
it must reproduce the committed probe byte for byte (printed as `selfcheck identical=True`).
Usage: regen_core_cpu_probe.py <committed_probe> <d116ac0_root> <root> <out.cpp>
Prior art: the reviewer's generator (P1T-REV2, logs/p1t_rev2_r1.log); no prior art known to me for the split-and-rejoin.
"""
import sys
from pathlib import Path

MARKS = [('tensor_cuda/src/kernels.cu', 'size_t dtype_size', 'namespace {\ntemplate <typename T> __device__'),
         ('tensor_cuda/src/autograd.cpp', 'Tensor Tensor::make', 'Tensor Tensor::from_op'),
         ('tensor_cuda/src/bindings.cpp', 'std::string shape_str(const int64_t*', '// ---- export:'),
         ('tensor_cuda/src/bindings.cpp', 'Tensor tensor_empty', 'const char* const kDlpackDoc')]


def slices(root):
    out = []
    for f, a, b in MARKS:
        s = (Path(root) / f).read_text()
        lo = s.index(a)
        hi = s.index(b, lo)
        out.append(s[lo:hi])
    return out


def frame(committed, base_slices):
    """The reviewer's fixed text: the committed probe split around the d116ac0 slices (asserted to occur in order)."""
    text = Path(committed).read_text()
    pieces, pos = [], 0
    for sl in base_slices:
        i = text.index(sl, pos)
        pieces.append(text[pos:i])
        pos = i + len(sl)
    pieces.append(text[pos:])
    return text, pieces


def join(pieces, sl):
    out = pieces[0]
    for k, s in enumerate(sl):
        out += s + pieces[k + 1]
    return out


def main():
    committed, base_root, root, out = sys.argv[1:5]
    text, pieces = frame(committed, slices(base_root))
    print('selfcheck identical=%s' % (join(pieces, slices(base_root)) == text))
    new = join(pieces, slices(root))
    Path(out).write_text(new)
    print('wrote %s (%d bytes) from %s' % (out, len(new), root))


if __name__ == '__main__':
    main()
