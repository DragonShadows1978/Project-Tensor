"""P1T-FIX1 G3: build the four registered mutants OUTSIDE the worktree, one asserted edit each.

Copies the worktree's tc_signal/ (mutants a, b) or tensor_cuda/ (mutants c, d) source tree (no build/, no .so, no
caches) to /tmp/p1t_fix1_mut_<m>/, applies exactly one string replacement that must match once, and writes the unified
diff to artifacts/p1t_fix1/g3/mut_<m>.diff. Prior art: DeMillo, Lipton & Sayward (1978) mutation testing; the M1 seat's
make_mutants.py pattern (artifacts/p1t_m1/g4). Usage: python3 make_mutants.py a b c d
"""
import difflib
import os
import shutil
import sys

W = '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix1'
G3 = os.path.join(W, 'artifacts/p1t_fix1/g3')
IGNORE = shutil.ignore_patterns('build', '*.so', '__pycache__', '.pytest_cache')

MUTANTS = {
    'a': ('tc_signal', 'src/dlpack_operand.cpp',
          'RetainUntilComplete::~RetainUntilComplete() {\n  PendingState& st = pending();',
          'RetainUntilComplete::~RetainUntilComplete() {\n  return;  // MUTANT (a): F01 fix removed: immediate release '
          'restored (the Operands release at scope exit)\n  PendingState& st = pending();',
          'F01 deferred release removed: every consumed manager is released right after the launches again'),
    'b': ('tc_signal', 'src/dlpack_operand.cpp',
          'void require_operand(py::handle obj, const char* role) {\n  // F02 (P1T-FIX1): None is never a tensor.',
          'void require_operand(py::handle obj, const char* role) {\n  return;  // MUTANT (b): F02 None guard removed\n'
          '  // F02 (P1T-FIX1): None is never a tensor.',
          'F02 None guard removed (require_operand is a no-op; the absent-Operand accessor defence stays)'),
    'c': ('tensor_cuda', 'src/bindings.cpp',
          '  if (shape.size() > static_cast<size_t>(TC_MAX_DIMS))\n    throw py::value_error("empty: "',
          '  if (false && shape.size() > static_cast<size_t>(TC_MAX_DIMS))  // MUTANT (c): F03 rank check removed\n'
          '    throw py::value_error("empty: "',
          'F03 empty() rank check removed (count/byte checks stay)'),
    'd': ('tensor_cuda', 'src/kernels.cu',
          '  if (e == cudaSuccess && prev != dev) {\n    e = cudaSetDevice(dev);\n    switched = (e == cudaSuccess);\n  }',
          '  if (false && e == cudaSuccess && prev != dev) {  // MUTANT (d): F05 device switch removed\n'
          '    e = cudaSetDevice(dev);\n    switched = (e == cudaSuccess);\n  }',
          'F05 device switch removed: the current device\'s legacy stream is drained, nothing is restored'),
    # Informational (not one of the order's registered G3 mutants): F04 with BOTH lifecycle checks removed.
    'e': ('tensor_cuda', 'src/bindings.cpp',
          'bool interpreter_accepts_gil_state() {\n  if (!Py_IsInitialized()) return false;',
          'bool interpreter_accepts_gil_state() {\n  return true;  // MUTANT (e): F04 finalization/initialized checks removed\n'
          '  if (!Py_IsInitialized()) return false;',
          'F04 lifecycle checks removed: PyGILState_Ensure is attempted even while the interpreter finalizes'),
}

for m in sys.argv[1:]:
    pkg, rel, old, new, what = MUTANTS[m]
    root = f'/tmp/p1t_fix1_mut_{m}'
    os.makedirs(root, exist_ok=True)
    dst = os.path.join(root, pkg)
    if os.path.exists(dst):
        sys.exit(f'{dst} exists; refusing to overwrite (no deletes)')
    shutil.copytree(os.path.join(W, pkg), dst, ignore=IGNORE)
    path = os.path.join(dst, rel)
    src = open(path).read()
    assert src.count(old) == 1, (m, src.count(old))
    mut = src.replace(old, new)
    open(path, 'w').write(mut)
    diff = ''.join(difflib.unified_diff(src.splitlines(True), mut.splitlines(True), f'a/{pkg}/{rel}', f'b/{pkg}/{rel}'))
    open(os.path.join(G3, f'mut_{m}.diff'), 'w').write(f'# mutant ({m}): {what}\n# tree: {dst}\n' + diff)
    print(f'mutant {m}: {what}\n  tree {dst}\n  edit {pkg}/{rel} ({diff.count(chr(10))} diff lines)')
