"""P1T-M4 G3 mutant builder: copies <worktree>/tc_signal (sources only) to /tmp/p1t_m4_mut_<id>/tc_signal and applies one
registered defect per mutant. Each replacement must match exactly once (assert); a unified diff is written next to the
copy as the receipt. Prior art: mutation testing (DeMillo, Lipton & Sayward 1978); defects (a)-(f) are the order's
registered list, (g) is mine (the shift removed)."""
import difflib
import shutil
import sys
from pathlib import Path

SRC = Path(sys.argv[1])            # <worktree>/tc_signal
ONLY = sys.argv[2:]                # optional subset of mutant ids

STATS_ORIG = '''    const float K = xr[0];

    // Pass 1: the mean of the shifted row.
    float acc = 0.0f;
    for (int64_t base = tid; base < L; base += chunk_span) {
      const int64_t end = min(L, base + chunk_span);
      float part = 0.0f;
      for (int64_t j = base; j < end; j += B) part = part + (xr[j] - K);
      acc = acc + part;
    }
    const float d = block_tree_sum<B>(acc, red) / Lf;

    // Pass 2: the centred sum of squares (biased variance).
    acc = 0.0f;
    for (int64_t base = tid; base < L; base += chunk_span) {
      const int64_t end = min(L, base + chunk_span);
      float part = 0.0f;
      for (int64_t j = base; j < end; j += B) {
        const float c = (xr[j] - K) - d;
        part = fmaf(c, c, part);
      }
      acc = acc + part;
    }
    const float var = block_tree_sum<B>(acc, red) / Lf;'''

STATS_ONE_PASS = '''    const float K = 0.0f;  // MUTANT (b): one-pass E[x^2] - E[x]^2 on the raw row (no shift), clamped at 0

    float acc = 0.0f, acc2 = 0.0f;
    for (int64_t base = tid; base < L; base += chunk_span) {
      const int64_t end = min(L, base + chunk_span);
      float part = 0.0f, part2 = 0.0f;
      for (int64_t j = base; j < end; j += B) {
        const float v = xr[j];
        part = part + v;
        part2 = fmaf(v, v, part2);
      }
      acc = acc + part;
      acc2 = acc2 + part2;
    }
    const float d = block_tree_sum<B>(acc, red) / Lf;
    const float ex2 = block_tree_sum<B>(acc2, red) / Lf;
    const float var = fmaxf(ex2 - d * d, 0.0f);'''

MUTANTS = {
    'a': ('unbiased variance (divide by L - 1)', [
        ('src/norms.cu', '    const float var = block_tree_sum<B>(acc, red) / Lf;',
         '    const float var = block_tree_sum<B>(acc, red) / (Lf - 1.0f);  // MUTANT (a): unbiased variance'),
    ]),
    'b': ('one-pass E[x^2] - E[x]^2 variance on the raw row (clamped at 0)', [
        ('src/norms.cu', STATS_ORIG, STATS_ONE_PASS),
    ]),
    'c': ('layer_norm eps forced to 1e-5', [
        ('src/m4_bindings.cpp', '        norm_run(NormKind::kLayer, xc, sc, bc, out, axis, eps);',
         '        norm_run(NormKind::kLayer, xc, sc, bc, out, axis, 1e-5);  // MUTANT (c): eps forced to 1e-5'),
    ]),
    'd': ('rint via the C library half-away-from-zero rounding', [
        ('src/small_ops.cu', '    y[i] = rintf(x[i]);', '    y[i] = roundf(x[i]);  // MUTANT (d)'),
    ]),
    'e': ('reflection pad as symmetric pad (edge sample repeated)', [
        ('src/small_ops.cu', '''    if (s < 0)
      s = -s;
    else if (s >= T)
      s = 2 * (T - 1) - s;''', '''    if (s < 0)
      s = -s - 1;  // MUTANT (e): symmetric pad
    else if (s >= T)
      s = 2 * T - 1 - s;  // MUTANT (e): symmetric pad'''),
    ]),
    'f': ('upsample source index by rounding instead of floor (clamped)', [
        ('src/small_ops.cu', '    y[i] = x[r * T + t / scale];',
         '    y[i] = x[r * T + min((t + scale / 2) / scale, T - 1)];  // MUTANT (f): rounding, clamped'),
    ]),
    'g': ('shift removed: plain fp32 two-pass (K = 0)', [
        ('src/norms.cu', '    const float K = xr[0];',
         '    const float K = 0.0f;  // MUTANT (g): no shift (plain fp32 two-pass)'),
    ]),
}

for mid, (desc, edits) in MUTANTS.items():
    if ONLY and mid not in ONLY:
        continue
    root = Path(f'/tmp/p1t_m4_mut_{mid}')
    dst = root / 'tc_signal'
    if dst.exists():
        raise SystemExit(f'{dst} exists; use a fresh root')
    shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns('build', '_tc_signal*.so', '__pycache__'))
    diffs = []
    for rel, old, new in edits:
        p = dst / rel
        s = p.read_text()
        assert s.count(old) == 1, f'mutant {mid}: pattern in {rel} matched {s.count(old)} times'
        t = s.replace(old, new)
        p.write_text(t)
        diffs += difflib.unified_diff(s.splitlines(True), t.splitlines(True), f'a/tc_signal/{rel}', f'b/tc_signal/{rel}')
    (root / f'mutant_{mid}.diff').write_text(''.join(diffs))
    (root / 'MUTANT.txt').write_text(f'mutant {mid}: {desc}\n')
    print(f'mutant {mid} ({desc}): {root}  diff lines {len(diffs)}')
