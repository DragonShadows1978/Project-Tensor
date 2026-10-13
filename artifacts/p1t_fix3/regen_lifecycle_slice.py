"""P1T-FIX3: regenerate the round-3 reviewer's lifecycle_slice.cpp from another source tree, exactly as the reviewer made it.

The committed probe support file (artifacts/p1t_rev3/probes/lifecycle_slice.cpp, read-only here) is the reviewer's
fixed text (a 10-line header and a CPU driver tail) around ONE verbatim slice of 5f9f955's tensor_cuda/src/bindings.cpp
(lines 571-683, sha256 0a56c6f3... in artifacts/p1t_rev3/source_slices.json): from
'std::atomic<unsigned long long> g_import_released{0};' up to (not including) 'void import_release_noop(void*) {}'.
Because it embeds the OLD lifecycle code, compiling it unmodified can only ever test 5f9f955. This script splits the
committed file around the 5f9f955 slice (asserted to occur exactly once), cuts the same region from <root>'s
bindings.cpp with the same two markers, and joins them. The driver tail (`close` bound to dlpack_begin_shutdown with
one double argument, `start`, `stats`) is kept byte for byte; FIX3 kept dlpack_begin_shutdown's single-argument
signature so the driver applies unchanged.
Self-check: with <root> = the 5f9f955 tree it must reproduce the committed file byte for byte (`selfcheck identical=True`).
Usage: regen_lifecycle_slice.py <committed_slice> <5f9f955_root> <root> <out.cpp>
Prior art: the reviewer's own slice generator (P1T-REV3, source_slices.json) and FIX2's regen_core_cpu_probe.py (same
split-and-rejoin); no other prior art known to me.
"""
import hashlib
import sys
from pathlib import Path

START = 'std::atomic<unsigned long long> g_import_released{0};'
END = 'void import_release_noop(void*) {}'


def region(root):
    s = (Path(root) / 'tensor_cuda/src/bindings.cpp').read_text()
    assert s.count(START) == 1 and s.count(END) == 1
    lo = s.index(START)
    return s[lo:s.index(END, lo)]


def main():
    committed, base_root, root, out = sys.argv[1:5]
    text = Path(committed).read_text()
    base = region(base_root)
    print('base slice sha256 %s (review pin 0a56c6f3...)' % hashlib.sha256(base.encode()).hexdigest())
    assert text.count(base) == 1
    head, tail = text.split(base)
    print('selfcheck identical=%s' % (head + region(base_root) + tail == text))
    new = head + region(root) + tail
    Path(out).write_text(new)
    print('wrote %s (%d bytes) from %s; slice sha256 %s' % (out, len(new), root,
                                                          hashlib.sha256(region(root).encode()).hexdigest()))


if __name__ == '__main__':
    main()
