"""P1T-FIX5: regenerate the two round-5 reviewer probes that EMBED product code instead of including it, from a given
tree's sources, so that they test that tree (FIX4 D9 precedent: core_interop_cpu.cpp).

  * artifacts/p1t_rev5/probes/handle_insert_oom_cpu.cpp embeds conv.cu `handle_for` and lstm.cu `lstm_handle_for`
    verbatim (1475fae). regen_handle_probe() replaces each embedded function with the function extracted from the given
    conv.cu / lstm.cu (from its signature line to the first following line that is exactly "}"). If the extracted text
    calls note_cublas_handle_destroy_failure (P1T-FIX5), a counting stub of it is inserted before `namespace conv {`;
    if it calls check_cublas_on (P1T-FIX7), a throwing stub of it (the probe's own check_cublas, plus the device
    argument) is inserted there too; nothing else changes. Self-check: regenerating from 1475fae's sources reproduces the probe byte for byte.
  * artifacts/p1t_rev5/probes/fill_grid_cpu.cpp embeds the baseline `nblk` helper that 1475fae's fill_ launched with.
    regen_fill_probe() reads the given kernels.cu fill_: if fill_ launches with nblk(n) (1475fae), the probe is returned
    unchanged; if it uses P1T-FIX5's fill_grid_blocks, that function and its kFillMaxGridX constant are extracted
    verbatim and inserted after nblk, and main's geometry line `int blocks=nblk(n);` becomes the fill_ path's geometry
    (fill_grid_blocks, its std::invalid_argument refusal counted as the probe header's "refused" outcome, and the
    launch's unsigned conversion). This touches main, beyond a slice substitution: reported as such (FIX5 deviation).

Prior art: the FIX4 regen_core_interop_import.py approach (extract the arm's function text, substitute, self-check).
No other prior art claimed (text plumbing).
"""
import re


def _extract_function(src, signature):
    lines = src.splitlines(True)
    for i, ln in enumerate(lines):
        if ln.rstrip('\n') == signature:
            for j in range(i + 1, len(lines)):
                if lines[j].rstrip('\n') == '}':
                    return ''.join(lines[i:j + 1])
    raise ValueError(f'function not found: {signature!r}')


def regen_handle_probe(probe, conv_src, lstm_src):
    conv_sig = 'cublasHandle_t handle_for(int device) {'
    lstm_sig = 'cublasHandle_t lstm_handle_for(int device) {'
    old_conv = _extract_function(probe, conv_sig)
    old_lstm = _extract_function(probe, lstm_sig)
    new_conv = _extract_function(conv_src, conv_sig)
    new_lstm = _extract_function(lstm_src, lstm_sig)
    out = probe.replace(old_conv, new_conv, 1).replace(old_lstm, new_lstm, 1)
    if 'note_cublas_handle_destroy_failure' in new_conv + new_lstm:
        stub = ('// P1T-FIX5 regeneration: counting stub for the product\'s note_cublas_handle_destroy_failure\n'
                'static int note_destroy_failures=0;\n'
                'void note_cublas_handle_destroy_failure(int) noexcept {++note_destroy_failures;}\n')
        out = out.replace('namespace conv {\n', stub + 'namespace conv {\n', 1)
    if 'check_cublas_on(' in new_conv + new_lstm:
        # P1T-FIX7 (sweep (j)): the product's status check now also names the device (check_cublas_on); stub it the way
        # the probe stubs check_cublas (any failure throws), so the extracted functions compile unchanged.
        stub = ('// P1T-FIX7 regeneration: stub for the product\'s device-naming check_cublas_on\n'
                'static void check_cublas_on(cublasStatus_t s,const char*,int) {if(s!=CUBLAS_STATUS_SUCCESS)'
                'throw std::runtime_error("cuBLAS error");}\n')
        out = out.replace('namespace conv {\n', stub + 'namespace conv {\n', 1)
    return out


def regen_fill_probe(probe, kernels_src):
    fill = _extract_function(kernels_src, 'void fill_(NDArray& a, double value) {')
    if 'fill_grid_blocks' not in fill:
        assert 'nblk(n)' in fill, 'fill_ uses neither nblk(n) nor fill_grid_blocks'
        return probe
    const = re.search(r'^constexpr int64_t kFillMaxGridX = [^\n]*\n', kernels_src, re.M).group(0)
    helper = _extract_function(kernels_src, 'int64_t fill_grid_blocks(int64_t n) {')
    nblk = 'inline int nblk(int64_t n) { return static_cast<int>((n + kT - 1) / kT); }\n'
    assert probe.count(nblk) == 1
    out = probe.replace('#include <cstdint>\n', '#include <cstdint>\n#include <stdexcept>\n#include <string>\n', 1)
    out = out.replace(nblk, nblk + '// P1T-FIX5 regeneration: the fill_ path\'s geometry, extracted verbatim\n' +
                      const + helper, 1)
    old = 'int blocks=nblk(n);int64_t coverage=static_cast<int64_t>(static_cast<unsigned int>(blocks))*kT;'
    new = ('int64_t blocks;try{blocks=fill_grid_blocks(n);}catch(const std::invalid_argument& e){'
           'std::cout<<"n="<<n<<" refused: "<<e.what()<<"\\n";continue;}'
           'int64_t coverage=static_cast<int64_t>(static_cast<unsigned int>(blocks))*kT;')
    assert out.count(old) == 1
    return out.replace(old, new, 1)
