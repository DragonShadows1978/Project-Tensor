# Prior art: ordinary source call-path analysis (no prior art known to me for this order-specific table);
# GNU unified diff; GN1 A8 completed-child supervision. No algorithm change or new proof technique claimed.
import json,re
from pathlib import Path
root=Path.cwd();rev=root/'artifacts/eng1/rev1';c=json.loads((rev/'census_scope.json').read_text())
raw=(rev/'callers_raw.txt').read_text().splitlines()
out=['# E4, E5 and C++ reshape call-path audit','',
'Reasoning evidence: the three non-topk functions have identical nonempty paths; metadata allocation uses the same input shape, dtype and device. An empty output has no elements to read. No caller can legitimately require the former host division by zero. Topk changes only out-of-range k and no launch for empty output. Unknown external users are not covered.','',
'## Every direct E4 call found in worktree source','',
'| File:line | Source | Caller-specific conclusion |','|---|---|---|']
for ln in raw:
    m=re.match(r'(.*?):(\d+):(.*)',ln);p,n,text=m.groups();p=p.removeprefix('./')
    if not any(s in text for s in ('rms_norm(', 'causal_softmax(', 'apa_blend_softmax(','.topk(')):continue
    if '/src/' in p or '/tests/test_eng1.py' in p or 'def ' in text:continue
    if 'torch.topk' in text:why='PyTorch call, does not invoke NDArray topk; unaffected.'
    elif 'tc._C.' in text:why='Thin binding; directly forwards shape and dtype, no prior empty-result assumption.'
    elif '.topk(' in text:
        why='Model routing chooses configured k expert scores; valid 0<=k<=expert_count results unchanged. Invalid configs now refuse instead of emitting -1 sentinels; no intentional sentinel use found.' if 'core/' in p else 'phase9 uses k=3 on last extent 5; range is valid, nonempty kernel path unchanged.'
    elif 'rms_norm(' in text:why='RMS inference passes fp32 weights and preserves x dtype; nonempty feature dimension path unchanged; an empty batch retains shape.'
    elif 'apa_blend_softmax(' in text:why='Attention scores/bulk/rank share (...,S). Nonempty launch identical; empty returns same score shape/dtype and can be reshaped as before.'
    else:why='Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype.'
    out.append(f"| {p}:{n} | `{text.strip().replace('|','/')}` | {why} |")
out += ['', 'Callback site functional.py:170 passes tc.causal_softmax through _guarded_softmax; same S>=L admission guard, same nonempty path. Script interposer apa_sp4g_a3_model.py:29/62/68 saves/restores the callable without changing its metadata contract. Mutation-template string hits in apa_mf2_2b_mutation.py are source text, not runtime callers. The sink variant (kernels.cu:4012–4044) uses rows=B*H*L with no host division by S, so D2 correctly leaves it unchanged.',
'', '## Every make_dims caller','',
'| Call | Entry line in conv.cu | Allocation/launch and regression argument |','|---|---|---|',
'| im2col | 158 | Helper first; then allocate (N,C*kh*kw,OH*OW), launch iff nonzero. Public refusal IDs and valid im2col reference pin this. |',
'| col2im | 166 | Helper first; then zero input gradient, scatter from cols. Conv/im2col backward routes here; invalid geometry cannot get past helper. |',
'| avgpool2d | 173 | Helper first; then allocate (N,C,OH,OW). Positive geometry uses unchanged kernel, checked against independent windows. |',
'| avgpool2d_bwd | 180 | Helper first with captured input shape and forward geometry. Invalid forward already refuses; valid backward scatter unchanged and value-tested. |',
'| maxpool2d | 187 | Helper first; then values and argmax allocation. Invalid geometry refused before either; valid forward/backward value-tested. |',
'', 'maxpool2d_bwd (conv.cu:198) does not call make_dims: it consumes the captured valid-forward argmax and x_shape. The order premise that both backward pools directly call make_dims is imprecise; max-pool backward is protected by its forward.',
'', 'For positive strides, floor(A/s)+1>0 iff A>=0. Thus every candidate-accepted geometry has the same extent under old truncation. The old accepted set also contained -s<A<0; those cases are now correctly refused (H=3,k=4,s=2,p=0). Invalid zero/negative strides and >int32 image sizes are inherited limits, not claimed repaired.',
'', '## Additional C++ metadata reshape sites','',
'| File:line | Check |','|---|---|',
'| ops.cpp:812 | sum backward restores reduced singleton axes kshape; product preserved. |',
'| ops.cpp:827,830 | public reshape passes shape to NDArray; backward restores the captured original nonnegative shape. |',
'| ops.cpp:912,925 | extrema backward restores keepdim layout; product preserved. |',
'| ops.cpp:943,944 | product backward restores gradient and product to keepdim layout; same count. |',
'| ops.cpp:1008 | squeeze removes only size-one axes. |',
'| ops.cpp:1015 | unsqueeze inserts a size-one axis. |',
'| ops.cpp:1034 | flatten replaces a contiguous axis range by its explicit product. |',
'| kernels.cu:1127 | reduction removes singleton keepdim axes. |',
'| kernels.cu:1154 | reduce_to restores target after same-count reduction. |',
'| kernels.cu:5093 | grouped KV: rows*G*group = B*KV*S*D under D%group==0. |',
'| kernels.cu:5099 | scales rows*G restored to B*KV*S*G. |',
'| gemm_apa.cu:778 | grouped heads: H*L = KVH*M; probability element count preserved. |',
'| gemm_apa.cu:780 | output grouping reversed with the same head/query identity. |',
'', 'All previously valid explicit nonnegative shapes take the same final count comparison and share the same storage. Negative extents only arose from already-invalid metadata; their new refusals are intentional. No Python validator intercepts -1: bindings.cpp:1879 forwards the int64 vector.']
(rev/'CORE_CALLERS.md').write_text('\n'.join(out)+'\n')
notes=[
'Header/import changes and removed GN1-only program binding; completed-child outcome checks remain.',
'RUNS scope only.',
'Per-arm source/module seals; original shared-engine design replaced.',
'START audit binding and ID normalization; no fabricated outcome.',
'JUnit dictionary initialization only.',
'Comment removal around duplicate failure/teardown entries; guard remains.',
'TMPDIR naming and deleted comments.',
'Timeout attribution comment removal only.',
'Remove GroupNorm-specific killer filters; introduce literal bracket matcher with registered mutants.',
'Adapt child arm/seal/census checks and baseline/mutant/candidate roles. JUnit equality and actual rc agreement retained.',
'Remove old runtime/program conjunctions from evaluate; final device classification now occurs in recompute_line.',
'Remove GN1-only package validation; retain exact baseline-passing ID comparison.',
'Completed-parent integrity, raw/normalized exit, ordered timestamps, dependencies and device evidence checked by scorer.',
'Parent docstring only.',
'Parent imports adapt evidence functions.',
'Parent environment, selected arm/suite and dependency changes.',
'Parent foreground subprocess.run returns before receipt construction/publication; actual exit and timeout retained.',
'Child docstring only.',
'Child imports/report exception fields adapted.',
'Pinned arm preload and suite selection. Child returns pytest exit after writing JUnit/summary; D4 subprocess inheritance is R1-F02.',
'Scorer changes roles, control and census reporting; requires valid parent+child+JUnit and positive device evidence.',
'START docstring only.',
'START immutable record changes GN1 flags to ENG1 audit mode; retains seal/date bindings.',
'Stand-in environment naming; explicitly CPU audit only.',
'Device evidence docstring.',
'Device query pin is selected per arm.',
'22-line registration replaces GN1 labels and bounds; expected child exits remain separate from parent success.'
]
assert len(notes)==len(c['tool_hunks'])==27
o=['# Review of all 27 tooling hunks','',
'All frozen originals match accepted artifacts/gn1/a8 byte for byte; all 27 diff hunks reconstruct the live adapted files exactly (census_scope.json). Scope is the requested completed-child property, not GN1 contract-C equivalence.','',
'| Hunk | File / header | Review |','|---|---|---|']
for h,n in zip(c['tool_hunks'],notes):o.append(f"| {h['number']} | {h['file']} {h['header']} | {n} |")
o+=['','run_gpu_line.py:65–70 waits for the foreground subprocess, derives rc from returncode, and only then evaluates receipts and publishes parent_result.json at :108–130. evidence.py:229–249 requires the observed exit, matching child pytest_exit, exact collected IDs and JUnit outcomes; :278–280 cross-checks rc against recorded failures. recompute_line :320–342 refuses missing parent completion and validates start <= child_exited <= recorded. The parent never treats a mutant kill or a baseline expected failure as a child success; expected exit 1 is explicit. Device failures, missing tests, collection failures and unregistered skips are refused outside the regression role.',
'','The existing audit scorer was executed unchanged against its original audit root. Only its attempted G5_SUMMARY.json write was redirected to rev1/audit_rescore.json to respect read-only receipts. It returned 1: candidate_eng1 passed in CPU mode, all other lines were absent, and audit data could not establish GPU acceptance.']
(rev/'TOOLING_REVIEW.md').write_text('\n'.join(o)+'\n')
print('Wrote itemized E4/E5/C++ caller audit and 27-hunk tooling review')
