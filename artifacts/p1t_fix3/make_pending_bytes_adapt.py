"""P1T-FIX3: the round-3 reviewer's pending_bytes.cpp with its "entry" branch adapted, as the reviewer instructed
("The per-entry aggregate statement is pinned to this reviewed revision; adapt the driver when reviewing a changed
byte-accounting implementation", docs/P1T_REV3_REPORT.md, Probe list). The original stays byte-identical.

The original "entry" branch does not call product code: it COPIES 5f9f955's destructor statement
`e.bytes += op->numel() * static_cast<int64_t>(sizeof(float));` into the probe, so it overflows on every build by
construction. The adaptation replaces exactly that loop (asserted to occur once) with a call of the REAL retainer on
the same two validated, disjoint (2**61-1,) fp32 operands: `tcs::RetainUntilComplete keep(0, {&X, &Y});` inside a
try block. No kernel is launched.
  * On 5f9f955 the retainer accepts the pair; with a CUDA device visible its destructor then performs the summation at
    dlpack_operand.cpp:731 (UBSan: signed integer overflow, rc != 0). With no device, event creation fails first
    (std::runtime_error, printed), so the 5f9f955 witness needs the GPU-visible condition.
  * On FIX3 the constructor refuses with ValueError before anything is acquired: "SAFE REFUSAL: ... aggregate ...
    overflows ...", rc 0, in both conditions.
The "queue" branch (actual note_push) and everything else are unchanged.
Usage: make_pending_bytes_adapt.py <original> <out.cpp>
No prior art claimed (plain text substitution).
"""
import difflib
import sys
from pathlib import Path

OLD = '''   // Verbatim arithmetic statement from RetainUntilComplete destructor, dlpack_operand.cpp:731.
   for(tcs::Operand* op:{&X,&Y}) {
     e.bytes += op->numel() * static_cast<int64_t>(sizeof(float));
   }
   std::cout<<e.bytes<<std::endl;'''
NEW = '''   // P1T-FIX3 adaptation: the per-entry sum is formed by the REAL retainer (5f9f955: its destructor,
   // dlpack_operand.cpp:731; FIX3: its constructor's checked pre-launch aggregate). No kernel is launched.
   try {
     tcs::RetainUntilComplete keep(0, {&X, &Y});
     std::cout<<"retainer accepted the pair (destructor runs next)"<<std::endl;
   } catch(const std::exception& ex) {
     std::cout<<"SAFE REFUSAL: "<<ex.what()<<std::endl;
   }'''

orig, out = sys.argv[1:3]
src = Path(orig).read_text()
assert src.count(OLD) == 1, src.count(OLD)
new = src.replace(OLD, NEW)
Path(out).write_text(new)
sys.stdout.writelines(difflib.unified_diff(src.splitlines(True), new.splitlines(True), orig, out))
