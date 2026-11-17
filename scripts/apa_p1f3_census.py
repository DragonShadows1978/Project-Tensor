"""APA-P1-F3 operand censuses (Mechanism & scope, L82, L16):
  PRODUCT_CENSUS.json : every host product (and every sum or difference that feeds one) that is range-checked in
                        tensor_cuda/src/apa_spo (R3'), starting from `rg -n 'numeric_limits<int>'` and the F2 range
                        helpers; per row: file, line now and in the committed tree (25f59b9c, read with git show),
                        the operand types, whether it is formed in int64 before the check and before any narrowing,
                        and the disposition.
  OPERAND_CENSUS.json : prepare's norm store (R2') and every read of the margin mapping in apa_packed.py (R1').
Each row is located by an anchor text that must occur exactly once in its function; the census refuses otherwise.
Prior art: none known to me for the census itself (a grep-anchored table); ISO C++17 [expr]/4 and [conv.integral]
for the classification. CPU only; read-only git.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_p1f3'
BASE = '25f59b9c'
SPO = 'tensor_cuda/src/apa_spo/'

ROWS = [
    # file, function head, anchor now, anchor in base (or None if the row is new), types, int64 now, int64 before, disposition
    ('eb2b.cuh', 'std::vector<NDArray> apa_eb2b_run(', 'apa_spo_product({q.shape[0],q.shape[1]}),kApaSpoIntMax/4);',
     'B*H>std::numeric_limits<int>::max()/4', 'before: int B,H narrowed from int64 extents, product in int (2^16*2^16 '
     'undefined, wraps to 0); now: int64 extents through apa_spo_product (saturating)', True, False,
     "FIXED (R3'): formed in int64 before the check and before `int B=`; ValueError naming apa_eb2b and B*H"),
    ('spq4.cuh', 'std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq4_tc(', 'B*H>std::numeric_limits<int>::max()/4',
     'B*H>std::numeric_limits<int>::max()/4', 'const int64_t B,H from the extents (each <= INT_MAX after the extent '
     'checks), product < 2^62', True, True, 'no change: already int64; narrowed (rows=(int)(B*H)) only after the check'),
    ('spq6.cuh', 'std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq6_tq4_tc(', 'B*H>std::numeric_limits<int>::max()/4',
     'B*H>std::numeric_limits<int>::max()/4', 'const int64_t B,H', True, True, 'no change: already int64'),
    ('eb3.cuh', 'std::vector<NDArray> apa_eb3_run(', 'B*H>std::numeric_limits<int>::max()/4',
     'B*H>std::numeric_limits<int>::max()/4', 'const int64_t B,H', True, True, 'no change: already int64'),
    ('eb4.cuh', 'std::vector<NDArray> apa_eb4_run(', 'B*H>std::numeric_limits<int>::max()/4',
     'B*H>std::numeric_limits<int>::max()/4', 'const int64_t B,H; (int64_t)BLK*NSORT_MAX', True, True,
     'no change: already int64'),
    ('launcher.cuh', 'inline void apa_spq1_core_check(', 'B > std::numeric_limits<int>::max() / H / L',
     'B > std::numeric_limits<int>::max() / H / L', 'const int64_t B,H,L; division, no product formed', True, True,
     'no change: the bound is tested by int64 division'),
    ('launcher.cuh', 'inline void apa_spo_require_key_range(', 'S+(int64_t)TC_APA_SPLITK_PART_KEYS-1',
     'S+(int64_t)TC_APA_SPLITK_PART_KEYS-1', 'int64 S + int64 constant', True, True, 'no change: int64 sum'),
    ('packed_launcher.cuh', 'void apa_spq1_buffer_check(', 'apa_spo_require_int_range("apa_spq1 packed buffers","capacity (codes.shape[2])",codes.shape[2]);',
     'apa_spo_require_int_range("apa_spq1 packed buffers","capacity (codes.shape[2])",codes.shape[2]);',
     'int64 extent, no product', True, True, 'no change'),
    ('packed_launcher.cuh', 'NDArray apa_spq1_rotate(', 'apa_spo_product({q.shape[0],q.shape[1]})',
     'apa_spo_product({q.shape[0],q.shape[1]})', 'int64 extents through apa_spo_product', True, True, 'no change'),
    ('packed_launcher.cuh', 'std::pair<NDArray,NDArray> apa_spq1_prepare(', 'apa_spo_product({k.shape[0],k.shape[1],N64})',
     'apa_spo_product({(int64_t)B,(int64_t)H,(int64_t)N})', 'before: factor N = S - start formed in int before the '
     'start < 0 refusal (S - start overflows for a large negative start); now N64 = k.shape[2] - (int64_t)start and the '
     'product from the int64 extents', True, False,
     "FIXED (R3' like site): N formed in int64; same refusals for every defined input (start < 0 still raises the "
     "RuntimeError 'prepare geometry/rotation')"),
    ('packed_launcher.cuh', 'void apa_spq1_pack(', 'apa_spo_product({k.shape[0],k.shape[1]})',
     'apa_spo_product({k.shape[0],k.shape[1]})', 'int64 extents through apa_spo_product (B*KV)', True, True, 'no change'),
    ('packed_launcher.cuh', 'void apa_spq1_pack(', 'apa_spo_ceil_div(apa_spo_product({(int64_t)BH,(int64_t)N,(int64_t)(D/2)}),128)',
     'apa_spo_ceil_div(apa_spo_product({(int64_t)BH,(int64_t)N,(int64_t)(D/2)}),128)', 'before: factor N = S - start '
     'formed in int before the start < 0 refusal; now N = (int)N64 after the start checks (0 <= N <= S <= INT_MAX), '
     'BH = the checked int64 B*KV narrowed after its check', True, False,
     "FIXED (R3' like site): N64 formed in int64 before the start check and narrowed after it"),
    ('packed_launcher.cuh', 'std::pair<NDArray,NDArray> apa_spq1_int4_stats(', 'q.shape[0]*q.shape[1]>std::numeric_limits<int>::max()',
     'q.shape[0]*q.shape[1]>std::numeric_limits<int>::max()', 'int64 extents, each <= INT_MAX after the preceding '
     'per-extent check, product < 2^62', True, True, 'no change'),
    ('eb2b.cuh', 'std::pair<NDArray,NDArray> apa_eb2b_quantize(', 'v.numel()/128>std::numeric_limits<int>::max()',
     'v.numel()/128>std::numeric_limits<int>::max()', 'int64 numel (checked at import), division', True, True, 'no change'),
    ('eb2b.cuh', 'NDArray apa_eb2b_values(', 'v.numel()/8+127', 'v.numel()/8+127', 'int64 numel', True, True, 'no change'),
]

EXTENT_ROWS = [
    (f, head, entry) for f, head, entry in (
        ('packed_launcher.cuh', 'std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq1_run(', 'apa_spq1 packed r2 (packed_run)'),
        ('spq4.cuh', 'std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq4_run(', 'apa_spq4_run (packed_run)'),
        ('spq4.cuh', 'std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq4_tc(', 'apa_spq4 tc int4'),
        ('spq6.cuh', 'std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq6_tq4_tc(', 'apa_spq6 tc tq4'),
        ('eb2b.cuh', 'std::vector<NDArray> apa_eb2b_run(', 'apa_eb2b'),
        ('eb3.cuh', 'std::vector<NDArray> apa_eb3_run(', 'apa_eb3'),
        ('eb4.cuh', 'std::vector<NDArray> apa_eb4_run(', 'apa_eb4'))]

OPEN = re.compile(r'\)\s*\{\n')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def body_span(text, head):
    start = text.index(head)
    m = OPEN.search(text, start)
    end = text.index('\n}\n', m.end())
    return m.end(), end


def locate(text, head, anchor):
    a, b = body_span(text, head)
    body = text[a:b]
    assert body.count(anchor) == 1, (head, anchor, body.count(anchor))
    pos = a + body.index(anchor)
    return text.count('\n', 0, pos) + 1


def base_text(f):
    return subprocess.run(['git', 'show', f'{BASE}:{SPO}{f}'], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


def main():
    rows, files = [], {}
    for f, head, now_anchor, base_anchor, types, now64, base64, disp in ROWS:
        now = (ROOT / SPO / f).read_text()
        base = base_text(f)
        files[f] = dict(now=sha(now.encode()), base=sha(base.encode()))
        rows.append(dict(file=SPO + f, function=head.rstrip('('), anchor_now=now_anchor, line_now=locate(now, head, now_anchor),
                         anchor_base=base_anchor, line_base=locate(base, head, base_anchor), operand_types=types,
                         formed_in_int64_before_check_and_narrowing_now=now64,
                         formed_in_int64_before_check_and_narrowing_base=base64, disposition=disp))
    extents = []
    for f, head, entry in EXTENT_ROWS:
        now = (ROOT / SPO / f).read_text()
        a, b = body_span(now, head)
        code = [l.strip() for l in now[a:b].splitlines() if l.strip() and not l.strip().startswith('//')]
        guard = f'apa_spo_require_attention_extents("{entry}",{{{{"q",&q}},{{"k",&k}},{{"v",&v}}}});'
        extents.append(dict(file=SPO + f, function=head.rstrip('('), entry=entry, first_statement_is_guard=code[0] == guard,
                            line=locate(now, head, guard)))
    grep = subprocess.run(['rg', '-n', 'numeric_limits<int>', SPO], cwd=ROOT, capture_output=True, text=True)
    census = dict(ruling="R3' (P1R3-F03)", base=BASE, rows=rows, files_sha256=files,
                  rows_fixed=sum(not r['formed_in_int64_before_check_and_narrowing_base'] for r in rows),
                  all_formed_in_int64_now=all(r['formed_in_int64_before_check_and_narrowing_now'] for r in rows),
                  attention_extent_guards=extents,
                  start_grep=dict(cmd="rg -n 'numeric_limits<int>' tensor_cuda/src/apa_spo", exit=grep.returncode,
                                  lines=grep.stdout.splitlines()),
                  note=('Range-checked host products only. Arena sizes (int64 RS/RC/RP products) are allocation sizes, '
                        'not range checks, and are out of this census. The attention extents themselves are guarded by '
                        'apa_spo_require_attention_extents as each entry\'s first statement (listed).'))
    with (ART / 'PRODUCT_CENSUS.json').open('x') as f:
        json.dump(census, f, indent=2)
        f.write('\n')
    # R2' and R1' operand census
    pk = (ROOT / SPO / 'packed.cuh').read_text()
    store = [i + 1 for i, l in enumerate(pk.splitlines()) if 'norms[row]=' in l]
    ap = (ROOT / 'tensor_cuda/tensor_cuda/apa_packed.py').read_text().splitlines()
    reads = [dict(line=i + 1, text=l.strip()) for i, l in enumerate(ap) if re.search(r'\bentries\b', l) and
             not l.strip().startswith('#') and 'PACKED_PINNED_ENTRIES' not in l]
    op = dict(norm_store=dict(ruling="R2' (P1R3-F02)", file=SPO + 'packed.cuh', store_lines=store,
                              text=[pk.splitlines()[i - 1].strip() for i in store],
                              note=('the one TQ4 key-norm store of the packed route (F2 TQ4_CENSUS: prepare owns the '
                                    'norm; pack_tq copies norms[] verbatim)')),
              margin_reads=dict(ruling="R1' (P1R3-F01)", file='tensor_cuda/tensor_cuda/apa_packed.py', lines=reads,
                                note=('`entries` is read in load() (building the loaded table), in the dataclass field, '
                                      'and in check_pinned_margins (validation reads); lookup returns '
                                      'PACKED_PINNED_ENTRIES and packed_attention reaches margin values only through '
                                      'lookup')))
    with (ART / 'OPERAND_CENSUS.json').open('x') as f:
        json.dump(op, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(rows=len(rows), fixed=census['rows_fixed'], all_int64_now=census['all_formed_in_int64_now'],
                          guards=[(e['entry'], e['first_statement_is_guard']) for e in extents],
                          norm_store=store, margin_reads=[r['line'] for r in reads]), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
