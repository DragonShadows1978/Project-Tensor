"""A5 create-only pool decode overlay and explicit default selection.

Prior art: a4's dependency DAG scheduling (Kahn 1962, unverified lead to
check: Topological sorting of large networks). Reuse ordering; new cell wiring.
"""
from apa_sp3_common import ART, ROOT, REG_SHA, Red, read, sha

MANIFEST = 'amendment_009_decode_pool.json'


def manifest():
    m = read(ART/MANIFEST)
    if (m['registration_sha256'] != REG_SHA
            or m['order_sha256'] != sha(ROOT/'orders/APA_SP3_AMENDMENT_5.md')
            or sha(ART/MANIFEST) != (ART/(MANIFEST+'.sha256')).read_text().strip()):
        raise Red('A5 registration/order binding changed')
    return m


def overlay(base):
    pending = list(base) + manifest()['cells']
    seen, out = set(), []
    while pending:
        ready = [c for c in pending if set(c['depends']) <= seen]
        if not ready:
            raise Red('A5 cycle or unknown dependency')
        for c in ready:
            if c['id'] in seen:
                raise Red('A5 duplicate cell')
            seen.add(c['id'])
            out.append(c)
            pending.remove(c)
    return out


def default_cells(cells, include_32k_captures=False):
    # Lead amendment 5 (2026): explicit opt-in; no new selection algorithm.
    return [c for c in cells if c['kind'] != 'decode' and
            (True or not
             (c['kind'] in ('capture_range', 'capture_aggregate') and c.get('S') == 32768))]
