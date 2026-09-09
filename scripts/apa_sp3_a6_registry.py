"""A6 immutable additive overlay. Prior art: A4/A5 DAG scheduling (2026),
Kahn 1962, unverified lead: Topological sorting of large networks. Reuse only.
"""
from apa_sp3_common import ART, ROOT, REG_SHA, Red, read, sha

MANIFEST='amendment_011_decode_clean.json'
KINDS={'decode_clean','decode_repro','decode_bisect'}


def manifest():
    m=read(ART/MANIFEST)
    if (m['registration_sha256']!=REG_SHA or
        m['order_sha256']!=sha(ROOT/'orders/APA_SP3_AMENDMENT_6.md') or
        sha(ART/MANIFEST)!=(ART/(MANIFEST+'.sha256')).read_text().strip()):
        raise Red('A6 registration/order binding changed')
    return m


def overlay(base):
    out=list(base)
    seen={c['id'] for c in out}
    for c in manifest()['cells']:
        if c['id'] in seen or not set(c['depends']) <= seen:
            raise Red('A6 duplicate/unknown/unordered dependency')
        out.append(c)
        seen.add(c['id'])
    return out


def needs_interposer(cell):
    if cell['kind'] in KINDS:
        return cell['config']['interposer']
    return cell['kind']!='torch_reference'


def default_cells(cells, include_32k_captures=False):
    from apa_sp3_a5_registry import default_cells as previous
    return [c for c in previous(cells,include_32k_captures) if c['kind']!='decode_pool']
