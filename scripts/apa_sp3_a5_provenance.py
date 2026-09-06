"""Reviewed a4-to-a5 source endpoints; unknown edits fail closed.

Prior art: a4's dependency-directed hash bridge, Make (Feldman 1979), Nix
(Dolstra et al. 2004); unverified leads to check. New SP3 transition audit.
"""
import hashlib
from apa_sp3_common import ART, REG_SHA, Red, read, sha
from apa_sp3_a5_registry import MANIFEST, manifest

BRIDGE = 'amendment_010_pool_fingerprint.json'


def extend_bridge(parent):
    registration = manifest()
    m = read(ART/BRIDGE)
    if (m['registration_sha256'] != REG_SHA
            or m['order_sha256'] != registration['order_sha256']
            or m['cell_amendment_sha256'] != sha(ART/MANIFEST)
            or m['parent_effective_sha256'] != parent['effective_sha256']
            or registration['parent_a4_effective_sha256'] != parent['effective_sha256']
            or sha(ART/BRIDGE) != (ART/(BRIDGE+'.sha256')).read_text().strip()):
        raise Red('A5 fingerprint amendment binding changed')
    return dict(parent, a5_transition=m, parent_effective_sha256=parent['effective_sha256'],
                effective_sha256=hashlib.sha256((parent['effective_sha256']+sha(ART/BRIDGE)).encode()).hexdigest())


def previous_current(cell, current, amendment):
    """Project ONLY exact reviewed endpoints onto the prior closure."""
    previous = dict(current)
    for path, transition in amendment['file_transitions'].items():
        if path not in previous:
            continue
        if (previous[path] != transition['after_sha256']
                or cell['kind'] not in transition['unchanged_kinds']):
            return None
        if transition['before_sha256'] is None:
            del previous[path]
        else:
            previous[path] = transition['before_sha256']
    return previous
