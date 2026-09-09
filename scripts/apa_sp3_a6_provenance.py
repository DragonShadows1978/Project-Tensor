"""Reviewed A5-to-A6 endpoints. Prior art: A4/A5 source hash bridges (2026),
Make (Feldman 1979) / Nix (Dolstra 2004), unverified lead: dependency hashing.
Reuse exact endpoint projection; not a general semantic equivalence proof.
"""
import hashlib
from apa_sp3_common import ART, REG_SHA, Red, read, sha
from apa_sp3_a6_registry import MANIFEST, manifest

BRIDGE='amendment_012_clean_fingerprint.json'


def extend_bridge(parent):
    reg=manifest();m=read(ART/BRIDGE)
    if (m['registration_sha256']!=REG_SHA or m['order_sha256']!=reg['order_sha256']
        or m['cell_amendment_sha256']!=sha(ART/MANIFEST)
        or m['parent_effective_sha256']!=parent['effective_sha256']
        or reg['parent_a5_effective_sha256']!=parent['effective_sha256']
        or sha(ART/BRIDGE)!=(ART/(BRIDGE+'.sha256')).read_text().strip()):
        raise Red('A6 fingerprint amendment binding changed')
    return dict(parent,a6_transition=m,a6_parent_effective_sha256=parent['effective_sha256'],
                effective_sha256=hashlib.sha256((parent['effective_sha256']+sha(ART/BRIDGE)).encode()).hexdigest())


def previous_current(cell,current,amendment):
    # Same conservative A5 projection: exact hashes and explicitly reviewed kinds.
    from apa_sp3_a5_provenance import previous_current as project
    return project(cell,current,amendment)
