"""Prior art: SP4G A4/A5 (2026), immutable experiment DAG; wiring only."""
from apa_sp4g_common import A, read

REGISTRATION = A/'amendment_019_a6_ruling.json'
REGISTRATION_SHA = 'ac2cc1519634cd74e0540790dce7b5aa2a4900a8922d52c6a1f1d8305fa130b3'
SEAL = A/'amendment_020_a6_fingerprint.json'
DIAG = 'diag_a6_propagation_A32_vs_A_2048_w0'
CPU_KINDS = {'aggregate', 'freeze', 'eq'}


def cells():
    return read(REGISTRATION)['cells']


def by_id():
    return {c['id']: c for c in cells()}
