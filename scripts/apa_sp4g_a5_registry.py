"""Prior art: SP4G (2026) immutable registered experiment cells; wiring only."""
from apa_sp4g_common import A, read

REGISTRATION = A/'amendment_017_a5_registration.json'
REGISTRATION_SHA = '68835c48c2fb800ed475bd25affc12920a8c1b54b2292dedefb3f7de70b4ccf0'
GATE = 'ppl_a4_D32_2048_w0'
CALLS = ['diag_a5_call_l05_b00', 'diag_a5_call_l05_b15', 'diag_a5_call_l47_b15']
COMPLETED = 'a4_completed:' + GATE


def cells():
    return read(REGISTRATION)['cells']


def by_id():
    return {c['id']: c for c in cells()}
