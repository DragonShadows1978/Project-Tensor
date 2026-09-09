"""A3 immutable cells. Prior art: SP3 (2026) registered bounded DAGs."""
from apa_sp4g_common import A, read

REGISTRATION = A / 'amendment_010_a3_registration.json'
REGISTRATION_SHA = 'd2da08a10e08413887bced5da9829cbfbca0e1ffb5545f1cdae3f63db5e46852'

def cells():
    return read(REGISTRATION)['cells']

def by_id():
    return {c['id']: c for c in cells()}
