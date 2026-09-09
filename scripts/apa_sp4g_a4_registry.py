"""A4 immutable DAG. Prior art: SP3 (2026) registered bounded dispatch."""
from apa_sp4g_common import A, read
REGISTRATION=A/'amendment_013_a4_registration.json'
REGISTRATION_SHA='9fce2b05849eb4e69e8cb5cc2a3434e4c05510677128473aa1f49a4c28af2b1a'
GATE='ppl_a4_D32_2048_w0'
CPU_KINDS={'aggregate','freeze','eq'}
def cells():return read(REGISTRATION)['cells']
def by_id():return {c['id']:c for c in cells()}
