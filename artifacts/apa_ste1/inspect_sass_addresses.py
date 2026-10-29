"""Describe literal SASS mismatches without weakening the identity check.
Prior art: NVIDIA cuobjdump; address interpretation is reasoning, unverified.
"""
import runpy
import re
import json
from pathlib import Path

P=Path(__file__).resolve().parent
r=runpy.run_path(str(P/'check_sass.py'))
records=[]
for row in r['rows']+r['forward']:
    if row['PASS']:continue
    a=r['old'][row.get('base_name',row['name'])].splitlines();b=r['new'][row['name']].splitlines()
    changes=[];other=[]
    for x,y in zip(a,b):
        if x==y:continue
        if not re.match(r'/\*[0-9a-f]+\*/',x):other.append([x,y]);continue
        left=x.split('*/',1)[1].split('/*')[0].strip();right=y.split('*/',1)[1].split('/*')[0].strip()
        opcode=left.split()[1] if left.startswith('@') else left.split()[0]
        only=re.sub(r'0x[0-9a-f]+','IMMEDIATE',left)==re.sub(r'0x[0-9a-f]+','IMMEDIATE',right)
        allowed=opcode in ('MOV','UMOV') or opcode.startswith(('LDS','STS'))
        changes.append({'base':left,'arm':right,'address_instruction_immediate_only':only and allowed})
    records.append({'name':row['name'],'same_instruction_count':len(a)==len(b),'address_immediates_only':len(a)==len(b) and not other and all(c['address_instruction_immediate_only'] for c in changes),'changes':changes,'other_differences':other})
(P/'receipts/SASS_ADDRESS_DIFFS.json').write_text(json.dumps({'literal_identity':'RED for these specializations; SASS_IDENTITY.json unchanged','evidence_class':'static instruction classification; shared-allocation interpretation is reasoning','records':records},indent=2)+'\n')
print('Address-immediate-only mismatches:',sum(x['address_immediates_only'] for x in records),'of',len(records))
