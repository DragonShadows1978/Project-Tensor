# Prior art: NVIDIA cuobjdump SASS disassembly; exact instruction census.
# Project-Tensor AB-1 (2026) normalizes anonymous-namespace source hashes.
from pathlib import Path
import re,json,subprocess,collections,hashlib
A=Path('artifacts/apa_p1')
# These namespaces are the entire new header closure (not production kernels).
namespace=re.compile(r'^_ZN2tc(?:[0-9]+(?:spo[245]|spq[146]|eb[234]b?)[0-9]|[0-9]+apa_spo2_baseline_dump_kernel)')
def read(p):
    result={}; key=None
    for line in p.open():
        m=re.search(r'Function : (\S+)',line)
        if m:
            key=m[1] if namespace.search(m[1]) else None
            if key: assert key not in result,key; result[key]=[]
        elif key:
            m=re.match(r'\s*/\*[0-9a-f]+\*/\s+(.*?);',line)
            if m: result[key].append(m[1].strip())
    return result
port=read(A/'sass_port.txt'); lab=read(A/'sass_lab.txt')
assert port and lab
names=sorted(set(port)|set(lab)); demangled=subprocess.check_output(['c++filt',*names],text=True).splitlines()
rows=[]
for name,label in zip(names,demangled):
    p=port.get(name,[]); l=lab.get(name,[])
    hist=lambda lines:dict(sorted(collections.Counter(re.sub(r'^@!?P\d+\s+','',x).split()[0] for x in lines).items()))
    ph,lh=hist(p),hist(l)
    rows.append(dict(symbol=name,kernel=label,port_count=len(p),lab_count=len(l),delta=len(p)-len(l),opcode_histogram_equal=ph==lh,instruction_text_equal=p==l,port_opcode_histogram=ph,lab_opcode_histogram=lh,cause=None if len(p)==len(l) and ph==lh else 'UNRESOLVED'))
diffs=[x for x in rows if x['delta'] or not x['opcode_histogram_equal'] or not x['port_count'] or not x['lab_count']]
r=dict(status='PASS' if not diffs else 'FAIL',kernels=len(rows),port_total=sum(x['port_count'] for x in rows),lab_total=sum(x['lab_count'] for x in rows),count_differences=len(diffs),instruction_text_differences=sum(not x['instruction_text_equal'] for x in rows),namespace_filter=namespace.pattern,rows=rows,differences=diffs,method='cuobjdump 12.6 --dump-sass --gpu-architecture sm_89; count address-prefixed instructions including alignment NOPs; match mangled symbols. Full instruction text and opcode histograms also checked.')
(A/'sass_comparison.json').write_text(json.dumps(r,indent=2)+'\n')
with (A/'sass_counts.tsv').open('w') as f:
    f.write('kernel\tport_instructions\tlab_instructions\tdelta\topcode_histogram_equal\tinstruction_text_equal\n')
    for x in rows:f.write(f"{x['kernel']}\t{x['port_count']}\t{x['lab_count']}\t{x['delta']}\t{x['opcode_histogram_equal']}\t{x['instruction_text_equal']}\n")
print(json.dumps({k:v for k,v in r.items() if k not in ('rows','differences')},indent=2))
for d in diffs:print(d['kernel'],d['port_count'],d['lab_count'])
