# Lead check: per-kernel SASS diff of two cuobjdump dumps against a registered expectation file.
# Independent of the seat's script: anonymous-namespace suffixes _cu_<8 hex> are replaced by a placeholder in names and
# bodies (a coarser map than P1-D's bijective one, so it can only hide differences, never invent them); bodies compared
# as exact text otherwise.
import re, sys, json, subprocess, collections
SUF = re.compile(r'_cu_[0-9a-f]{8}(_[0-9]+)?')
def read(p):
    blocks, key, buf = {}, None, []
    for line in open(p):
        m = re.search(r'Function : (\S+)', line)
        if m:
            if key: blocks[key] = ''.join(buf)
            key, buf = SUF.sub('_cu_X', m.group(1)), []
            continue
        if key is not None: buf.append(SUF.sub('_cu_X', line))
    if key: blocks[key] = ''.join(buf)
    return blocks
B, F = read(sys.argv[1]), read(sys.argv[2])
exp = json.load(open(sys.argv[3]))['sass']
names = sorted(set(B) | set(F))
dem = dict(zip(names, subprocess.run(['c++filt'], input='\n'.join(names), capture_output=True, text=True).stdout.split('\n')))
def listed(section):
    out = set()
    for v in exp.get(section, {}).values(): out |= set(v)
    return out
def match(n, lst):
    d = dem[n]
    return any(d == e or d.startswith(e + '(') for e in lst)
chg = [n for n in names if n in B and n in F and B[n] != F[n]]
rem = [n for n in names if n in B and n not in F]
add = [n for n in names if n in F and n not in B]
EC, ER, EA = listed('expected_changed'), listed('expected_removed'), listed('expected_added')
r = dict(B=len(B), F=len(F), changed=len(chg), removed=len(rem), added=len(add),
         unlisted_changed=[dem[n] for n in chg if not match(n, EC)],
         unlisted_removed=[dem[n] for n in rem if not match(n, ER)],
         unlisted_added=[dem[n] for n in add if not match(n, EA)])
seen = lambda lst, pool: [e for e in lst if not any(dem[n] == e or dem[n].startswith(e + '(') for n in pool)]
r['listed_changed_not_changed'] = seen(EC, chg)
r['listed_removed_not_removed'] = seen(ER, rem)
r['listed_added_not_added'] = seen(EA, add)
r['n_unlisted_changed'] = len(r['unlisted_changed'])
r['unlisted_changed'] = r['unlisted_changed'][:8]
print(json.dumps(r, indent=1))
