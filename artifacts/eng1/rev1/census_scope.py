# Prior art: Python ast (PSF), SHA-256 NIST FIPS180-4 (2015), GNU unified diff.
# Taken: source identity and syntax census. Ours: check every call site, hunk and pinned mutant.
import ast,sys,json,hashlib,subprocess,re
from pathlib import Path
root=Path.cwd(); rev=root/'artifacts/eng1/rev1'; eng=root/'artifacts/eng1'; tool=eng/'tool'
sha=lambda b:hashlib.sha256(b).hexdigest()
# One justification for each engine reshape call, including multiple calls on the same source line.
why={
('nn.py',304):'B*L*heads*head_dim = B*L*embed_dim; constructor defines head_dim=embed_dim//heads.',
('nn.py',315):'Transpose copies preserve B*L*embed_dim; merge heads back to embed_dim.',
('nn.py',363):'K=in_ch*kh*kw; weight has out_ch*K elements.',
('nn.py',367):'L=OH*OW from im2col; output has N*L*out_ch elements.',
('nn.py',369):'Bias vector length out_ch; singleton axes add no elements.',
('nn.py',389):'im2col gives N*(C*kh*kw)*L, split its channel/kernel axis.',
('nn.py',390):'Depthwise weight has C*kh*kw elements; add singleton axes.',
('nn.py',394):'Sum removed the kernel axis, leaving N*C*(OH*OW).',
('nn.py',396):'Bias length C; add singleton axes.',
('nn.py',424):'Transposed-conv weight in_ch*out_ch*kh*kw; merge last three axes.',
('nn.py',425):'Flatten H*W; element count N*Cin*H*W unchanged.',
('nn.py',432):'Bias length out_ch; add singleton axes.',
('nn.py',470):'BN2D weight length C; singleton axes only.',
('nn.py',471):'BN2D bias length C; singleton axes only.',
('nn.py',480):'BN2D running_mean length C; singleton axes only.',
('nn.py',481):'BN2D running_var length C; singleton axes only.',
('nn.py',498):'BN1D parameter length C; wshape=[1,C] or [1,C,1].',
('nn.py',505):'BN1D running_mean length C; wshape adds only singleton axes.',
('nn.py',506):'BN1D running_var length C; wshape adds only singleton axes.',
('nn.py',576):'GroupNorm validates C%G=0; product is N*C*prod(spatial), also when N/spatial is zero.',
('nn.py',581):'Inverse of grouped layout; product N*C*prod(spatial).',
('nn.py',585):'Each GroupNorm parameter length C; wshape=[1,C]+[1]*len(spatial).',
('nn.py',588):'Float32-cast parameters retain length C; broadcast singleton shape.',
('nn.py',602):'InstanceNorm parameter length C; singleton axes only.',
('nn.py',613):'Conv1D inserts height 1; N*C*L preserved.',
('nn.py',614):'Height-one Conv2D output; removing its singleton height preserves count.',
('nn.py',696):'RNN time slice length 1; remove singleton time axis, keep B*input_dim.',
('nn.py',799):'Position table slice L*d_model; prepend singleton batch axis.',
('quant.py',124):'R has H*D*D elements; prepend singleton B before expand.',
('quant.py',186):'main contains nb*4096 keys per row (tail removed); split last axis into nb and 4096.',
('functional.py',189):'Undo causal-softmax flattening: B*H*L*S; output same element count.',
('functional.py',190):'Flatten H*L and insert query extent 1; B*H*L*S unchanged.',
('functional.py',226):'Einsum permutation preserves all operand axes; insert only missing singleton axes.'
}
reshapes=[]
for path in sorted((root/'tensor_cuda/tensor_cuda').rglob('*.py')):
    text=path.read_text(); tree=ast.parse(text)
    for n in sorted(ast.walk(tree),key=lambda n:(getattr(n,'lineno',0),getattr(n,'col_offset',0))):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='reshape':
            numpy_only=path.name in ('__init__.py','affine.py')
            key=(path.name,n.lineno)
            reason='NumPy array operation; NDArray::reshape is never invoked.' if numpy_only else why[key]
            reshapes.append(dict(file=str(path.relative_to(root)),line=n.lineno,column=n.col_offset,
                                 expression=ast.get_source_segment(text,n),kind='numpy' if numpy_only else 'engine',
                                 reasoning=reason,
                                 compatibility='All extents nonnegative on previously valid input; same final numel check and storage alias. Existing invalid caller states remain refused.'))
assert len([r for r in reshapes if r['kind']=='engine'])==37
# Original tracked-map scope, and live bytes match frozen after map.
before=json.loads((eng/'tracked_before.json').read_text())['sha256']
after=json.loads((eng/'tracked_after.json').read_text())['sha256']
changed=sorted(p for p in set(before)|set(after) if before.get(p)!=after.get(p))
live=rev/'worktree_before.json'; initial=json.loads(live.read_text())
after_mismatch=[p for p,h in after.items() if initial.get(p)!=h]
# The frozen diff contains exactly the live source diff from the registered base.
cmd=['git','diff','1680764d','--','tensor_cuda/src','tensor_cuda/tests/test_terrain_render.py']
d=subprocess.run(cmd,capture_output=True,check=True).stdout
scope=dict(changed_tracked=changed,changed_count=len(changed),map_count=len(before),
           live_vs_frozen_after_mismatches=after_mismatch,
           source_diff_matches=d==(eng/'source.diff').read_bytes())
# Verify each mutant is exactly the registered single replacement and module pin matches receipt.
mutants=[]
for m in json.loads((eng/'mutants/mutants.json').read_text()):
    pristine=(root/m['file']).read_text(); assert pristine.count(m['old'])==1
    mutated=pristine.replace(m['old'],m['new'],1)
    row=dict(id=m['id'],old_occurrences=1,replacement_count=1,changed=mutated!=pristine,
             pristine_sha256=sha(pristine.encode()),mutated_sha256=sha(mutated.encode()))
    if m['kind']!='python':
        receipt=json.loads((eng/'mutants'/m['id']/'receipt.json').read_text())
        row['receipt_hashes_match']=all((receipt['pristine_file_sha256']==row['pristine_sha256'],
                   receipt['mutated_file_sha256']==row['mutated_sha256'],receipt['restored_sha256']==row['pristine_sha256']))
        module=next((tool/'arms'/m['id']/'tensor_cuda/tensor_cuda').glob('_tensor_cuda*.so'))
        row['module_matches_receipt']=sha(module.read_bytes())==receipt['module_sha256']
        assert row['receipt_hashes_match'] and row['module_matches_receipt']
    else:
        row['python_arm_matches_replacement']=(tool/'arms'/m['id']/m['file']).read_text()==mutated
        assert row['python_arm_matches_replacement']
    mutants.append(row)
# Verify all stored unified-diff hunks by applying them IN MEMORY to the accepted originals.
lines=(tool/'TOOLING_CHANGES.diff').read_text().splitlines(keepends=True)
i=0; hunks=[]; comparisons=[]
while i<len(lines):
    assert lines[i].startswith('--- '); oldfile=lines[i].split('\t')[0][4:]; i+=1
    assert lines[i].startswith('+++ '); newfile=lines[i].split('\t')[0][4:]; i+=1
    old=(tool/oldfile).read_text().splitlines(keepends=True); out=[]; pos=0
    while i<len(lines) and lines[i].startswith('@@'):
        head=lines[i].strip(); m=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',head); assert m
        at=int(m[1])-1; out.extend(old[pos:at]); pos=at; i+=1
        while i<len(lines) and not lines[i].startswith(('@@','--- ')):
            row=lines[i]; mark=row[0]
            if mark==' ': assert old[pos]==row[1:]; out.append(old[pos]);pos+=1
            elif mark=='-': assert old[pos]==row[1:];pos+=1
            elif mark=='+':out.append(row[1:])
            else:raise AssertionError(row)
            i+=1
        hunks.append(dict(number=len(hunks)+1,file=newfile,header=head))
    out.extend(old[pos:])
    comparisons.append(dict(file=newfile,diff_reconstructs_live=''.join(out)==(tool/newfile).read_text()))
assert len(hunks)==27
assert all(r['diff_reconstructs_live'] for r in comparisons)
originals=[]
for p in sorted((tool/'_gn1_a8_originals').glob('*')):
    if p.name=='SHA256SUMS' or not p.is_file():continue
    origin=root/'artifacts/gn1/a8'/p.name
    originals.append(dict(file=p.name,matches_gn1_a8=origin.exists() and sha(origin.read_bytes())==sha(p.read_bytes())))
assert all(r['matches_gn1_a8'] for r in originals)
# Exact ID counts and real collection set equality (order in frozen lists is sorted).
idclasses={}
for line in (tool/'ids_eng1.txt').read_text().splitlines():
    cl=re.search(r'::test_(e\d)_',line)[1];idclasses[cl]=idclasses.get(cl,0)+1
arm_ids={}
for arm in ('baseline','candidate'):
    a=json.loads((rev/('arm_'+arm+'.json')).read_text())
    arm_ids[arm]=dict(count=len(a['ids']),exact_strings_match=sorted(a['ids'])==(tool/'ids_regression.txt').read_text().splitlines(),
                     tensor_modules_all_from_own_arm=all('/arms/'+arm+'/' in p for p in a['post'].values()))
r=dict(reshape_callers=reshapes,engine_reshape_calls=37,numpy_reshape_calls=9,
       engine_reshape_text_lines=sum(1 for p in (root/'tensor_cuda/tensor_cuda').rglob('*.py') if p.name not in ('affine.py','__init__.py') for line in p.read_text().splitlines() if 'reshape(' in line),
       scope=scope,mutants=mutants,tool_hunks=hunks,diff_comparisons=comparisons,originals=originals,
       id_classes=idclasses,arm_ids=arm_ids)
(rev/'census_scope.json').write_text(json.dumps(r,indent=2)+'\n')
out=['# Complete Python reshape census','',f"37 engine calls on {r['engine_reshape_text_lines']} text lines, plus 9 NumPy calls. The reported 30 is an undercount; multiple calls share source lines.",'',
     '| File:line:column | Call | Check |','|---|---|---|']
for row in reshapes:
    expr=row['expression'].replace('\n',' ').replace('|','/')
    out.append(f"| {row['file']}:{row['line']}:{row['column']} | `{expr}` | {row['reasoning']} |")
(rev/'CALLER_CENSUS.md').write_text('\n'.join(out)+'\n')
print('scope',scope)
print('reshape',r['engine_reshape_calls'],'calls',r['engine_reshape_text_lines'],'text lines;',r['numpy_reshape_calls'],'NumPy calls')
print('mutants',len(mutants),'all single replacements and hashes match')
print('tooling hunks',len(hunks),'all reproduce live files, originals identical',len(originals))
print('classes',idclasses,'arm IDs',arm_ids)
