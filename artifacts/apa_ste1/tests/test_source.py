"""Source operand checks, distinct from GPU byte-identity evidence.
Prior art: textual regression comparison, no prior art known to me for this check.
"""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/apa_ste1/scratch/original'


def body(s,name):
    at=s.index(name+'(');start=s.index('{',at);level=1;end=start+1
    while level:
        if s[end]=='{':level+=1
        if s[end]=='}':level-=1
        end+=1
    return s[start:end]


def tokens(s):
    s=re.sub(r'//[^\n]*|/\*.*?\*/','',s,flags=re.S)
    return re.sub(r'\s+','',s)


def test_forward_untouched():
    for filename,names in [('kernels.cu',['apa_selective_fwd_train_kernel','apa_selective_fwd_train','apa_selective_fwd_bk4','apa_selective_fwd_train_variant']),('attention_tf32.cu',['forward','apa_selective_fwd_tf32'])]:
        old=(BASE/'src'/filename).read_text();new=(ROOT/'tensor_cuda/src'/filename).read_text()
        for name in names:assert body(old,name)==body(new,name),name
    # bk4 forward is a namespace-scoped second occurrence, checked separately.
    a=(BASE/'src/kernels.cu').read_text();b=(ROOT/'tensor_cuda/src/kernels.cu').read_text()
    assert body(a[a.index('namespace bk4'):],'forward')==body(b[b.index('namespace bk4'):],'forward')


def test_off_kernel_operands_unchanged():
    for filename,names in [('kernels.cu',['apa_selective_bwd_kernel','apa_selective_bwd_bk1_kernel','apa_selective_bwd_bk2_kv_kernel','backward']),('attention_tf32.cu',['backward'])]:
        old=(BASE/'src'/filename).read_text();new=(ROOT/'tensor_cuda/src'/filename).read_text()
        for name in names:
            x=body(new,name)
            x=x.replace('if constexpr (WRITE_KV || DK_STE)', 'if constexpr (WRITE_KV)')
            x=x.replace('(WRITE_KV || DK_STE)', 'WRITE_KV')
            x=x.replace('((sel || DK_STE) && (WRITE_KV || !sel))','(sel)')
            x=x.replace('(sel || DK_STE)', '(sel)').replace('(selected || DK_STE)', '(selected)')
            x=x.replace('(selected || (KEY && DK_STE))', 'selected')
            assert tokens(body(old,name))==tokens(x),name
