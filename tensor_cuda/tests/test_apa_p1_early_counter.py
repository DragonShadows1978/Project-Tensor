"""Deterministic survey-S3 poisoned-partial handshake, compiled structural and GPU.

Prior art: DeMillo, Lipton, Sayward (1978) mutation testing; CUDA happens-before;
APA-2P survey S3 (2026). Taken: its ordered two-CTA design. No prior art known
to me for this exact fixture. Unverified -- lead to check: mutation testing
1978; CUDA memory fence atomic release acquire.
"""
from pathlib import Path
import os
import re
import subprocess

import numpy as np
import pytest

from test_apa_p1_tickets import ROOT, ART, GPU, compile_cpp, gpu_runtime, float_bits, assert_informative


def test_early_counter_compiled_structure(tmp_path):
    """Compile the actual device handshake and each actual ticket predicate.

    One CPU thread represents a CTA leader; CUDA barriers/fences are CPU
    synchronization instruments. This is structural coverage, not a GPU run.
    """
    text3 = (ROOT / 'tensor_cuda/src/apa_spo/eb3.cuh').read_text()
    text4 = (ROOT / 'tensor_cuda/src/apa_spo/eb4.cuh').read_text()
    helpers = text3.split('// APA_P1_EARLY_DEVICE_BEGIN', 1)[1].split('\n', 1)[1].split('// APA_P1_EARLY_DEVICE_END', 1)[0]
    predicates = {}
    for name, text in [('eb3', text3), ('eb4', text4)]:
        kernel = text[text.index('__global__ void __launch_bounds__(128' + (',MINB)' if name == 'eb3' else ',eb3::tune::SEL_MINB)')):]
        # Select the last-writer predicate, not the optional bulk-choose ticket.
        if name == 'eb3': kernel = text[text.index(' select_pv('):]
        else: kernel = text[text.index(' select_mixed('):]
        start = kernel.index('  if(tid==0){\n', kernel.index('__shared__ unsigned last_s;'))
        end = kernel.index('\n  }', start)
        predicates[name] = kernel[start:end + 4]
        assert kernel.index('early_counter_wait(test_hook,c)') < kernel.index('pm[row*NC+c]=m')
        early = kernel[kernel.index('if(!last_s)'):kernel.index('if(!last_s)') + 230]
        assert 'early_counter_done(test_hook,c)' in early
        # The second signal follows all merge output stores.
        assert kernel.rindex('early_counter_done(test_hook,c)') > kernel.rindex('o[3]=__float2bfloat16_rn')
    prefix = r'''
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#define __device__
#define __forceinline__ inline
#define __shared__ thread_local
struct {int x=0;} threadIdx;
void __syncthreads(){std::atomic_thread_fence(std::memory_order_seq_cst);}
void __threadfence(){std::atomic_thread_fence(std::memory_order_seq_cst);}
int __syncthreads_or(int value){std::atomic_thread_fence(std::memory_order_seq_cst);return value!=0;}
unsigned long long clock64(){return std::chrono::steady_clock::now().time_since_epoch().count();}
unsigned atomicAdd(unsigned* p,unsigned v){return __atomic_fetch_add(p,v,__ATOMIC_SEQ_CST);}
unsigned atomicExch(unsigned* p,unsigned v){return __atomic_exchange_n(p,v,__ATOMIC_SEQ_CST);}
unsigned atomicInc(unsigned* p,unsigned limit){unsigned old=__atomic_load_n(p,__ATOMIC_SEQ_CST);for(;;){unsigned next=old>=limit?0:old+1;if(__atomic_compare_exchange_n(p,&old,next,false,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST))return old;}}
'''
    main = r'''
bool takes_ticket(unsigned* ctr){int tid=0,bh=0,NC=2;unsigned last_s=0;
PREDICATE
return last_s;
}
int main(){
 unsigned state[2]={0,0},counter=0;float partial[2]={NAN,NAN},out=NAN;
 EarlyCounterHook hook{state,5000000000ull};
 std::thread delayed([&]{assert(early_counter_wait(hook,1));partial[1]=7.f;if(takes_ticket(&counter))out=partial[0]+partial[1];});
 partial[0]=3.f;
 if(takes_ticket(&counter))out=partial[0]+partial[1];
 early_counter_done(hook,0);delayed.join();
 assert(counter==0 && state[0]==1 && state[1]==0);
 // Timeout is a harness failure, never counted as a successful mutant kill.
 unsigned timed[2]={0,0};assert(!early_counter_wait({timed,1},1) && timed[1]==1);
 if(out!=10.f){std::cout << "deterministic poisoned-partial output mismatch\n";return 1;}
 std::cout << "PASS ordered two-CTA structural check\n";
}
'''
    rows = []
    for family in ('eb3', 'eb4'):
        for mutant in (False, True):
            predicate = predicates[family]
            defines = '\n#define APA_EB3_MUTANT ' + ('3' if mutant and family == 'eb3' else '0') + '\n'
            if mutant and family == 'eb4':
                assert predicate.count('last_s=prev==(unsigned)(NC-1);') == 1
                predicate = predicate.replace('last_s=prev==(unsigned)(NC-1);', 'last_s=prev==(unsigned)(NC-2);')
            exe = compile_cpp(tmp_path, prefix + defines + helpers + main.replace('PREDICATE', predicate), family + ('_early' if mutant else '_correct'))
            result = subprocess.run(['timeout', '10s', str(exe)], capture_output=True, text=True)
            expected = 1 if mutant else 0
            assert result.returncode == expected, result.stdout + result.stderr
            rows.append((family, mutant, result.returncode, result.stdout.strip()))
    # Retain actual CPU evidence under pytest's authorized /tmp basetemp.
    (tmp_path / 'structural_results.txt').write_text(repr(rows) + '\n')


def qwen_two_cta_input():
    """Read only on the lead GPU host. Paths/hashes are inherited A1 records.

    The CPU seat reads only A1's metadata, not the external SPQ3 captures.
    Real post-norm/post-RoPE Qwen3-1.7B L05: two q heads and one KV head,
    query at row 256, real K/V prefix [0,257), no synthetic replacement.
    """
    import json
    import hashlib
    record = json.loads((ROOT / 'artifacts/apa_p1_a1/INPUT_CENSUS.json').read_text())[0]
    result = []
    for name in 'qkv':
        row = record['files'][name]
        p = ROOT / row['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256'], str(p)
        a = np.load(p, mmap_mode='r', allow_pickle=False)
        assert a.dtype == np.uint16 and list(a.shape) == row['shape']
        u = a[:2, 256:257] if name == 'q' else a[:1, :257]
        f = (np.ascontiguousarray(u).astype(np.uint32) << 16).view(np.float32)[None]
        assert np.isfinite(f).all()
        assert_informative(f.view(np.uint32))
        result.append(f)
    return result


def test_two_cta_operand_census_cpu():
    import json
    case = json.loads((ROOT / 'artifacts/apa_p1_a1/INPUT_CENSUS.json').read_text())[0]
    assert case['model'] == 'Qwen3-1.7B' and case['layer'] == 5
    assert case['files']['q']['shape'][0] >= 2
    for name in 'qkv':
        assert case['files'][name]['dtype'] == 'uint16'
        assert case['files'][name]['shape'][1] >= 257
        assert case['files'][name]['shape'][2] == 128
        assert len(case['files'][name]['sha256']) == 64
    assert (257 + 255) // 256 == 2


@pytest.mark.skipif(not GPU, reason='staged lead GPU test; CPU seat has no GPU')
@pytest.mark.parametrize('family', ['eb3', 'eb4'])
def test_early_counter_gpu(family):
    tc, native = gpu_runtime()
    q, k, v = qwen_two_cta_input()
    T = lambda x: tc.tensor(np.ascontiguousarray(x), dtype='bfloat16')
    qt, kt, vt = T(q), T(k), T(v)
    w = tc.ApaSpq1Workspace(kt, 257, fmt='tq4')
    vi, vs = tc.apa_eb2b_quantize(vt)
    scale = float(np.float32(1 / np.sqrt(128)))
    os.environ.pop('APA_P1_EARLY_COUNTER', None)
    # EB3 variant 8 is the same ordered merge in a separate kernel and is
    # unaffected by EB3 mutant 3 or the EB4 early-counter analogue.
    ref = float_bits(native._p1_test_run(False, qt, kt, vt, vi, vs, w.packed, scale, 8.1, 1 / 32, True, 8)[0])
    assert_informative(ref)
    os.environ['APA_P1_EARLY_COUNTER'] = '1'
    try:
        got = float_bits(native._p1_test_run(family == 'eb4', qt, kt, vt, vi, vs, w.packed, scale, 8.1, 1 / 32, True, 0)[0])
    finally:
        os.environ.pop('APA_P1_EARLY_COUNTER', None)
    # Named negative controls: EB3 APA_EB3_MUTANT=3; EB4 select_mixed's
    # last_s predicate NC-1 -> NC-2. Both MUST fail this exact assertion.
    assert np.array_equal(ref, got), 'deterministic poisoned-partial output mismatch'
