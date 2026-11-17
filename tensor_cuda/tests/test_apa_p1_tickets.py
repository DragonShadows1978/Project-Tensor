"""P1 per-device ticket ownership and lead-only native concurrency test.

Prior art: NVIDIA CUDA Samples threadFenceReduction and stream-ordered lifetime
APIs (2021); ISO C++11 mutex/map. Taken: existing tickets and legacy stream;
new here: device ownership unit coverage. Literature unverified -- lead to
check: CUDA 11.2 stream ordered memory allocator 2021.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/apa_p1_s3_r2'
GPU = os.environ.get('APA_P1_GPU') == '1'


def compile_cpp(tmp_path, source, name):
    p = tmp_path / (name + '.cpp')
    p.write_text(source)
    exe = tmp_path / name
    cmd = ['timeout', '45s', 'c++', '-std=c++17', '-O2', '-pthread', str(p), '-o', str(exe)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return exe


def test_per_device_ticket_ownership(tmp_path):
    """Compile the actual registry, EB4 wrapper and existing checked guard.

    The CUDA API recorder is a CPU unit instrument, not evidence of a device
    runtime. GPU ownership/concurrency is reserved for the staged lead run.
    """
    eb3 = (ROOT / 'tensor_cuda/src/apa_spo/eb3.cuh').read_text()
    eb4 = (ROOT / 'tensor_cuda/src/apa_spo/eb4.cuh').read_text()
    core = (ROOT / 'tensor_cuda/src/kernels.cu').read_text()
    host = eb3.split('// APA_P1_HOST_TICKETS_BEGIN', 1)[1].split('\n', 1)[1].split('// APA_P1_HOST_TICKETS_END', 1)[0]
    guard = core[core.index('class CheckedDeviceGuard {'):]
    guard = guard[:guard.index('\n};') + 3]
    choose = eb4[eb4.index('inline unsigned* counters('):]
    choose = choose[:choose.index('\n}') + 2]
    prefix = r'''
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <iostream>
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaDevAttrL2CacheSize=1;
thread_local int current_device=0;
std::map<void*,int> owner;
std::map<int,int> syncs,attributes,allocations;
std::map<void*,size_t> sizes;
std::atomic<unsigned long long> g_dlpack_device_restore_failures{0};
int cudaGetDevice(int* d){*d=current_device;return 0;}
int cudaSetDevice(int d){current_device=d;return 0;}
int cudaGetLastError(){return 0;}
[[noreturn]] void raise_cuda_failure(const std::string& msg,int){throw std::runtime_error(msg);}
void note_core_unraisable(const char*,int,int){}
int cudaMalloc(unsigned** p,size_t n){*p=new unsigned[n/4];owner[*p]=current_device;sizes[*p]=n;allocations[current_device]++;std::memset(*p,0xcd,n);return 0;}
int cudaMemset(void* p,int v,size_t n){assert(owner.at(p)==current_device && sizes.at(p)==n);std::memset(p,v,n);return 0;}
int cudaFree(void* p){assert(owner.at(p)==current_device);return 0;} // keep CPU instrument allocations for inspection
int cudaDeviceSynchronize(){syncs[current_device]++;return 0;}
int cudaDeviceGetAttribute(int* v,int,int d){assert(d==current_device);*v=(d+1)*1000000;attributes[d]++;return 0;}
'''
    main = r'''
int main(){
  using eb3::TicketDeviceGuard;
  int64_t bytes=0;
  auto* m0=eb3::counters(1,&bytes);auto* c0=eb4::counters(2);
  assert(bytes==4096 && m0!=c0 && owner[m0]==0 && owner[c0]==0);
  for(int i=0;i<1024;++i)assert(m0[i]==0 && c0[i]==0);
  assert(eb3::l2_bytes()==1000000 && eb3::l2_bytes()==1000000 && attributes[0]==1);
  unsigned *m1,*c1;
  {TicketDeviceGuard g(1,"unit");m1=eb3::counters(3);c1=eb4::counters(4);assert(m1!=c1 && m1!=m0 && c1!=c0);assert(owner[m1]==1 && owner[c1]==1);assert(eb3::l2_bytes()==2000000 && eb3::l2_bytes()==2000000 && attributes[1]==1);g.restore();}
  assert(current_device==0);
  try{TicketDeviceGuard g(1,"unit unwind");throw std::runtime_error("unit");}catch(const std::runtime_error&){}
  assert(current_device==0);
  auto* grown=eb3::counters(2049,&bytes);
  assert(grown!=m0 && bytes==2049*4 && syncs[0]==1 && syncs[1]==0);
  assert(eb4::counters(0)==c0 && owner[grown]==0);
  for(int i=0;i<2049;++i)assert(grown[i]==0);
  {TicketDeviceGuard g(1,"unit");assert(eb3::counters(0)==m1);assert(eb4::counters(2049)!=c1);assert(syncs[1]==1);g.restore();}
  auto worker=[](int d){for(int i=0;i<100;++i){std::lock_guard<std::recursive_mutex> lock(eb3::ticket_mutex());TicketDeviceGuard g(d,"unit threads");auto* m=eb3::counters(2);auto* c=eb4::counters(2);assert(m!=c && owner[m]==d && owner[c]==d);g.restore();assert(current_device==0);}};
  std::thread a(worker,0),b(worker,1);a.join();b.join();
  assert(eb3::device_tickets().size()==2 && allocations[0]==3 && allocations[1]==3);
  std::cout << "PASS: two devices; distinct merge/choose; first-use zero; device-local growth; per-device L2; restore/unwind; native host threads\n";
}
'''
    source = prefix + guard + '\nnamespace eb3 { namespace tune {constexpr int COUNTER_MIN_ENTRIES=1024;}\n' + host + '\n}\nnamespace eb4 {namespace tune {constexpr int COUNTER_MIN_ENTRIES=1024;}\n' + choose + '\n}\n' + main
    exe = compile_cpp(tmp_path, source, 'ticket_ownership')
    result = subprocess.run(['timeout', '10s', str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    # The actual native entry must hold its lock over lookup and submission.
    for text, name in [(eb3, 'apa_eb3_run'), (eb4, 'apa_eb4_run')]:
        launcher = text[text.index('std::vector<NDArray> ' + name):]
        assert launcher.index('ticket_launch_lock') < launcher.index('NDArray arena')
        assert launcher.index('operand_device(q.device.index') < launcher.index('NDArray arena')
        assert launcher.index('ticket_launch_lock') < launcher.index('<<<')
        assert 'operand_device.restore();' in launcher


def gpu_runtime():
    assert GPU, 'Lead-only GPU test; set APA_P1_GPU=1 under forge-gpu-run'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != '', 'GPU visibility is empty'
    import tensor_cuda as tc
    native = tc._spo2
    assert hasattr(native, '_p1_test_run'), 'This round needs TC_APA_P1_TEST_HOOKS=1'
    return tc, native


def float_bits(tensor):
    return tensor.numpy().astype(np.float32).view(np.uint32).copy()


def assert_informative(a):
    assert np.any(a != 0), 'zero reference'
    assert a.size > 1 and np.any(a != a.flat[0]), 'constant reference'


@pytest.mark.skipif(not GPU, reason='staged lead GPU test; CPU seat has no GPU')
def test_two_host_threads_bitwise():
    """Native GIL-releasing S2 test entry exercises the same launch lock.

    Prior art: serial-reference concurrency testing; no prior art known to me
    for this exact packed fixture. One legacy default stream on each device.
    """
    tc, native = gpu_runtime()
    os.environ.pop('APA_P1_EARLY_COUNTER', None)
    tc.set_alloc_pooling(True)
    from test_apa_p1_exact_tie import tie_fixture
    q, k, v = tie_fixture()
    T = lambda x: tc.tensor(np.ascontiguousarray(x), dtype='bfloat16')
    qt, kt, vt = T(q), T(k), T(v)
    w = tc.ApaSpq1Workspace(kt, 2049, fmt='tq4')
    vi, vs = tc.apa_eb2b_quantize(vt)
    scale = float(np.float32(1 / np.sqrt(128)))
    def run(mixed):
        return native._p1_test_run(mixed, qt, kt, vt, vi, vs, w.packed, scale, 8.1, 1 / 32, False, 0)
    refs = {mixed: float_bits(run(mixed)[0]) for mixed in (False, True)}
    for ref in refs.values(): assert_informative(ref)
    barrier = threading.Barrier(2, timeout=20)
    def worker(mixed):
        for _ in range(64):
            barrier.wait()
            # Calls release the GIL inside the native test binding.
            out = run(mixed)[0]
            assert np.array_equal(float_bits(out), refs[mixed]), 'thread/serial bitwise mismatch'
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, mixed) for mixed in (False, True)]
        for f in futures: f.result(timeout=120)
