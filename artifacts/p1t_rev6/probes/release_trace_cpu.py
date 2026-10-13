"""PASS iff safe-release/error/lifecycle checks hold. diagnostic_oom: PASS iff original transient CUDA error
is retained through message OOM and reported by the following synchronize (exit1 means lost error).
Includes actual product dlpack_operand.cpp in trace module; runtime replacements prove host control flow only.
Prior art: DLPack2017 ownership, CUDA12.6 error slots, CPython3.12 lifecycle, SQLite OOM injection.
Ours: failure combinations and per-device host trace. Run each mode in a fresh process under timeout15.
"""
import _p1t_rev6_trace as n
import sys, json, threading, time, atexit, types
mode=sys.argv[1]
if mode in ('diagnostic_oom_query','diagnostic_oom_sync','diagnostic_oom_device'):
    calls=[]
    if mode.endswith('query'): n.queue(lambda:calls.append(1));n.inject('query')
    elif mode.endswith('sync'):n.inject('sync')
    else:n.inject('device')
    n.inject('message')
    try:n.synchronize()
    except MemoryError:print('FIRST MemoryError',flush=True)
    else:raise AssertionError('expected message allocation failure')
    st=dict(n.stats());print('AFTER_OOM',json.dumps(st),flush=True)
    try:n.synchronize()
    except RuntimeError as e:
        assert 'cudaErrorInvalidValue' in str(e);print('PASS retained error',e,flush=True)
        n.synchronize()
    else:
        print('FAIL second synchronize returned success: transient CUDA diagnostic disappeared',flush=True);sys.exit(1)
elif mode=='query_once':
    calls=[];n.queue(lambda:calls.append(1));n.inject('query');n.drain()
    assert n.stats()['event_query_failures']==1 and n.slot()==0
    try:n.synchronize()
    except RuntimeError as e:assert 'cudaEventQuery' in str(e)
    else:raise AssertionError('error lost')
    assert calls==[1];n.synchronize();print('PASS query once')
elif mode=='multidevice_restore':
    calls=[];n.queue(lambda:calls.append(0),0);n.queue(lambda:calls.append(2),2);n.set_device(1);n.inject('restore')
    try:n.synchronize()
    except RuntimeError as e:assert 'restoring' in str(e)
    else:raise AssertionError('restore failure hidden')
    assert calls==[0,2];assert set(n.syncs())=={0,1,2};n.synchronize();print('PASS per-device sync',n.syncs())
elif mode=='prior_oom':
    n.inject('prior');n.inject('message')
    try:n.synchronize()
    except MemoryError:pass
    else:raise AssertionError('expected MemoryError')
    try:n.synchronize()
    except RuntimeError as e:assert 'already pending' in str(e)
    else:raise AssertionError('prior lost')
    n.synchronize();print('PASS prior persisted through OOM')
elif mode=='deferred_exit':
    started=threading.Event();done=threading.Event()
    def slow():
        print('DELETER started',flush=True);started.set();time.sleep(.6);done.set();print('DELETER finished',flush=True)
    n.queue(slow)
    def hook():
        n.synchronize();n.close(5.0);assert done.is_set();print('HOOK returned',flush=True)
    atexit.register(hook)
    def start():
        threading.Thread(target=n.drain,daemon=True).start();assert started.wait(2)
        assert n.stats()['entries']==0 and n.stats()['releases_in_flight']==1
    atexit.register(start)
    class Final:
        def __init__(self):self.sleep,self.fin=time.sleep,sys.is_finalizing
        def __del__(self):print('FINALIZING',self.fin(),flush=True);self.sleep(.3)
    h=types.ModuleType('_p1t_rev6_holder');h.keep=Final();sys.modules[h.__name__]=h
elif mode=='inside_callback':
    n.queue(lambda:(n.synchronize(),n.close(5.0)))
    n.drain();assert n.stats()['release_admission_closed'] and n.stats()['releases_in_flight']==0
    n.queue(lambda: (_ for _ in ()).throw(AssertionError('late deleter called')));n.drain()
    assert n.stats()['deleters_leaked_after_close']==1;print('PASS self exclusion and late release')
else:raise ValueError(mode)
