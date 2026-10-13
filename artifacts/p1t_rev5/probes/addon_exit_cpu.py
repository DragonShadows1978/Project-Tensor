"""PASS iff an admitted add-on deleter finishes before finalization, child exits 0.
Prior art: CPython 3.12 lifecycle/GIL and C++17 noexcept; ours: queue-hidden callback at exit.
CPU call trace: actual dlpack_operand.cpp, stubbed successful CUDA completion. No GPU execution.
"""
import atexit, sys, time, threading, types, resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import _p1t_rev5_trace as n
started=threading.Event()
def deleter():
    print('DELETER started',flush=True)
    started.set()
    time.sleep(0.6)
    print('DELETER finished',flush=True)
n.queue(deleter)
def close():
    n.synchronize()
    print('HOOK returned entries='+str(n.stats()['entries']),flush=True)
atexit.register(close)
def start():
    threading.Thread(target=n.drain,daemon=True).start()
    assert started.wait(2)
    print('CALLBACK IN PROGRESS entries='+str(n.stats()['entries']),flush=True)
atexit.register(start)
class Finalizer:
    def __init__(self):self.sleep,self.fin=time.sleep,sys.is_finalizing
    def __del__(self):
        print('FINALIZING '+str(self.fin()),flush=True)
        self.sleep(1.5)
holder=types.ModuleType('_p1t_rev5_holder');holder.keep=Finalizer();sys.modules[holder.__name__]=holder;del holder
print('ARMED',flush=True)
