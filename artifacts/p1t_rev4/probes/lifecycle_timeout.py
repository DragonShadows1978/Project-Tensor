"""PASS iff the actual lifecycle slice lets finite deleters finish before finalization, exit 0.
Prior art: CPython 3.12 atexit/finalization rules; round2 core_shutdown staging.
Ours: finite timeout overrun, without os._exit (which would conceal normal finalization).
Run: PYTHONPATH=/tmp/p1t_rev3_build timeout 15 python this_file.py [control|short|default].
No CUDA API or GPU execution. The native helper is a hash-pinned unchanged lifecycle slice.
"""
import atexit, sys, time, types, resource
resource.setrlimit(resource.RLIMIT_CORE, (0,0))
import _p1t_rev4_lifecycle as n
mode=sys.argv[1] if len(sys.argv)>1 else 'short'
bound=5.0 if mode=='default' else (1.0 if mode=='control' else .1)
delay=5.5 if mode=='default' else .6
started=False
def deleter():
    global started
    started=True
    print('DELETER started',flush=True)
    time.sleep(delay)
    print('DELETER finished',flush=True)
def close():
    print('HOOK',dict(n.close(bound)),flush=True)
atexit.register(close)
def start():
    n.start(deleter)
    t=time.monotonic()
    while not started:
        assert time.monotonic()-t < 3
        time.sleep(.005)
    print('ADMITTED',dict(n.stats()),flush=True)
atexit.register(start)
class Finalizer:
    def __init__(self): self.sleep,self.fin=time.sleep,sys.is_finalizing
    def __del__(self):
        print('FINALIZING',self.fin(),flush=True)
        self.sleep(1.5)
holder=types.ModuleType('_p1t_rev3_holder');holder.keep=Finalizer();sys.modules[holder.__name__]=holder;del holder
print('ARMED',mode,flush=True)
