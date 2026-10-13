"""PASS iff host boundaries and shutdown holds; casts_exit: exits0 without abort/segfault.
Runs current-source verbatim core host slices, CUDA-labelled zero-size storage, no CUDA allocation or GPU execution.
Prior art: DLPack/Python array API (2017/2021), C++17 checked narrowing, CPython3.12 (PSF2023).
Taken: accepted argument conversions and lifecycle contracts; ours: finite __index__/__bool__ daemon exit scenario.
Use under timeout15, fresh process per case; --control joins before ordinary exit.
"""
import _p1t_rev6_core as n
import ctypes as C, sys, atexit, threading, time, types, resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
mode=sys.argv[1]
if mode=='arithmetic':
    limit=((1<<31)-1)*256
    for v in [1,255,256,257,limit-1,limit]:
        blocks=n.fill_blocks(v);assert blocks*256>=v and 0<blocks<=((1<<31)-1)
    for v in [limit+1,1<<40,(1<<40)+1,(1<<63)-1]:
        try:n.fill_blocks(v)
        except ValueError:pass
        else:raise AssertionError(v)
    for shape in [[0,1<<62,4],[1<<62,0,4],[1<<62,4,0],[0,1<<61],[1<<63-1]]:
        try:n.empty(shape,'float32','cpu')
        except (ValueError,OverflowError):pass
        else:raise AssertionError(shape)
    for shape in [[],[0],[0,1<<60],[1<<60,0],[1,2,3]]:
        t=n.empty(shape,'float32','cpu');assert t is not None
    print('PASS fill grid rail, scalar/empty and zero-neighbor products')
elif mode=='casts_exit':
    started=threading.Event()
    class Flag:
        def __bool__(self):
            print('BOOL started',flush=True);started.set();time.sleep(.6);print('BOOL finished',flush=True);return False
    class Index:
        def __index__(self):
            print('INDEX started',flush=True);started.set();time.sleep(.6);print('INDEX finished',flush=True);return 1
    t=n.zero_cuda_label()
    def worker():
        if sys.argv[2]=='bool':t.__dlpack__(copy=Flag())
        else:t.__dlpack__(max_version=(Index(),3))
        print('EXPORT returned',flush=True)
    def close():print('CORE HOOK',n.close(5.0),flush=True)
    atexit.register(close)
    def start():
        th=threading.Thread(target=worker,daemon=True);th.start();assert started.wait(2)
        print('IN_FLIGHT',n.in_flight(),flush=True)
        if '--control' in sys.argv:th.join(2);assert not th.is_alive()
    atexit.register(start)
    class Final:
        def __init__(self):self.sleep,self.fin=time.sleep,sys.is_finalizing
        def __del__(self):print('FINALIZING',self.fin(),flush=True);self.sleep(1.5)
    holder=types.ModuleType('_p1t_rev6_casts_final');holder.keep=Final();sys.modules[holder.__name__]=holder
else:raise ValueError(mode)
