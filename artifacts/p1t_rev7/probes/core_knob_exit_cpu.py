"""PASS iff the actual core Python exit hook waits for the finite admitted release and exits0.
Build build_core_host.py first; sources under review dynamically extracted, no frozen product code.
--control uses default knob. No GPU execution (direct exact-source import-release lifecycle).
Prior art: CPython3.12 lifecycle PSF2023, DLPack DMLC2017; taken: REV5/6 finite-release staging;
ours: conversion overflow at the real Python hook before its raw native close function.
"""
from pathlib import Path
import ast,sys,atexit,threading,time,types,resource
import _p1t_rev7_core as n
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
root=Path(__file__).resolve().parents[3]
src=(root/'tensor_cuda/tensor_cuda/__init__.py').read_text();tree=ast.parse(src)
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_dlpack_shutdown_hook')
ns={'_C':n,'sys':sys,'_DLPACK_SHUTDOWN_WAIT_S':5.0 if '--control' in sys.argv else 1<<4096}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(root/'tensor_cuda/tensor_cuda/__init__.py'),'exec'),ns)
started=threading.Event()
def slow():
    print('DELETER started',flush=True);started.set();time.sleep(.6);print('DELETER finished',flush=True)
def start():
    threading.Thread(target=lambda:n.release(slow),daemon=True).start();assert started.wait(2)
    print('IN_FLIGHT',n.in_flight(),flush=True)
atexit.register(ns['_dlpack_shutdown_hook']);atexit.register(start)
class Final:
    def __init__(self):self.sleep,self.fin=time.sleep,sys.is_finalizing
    def __del__(self):print('FINALIZING',self.fin(),flush=True);self.sleep(1.5)
h=types.ModuleType('_p1t_rev7_core_final');h.keep=Final();sys.modules[h.__name__]=h
print('ARMED core actual-source host lifecycle',flush=True)
