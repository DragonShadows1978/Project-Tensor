"""LEAD GPU ONLY: imported Storage last-release on a foreign thread, normal and finalizing.
Prior art: CPython 3.12 finalization/GIL API; DLPack lifetime. timeout 30 per subprocess required.
Compile release_worker.cpp into /tmp/p1t_rev_native using the command in README.md first.
Run once with --normal, then once without; nonzero exit/hang is a failure. No source edits.
"""
import os, subprocess, sys
if '--child' not in sys.argv:
    result=subprocess.run([sys.executable,__file__,'--child']+(['--normal'] if '--normal' in sys.argv else []),
                          capture_output=True,text=True,timeout=25,env=os.environ.copy())
    print(result.stdout,end=''); print(result.stderr,end='',file=sys.stderr)
    marker='PASS: ordinary foreign-thread release' if '--normal' in sys.argv else 'PASS: shutdown foreign-thread release completed'
    assert result.returncode==0 and marker in result.stdout, 'F04: release aborted or did not complete'
    if '--normal' not in sys.argv:
        assert 'finalizing: True' in result.stdout, 'INCONCLUSIVE: shutdown finalization was not exercised'
    raise SystemExit(0)
import torch, tensor_cuda as tc
sys.path.insert(0,'/tmp/p1t_rev_native')
from p1t_rev_release_worker import Held
p=torch.full((1024,),3.,device='cuda')
u=tc.from_dlpack(p)
h=Held(u.__dlpack__(max_version=(1,3)))
del u,p
if '--normal' in sys.argv:
    h.release_worker()
    print('PASS: ordinary foreign-thread release')
else:
    class Finalizer:
        def __init__(self,held):
            self.held=held
            self.is_finalizing=sys.is_finalizing
        def __del__(self):
            print('finalizing:',self.is_finalizing(),flush=True)
            self.held.release_worker()
            print('PASS: shutdown foreign-thread release completed',flush=True)
    finalizer=Finalizer(h)
    del h
    print('Armed: finalizer retains the last core-export Storage reference',flush=True)
