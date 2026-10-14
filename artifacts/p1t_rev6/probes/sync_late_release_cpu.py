"""PASS iff synchronize waits only releases admitted before the call; exit1 if a later release extends that wait.
Host-control trace against actual dlpack_operand.cpp. No GPU execution. Fresh process under timeout15.
Prior art: RCU grace periods (McKenney/Slingwine1998; unverified lead to check synchronize_rcu pre-existing readers).
Taken: distinguish pre-call from later work; ours: finite two-thread overlap with recorded begin/end ordering.
"""
import _p1t_rev6_trace as n
import threading,time,sys
first_started=threading.Event();first_end=threading.Event();late_started=threading.Event();late_end=threading.Event()
def first():
 first_started.set();assert late_started.wait(3);first_end.set();print('FIRST completed',flush=True)
def late():
 late_started.set();time.sleep(.6);late_end.set();print('LATER completed',flush=True)
n.queue(first);a=threading.Thread(target=n.drain);a.start();assert first_started.wait(2)
def add_late():
 time.sleep(.1);print('ADMIT later release AFTER synchronize began',flush=True);n.queue(late);n.drain()
b=threading.Thread(target=add_late);b.start()
t0=time.monotonic();print('SYNCHRONIZE begins',flush=True);n.synchronize();elapsed=time.monotonic()-t0
was_late_finished=late_end.is_set();a.join(2);b.join(2)
print('elapsed=%.3f waited_for_later=%s' % (elapsed,was_late_finished),flush=True)
sys.exit(1 if was_late_finished else 0)
