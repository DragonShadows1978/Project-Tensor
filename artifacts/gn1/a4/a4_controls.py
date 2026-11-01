"""A4 G4 new targets, through the real A4 parent and scorer.

Prior art: REV3 B6/B7/B7b/B8/C7b/E5 (2026); Meszaros fault injection (2007);
GNU timeout/POSIX signals, BSD flock (year unverified, lead to check).
The installed runner uses only CPU children and scratch ownership files.
"""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from evidence import TOOL, lines, read, sha, evaluate, verify_start

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
BASE = TOOL / 'a4_runs'
BASE.mkdir(exist_ok=False)
SCRATCH = Path(os.environ['TMPDIR']) / 'a4_controls'
SCRATCH.mkdir(exist_ok=False)
PY = sys.executable
RESULT = []


def env(runs, **extra):
    e = dict(os.environ, RUNS=str(runs), REV2_MODE='normal', REV2_NO_LOG='1',
             FORGE_GPU_SUMMARY=str(SCRATCH / 'summary.log'),
             FORGE_GPU_WINDOW_FILE=str(SCRATCH / 'GPU_WINDOW'),
             FORGE_GPU_LOCK=str(SCRATCH / 'lock'), FORGE_GPU_POLL='0.05',
             FORGE_GPU_KILL_GRACE='0.2')
    e.pop('GN1_AUDIT', None)
    e.update(extra)
    assert all(Path(e[k]).resolve().is_relative_to(SCRATCH) for k in
               ('FORGE_GPU_SUMMARY', 'FORGE_GPU_WINDOW_FILE', 'FORGE_GPU_LOCK'))
    return e


def launch(args, e, bound=100, popen=False):
    with (BASE / 'LAUNCHES.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               args=args, bound=bound, RUNS=e['RUNS'], mode=e['REV2_MODE'])) + '\n')
    cmd = ['timeout', '--kill-after=2s', str(bound)+'s', *args]
    if popen:
        return subprocess.Popen(cmd, cwd=TOOL, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p = subprocess.run(cmd, cwd=TOOL, env=e, capture_output=True, text=True)
    return dict(exit=p.returncode, stdout=p.stdout[-1500:], stderr=p.stderr[-3000:])


def start(name):
    runs = BASE / name
    assert launch([PY, '-B', 'start_gpu_run.py'], env(runs))['exit'] == 0
    return runs


def command(label='candidate_focus', outer=None, maximum=None, minimum=None, inner=None, margin=1):
    line = lines()[label]
    return [PY, '-B', 'run_gpu_line.py', label,
            ','.join(map(str, line['expected_child_exits'])), str(line['outer'] if outer is None else outer),
            'forge-gpu-run', '--min', str(line['minimum'] if minimum is None else minimum),
            '--max', str(line['maximum'] if maximum is None else maximum), '--margin', str(margin),
            '--label', 'gn1-a4-cpu', '--', 'timeout', '--kill-after=0.2s',
            str(line['inner'] if inner is None else inner)+'s', PY,
            str(TOOL / 'rev2_forge.py'), 'run_gpu_tests.py', line['arm'], line['suite']]


def collect(runs, label, r):
    paths = sorted((runs / label).glob('attempt-*/parent_result.json'))
    r['parent'] = read(paths[-1]) if paths else None
    return r


def parent(runs, label='candidate_focus', mode='normal', **kw):
    e = kw.pop('extra', {})
    return collect(runs, label, launch(command(label, **kw), env(runs, REV2_MODE=mode, **e)))


def finish(p, runs, label='candidate_focus'):
    out, err = p.communicate(timeout=110)
    return collect(runs, label, dict(exit=p.returncode, stdout=out[-1500:], stderr=err[-3000:]))


def score(runs, label='candidate_focus'):
    r = launch([PY, '-B', 'score_gpu.py'], env(runs))
    r['line'] = read(runs / 'G5_SUMMARY.json')['lines'][label]
    return r


def record(name, rule, passed, **data):
    RESULT.append(dict(control=name, target=rule, passed=bool(passed), **data))
    (TOOL / 'G4_a4.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in RESULT), controls=RESULT), indent=2)+'\n')
    print(name, 'PASS' if passed else 'FAIL', flush=True)


def summary(event):
    (SCRATCH / 'summary.log').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+' | '+event+'\n')


def absent():
    p = SCRATCH / 'summary.log'
    if p.exists():
        p.rename(SCRATCH / ('summary.saved.'+str(time.time_ns())))


def wait_for(fn):
    deadline = time.monotonic()+15
    while not fn():
        assert time.monotonic() < deadline, 'control startup deadline'
        time.sleep(.02)


# REV3 B6: the actual runner is SIGTERM'd while flock is held; no row/marker.
runs = start('B6_lock_outer')
summary('COOLDOWN 3600s')
fd = os.open(SCRATCH / 'lock', os.O_CREAT | os.O_RDWR, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
try:
    first = parent(runs, outer=2, maximum=60, minimum=1)
finally:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
absent()
second = parent(runs)
s = score(runs)
record('B6_lock_outer_retry', 'D2 no start operands -> not admitted; retry admitted',
       first['parent']['admission']=='not_admitted' and first['parent']['exit']==124
       and second['exit']==0 and s['line']['passed'] and len(s['line']['attempts'])==2,
       first=first, retry=second, scorer=s)

# REV3 B7/B7b: deadline at 3s, outer at 4s, kill grace extends to 6s.
for name, mode in [('B7_grace_no_result', 'started_no_result'), ('B7b_grace_result', 'hang_after_result')]:
    runs = start(name)
    first = parent(runs, mode=mode, outer=4, maximum=6, minimum=1, margin=3,
                   extra=dict(FORGE_GPU_KILL_GRACE='3', A4_SLEEP='20'))
    retry = parent(runs)
    s = score(runs)
    row = (runs/'candidate_focus/attempt-0001/forge.log').read_text().splitlines()[-1].split('\t')[-2:]
    record(name, 'D3 outer fires in runner grace -> outer-timeout; D2 no retry',
           first['exit']==1 and first['parent']['timeout_cause']=='outer-timeout'
           and row==['timeout','143'] and retry['exit']==1 and not s['line']['passed'],
           first=first, retry=retry, scorer=s, raw_row=row)

# REV3 B8: change the scratch window after the CPU child marker.
runs = start('B8_window_grace')
summary('COOLDOWN 3600s')
p = launch(command(outer=3, maximum=60, minimum=1, margin=4),
           env(runs, REV2_MODE='started_no_result', FORGE_GPU_KILL_GRACE='4'), popen=True)
wait_for(lambda: (runs/'candidate_focus/attempt-0001/child_started.json').exists())
summary('RUN7 CYCLE 1 TRAINING')
first = finish(p, runs)
s = score(runs)
row = (runs/'candidate_focus/attempt-0001/forge.log').read_text().splitlines()[-1].split('\t')[-2:]
record('B8_window_grace', 'D3 window-closed/143 with exit124 -> outer-timeout',
       first['exit']==1 and first['parent']['timeout_cause']=='outer-timeout'
       and row==['window-closed','143'] and not s['line']['passed'], first=first, scorer=s, raw_row=row)
absent()

# C7b: kill only our own parent, leave the started child bounded by its timeout.
runs = start('C7b_started_parent_interrupted')
e = env(runs, REV2_MODE='started_no_result', REV2_NO_LOG='0')
cmd = command()
# This case uses the CPU stand-in directly, so its bounded process stays in
# the same group as the timeout supervising the parent (no orphan wait).
sep = cmd.index('forge-gpu-run')
cmd = cmd[:sep]+[PY, str(TOOL/'rev2_forge.py'), '--min','250','--max','300','--',
                 'timeout','240s','python3','run_gpu_tests.py','candidate','focus']
interrupted = launch(cmd, e, bound=1)
retry = parent(runs)
s = score(runs)
record('C7b_started_no_result', 'D2/D4 started incomplete child admitted and failed; no retry',
       (runs/'candidate_focus/attempt-0001/invocation.json').exists()
       and not (runs/'candidate_focus/attempt-0001/child_result.json').exists()
       and retry['exit']==1 and 'label already has an admitted attempt' in retry['stderr']
       and s['line']['attempts'][0]['admission']=='admitted' and not s['line']['passed'],
       interrupted=interrupted, retry=retry, scorer=s)

# E5 both original incomplete shape and a completed parent reading bad JSON.
runs = start('E5_truncated_incomplete')
att = runs/'candidate_focus/attempt-0001'
att.mkdir(parents=True)
(att/'child_result.json').write_text('{"pytest_exit":')
retry = parent(runs)
s = score(runs)
record('E5_truncated_incomplete', 'D4 JSON parse problem recorded; clean retry refusal',
       retry['exit']==1 and 'Traceback' not in retry['stderr']
       and any('Expecting value' in x for x in s['line']['reasons']) and not s['line']['passed'],
       retry=retry, scorer=s)
runs = start('E5_truncated_completed')
r = parent(runs, mode='truncated')
s = score(runs)
record('E5_truncated_completed', 'D4 evaluate never raises; JSON reason in parent and scorer',
       r['exit']==1 and any('Expecting value' in x for x in r['parent']['reasons'])
       and not s['line']['passed'], parent=r, scorer=s)

for mode, accept, needle in [('sealed_tamper',False,'runtime module tamper'),('unsealed',True,'device_only')]:
    runs = start('D5_'+mode)
    target = TOOL/'runtime_sentinel.py'
    original = target.read_bytes()
    try:
        r = parent(runs, mode=mode)
        altered = sha(target)
    finally:
        target.write_bytes(original)
    s = score(runs)
    ok = (r['exit']==0 and s['line']['passed'] and needle in str(s['line']['unsealed'])
          if accept else r['exit']==1 and needle in str(r['parent']['reasons']) and not s['line']['passed'])
    record('D5_'+mode, 'D5 '+('unknown origins listed without veto' if accept else 'sealed runtime digest mismatch veto'),
           ok, parent=r, scorer=s)

runs = start('D6_live')
assert parent(runs)['exit']==0
for label in ('E_npbool_g','E_npbool_c'):
    r = parent(runs, label)
    s = score(runs, label)
    live = read(runs/label/'attempt-0001/child_result.json')['equivalence_runtime']
    record('D6_live_'+label, 'D6 arm interpreter NumPy 2.4.2 rejects numpy.bool_ through operator.index',
           r['exit']==0 and s['line']['passed'] and live['numpy_version']=='2.4.2'
           and live['accepts_numpy_bool'] is False, parent=r, scorer=s, live=live)
runs = start('D6_invalidated')
assert parent(runs)['exit']==0
r = parent(runs, 'E_npbool_g', mode='equivalence_accepts')
s = score(runs, 'E_npbool_g')
record('D6_acceptance_invalidates', 'D6 live acceptance invalidates equivalence',
       r['exit']==1 and 'equivalence invalid' in str(r['parent']['reasons']) and not s['line']['passed'], parent=r, scorer=s)

# Start from an honest completed line and alter exactly one binding field.
runs = start('D7_fields')
assert parent(runs)['exit']==0
path = runs/'candidate_focus/attempt-0001/parent_result.json'
original = path.read_text()
for field in ('outer','maximum','minimum','inner','expected_exits'):
    receipt = json.loads(original)
    if field=='outer':
        receipt['outer_bound_seconds'] += 1
    elif field=='expected_exits':
        receipt['expected_child_exits'] = [1]
    else:
        cmd = receipt['command']
        if field in ('maximum','minimum'):
            at=cmd.index('--max' if field=='maximum' else '--min')+1
        else:
            at=cmd.index('--')+3  # timeout, --kill-after=..., duration
        cmd[at] = str(int(cmd[at].removesuffix('s'))+1)+('s' if field=='inner' else '')
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    s=score(runs)
    record('D7_'+field, 'D7 exact '+field+' binding to gpu_lines.json',
           not s['line']['passed'] and 'gpu_lines.json' in str(s['line']['reasons']), scorer=s)
    path.write_text(original)
assert score(runs)['line']['passed']

# D3 malformed attribution still traverses the real parent/scorer, no direct
# evaluator substitution. A4 must record a reason instead of crashing.
runs = start('D3_unattributable')
# A malformed row cannot be produced by the installed runner. Use the same
# synthetic runner transport as inherited malformed-log controls, through the
# unmodified real parent/scorer; the real-runner interruption cases above stay
# on the installed runner.
cmd = command()
at = cmd.index('forge-gpu-run')
cmd = cmd[:at] + [PY, str(TOOL/'rev2_forge.py'), *cmd[at+1:]]
r = collect(runs, 'candidate_focus', launch(cmd, env(runs, REV2_MODE='unattributable', REV2_NO_LOG='0')))
s = score(runs)
record('D3_evaluate_never_raises', 'D3 attribution exception becomes a parent reason; scorer rejects',
       r['exit']==1 and r['parent'] is not None
       and any('runner attribution' in x for x in r['parent']['reasons'])
       and not s['line']['passed'], parent=r, scorer=s)

# D2 operand isolation: each form of start evidence independently burns an
# incomplete label. The real parent recovery and scorer see the raw operand.
for operand in ('marker','invocation','ran','timeout','window-closed','junit'):
    runs=start('D2_operand_'+operand)
    att=runs/'candidate_focus/attempt-0001'
    att.mkdir(parents=True)
    if operand in ('ran','timeout','window-closed'):
        (att/'forge.log').write_text('x\tx\tx\tx\tx\tx\tx\t'+operand+'\t0\n')
    else:
        name={'marker':'child_started.json','invocation':'invocation.json','junit':'junit.xml'}[operand]
        (att/name).write_text('{}')
    retry=parent(runs)
    s=score(runs)
    record('D2_start_'+operand, 'D2 positive start evidence independently admits incomplete failure',
           retry['exit']==1 and 'label already has an admitted attempt' in retry['stderr']
           and s['line']['attempts'][0]['admission']=='admitted' and not s['line']['passed'],
           retry=retry,scorer=s)

sys.exit(0 if all(r['passed'] for r in RESULT) else 1)
