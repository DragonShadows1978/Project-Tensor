"""PT-DET-3 author baseline (CPU only), not blind verification or GPU certification.

Prior art: PT-DET-1/2 CPU tests (this fork, 2026) -- replay_fixture structural
receipts and the verify_repro tamper pattern, reused. Ours: cross-slot union
fixtures (pinned prior receipts + completion receipts), drift/missing/non-GREEN
counterexamples and the frozen env-block contract.
"""
import ast
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'tests'))
import pt_det_1 as d
import pt_det_3 as p3
from test_pt_det_1_cpu import replay_fixture

CPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')


# ---------------------------------------------------------------- registration

def test_sealed_registration_on_disk():
    r = p3.registration()
    assert d.sha(p3.REG) == p3.REG_SHA == p3.REG.with_suffix('.sha256').read_text().strip()
    lane = r['lane']
    assert lane['arms'] == ['bf16_off'] and lane['repeats'] == [1, 2] and lane['steps'] == 30
    assert lane['per_run_seconds'] <= 280 and lane['lane_seconds'] == 600
    assert r['prior']['dir'] == str(p3.PRIOR)
    assert sorted(r['prior']['receipt_sha256']) == sorted(
        f'{a}_{i}/receipt.json' for a in ('v3_on', 'bf16_on', 'v3_off') for i in (1, 2))
    assert all(len(h) == 64 for h in r['prior']['receipt_sha256'].values())
    assert r['inherited']['pt_det_1_registration_sha256'] == d.REG_SHA
    assert r['union']['status_map'] == dict(GREEN='GREEN', NOT_RECURRED='RED', RED='RED', BLOCKED='BLOCKED')
    assert d.sha(r['order']['path']) == r['order']['sha256']


# ---------------------------------------------------------------- env block

def test_env_block_frozen_for_bf16_off():
    base = dict(PYTHONPATH='/x', TC_TF32_GEMM='1', NVIDIA_TF32_OVERRIDE='0', CUDA_LAUNCH_BLOCKING='1',
                CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='0', KEEP_ME='k',
                TC_DETERMINISTIC='1', CC46B_ARM='a')
    out = Path('/frozen/out'); run = out/'bf16_off_2'
    m = dict(binary=dict(path='tensor_cuda/tensor_cuda/x.so', sha256='b'*64))
    r = dict(cc46_registration=dict(path='/frozen/cc46/REGISTRATION.json'))
    R = str(d.ROOT)
    expected = {
        'CUDA_VISIBLE_DEVICES': '0', 'KEEP_ME': 'k',
        'PYTHONPATH': R + '/tensor_cuda', 'PYTHONDONTWRITEBYTECODE': '1',
        'GRAPA_ENGINE_PATH': R + '/tensor_cuda', 'GRAPA_ENGINE_SHA256': 'b'*64,
        'PT_DET_EXPECT_SO': R + '/tensor_cuda/tensor_cuda/x.so',
        'TC_DETERMINISTIC': '0', 'TC_DET_EMBED_BWD': '0',
        'PT_DET_HARNESS_PATH': R + '/scripts',
        'PT_DET_LOSS_RECORD': '/frozen/out/bf16_off_2/loss.jsonl',
        'CC46_REGISTRATION': '/frozen/cc46/REGISTRATION.json',
        'CC46_TRACE_CKPT': '/frozen/out/bf16_off_2/replay.ckpt',
        'CC46B_ARM': 'd', 'CC46B_RECORD': '/frozen/out/bf16_off_2/probe.jsonl',
        'TMPDIR': '/frozen/out/tmp', 'CUDA_CACHE_PATH': '/frozen/out/cuda_cache',
        'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'}
    got = p3.run_environment('bf16_off', run, out, m, r, base=base)
    assert got == expected
    for k in ('TC_TF32_GEMM', 'NVIDIA_TF32_OVERRIDE', 'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG'):
        assert k not in got


def _env_shape(fn):
    """(removed keys, env.update keyword names, **splat call names) of an env block."""
    tree = ast.parse(inspect.getsource(fn).lstrip())
    removed = keys = splats = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.DictComp) and node.generators[0].ifs
                and isinstance(node.generators[0].ifs[0], ast.Compare)):
            c = node.generators[0].ifs[0].comparators[0]
            # The replica names the tuple; resolve it in its own module.
            removed = tuple(getattr(inspect.getmodule(fn), c.id) if isinstance(c, ast.Name) else ast.literal_eval(c))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'update'
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'env'):
            keys = [k.arg for k in node.keywords if k.arg]
            splats = [ast.unparse(k.value) for k in node.keywords if k.arg is None]
    return removed, keys, splats


def test_env_block_replicates_sealed_repro_source():
    removed, keys, splats = _env_shape(d.repro)
    removed3, keys3, splats3 = _env_shape(p3.run_environment)
    assert removed == removed3 == p3.REMOVED_ENV
    assert keys == keys3 and len(keys) == 15
    assert splats == ['family_environment(arm)'] and splats3 == ['t.family_environment(arm)']


# ---------------------------------------------------------------- union fixtures

@pytest.fixture(scope='session')
def pool(tmp_path_factory):
    base = tmp_path_factory.mktemp('pool'); m = None
    for arm in d.ARMS:
        for i in (1, 2):
            rec, m = replay_fixture(base/f'{arm}_{i}', arm, changed=arm.endswith('_off') and i == 2)
            d.create_json(base/f'{arm}_{i}/receipt.json', rec)
    rec, _ = replay_fixture(base/'bf16_off_same', 'bf16_off')
    d.create_json(base/'bf16_off_same/receipt.json', rec)
    return base, m


def p3_reg():
    return json.loads(p3.REG.read_text())


def write_completion(completion, extra=None):
    runs = sorted(completion.glob('*/receipt.json')); i = p3_reg()['inherited']
    done = dict(status='GREEN', pt_det_3_registration_sha256=p3.REG_SHA, registration_sha256=d.REG_SHA,
                binary=i['binary'], steps=30, manifest_sha256=i['manifest_sha256'],
                source_checkpoint_sha256=i['source_checkpoint_sha256'],
                runs={q.parent.name: json.loads(q.read_text())['status'] for q in runs},
                keep_ckpts=False, receipt_sha256={str(q.relative_to(completion)): d.sha(q) for q in runs})
    done.update(extra or {})
    path = completion/'completion.json'
    if path.exists(): path.unlink()
    d.create_json(path, done)


@pytest.fixture
def stage(tmp_path, monkeypatch, pool):
    base, m = pool
    art = tmp_path/'pt_det_1'; prior = art/'slot02/repro'; completion = art/'det3/completion'
    prior.mkdir(parents=True); completion.mkdir(parents=True)
    (art/d.MANIFEST_NAME).write_text('fictional CPU test manifest\n')
    monkeypatch.setattr(d, 'ART', art)
    monkeypatch.setattr(d, 'verify_manifest', lambda: m)
    for arm in d.ARMS:
        for i in (1, 2):
            shutil.copytree(base/f'{arm}_{i}', prior/f'{arm}_{i}')
    # The real slot 02 shape: bf16_off_2 timed out.
    (prior/'bf16_off_2/receipt.json').unlink()
    d.create_json(prior/'bf16_off_2/receipt.json', dict(status='BLOCKED_TIMEOUT', returncode=-15, seconds=23.8))
    d.create_json(prior/'summary.json', dict(
        verdict='BLOCKED', registration_sha256=d.REG_SHA, binary=m['binary'], steps=30,
        pt_det_2_registration_sha256=m['pt_det_2_registration_sha256'],
        deterministic_sites=['embedding', 'gather_topk'], manifest_sha256=d.sha(art/d.MANIFEST_NAME),
        source_checkpoint_sha256=d.registration()['repro']['checkpoint']['sha256'],
        runs={q.parent.name: json.loads(q.read_text())['status'] for q in prior.glob('*/receipt.json')},
        receipt_sha256={str(q.relative_to(prior)): d.sha(q) for q in prior.glob('*/receipt.json')}))
    monkeypatch.setattr(p3, 'REG', tmp_path/'pt_det_3/REGISTRATION.json')
    p3.register(prior)
    monkeypatch.setattr(p3, 'REG_SHA', d.sha(p3.REG))
    shutil.copytree(base/'bf16_off_1', completion/'bf16_off_1')
    shutil.copytree(base/'bf16_off_2', completion/'bf16_off_2')
    write_completion(completion)

    def verdict(out='union'):
        return p3.main(['verdict', '--prior', str(prior), '--completion', str(completion), '--out', str(art/out)])
    return dict(art=art, prior=prior, completion=completion, base=base, m=m, verdict=verdict)


def load(path):
    return json.loads(Path(path).read_text())


def test_union_green_when_control_differs(stage, capsys):
    assert stage['verdict']() == 0
    out = stage['art']/'union'
    assert 'PT_DET_3 UNION GREEN sealed_verdict=GREEN' in capsys.readouterr().out
    s = load(out/'summary.json'); u = load(out/'union_receipt.json')
    assert s['verdict'] == 'GREEN' and u == dict(pt_det_3_status='GREEN', sealed_verdict='GREEN', verify_repro='GREEN',
                                                 summary_sha256=d.sha(out/'summary.json'),
                                                 pt_det_3_registration_sha256=p3.REG_SHA)
    assert [s['pairs'][a]['bitwise'] for a in d.ARMS] == [True, True, False, False]
    assert s['runs'] == {f'{a}_{i}': 'GREEN' for a in d.ARMS for i in (1, 2)}
    sealed_keys = {'verdict', 'registration_sha256', 'binary', 'steps', 'pt_det_2_registration_sha256',
                   'deterministic_sites', 'manifest_sha256', 'source_checkpoint_sha256', 'pairs', 'runs',
                   'seconds', 'keep_ckpts', 'evidence_class', 'receipt_sha256'}
    assert set(s) == sealed_keys | {'completion'}
    assert s['completion']['prior_receipt_sha256'] == p3_reg()['prior']['receipt_sha256']
    # Copies, never links; only the files verify_repro reads; bf16_off from the completion lane.
    files = sorted(str(q.relative_to(out)) for q in out.rglob('*') if not q.is_dir())
    assert files == sorted([f'{a}_{i}/{f}' for a in d.ARMS for i in (1, 2) for f in p3.COPIED]
                           + ['summary.json', 'union_receipt.json'])
    assert not any(q.is_symlink() for q in out.rglob('*'))
    for i in (1, 2):
        assert d.sha(out/f'bf16_off_{i}/receipt.json') == d.sha(stage['completion']/f'bf16_off_{i}/receipt.json')
    assert d.verify_repro(out/'summary.json')['verdict'] == 'GREEN'
    # Create-only: a second verdict into the same union dir is refused.
    assert stage['verdict']() == 1


def test_union_red_when_control_bitwise(stage, capsys):
    c = stage['completion']
    shutil.rmtree(c/'bf16_off_2'); shutil.copytree(stage['base']/'bf16_off_same', c/'bf16_off_2')
    write_completion(c)
    assert stage['verdict']() == 1
    out = stage['art']/'union'
    assert 'PT_DET_3 UNION RED sealed_verdict=NOT_RECURRED' in capsys.readouterr().out
    assert load(out/'summary.json')['pairs']['bf16_off']['bitwise'] is True
    assert load(out/'union_receipt.json')['pt_det_3_status'] == 'RED'


def test_union_blocked_when_completion_run_missing(stage, capsys):
    c = stage['completion']; shutil.rmtree(c/'bf16_off_2'); write_completion(c)
    assert stage['verdict']() == 2
    s = load(stage['art']/'union/summary.json')
    assert s['verdict'] == 'BLOCKED' and s['runs']['bf16_off_2'] == 'MISSING'
    assert 'PT_DET_3 UNION BLOCKED' in capsys.readouterr().out


@pytest.mark.parametrize('status,expected,rc', [('BLOCKED_TIMEOUT', 'BLOCKED', 2), ('BLOCKED', 'BLOCKED', 2),
                                                ('RED', 'RED', 1)])
def test_union_non_green_completion_run(stage, capsys, status, expected, rc):
    c = stage['completion']; (c/'bf16_off_2/receipt.json').unlink()
    d.create_json(c/'bf16_off_2/receipt.json', dict(status=status, returncode=None, seconds=280.))
    write_completion(c)
    assert stage['verdict']() == rc
    s = load(stage['art']/'union/summary.json')
    assert s['verdict'] == expected and s['pairs']['bf16_off']['measured'] is False
    assert f'PT_DET_3 UNION {expected}' in capsys.readouterr().out


def test_union_blocked_on_prior_receipt_drift(stage, capsys):
    rec = stage['prior']/'v3_on_1/receipt.json'; text = rec.read_text(); rec.unlink(); rec.write_text(text + ' ')
    assert stage['verdict']() == 2
    assert 'prior receipt drift: v3_on_1/receipt.json' in capsys.readouterr().out
    assert not (stage['art']/'union').exists()


def test_union_blocked_on_prior_log_drift(stage, capsys):
    log = stage['prior']/'bf16_on_2/logs/train.log'; log.write_text(log.read_text() + 'tampered\n')
    assert stage['verdict']() == 2
    assert 'run file drift: bf16_on_2/logs/train.log' in capsys.readouterr().out
    assert not (stage['art']/'union').exists()


def test_union_blocked_on_completion_receipt_drift(stage, capsys):
    rec = stage['completion']/'bf16_off_1/receipt.json'; rec.write_text(rec.read_text() + ' ')
    assert stage['verdict']() == 2
    assert 'completion receipt drift' in capsys.readouterr().out
    assert not (stage['art']/'union').exists()


def test_union_blocked_on_unregistered_prior_or_foreign_completion(stage, capsys):
    other = stage['art']/'other_slot'; shutil.copytree(stage['prior'], other)
    assert p3.main(['verdict', '--prior', str(other), '--completion', str(stage['completion']),
                    '--out', str(stage['art']/'union')]) == 2
    assert 'not the registered prior slot' in capsys.readouterr().out
    write_completion(stage['completion'], dict(pt_det_3_registration_sha256='0'*64))
    assert stage['verdict']() == 2
    assert 'completion provenance differs' in capsys.readouterr().out
    assert not (stage['art']/'union').exists()


def test_register_is_create_only_and_sha_checked(stage, capsys):
    with pytest.raises(FileExistsError):
        p3.register(stage['prior'])
    assert p3.registration()['prior']['dir'] == str(stage['prior'])
    reg = p3.REG; text = reg.read_text(); reg.unlink()
    reg.write_text(text.replace('"lane_seconds": 600', '"lane_seconds": 900'))
    with pytest.raises(ValueError, match='registration drift'):
        p3.registration()
    assert p3.main(['plan']) == 2 and 'PT_DET_3 BLOCKED' in capsys.readouterr().out


def test_register_refuses_prior_receipt_not_matching_its_summary(stage, monkeypatch, tmp_path):
    rec = stage['prior']/'bf16_on_1/receipt.json'; rec.write_text(rec.read_text() + ' ')
    monkeypatch.setattr(p3, 'REG', tmp_path/'second/REGISTRATION.json')
    with pytest.raises(ValueError, match='differs from its slot summary'):
        p3.register(stage['prior'])
    assert not (tmp_path/'second/REGISTRATION.json').exists()


def test_verdict_refuses_out_outside_pt_det_1_art(stage, tmp_path):
    for bad in (tmp_path/'elsewhere', stage['art'], stage['art'].parent/'pt_det_3/union'):
        with pytest.raises(SystemExit) as e:
            p3.main(['verdict', '--prior', str(stage['prior']), '--completion', str(stage['completion']),
                     '--out', str(bad)])
        assert e.value.code == 2
    assert not (tmp_path/'elsewhere').exists() and not (stage['art'].parent/'pt_det_3/union').exists()


def test_verdict_refuses_out_outside_art_subprocess(tmp_path):
    out = tmp_path/'union'
    p = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/pt_det_3.py'), 'verdict', '--prior', str(p3.PRIOR),
                        '--completion', str(tmp_path/'c'), '--out', str(out)],
                       env=CPU_ENV, capture_output=True, text=True, timeout=30)
    assert p.returncode == 2 and 'out must be a fresh PT-DET-1 artifact subdirectory' in p.stderr
    assert not out.exists()


# ---------------------------------------------------------------- complete guards

@pytest.mark.parametrize('extra', [['--lead-gpu', '--lock-fd', '9'], ['--lock-fd', '9'], ['--lead-gpu']])
def test_complete_cpu_hidden_blocks_before_cuda_and_output(extra):
    out = d.ART/f'pt_det_3_guard_probe_{os.getpid()}'
    code = ('import sys; sys.path.insert(0, %r); import pt_det_3 as p; rc = p.main(sys.argv[1:]); '
            "print('CUDA_ENGINE_IMPORTED', any(k.startswith('tensor_cuda') for k in sys.modules)); sys.exit(rc)"
            % str(ROOT/'scripts'))
    p = subprocess.run([sys.executable, '-B', '-c', code, 'complete', '--out', str(out)] + extra,
                       env=CPU_ENV, capture_output=True, text=True, timeout=30)
    assert p.returncode == 2, p.stdout + p.stderr
    assert 'PT_DET_3 BLOCKED' in p.stdout and 'CUDA_ENGINE_IMPORTED False' in p.stdout
    assert not out.exists()


def test_complete_guard_order_in_process(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(d, 'verify_manifest', lambda: calls.append('manifest') or {})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    out = d.ART/f'pt_det_3_guard_probe_{os.getpid()}'
    assert p3.main(['complete', '--out', str(out), '--lead-gpu', '--lock-fd', '9']) == 2
    assert calls == [] and not out.exists()
    # Lead guard passes, manifest fails: still nothing created.
    monkeypatch.setattr(d, 'require_lead', lambda lead, fd: dict(fd=fd))
    def drift(): raise ValueError('PT_DET_1 source/binary/dependency drift: fixture')
    monkeypatch.setattr(d, 'verify_manifest', drift)
    assert p3.main(['complete', '--out', str(out), '--lead-gpu', '--lock-fd', '9']) == 1
    assert 'PT_DET_3 RED' in capsys.readouterr().out and not out.exists()


@pytest.fixture
def fake_lane(monkeypatch, tmp_path):
    reg = p3.registration()
    art = tmp_path/'pt_det_1'; art.mkdir()
    shutil.copyfile(ROOT/'artifacts/pt_det_1'/d.MANIFEST_NAME, art/d.MANIFEST_NAME)
    m = dict(binary=reg['inherited']['binary'],
             pt_det_2_registration_sha256=reg['inherited']['pt_det_2_registration_sha256'])
    monkeypatch.setattr(d, 'ART', art)
    monkeypatch.setattr(d, 'require_lead', lambda lead, fd: dict(fd=fd, target='fixture', exclusive_flock=True))
    monkeypatch.setattr(d, 'verify_manifest', lambda: m)
    source = tmp_path/'source.ckpt'; source.write_bytes(b'fixture')
    monkeypatch.setattr(p3, 'source_checkpoint', lambda r: (source, source.stat(), {}))
    calls = []
    def run_process(argv, cwd, env, log_path, seconds, stdin=None, pass_fds=(), new_process_group=True):
        calls.append(dict(argv=argv, cwd=cwd, env=env, log=log_path, seconds=seconds, stdin=stdin,
                          pass_fds=pass_fds, new_process_group=new_process_group))
        log_path.write_text('fixture\n'); (log_path.parent/'replay.ckpt').write_bytes(b'x')
        status = 'GREEN' if len(calls) == 1 else 'BLOCKED_TIMEOUT'
        return dict(status=status, returncode=0 if status == 'GREEN' else None, seconds=1.5, argv=argv, log=str(log_path))
    monkeypatch.setattr(d, 'run_process', run_process)
    monkeypatch.setattr(d, 'run_receipt', lambda run_dir, result, arm, schema, mm: dict(result))
    return art, calls


def test_complete_lays_out_registered_arm_only(fake_lane, capsys, monkeypatch):
    art, calls = fake_lane
    monkeypatch.delenv('PT_DET_SLOT_LANE_DEADLINE', raising=False)
    out = art/'det3/completion'
    assert p3.main(['complete', '--out', str(out), '--lead-gpu', '--lock-fd', '9']) == 2
    text = capsys.readouterr().out
    assert 'PT_DET_3 RUN bf16_off 1 remaining_seconds 280' in text and 'PT_DET_3 RUN bf16_off 2' in text
    assert 'PT_DET_3 COMPLETE BLOCKED bf16_off=None' in text
    assert sorted(q.name for q in out.iterdir()) == ['bf16_off_1', 'bf16_off_2', 'completion.json', 'lock_receipt.json', 'tmp']
    for i in (1, 2):
        run = out/f'bf16_off_{i}'
        assert (run/'logs').is_dir() and (run/'leg.stdout').is_file() and not (run/'replay.ckpt').exists()
    done = load(out/'completion.json')
    assert done['runs'] == dict(bf16_off_1='GREEN', bf16_off_2='BLOCKED_TIMEOUT')
    assert done['receipt_sha256'] == {f'bf16_off_{i}/receipt.json': d.sha(out/f'bf16_off_{i}/receipt.json') for i in (1, 2)}
    assert done['pt_det_3_registration_sha256'] == p3.REG_SHA and done['steps'] == 30
    assert len(calls) == 2
    for i, c in enumerate(calls, 1):
        run = out/f'bf16_off_{i}'
        assert c['argv'] == [sys.executable, '-B', '-'] + d.trainer_argv('bf16_off', run)
        assert '--fwd-fp32-blocks' not in c['argv'] and c['cwd'] == d.CC and c['stdin'] == d.PREAMBLE
        assert c['pass_fds'] == (9,) and c['new_process_group'] is True and 0 < c['seconds'] <= 280
        assert c['env']['TC_DETERMINISTIC'] == c['env']['TC_DET_EMBED_BWD'] == '0' and c['env']['CC46B_ARM'] == 'd'
        assert c['env']['CC46_TRACE_CKPT'] == str(run/'replay.ckpt')


def test_complete_inherits_slot_lane_deadline(fake_lane, monkeypatch):
    art, calls = fake_lane
    monkeypatch.setenv('PT_DET_SLOT_LANE_DEADLINE', str(time.monotonic() + 100))
    assert p3.main(['complete', '--out', str(art/'det3/c2'), '--lead-gpu', '--lock-fd', '9']) == 2
    assert len(calls) == 2 and all(c['new_process_group'] is False for c in calls)
    assert 80 < calls[0]['seconds'] <= 85
    monkeypatch.setenv('PT_DET_SLOT_LANE_DEADLINE', 'nan')
    assert p3.main(['complete', '--out', str(art/'det3/c3'), '--lead-gpu', '--lock-fd', '9']) == 1
    assert load(art/'det3/c3/completion.json')['status'] == 'RED' and len(calls) == 2


def test_plan_is_cpu_only_and_registered():
    p = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/pt_det_3.py'), 'plan'], env=CPU_ENV,
                       capture_output=True, text=True, timeout=30, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    plan = json.loads(p.stdout)
    assert list(plan['runs']) == ['bf16_off_1', 'bf16_off_2']
    assert plan['deadlines']['per_run_seconds'] == 280 and plan['deadlines']['lane_seconds'] == 600
    for run in plan['runs'].values():
        assert run['env_set']['TC_DETERMINISTIC'] == '0' and run['env_set']['CC46B_ARM'] == 'd'
        assert '--fwd-fp32-blocks' not in run['argv']
