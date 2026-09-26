#!/usr/bin/env python3
"""PT-DET-3 completion lane: arm-selectable replay runner + cross-slot union verdict.

Additive to the sealed PT-DET-1/2 harness: every guard, receipt, pair comparison
and verdict is the sealed pt_det_1 function; nothing here re-implements them,
except the per-run environment block of pt_det_1.repro, replicated verbatim
(same keys, same removals) and unit-tested against a frozen dict.

Prior art: PT-DET-1/2 harness (this fork, 2026) -- taken: lead/lock guard,
manifest seal, fresh-process replay, run receipts, compare_pair/assess_pairs,
verify_repro; CC46-B/C (GRAPA 2026) identical-checkpoint replay, via that
harness; SHA256 (NIST 2001) content pins. Ours: completing a registered
pair-set across two GPU slots by pinning the first slot's receipts by sha at
registration and re-deriving the sealed verdict over the union of copied
receipts. No prior art known to me for that cross-slot union
(unverified -- lead to check; search terms: "resumable certification",
"receipt union across runs", "reproducible builds rebuild attestation
aggregation", "in-toto multi-step attestation").
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt_det_1 as t

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / 'artifacts/pt_det_3/REGISTRATION.json'
REG_SHA = '8ee41a4543138111a63696f039e42f2acef3c64d84c0bc6ddbdb1106f7901eed'
ORDER = Path('/mnt/ForgeRealm/wt/pt-tf32/orders/PT_DET_3_COMPLETION.md')
PRIOR = Path('/mnt/ForgeRealm/wt/pt-det-cert/artifacts/pt_det_1/lead_slot_det2_02/repro')
# Registration-time defaults only; every run reads them back from the sealed
# REGISTRATION.json. 280 s/run holds the measured ~175 s bf16 runs; 600 s holds
# two runs plus the checkpoint digest and margins.
LANE = dict(arms=['bf16_off'], repeats=[1, 2], steps=30, per_run_seconds=280, lane_seconds=600,
            per_run_margin_seconds=10, slot_margin_seconds=5)
# Exactly the files pt_det_1.verify_repro reads per run directory.
COPIED = ('receipt.json', 'logs/train.log', 'leg.stdout', 'probe.jsonl', 'loss.jsonl')
# Replicated from pt_det_1.repro (lines 589-590), never edited independently.
REMOVED_ENV = ('PYTHONPATH', 'TC_TF32_GEMM', 'NVIDIA_TF32_OVERRIDE', 'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG')
# Sealed assess_pairs verdict -> PT-DET-3 registered status. A bitwise control
# (sealed NOT_RECURRED) is a PT-DET-3 RED: the control must diverge.
STATUS_MAP = dict(GREEN='GREEN', NOT_RECURRED='RED', RED='RED', BLOCKED='BLOCKED')


def registration():
    if t.sha(REG) != REG_SHA or REG.with_suffix('.sha256').read_text().strip() != REG_SHA:
        raise ValueError('PT_DET_3 registration drift')
    r = json.loads(REG.read_text())
    if t.sha(r['order']['path']) != r['order']['sha256']:
        raise ValueError('PT_DET_3 immutable order drift')
    if r['inherited']['pt_det_1_registration_sha256'] != t.REG_SHA:
        raise ValueError('PT_DET_3 inherited PT_DET_1 registration drift')
    t.registration()  # Sealed PT-DET-1 + PT-DET-2 registrations, unchanged.
    return r


def prior_runs(lane):
    return [f'{a}_{i}' for a in t.ARMS if a not in lane['arms'] for i in lane['repeats']]


def register(prior=PRIOR):
    t.registration()
    prior = Path(prior).resolve()
    summary = json.loads((prior / 'summary.json').read_text())
    if summary.get('registration_sha256') != t.REG_SHA or summary.get('steps') != LANE['steps']:
        raise ValueError('prior slot summary is not a PT_DET_1 30-step replay')
    receipts = {}
    for name in prior_runs(LANE):
        rel = f'{name}/receipt.json'; h = t.sha(prior / rel)
        if summary['receipt_sha256'].get(rel) != h:
            raise ValueError('prior receipt differs from its slot summary: ' + rel)
        if json.loads((prior / rel).read_text()).get('status') != 'GREEN':
            raise ValueError('prior run is not GREEN: ' + rel)
        receipts[rel] = h
    r = dict(
        schema='pt-det-3-registration-v1',
        written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        order=dict(path=str(ORDER), sha256=t.sha(ORDER)),
        inherited=dict(pt_det_1_registration_sha256=t.REG_SHA,
                       pt_det_2_registration_sha256=summary['pt_det_2_registration_sha256'],
                       manifest_name=t.MANIFEST_NAME, manifest_sha256=summary['manifest_sha256'],
                       binary=summary['binary'], source_checkpoint_sha256=summary['source_checkpoint_sha256']),
        lane=dict(LANE, runs=[f'{a}_{i}' for a in LANE['arms'] for i in LANE['repeats']],
                  runner='pt_det_1 trainer_argv/family_environment/run_process/run_receipt/PREAMBLE, '
                         'cwd pt_det_1.CC, env = pt_det_1.repro per-run block'),
        prior=dict(dir=str(prior), summary_sha256=t.sha(prior / 'summary.json'),
                   summary_verdict=summary['verdict'], summary_runs=summary['runs'],
                   receipt_sha256=receipts,
                   superseded_runs=[f'{a}_{i}' for a in LANE['arms'] for i in LANE['repeats']]),
        union=dict(
            rule='pairs[arm] = pt_det_1.compare_pair(receipt_1, receipt_2) for all four pt_det_1.ARMS, '
                 'receipts read from the union copies (prior six + completion two); '
                 'sealed_verdict = pt_det_1.assess_pairs(pairs); summary.json in the pt_det_1.repro schema; '
                 'final check pt_det_1.verify_repro(union/summary.json) when sealed_verdict is GREEN',
            status_map=STATUS_MAP,
            failure_states=[
                'a completion run not GREEN -> BLOCKED or RED exactly as compare_pair/assess_pairs derive from its receipt',
                'a completion run missing -> BLOCKED',
                'bf16_off pair bitwise True -> sealed NOT_RECURRED -> PT_DET_3 RED (the control must diverge)',
                'prior summary/receipt/log drift vs the shas pinned here or in the receipts -> BLOCKED',
                'completion receipt drift vs completion.json -> BLOCKED',
                'sealed GREEN but verify_repro rejects the union -> RED']),
        prior_art='PT-DET-1/2 harness (this fork, 2026): guards, receipts, pair logic, verify_repro, taken. '
                  'Cross-slot union of sha-pinned receipts: ours; no prior art known to me '
                  '(unverified -- lead to check: "resumable certification", "receipt union", '
                  '"in-toto multi-step attestation").')
    REG.parent.mkdir(parents=True, exist_ok=True)
    t.create_json(REG, r)
    with REG.with_suffix('.sha256').open('x') as f:
        f.write(t.sha(REG) + '\n')
    print('PT_DET_3 REGISTERED', t.sha(REG))


def run_environment(arm, run_dir, out, m, r, base=None):
    # Verbatim replica of the pt_det_1.repro per-run block (lines 589-599):
    # same removals, same keys, same values. Prior art: PT-DET-1 (2026), taken.
    base = os.environ if base is None else base
    ROOT = t.ROOT
    env = {k: v for k, v in base.items() if k not in REMOVED_ENV}
    env.update(PYTHONPATH=str(ROOT/'tensor_cuda'), PYTHONDONTWRITEBYTECODE='1',
               GRAPA_ENGINE_PATH=str(ROOT/'tensor_cuda'), GRAPA_ENGINE_SHA256=m['binary']['sha256'],
               PT_DET_EXPECT_SO=str(ROOT/m['binary']['path']), **t.family_environment(arm),
               PT_DET_HARNESS_PATH=str(ROOT/'scripts'),
               PT_DET_LOSS_RECORD=str(run_dir/'loss.jsonl'),
               CC46_REGISTRATION=r['cc46_registration']['path'], CC46_TRACE_CKPT=str(run_dir/'replay.ckpt'),
               CC46B_ARM='d' if arm.startswith('bf16') else 'a', CC46B_RECORD=str(run_dir/'probe.jsonl'),
               TMPDIR=str(out/'tmp'), CUDA_CACHE_PATH=str(out/'cuda_cache'),
               OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
    return env


def check_inherited(reg, m):
    i = reg['inherited']
    if (m['binary'] != i['binary'] or m['pt_det_2_registration_sha256'] != i['pt_det_2_registration_sha256']
            or t.sha(t.ART / t.MANIFEST_NAME) != i['manifest_sha256']
            or t.registration()['repro']['checkpoint']['sha256'] != i['source_checkpoint_sha256']):
        raise t.Blocked('BLOCKED: PT_DET_3 sealed provenance differs from registration')


def source_checkpoint(r):
    source = Path(r['checkpoint']['path'])
    if t.sha(source) != r['checkpoint']['sha256']: raise ValueError('source checkpoint drift')
    return source, source.stat(), t.tensor_schema(t.checkpoint_digest(source))


def complete(out, lock_fd, reg, m, keep_ckpts=False):
    # Deadline handling is pt_det_1.repro's, with the registered lane budget and
    # an added per-run cap (PT-DET-3 registration).
    lane = reg['lane']
    deadline = time.monotonic() + lane['lane_seconds']
    slot_deadline = os.environ.get('PT_DET_SLOT_LANE_DEADLINE')
    if slot_deadline is not None:
        inherited = float(slot_deadline)
        if not math.isfinite(inherited): raise ValueError('invalid slot deadline')
        deadline = min(deadline, inherited - lane['slot_margin_seconds'])
    r = t.registration()['repro']
    source, initial_source_stat, source_schema = source_checkpoint(r)
    pairs = {}; runs = {}; run_seconds = {}; start = time.monotonic()
    for arm in lane['arms']:
        pair = []
        for repeat in lane['repeats']:
            name = f'{arm}_{repeat}'
            run_dir = out / name; run_dir.mkdir(); (run_dir / 'logs').mkdir()
            env = run_environment(arm, run_dir, out, m, r)
            (out/'tmp').mkdir(exist_ok=True)
            argv = [sys.executable, '-B', '-'] + t.trainer_argv(arm, run_dir)
            remaining = min(lane['per_run_seconds'], deadline-time.monotonic()-lane['per_run_margin_seconds'])
            print('PT_DET_3 RUN', arm, repeat, 'remaining_seconds', round(remaining, 1), flush=True)
            try:
                st = source.stat()
                if (st.st_ino, st.st_size, st.st_mtime_ns) != (initial_source_stat.st_ino, initial_source_stat.st_size, initial_source_stat.st_mtime_ns):
                    raise ValueError('source checkpoint changed between replay runs')
                result = t.run_process(argv, t.CC, env, run_dir/'leg.stdout', remaining, t.PREAMBLE, (lock_fd,),
                                       new_process_group=slot_deadline is None)
                receipt = t.run_receipt(run_dir, result, arm, source_schema, m)
            except Exception as exc:
                receipt = dict(status='BLOCKED' if isinstance(exc, (t.Blocked, t.storage.StorageBlocked)) else 'RED', error=repr(exc))
            t.create_json(run_dir/'receipt.json', receipt); pair.append(receipt)
            runs[name] = receipt['status']; run_seconds[name] = receipt.get('seconds')
            # Digest all tensors before removing only the checkpoint we created.
            if not keep_ckpts and (run_dir/'replay.ckpt').exists(): (run_dir/'replay.ckpt').unlink()
        pairs[arm] = t.compare_pair(*pair)
    t.verify_manifest()  # Recheck trainer/config/engine bytes after all runs.
    status = ('GREEN' if all(s == 'GREEN' for s in runs.values()) else
              'RED' if any(s == 'RED' for s in runs.values()) else 'BLOCKED')
    result = dict(status=status, pt_det_3_registration_sha256=REG_SHA, registration_sha256=t.REG_SHA,
                  binary=m['binary'], steps=30, pt_det_2_registration_sha256=m['pt_det_2_registration_sha256'],
                  manifest_sha256=t.sha(t.ART/t.MANIFEST_NAME), source_checkpoint_sha256=r['checkpoint']['sha256'],
                  arms=lane['arms'], runs=runs, run_seconds=run_seconds, seconds=time.monotonic()-start,
                  deadlines=dict(lane_seconds=lane['lane_seconds'], per_run_seconds=lane['per_run_seconds'],
                                 slot_deadline_inherited=slot_deadline is not None),
                  pairs_informational=pairs, keep_ckpts=keep_ckpts,
                  evidence_class='fresh-process 30-step runs of the registered completion arm(s); '
                                 'the certification verdict is computed only by `verdict` over the union',
                  receipt_sha256={str(p.relative_to(out)): t.sha(p) for p in sorted(out.glob('*/receipt.json'))})
    t.create_json(out/'completion.json', result)
    print('PT_DET_3 COMPLETE', status, ' '.join(f'{a}={pairs[a].get("bitwise")}' for a in lane['arms']), flush=True)
    return 0 if status == 'GREEN' else 2 if status == 'BLOCKED' else 1


def copy_create(src, dst):
    # Create-only byte copy (never a symlink, never an overwrite).
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open('rb') as a, dst.open('xb') as b:
        for chunk in iter(lambda: a.read(8 << 20), b''):
            b.write(chunk)
    if t.sha(dst) != t.sha(src): raise t.Blocked('BLOCKED: copy changed bytes: ' + str(src))


def sources_for_union(reg, prior, completion):
    """Every pre-copy check; raises Blocked on any drift, creates nothing."""
    p = reg['prior']; lane = reg['lane']
    if prior != Path(p['dir']): raise t.Blocked('BLOCKED: --prior is not the registered prior slot')
    if not (prior/'summary.json').is_file() or t.sha(prior/'summary.json') != p['summary_sha256']:
        raise t.Blocked('BLOCKED: PT_DET_3 prior summary drift')
    for rel, h in p['receipt_sha256'].items():
        if not (prior/rel).is_file() or t.sha(prior/rel) != h:
            raise t.Blocked('BLOCKED: PT_DET_3 prior receipt drift: ' + rel)
    if not (completion/'completion.json').is_file():
        raise t.Blocked('BLOCKED: completion.json missing')
    done = json.loads((completion/'completion.json').read_text())
    i = reg['inherited']
    if (done.get('pt_det_3_registration_sha256') != REG_SHA or done.get('registration_sha256') != t.REG_SHA
            or done.get('binary') != i['binary'] or done.get('manifest_sha256') != i['manifest_sha256']
            or done.get('source_checkpoint_sha256') != i['source_checkpoint_sha256'] or done.get('steps') != 30):
        raise t.Blocked('BLOCKED: completion provenance differs from registration')
    sources = {}
    for arm in t.ARMS:
        for repeat in lane['repeats']:
            name = f'{arm}_{repeat}'; rel = f'{name}/receipt.json'
            if arm in lane['arms']:
                src = completion/name
                if (src/'receipt.json').is_file() != (rel in done['receipt_sha256']):
                    raise t.Blocked('BLOCKED: completion receipt set differs from completion.json: ' + rel)
                if not (src/'receipt.json').is_file():
                    sources[name] = None; continue
                if t.sha(src/'receipt.json') != done['receipt_sha256'][rel]:
                    raise t.Blocked('BLOCKED: completion receipt drift: ' + rel)
            else:
                src = prior/name
            rec = json.loads((src/'receipt.json').read_text())
            for fname, digest in (rec.get('files') or {}).items():
                if t.sha(src/fname) != digest: raise t.Blocked('BLOCKED: run file drift: ' + f'{name}/{fname}')
            sources[name] = src
    return done, sources


def union(prior, completion, out, reg, m):
    # Prior art: pt_det_1.repro/verify_repro (PT-DET-1, 2026) for the schema,
    # pair and verdict logic, taken unchanged. Ours: the pinned cross-slot union.
    lane = reg['lane']
    done, sources = sources_for_union(reg, prior, completion)
    out.mkdir(parents=True, exist_ok=False)
    receipts = {}; runs = {}
    for name, src in sources.items():
        if src is None:
            receipts[name] = {}; runs[name] = 'MISSING'; continue
        for fname in COPIED:
            if (src/fname).is_file(): copy_create(src/fname, out/name/fname)
        receipts[name] = json.loads((out/name/'receipt.json').read_text()); runs[name] = receipts[name].get('status')
    pairs = {arm: t.compare_pair(*(receipts[f'{arm}_{i}'] for i in lane['repeats'])) for arm in t.ARMS}
    sealed = t.assess_pairs(pairs)
    result = dict(verdict=sealed, registration_sha256=t.REG_SHA, binary=m['binary'], steps=30,
                  pt_det_2_registration_sha256=m['pt_det_2_registration_sha256'],
                  deterministic_sites=['embedding', 'gather_topk'],
                  manifest_sha256=t.sha(t.ART/t.MANIFEST_NAME),
                  source_checkpoint_sha256=t.registration()['repro']['checkpoint']['sha256'],
                  pairs=pairs, runs=runs,
                  seconds=sum(r.get('seconds') or 0. for r in receipts.values()), keep_ckpts=done.get('keep_ckpts', False),
                  evidence_class='two fresh-process 30-step runs per arm, saved tensor bytes and exact text; '
                                 'union of two GPU slots (seconds = sum of the eight run receipts)',
                  receipt_sha256={str(p.relative_to(out)): t.sha(p) for p in sorted(out.glob('*/receipt.json'))},
                  completion=dict(pt_det_3_registration_sha256=REG_SHA, prior_dir=str(prior),
                                  prior_summary_sha256=reg['prior']['summary_sha256'],
                                  prior_receipt_sha256=reg['prior']['receipt_sha256'],
                                  completion_dir=str(completion), completion_sha256=t.sha(completion/'completion.json'),
                                  completion_receipt_sha256=done['receipt_sha256'],
                                  superseded_prior_runs=reg['prior']['superseded_runs']))
    t.create_json(out/'summary.json', result)
    status = STATUS_MAP[sealed]; check = 'skipped: sealed verify_repro accepts only a GREEN replay'
    if sealed == 'GREEN':
        try:
            t.verify_repro(out/'summary.json'); check = 'GREEN'
        except Exception as exc:
            status = 'RED'; check = 'REJECTED: ' + repr(exc)
    t.create_json(out/'union_receipt.json', dict(pt_det_3_status=status, sealed_verdict=sealed, verify_repro=check,
                                                 summary_sha256=t.sha(out/'summary.json'),
                                                 pt_det_3_registration_sha256=REG_SHA))
    print('PT_DET_3 VERIFY_REPRO', check)
    print('PT_DET_3 UNION', status, 'sealed_verdict=' + sealed,
          ' '.join(f'{a}={pairs[a].get("bitwise")}' for a in t.ARMS), flush=True)
    return 0 if status == 'GREEN' else 2 if status == 'BLOCKED' else 1


def plan(reg, base):
    # CPU-only view: the manifest's binary is read, not verified (complete verifies).
    manifest = json.loads((t.ART/t.MANIFEST_NAME).read_text())
    m = dict(binary=manifest['binary']); r = t.registration()['repro']; lane = reg['lane']
    runs = {}
    for arm in lane['arms']:
        for i in lane['repeats']:
            run_dir = base/f'{arm}_{i}'
            runs[f'{arm}_{i}'] = dict(argv=[sys.executable, '-B', '-'] + t.trainer_argv(arm, run_dir),
                                      env_set=run_environment(arm, run_dir, base, m, r, base={}))
    return dict(pt_det_3_registration_sha256=REG_SHA, out=str(base), cwd=str(t.CC), stdin='pt_det_1.PREAMBLE',
                env_removed=list(REMOVED_ENV), env_note='complete inherits os.environ minus env_removed, then env_set',
                runs=runs, deadlines={k: lane[k] for k in ('per_run_seconds', 'lane_seconds', 'per_run_margin_seconds',
                                                           'slot_margin_seconds')} | dict(slot_env='PT_DET_SLOT_LANE_DEADLINE'),
                prior=reg['prior'], union=reg['union'], manifest_verified=False)


def fresh_under(out, bases):
    return any(out.is_relative_to(b) and out != b for b in bases)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('register', 'plan', 'complete', 'verdict'))
    p.add_argument('--out', type=Path)
    p.add_argument('--prior', type=Path)
    p.add_argument('--completion', type=Path)
    p.add_argument('--lead-gpu', action='store_true')
    p.add_argument('--lock-fd', type=int)
    p.add_argument('--keep-ckpts', action='store_true')
    args = p.parse_args(argv)
    if args.command == 'register':
        register(args.prior or PRIOR); return 0
    try:
        reg = registration()
    except Exception as exc:
        print('PT_DET_3 BLOCKED', str(exc), flush=True); return 2
    if args.command == 'plan':
        print(json.dumps(plan(reg, (args.out or t.ART/'lead_slot_det3/completion').resolve()), indent=2)); return 0
    if args.command == 'verdict':
        if args.prior is None or args.completion is None or args.out is None:
            p.error('verdict requires --prior, --completion and --out')
        out = args.out.resolve()
        if not fresh_under(out, (t.ART,)):
            p.error('out must be a fresh PT-DET-1 artifact subdirectory (pt_det_1.verify_repro requires it)')
        try:
            m = t.verify_manifest(); check_inherited(reg, m)
            return union(args.prior.resolve(), args.completion.resolve(), out, reg, m)
        except Exception as exc:
            status = 'BLOCKED' if isinstance(exc, (t.Blocked, t.storage.StorageBlocked)) else 'RED'
            print('PT_DET_3 UNION', status, str(exc), flush=True)
            return 2 if status == 'BLOCKED' else 1
    # complete. Guard order: registration -> require_lead -> verify_manifest -> mkdir out.
    if args.out is None: p.error('complete requires --out')
    try:
        lock = t.require_lead(args.lead_gpu, args.lock_fd)
        m = t.verify_manifest(); check_inherited(reg, m)
    except Exception as exc:
        status = 'BLOCKED' if isinstance(exc, (t.Blocked, t.storage.StorageBlocked)) else 'RED'
        print('PT_DET_3', status, str(exc), flush=True)
        return 2 if status == 'BLOCKED' else 1
    out = args.out.resolve()
    if not fresh_under(out, (t.ART, REG.parent)):
        p.error('out must be a fresh PT-DET-1 or PT-DET-3 artifact subdirectory')
    out.mkdir(parents=True, exist_ok=False)
    try:
        t.create_json(out/'lock_receipt.json', lock)
        return complete(out, args.lock_fd, reg, m, args.keep_ckpts)
    except Exception as exc:
        status = 'BLOCKED' if isinstance(exc, (t.Blocked, t.storage.StorageBlocked)) else 'RED'
        if not (out/'completion.json').exists():
            t.create_json(out/'completion.json', dict(status=status, reason=str(exc), pt_det_3_registration_sha256=REG_SHA,
                                                      gpu_executed=False if isinstance(exc, t.Blocked) else 'unknown'))
        print('PT_DET_3', status, str(exc), flush=True)
        return 2 if status == 'BLOCKED' else 1


if __name__ == '__main__':
    sys.exit(main())
