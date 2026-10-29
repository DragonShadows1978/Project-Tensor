#!/usr/bin/env python3
"""Lead-only governed STE5 captures; each run is a fresh interpreter.

Prior art: STE1/STE2 capture isolation (Project-Tensor, 2026), reused;
NIST FIPS 180-4 (2015) SHA-256; IEEE 754-2019 raw widths.
External standards unverified - lead to check. No new kernel algorithm.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

import common as c


def capture(args):
    c.environment()
    c.check_pins()
    which, switch, all_selected = c.CONFIGS[args.config]
    actual = os.environ.get('TC_APA_DK_STE')
    assert actual == (None if switch is None else str(switch)), 'wrong switch environment'
    assert os.environ.get('TC_DETERMINISTIC') == '1'
    assert os.environ.get('TC_DET_EMBED_BWD') == '1'
    assert 0 <= args.run < c.N
    if args.config!='base': c.require_premise()
    target = c.folder(args.config, args.suite, args.run)
    target.mkdir(parents=True, exist_ok=False)
    old = c.import_old_driver()
    tc = old.load_tc(which)
    # Binding must run INSIDE the worker before its first attention call.
    tc._C.bp_kernel_2_set_variant('g1')
    assert tc._C.bp_kernel_2_get_variant() == 'g1'
    tc._C.bp_kernel_4_set_variant('h')
    assert tc._C.bp_kernel_4_get_variant() == 'h'
    identity = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,driver_version',
                               '--format=csv,noheader'], capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    started = time.monotonic()
    records = []
    for name in c.names(args.suite):
        with np.load(c.fixture(name), allow_pickle=False) as fixture:
            tensors = [tc.tensor(fixture[key], dtype='bfloat16', device='cuda') for key in c.INPUTS]
        z = -1e6 if all_selected else c.Z
        outputs = old.ops(tc, tensors, c.SHAPES[name][4]**-.5, z, name != 'small1', 'h_g1')
        arrays = {key: old.raw(x, 'float32' if key in ('lse','thr') else 'bfloat16')
                  for key, x in zip(c.KEYS, outputs)}
        arrays.update({key: old.raw(x, 'bfloat16') for key, x in zip(c.INPUTS, tensors)})
        for key, words in arrays.items():
            c.healthy(c.decode(words))
        expected = c.input_words(name)
        assert all(arrays[key].tobytes() == expected[key].tobytes() for key in c.INPUTS)
        path = target / f'{name}.npz'
        with path.open('xb') as f:
            np.savez(f, **arrays)
        records.append({'name': name, 'sha256': c.digest(path), 'zthr': z,
                        'shape': c.SHAPES[name], 'variant': 'g1', 'forward_variant': 'h',
                        'dtype': 'bfloat16', 'causal': name != 'small1'})
    c.put(target / 'capture.json', {
        'evidence_class': 'raw GPU capture only; not a G-gate verdict',
        'config': args.config, 'suite': args.suite, 'run': args.run, 'which': which,
        'switch_environment': actual, 'all_selected': all_selected, 'variant': 'g1',
        'so_sha256': c.digest((c.BASE_PACKAGE if which == 'base' else c.ARM_PACKAGE)/c.SO_NAME),
        'driver_sha256': c.digest(__file__), 'input_pins_sha256': c.digest(c.HERE/'INPUT_PINS.json'),
        # Crash amendment STE5B: preserve the verifier's strict amendment binding.
        # Prior art: no prior art known to me; no new algorithm.
        'effective_pins_sha256': c.evidence_pins_digest(),
        'pid': os.getpid(), 'host': socket.gethostname(), 'gpu_identity': identity,
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'TC_DETERMINISTIC': os.environ['TC_DETERMINISTIC'],
        'TC_DET_EMBED_BWD': os.environ['TC_DET_EMBED_BWD'],
        'cases': records, 'elapsed_s': time.monotonic()-started})


def batch(args):
    # GNU timeout bounds the foreground process group. No detached workers,
    # polling loop or background jobs; each child is awaited synchronously.
    started = time.monotonic()
    for run in range(c.N):
        remaining = int(270 - (time.monotonic()-started))
        assert remaining > 0, '270s internal batch budget exhausted'
        subprocess.run(['timeout', '--foreground', f'{remaining}s', sys.executable, '-B',
                        __file__, 'capture', '--config', args.config, '--suite', args.suite,
                        '--run', str(run)], check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['capture','batch'])
    p.add_argument('--config', choices=c.CONFIGS, required=True)
    p.add_argument('--suite', choices=['small','grapa'], required=True)
    p.add_argument('--run', type=int)
    args = p.parse_args()
    assert os.environ.get('APA_STE5_LEAD_GPU') == '1', 'lead-only GPU opt-in required'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'registered visible GPU 0 required'
    (capture if args.command == 'capture' else batch)(args)
