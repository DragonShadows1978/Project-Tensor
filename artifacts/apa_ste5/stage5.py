#!/usr/bin/env python3
"""Stage exact foreground lead commands; CPU preflight checks pinned inputs.

Prior art: STE1/4 governed fresh-process capture protocol (2026); GNU timeout
(2008+, unverified - lead to check). Ours: base-only g1/h premise before arms.
Thresholds come unchanged from the immutable lead order; no seat gate added.
"""
import json
import common as c
import bundle5


def stage():
    c.cpu_environment()
    lines=['#!/usr/bin/env bash','# LEAD ONLY: do not run on the CPU seat.',
        '# Prior art: STE1/4 governed captures (2026); GNU timeout (2008+, unverified - lead to check).',
        'set -euo pipefail',f'cd {c.ROOT}',
        'export TMPDIR="$PWD/artifacts/apa_ste5/scratch"',
        'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
        'mkdir -p artifacts/apa_ste5/lead',
        'export FORGE_GPU_RUN_LOG="$PWD/artifacts/apa_ste5/lead/governor.tsv"',
        'CUDA_VISIBLE_DEVICES= timeout --foreground 120s python3 -B artifacts/apa_ste5/stage5.py --check',
        '# 1. Base only, then G-G1 premise. STOP on any failure.']
    for cfg in c.CONFIGS:
        for suite in ('small','grapa'):
            _,switch,_=c.CONFIGS[cfg]
            unset='-u TC_APA_DK_STE ' if switch is None else ''
            value='' if switch is None else f'TC_APA_DK_STE={switch} '
            lines.append(f'forge-gpu-run --min 300 --max 600 --label {c.label(cfg,suite)} -- env '
                f'{unset}CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 '
                f'{value}timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config {cfg} --suite {suite}')
        if cfg=='base':
            lines.extend(['CUDA_VISIBLE_DEVICES= timeout --foreground 300s python3 -B artifacts/apa_ste5/verify5.py --premise --output artifacts/apa_ste5/lead/PREMISE.json',
                          '# 2. Arm configurations only after premise PASS.'])
    lines.append('# 3. Timing: one h forward + g1 backward pair; warmup, then 2 samples.')
    for switch in (0,1):
        lines.append(f'forge-gpu-run --min 300 --max 600 --label ste5v1-timing-{switch} -- env CUDA_VISIBLE_DEVICES=0 '
            f'APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE={switch} '
            f'timeout --foreground 300s python3 -B artifacts/apa_ste5/timing5.py --switch {switch}')
    lines.extend(['# 4. CPU verify recomputes gates, timing difference, then refreshes bundle only on PASS.',
        'CUDA_VISIBLE_DEVICES= timeout --foreground 1800s python3 -B artifacts/apa_ste5/verify5.py --output artifacts/apa_ste5/lead/VERIFY5.json'])
    with (c.HERE/'GPU_COMMANDS.sh').open('x') as f: f.write('\n'.join(lines)+'\n')
    c.put(c.HERE/'STAGING.json',{'GPU_commands':14,'fresh_processes_per_configuration_per_suite':3,
        'GPU_command_budget_seconds':300,'internal_capture_batch_budget_seconds':270,
        'budget_basis':'g1/h small runs already observed in STE1; GRAPA runtime not measured by this seat. 300 s is a hard budget, not a completed timing claim.',
        'command_file_sha256':c.digest(c.HERE/'GPU_COMMANDS.sh'),'no_GPU_run_by_seat':True})


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--check']:
        c.cpu_environment(); c.check_pins(); bundle5.validate()
        st=json.loads((c.HERE/'STAGING.json').read_text())
        assert st['command_file_sha256']==c.digest(c.HERE/'GPU_COMMANDS.sh')
        print('pinned inputs, staged commands and bundle hashes PASS')
    else:
        assert not sys.argv[1:]
        stage()
