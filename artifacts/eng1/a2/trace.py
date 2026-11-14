#!/usr/bin/env python3
"""Run REV2 code unchanged with A2 output/scratch routing only.
Prior art: REV2 scripts (2026), PEP 578 (Dower, 2019), Python runpy/sys.settrace.
Taken: unchanged read attribution, write fence and classification. New: runtime
SCRATCH relocation at the first line after its assignment, before any use.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

A = Path(__file__).resolve().parent
R = A.parents[2]
V = R / 'artifacts/eng1/rev2'
mode = sys.argv[1]
name = 'q2_source_readers.py' if mode in 'ABC' and len(mode) == 1 else 'q2_classify_reads.py'
assert mode in ('A', 'B', 'C', 'classify')
script = A / name
assert script.read_bytes() == (V / name).read_bytes(), 'REV2 script changed'
if mode == 'classify':
    sys.argv = [str(script)]
else:
    # The unchanged fence permits writes only below SCRATCH. Keep tempfile's
    # output below that same relocated root before pytest/plugin initialization.
    tmp = Path('/tmp/eng1_a2_work/trace/tmp')
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(tmp)
    data = json.loads((V / ('q2_readers_' + mode + '.json')).read_text())
    sys.argv = [str(script), 'q2_readers_' + mode, *data['files']]
    def relocate(frame, event, arg):
        if (event == 'line' and frame.f_code.co_filename == str(script)
                and frame.f_globals.get('SCRATCH') == '/tmp/eng1_rev2_work'):
            frame.f_globals['SCRATCH'] = '/tmp/eng1_a2_work/trace'
            sys.settrace(None)
            return None
        return relocate
    sys.settrace(relocate)
runpy.run_path(str(script), run_name='__main__')
