"""Exact SASS Function-block parser copied from PF2-F1 (2026)."""
from pathlib import Path
import re
import subprocess

PATTERNS = [(re.compile(r'([A-Za-z]*)\d+_GLOBAL__N__[0-9a-f]{8}_'), r'\1N_GLOBAL__N__H_'),
            (re.compile(r'_GLOBAL__N__[0-9a-f]{8}_'), '_GLOBAL__N__H_'),
            (re.compile(r'__nv_static_\d+__[0-9a-f]{8}_'), '__nv_static_N__H_'),
            (re.compile(r'_cu_[0-9a-f]{8}(?:_\d+)?'), '_cu_H')]


def norm(text):
    for pat, rep in PATTERNS:
        text = pat.sub(rep, text)
    return text


def read(path):
    blocks, names, key, lines = {}, {}, None, []
    for line in Path(path).open():
        m = re.search(r'Function : (\S+)', line)
        if m:
            key = norm(m[1])
            assert key not in names, ('non-bijective symbol map', key)
            names[key] = m[1]
            lines = []
        elif key and line.strip() == '..........':
            blocks[key] = norm(''.join(lines))
            key = None
        elif key:
            lines.append(line)
    return blocks, names


def demangle(names):
    p = subprocess.run(['timeout', '60s', 'c++filt'], input='\n'.join(names) + '\n', capture_output=True, text=True, check=True)
    return [re.sub(r'\(.*$', '', line) for line in p.stdout.splitlines()]

