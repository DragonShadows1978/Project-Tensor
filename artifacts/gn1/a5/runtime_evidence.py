"""A4 D2/D4/D5/D6 evidence primitives; no engine import.

Prior art: atomic publication by POSIX rename(2), IEEE Std 1003.1
(year unverified, lead to check 'POSIX rename atomic replacement'); Python
sys.modules/__file__ census, PSF (year unverified, lead to check 'Python
sys.modules module origins'); SHA-256 FIPS 180-4 (2015); operator.index,
PEP 357 (2006). GN1-specific receipt fields are the application here.
"""
import hashlib
import json
import os
from pathlib import Path
import sys


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def module_census(root):
    rows = []
    for name, module in sorted(list(sys.modules.items())):
        origin = getattr(module, '__file__', None)
        if not origin:
            continue
        path = Path(origin).resolve()
        display = (str(path.relative_to(root)) if path.is_relative_to(root)
                   else str(path).replace(str(Path.home()) + '/', '$HOME/', 1))
        row = dict(module=name, path=display)
        try:
            row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            row['error'] = type(exc).__name__ + ': ' + str(exc)
        rows.append(row)
    return rows


def npbool_index_check():
    import operator
    import numpy
    result = dict(numpy_version=numpy.__version__)
    try:
        result['value'] = operator.index(numpy.bool_(True))
        result['accepts_numpy_bool'] = True
    except TypeError as exc:
        result.update(accepts_numpy_bool=False, exception_class='TypeError',
                      message=str(exc))
    return result


def runtime_modules(rows, seal, tool, root):
    """Compare every recorded origin with all seal namespaces, list unknowns."""
    known = dict(seal['external_modules'], **seal['import_modules'])
    prefix = str(tool.relative_to(root)) + '/'
    known.update({prefix + n: h for n, h in seal['tool_files'].items()})
    known.update(seal['test_files'])
    for arm, files in seal['arms'].items():
        known.update({prefix + 'arms/' + arm + '/tensor_cuda/' + n: h
                      for n, h in files.items()})
    errors, unsealed = [], []
    assert isinstance(rows, list) and rows, 'missing runtime module census'
    for row in rows:
        path = row['path']
        if row.get('error'):
            errors.append('runtime module unreadable: ' + path)
        elif path in known:
            if row['sha256'] != known[path]:
                errors.append('runtime module tamper: ' + path)
        else:
            unsealed.append(row)
    return errors, unsealed
