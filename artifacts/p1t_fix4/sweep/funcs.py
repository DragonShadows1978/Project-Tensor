"""P1T-FIX4 G2 helper: enclosing C/C++ function of a source line, by brace matching (a '{' that follows ')' plus optional
qualifiers opens a function body when the identifier before the matching '(' is not a control keyword or a lambda).
Comments and string literals are blanked first. No prior art claimed."""
import bisect
import functools
import re
from pathlib import Path

KW = {'if', 'for', 'while', 'switch', 'catch', 'return', 'sizeof', 'defined'}


def blank(text):
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        if text.startswith('//', i):
            j = text.find('\n', i); j = n if j < 0 else j
            out[i:j] = ' ' * (j - i); i = j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2); j = n if j < 0 else j + 2
            out[i:j] = [c if c == '\n' else ' ' for c in text[i:j]]; i = j
        elif text[i] in '"\'':
            q = text[i]; j = i + 1
            while j < n and text[j] != q:
                j += 2 if text[j] == '\\' else 1
            out[i + 1:j] = ' ' * (j - i - 1); i = j + 1
        else:
            i += 1
    return ''.join(out)


@functools.lru_cache(maxsize=None)
def functions(path):
    text = Path(path).read_text()
    b = blank(text)
    nl = [i for i, c in enumerate(b) if c == '\n']
    lineno = lambda pos: bisect.bisect_left(nl, pos) + 1
    res = []
    for m in re.finditer(r'\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?::[^{;]*)?\{', b):
        close = m.start()
        depth, k = 0, close
        while k >= 0:
            depth += {')': 1, '(': -1}.get(b[k], 0)
            if depth == 0:
                break
            k -= 1
        name_m = re.search(r'(~?[A-Za-z_][\w:~]*)\s*$', b[max(0, k - 200):k])
        if not name_m or name_m.group(1) in KW or b[max(0, k - 200):k].rstrip().endswith(']'):
            continue
        name = name_m.group(1)
        open_brace = m.end() - 1
        depth, e = 0, open_brace
        while e < len(b):
            depth += {'{': 1, '}': -1}.get(b[e], 0)
            if depth == 0:
                break
            e += 1
        res.append((name, lineno(open_brace), lineno(e)))
    return res


def enclosing(path, line):
    best = None
    for name, s, e in functions(path):
        if s <= line <= e and (best is None or s >= best[1]):
            best = (name, s, e)
    return best[0] if best else '(file scope)'
