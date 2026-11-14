#!/usr/bin/env python3
"""ENG2-A3 R2: read-only census of consumer call sites that pass a dtype into tensor_cuda's dtype entries.

Repositories (main checkouts, tracked files only, read through git and Python's ast; nothing is written there):
  /mnt/ForgeRealm/Project-Voice, /mnt/ForgeRealm/GRAPA-Native-LLM, /mnt/ForgeRealm/GraftRepository.
File listing: `git -C <repo> grep -l -E <import pattern> -- '*.py' ':!WorkTree'` on the checkout's tracked files, then
any path with a `WorkTree/` component is dropped; untracked files are never seen (git grep lists tracked files only).
Each listed file is parsed with ast from its working-tree bytes; `git show HEAD:<path>` is compared byte for byte and a
difference is recorded (modified_vs_HEAD). Multi-line string constants that mention tensor_cuda and import, and parse
as Python (subprocess preambles), are censused too (in_string_code). Beyond the order's import listing (group
"import"), two further groups: "cross-file" -- tracked files that import a tensor_cuda alias from another module of the
same repository (`from core.mistral7b_tc import F, tc`; resolved by a fixpoint over every tracked .py file, see
cross_file_aliases; listed files get these aliases too), and "supplementary" -- tracked files whose text calls
`tc.<entry>(` but bind no alias at all (the module is passed in); analysed with `tc` as the alias.

Entries (artifacts/eng2/L60_dtype_census_candidate.json and its A1/A2 completion a2/L60_dtype_census_candidate.json):
  CHANGED by ENG2 (baseline served every request, candidate refuses every non-string and every string outside the six
  supported names float32, float16, int64, bool, uint8, bfloat16): tc.tensor, tc.zeros, tc.ones, tc.randn, tc.rand,
  tc.nn.parameter, tc.functional._causal_mask.
  UNCHANGED (baseline rows == candidate rows for every probe in the A2 census): tc.empty, Tensor.astype,
  tc.int4_dequant / tc.intn_dequant / tc.kv_int4_unpack (out_dtype), quantization.affine quantize_affine_per_group
  (scale_dtype), dequantize_affine_per_group / dequantize_symmetric_per_group (out_dtype), apa_sp2
  RegisteredMargins.lookup. Their sites are censused and classified, but ENG2 cannot newly refuse them.
Callee resolution: names bound by `import tensor_cuda[.x] [as a]`, `from tensor_cuda[.x] import y [as a]`,
`from tensor_cuda import *`, `a = importlib.import_module("tensor_cuda")` / `__import__`, and `b = a` copies, anywhere in
the file (flow- and scope-insensitive: a simplification, stated); attribute receivers whose last name is such an alias
(`self.tc.zeros`) count, flagged. `.astype(x)` and `.lookup(...)` are methods: every such call is censused with its
receiver text (it may be a NumPy array, not a Tensor).
Argument classes: supported-literal | unsupported-literal | non-string-literal (None, a NumPy/torch dtype or type
object, a builtin type, a number) | traced-supported | traced-unsupported | traced-non-string | unresolved. Tracing
(within the file only, depth <= 6): assignments in the enclosing function then the module; literal for-loop targets;
pytest parametrize values on the enclosing function; a parameter's in-file call sites (one level per depth step) else
unresolved with its default noted; IfExp / BoolOp branches; constant-key subscripts of module dict literals (all
values when the key is not constant); `<name>.dtype` where <name> is assigned from a tensor_cuda call (Tensor.dtype is
one of the six names). Everything else is unresolved and listed.
Newly refused (ENG2) = a CHANGED entry whose argument class is unsupported-literal, non-string-literal,
traced-unsupported or traced-non-string; a CHANGED call with **kwargs (dtype may be inside) is listed as unresolved.
Roles (L90): receipt (any path under an `artifacts/` component), test (a tests/ or test/ component, test_*.py,
*_test.py, conftest.py), script (a scripts/, tools/, bin/, bench/, benchmarks/, experiments/, probes/ component),
product (everything else: package or engine code). Live = every role except receipt.
Usage: consumer_census.py OUT.json   (stdout: the tables)
Prior art: static call-site discovery over Python ASTs, as in linters and API-migration tools (pylint/flake8 plugins,
Bowler/LibCST codemods, "pyupgrade"; Python's own ast module docs); reaching-definition style constant tracing
(Kildall 1973 dataflow; Aho, Sethi and Ullman's "Dragon Book" constant propagation) in a deliberately crude, in-file
form. Unverified -- lead to check ("Bowler codemod", "LibCST call site", "Kildall 1973 constant propagation").
Ours: the entry table, the argument classes and the ENG2 newly-refused rule.
"""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import textwrap

REPOS = ["/mnt/ForgeRealm/Project-Voice", "/mnt/ForgeRealm/GRAPA-Native-LLM", "/mnt/ForgeRealm/GraftRepository"]
SUPPORTED = {"float32", "float16", "int64", "bool", "uint8", "bfloat16"}
IMPORT_RE = r"(^|[^A-Za-z0-9_])(import|from)[[:space:]]+tensor_cuda|import_module\(.tensor_cuda|__import__\(.tensor_cuda"
SUPP_RE = r"(^|[^A-Za-z0-9_.])tc\.(tensor|zeros|ones|randn|rand|empty|int4_dequant|intn_dequant|kv_int4_unpack)\("
T = "tensor_cuda."
# qualified callee -> (entry, changed_by_eng2, dtype keyword, positional index or None)
ENTRIES = {
    T + "tensor": ("tc.tensor", True, "dtype", None),
    T + "zeros": ("tc.zeros", True, "dtype", None),
    T + "ones": ("tc.ones", True, "dtype", None),
    T + "randn": ("tc.randn", True, "dtype", None),
    T + "rand": ("tc.rand", True, "dtype", None),
    T + "nn.parameter": ("tc.nn.parameter", True, "dtype", None),
    T + "functional._causal_mask": ("functional._causal_mask", True, "dtype", 3),
    T + "empty": ("tc.empty", False, "dtype", None),
    T + "int4_dequant": ("tc.int4_dequant(out_dtype)", False, "out_dtype", 4),
    T + "intn_dequant": ("tc.intn_dequant(out_dtype)", False, "out_dtype", 6),
    T + "kv_int4_unpack": ("tc.kv_int4_unpack(out_dtype)", False, "out_dtype", 5),
}
for mod in ("quantization.", "quantization.affine."):
    ENTRIES[T + mod + "quantize_affine_per_group"] = ("quantization.affine.quantize_affine_per_group(scale_dtype)", False, "scale_dtype", None)
    ENTRIES[T + mod + "dequantize_affine_per_group"] = ("quantization.affine.dequantize_affine_per_group(out_dtype)", False, "out_dtype", None)
    ENTRIES[T + mod + "dequantize_symmetric_per_group"] = ("quantization.affine.dequantize_symmetric_per_group(out_dtype)", False, "out_dtype", None)
METHODS = {"astype": ("Tensor.astype", False, "dtype", 0), "lookup": ("apa_sp2.RegisteredMargins.lookup", False, "dtype", 2)}
STAR_NAMES = {"tensor", "zeros", "ones", "randn", "rand", "empty", "int4_dequant", "intn_dequant", "kv_int4_unpack"}
NUMPYISH = {"numpy", "torch", "jax", "jax.numpy", "ml_dtypes", "cupy"}
BUILTIN_TYPES = {"float", "int", "bool", "str", "complex", "bytes", "object"}
NEW_REFUSAL = {"unsupported-literal", "non-string-literal", "traced-unsupported", "traced-non-string"}


def git(repo, *args, check=True):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, timeout=300)
    if check and p.returncode not in (0, 1):
        raise RuntimeError(p.stderr.decode()[-500:])
    return p


def role_of(path):
    parts = path.split("/")
    base = parts[-1]
    if "artifacts" in parts[:-1]:
        return "receipt"
    if {"tests", "test"} & set(parts[:-1]) or base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py":
        return "test"
    if {"scripts", "tools", "bin", "bench", "benchmarks", "experiments", "probes"} & set(parts[:-1]):
        return "script"
    return "product"


def dotted(node):
    out = []
    while isinstance(node, ast.Attribute):
        out.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        out.append(node.id)
        return ".".join(reversed(out))
    return None


class FileCensus:
    def __init__(self, src, extra_alias=None):
        self.tree = ast.parse(src)
        self.src = src
        self.alias = dict(extra_alias or {})   # local name -> qualified tensor_cuda path
        self.other = {}                        # local name -> other module (numpy, torch, ...)
        self.star = False
        self.parents = {}
        for p in ast.walk(self.tree):
            for c in ast.iter_child_nodes(p):
                self.parents[c] = p
        self._aliases()
        self.funcs = {}
        for n in ast.walk(self.tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcs.setdefault(n.name, []).append(n)

    def _aliases(self):
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    root = a.name.split(".")[0]
                    if a.name == "tensor_cuda" or a.name.startswith(T):
                        if a.asname:
                            self.alias[a.asname] = a.name
                        else:
                            self.alias["tensor_cuda"] = "tensor_cuda"
                    elif a.name in NUMPYISH or root in NUMPYISH:
                        self.other[a.asname or root] = a.name if a.asname else root
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                if n.module == "tensor_cuda" or n.module.startswith(T):
                    for a in n.names:
                        if a.name == "*":
                            self.star = True
                        else:
                            self.alias[a.asname or a.name] = n.module + "." + a.name
                elif n.module in NUMPYISH or n.module.split(".")[0] in NUMPYISH:
                    for a in n.names:
                        self.other[a.asname or a.name] = n.module + "." + a.name
        changed = True
        while changed:  # x = importlib.import_module("tensor_cuda") / __import__ / y = x copies
            changed = False
            for n in ast.walk(self.tree):
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    name, v = n.targets[0].id, n.value
                    q = None
                    if isinstance(v, ast.Call) and dotted(v.func) in ("importlib.import_module", "import_module", "__import__") \
                            and v.args and isinstance(v.args[0], ast.Constant) and isinstance(v.args[0].value, str) \
                            and (v.args[0].value == "tensor_cuda" or v.args[0].value.startswith(T)):
                        q = v.args[0].value
                    elif isinstance(v, (ast.Name, ast.Attribute)):
                        q = self.qualify(v)
                        if q and not (q == "tensor_cuda" or q.startswith(T)):
                            q = None
                    if q and self.alias.get(name) != q:
                        self.alias[name] = q
                        changed = True

    def qualify(self, func):
        """Qualified tensor_cuda path of a callee expression, or None. Returns (path, via_attribute_alias)."""
        d = dotted(func)
        if d is None:
            return None
        head, *rest = d.split(".")
        if head in self.alias:
            return ".".join([self.alias[head], *rest])
        if self.star and not rest and head in STAR_NAMES:
            return T + head
        return None

    def qualify_attr_alias(self, func):
        """self.tc.zeros -> tensor_cuda.zeros when the second-to-last name is a tensor_cuda alias."""
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
            if func.value.attr in self.alias:
                return self.alias[func.value.attr] + "." + func.attr
            if func.value.attr in ("tc", "tensor_cuda"):  # e.g. M.tc.zeros after `import core.m as M` (cross-file)
                return "tensor_cuda." + func.attr
        return None

    def enclosing(self, node, kinds=(ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        p = self.parents.get(node)
        while p is not None and not isinstance(p, kinds):
            p = self.parents.get(p)
        return p

    def seg(self, node):
        return (ast.get_source_segment(self.src, node) or ast.unparse(node))[:160]

    # ------------------------------------------------------------------------------------------------ classification
    def classify(self, expr, scope, depth=0, seen=None):
        """-> set of leaf kinds: 's:<name>' strings, 'ns:<what>' non-strings, 'u:<why>' unresolved."""
        seen = set() if seen is None else seen
        if depth > 6:
            return {"u:depth"}
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, str):
                return {"s:" + expr.value}
            return {"ns:" + repr(expr.value)}
        if isinstance(expr, ast.JoinedStr):
            return {"u:f-string"}
        if isinstance(expr, ast.IfExp):
            return self.classify(expr.body, scope, depth + 1, seen) | self.classify(expr.orelse, scope, depth + 1, seen)
        if isinstance(expr, ast.BoolOp):
            # `a or b`: a falsy value of a non-last operand (None, "", 0, False) is never the result (A3 refinement after
            # GraftRepository core/deepseek_v2_lite_tc.py:108 `dt = dtype or BlockTC.COMPUTE_DTYPE` read as None).
            out = set()
            for i, v in enumerate(expr.values):
                kinds = self.classify(v, scope, depth + 1, seen)
                if isinstance(expr.op, ast.Or) and i < len(expr.values) - 1:
                    kinds = {k for k in kinds if k not in ("ns:None", "s:", "ns:0", "ns:False", "ns:0.0")}
                out |= kinds
            return out
        d = dotted(expr) if isinstance(expr, (ast.Name, ast.Attribute)) else None
        if d is not None:
            head = d.split(".")[0]
            if head in self.other:
                return {"ns:" + d}
            if isinstance(expr, ast.Name) and d in BUILTIN_TYPES:
                return {"ns:type " + d}
        if isinstance(expr, ast.Call):
            fd = dotted(expr.func)
            if fd and fd.split(".")[0] in self.other:
                return {"ns:" + fd + "(...)"}
            if fd == "str":
                return {"u:str(...)"}
            return {"u:call " + (fd or "?")}
        if isinstance(expr, ast.Attribute) and expr.attr == "dtype":
            recv = expr.value
            if isinstance(recv, ast.Name):
                vals = self.assigned(recv.id, scope)
                if vals and all(isinstance(v, ast.Call) and (self.qualify(v.func) or self.qualify_attr_alias(v.func)
                                                             or (isinstance(v.func, ast.Attribute) and v.func.attr in ("astype", "cuda", "cpu", "to")
                                                                 and isinstance(v.func.value, ast.Name) and v.func.value.id == recv.id))
                                for v in vals if not isinstance(v, str)) and any(not isinstance(v, str) for v in vals):
                    if all(isinstance(v, ast.Call) and (self.qualify(v.func) or self.qualify_attr_alias(v.func)) for v in vals
                           if not isinstance(v, str)):
                        return {"s:<Tensor.dtype>"}
            return {"u:attribute " + self.seg(expr)}
        if isinstance(expr, ast.Subscript):
            base = expr.value
            if isinstance(base, ast.Name):
                vals = self.assigned(base.id, None)
                dicts = [v for v in vals if isinstance(v, ast.Dict)]
                if dicts and len(dicts) == len(vals):
                    key = expr.slice
                    out = set()
                    for dct in dicts:
                        for k, v in zip(dct.keys, dct.values):
                            if isinstance(key, ast.Constant) and isinstance(k, ast.Constant) and k.value != key.value:
                                continue
                            out |= self.classify(v, scope, depth + 1, seen)
                    if out:
                        return out
            return {"u:subscript " + self.seg(expr)}
        if isinstance(expr, ast.Name):
            key = (expr.id, id(scope))
            if key in seen:
                return {"u:cycle"}
            seen = seen | {key}
            vals = self.assigned(expr.id, scope)
            if not vals:
                return {"u:name " + expr.id + " (no in-file binding)"}
            out = set()
            for v in vals:
                if isinstance(v, str):
                    out.add(v)
                else:
                    out |= self.classify(v, self.enclosing(v) if v in self.parents else scope, depth + 1, seen)
            return out
        return {"u:expr " + type(expr).__name__}

    def assigned(self, name, scope):
        """Value expressions bound to `name` in `scope` (a function node or None = module), falling back to the module.
        Returns ast nodes, or 'u:...' strings for bindings that cannot be followed."""
        for sc in ([scope] if scope is not None else []) + [None]:
            vals = self._bindings(name, sc)
            if vals:
                return vals
        return []

    def _bindings(self, name, scope):
        root = scope if scope is not None else self.tree
        out = []
        for n in ast.walk(root):
            if scope is None and self.enclosing(n) is not None:
                continue  # module scope: skip nodes inside functions
            if scope is not None and n is not scope and self.enclosing(n) is not scope:
                continue
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        out.append(n.value)
                    elif isinstance(t, (ast.Tuple, ast.List)) and any(isinstance(e, ast.Name) and e.id == name for e in t.elts):
                        i = [isinstance(e, ast.Name) and e.id == name for e in t.elts].index(True)
                        if isinstance(n.value, (ast.Tuple, ast.List)) and len(n.value.elts) == len(t.elts):
                            out.append(n.value.elts[i])
                        else:
                            out.append("u:tuple-unpack " + name)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == name and n.value:
                out.append(n.value)
            elif isinstance(n, (ast.For, ast.comprehension)):
                tgt, it = n.target, n.iter
                idx = None
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    idx = -1
                elif isinstance(tgt, ast.Tuple) and any(isinstance(e, ast.Name) and e.id == name for e in tgt.elts):
                    idx = [isinstance(e, ast.Name) and e.id == name for e in tgt.elts].index(True)
                if idx is None:
                    continue
                if isinstance(it, ast.Name):
                    lit = [v for v in self.assigned(it.id, scope) if isinstance(v, (ast.List, ast.Tuple, ast.Set))]
                    it = lit[0] if len(lit) == 1 else it
                if isinstance(it, ast.Call) and dotted(it.func) in ("enumerate", "zip"):
                    out.append("u:loop over " + dotted(it.func))
                elif isinstance(it, (ast.List, ast.Tuple, ast.Set)):
                    for e in it.elts:
                        if idx == -1:
                            out.append(e)
                        elif isinstance(e, (ast.Tuple, ast.List)) and len(e.elts) > idx:
                            out.append(e.elts[idx])
                        else:
                            out.append("u:loop element")
                elif isinstance(it, ast.Dict) and idx == -1:
                    out.extend(it.keys)
                else:
                    out.append("u:loop over " + type(it).__name__)
        if scope is not None and isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = scope.args
            params = args.posonlyargs + args.args + args.kwonlyargs
            if name in [a.arg for a in params]:
                out.extend(self._param_values(scope, name))
        return out

    def _param_values(self, fn, name):
        args = fn.args
        pos = args.posonlyargs + args.args
        defaults = dict(zip([a.arg for a in pos][len(pos) - len(args.defaults):], args.defaults))
        defaults.update({a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None})
        # pytest parametrize on this function
        vals = []
        for dec in fn.decorator_list:
            if isinstance(dec, ast.Call) and (dotted(dec.func) or "").endswith("parametrize") and len(dec.args) >= 2:
                names = dec.args[0]
                names = ([s.strip() for s in names.value.split(",")] if isinstance(names, ast.Constant) and isinstance(names.value, str)
                         else [e.value for e in names.elts if isinstance(e, ast.Constant)] if isinstance(names, (ast.List, ast.Tuple)) else [])
                if name not in names:
                    continue
                i = names.index(name)
                values = dec.args[1]
                if isinstance(values, ast.Name):
                    lit = [v for v in self.assigned(values.id, None) if isinstance(v, (ast.List, ast.Tuple))]
                    values = lit[0] if len(lit) == 1 else values
                if isinstance(values, (ast.List, ast.Tuple)):
                    for e in values.elts:
                        if isinstance(e, ast.Call) and dotted(e.func) in ("pytest.param", "param") and e.args:
                            e = e.args[0] if len(names) == 1 else (e.args[i] if len(e.args) > i else None)
                            if e is None:
                                vals.append("u:pytest.param")
                                continue
                            vals.append(e)
                            continue
                        if len(names) == 1:
                            vals.append(e)
                        elif isinstance(e, (ast.Tuple, ast.List)) and len(e.elts) > i:
                            vals.append(e.elts[i])
                        else:
                            vals.append("u:parametrize row")
                else:
                    vals.append("u:parametrize values " + type(values).__name__)
        if vals:
            return vals
        # in-file call sites of this function name
        calls = []
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id == fn.name)
                                            or (isinstance(n.func, ast.Attribute) and n.func.attr == fn.name)):
                calls.append(n)
        pnames = [a.arg for a in pos]
        is_method = bool(pnames) and pnames[0] in ("self", "cls")
        for c in calls:
            got = None
            for kw in c.keywords:
                if kw.arg == name:
                    got = kw.value
                elif kw.arg is None:
                    got = got or "u:**kwargs at call site"
            if got is None and name in pnames:
                i = pnames.index(name) - (1 if is_method and isinstance(c.func, ast.Attribute) else 0)
                if any(isinstance(a, ast.Starred) for a in c.args):
                    got = "u:*args at call site"
                elif 0 <= i < len(c.args):
                    got = c.args[i]
            if got is None:
                got = defaults.get(name, "u:parameter " + name + " (call site omits it, no default)")
            vals.append(got)
        if not calls:
            d = defaults.get(name)
            vals.append("u:parameter %s of %s (no in-file call site%s)" % (
                name, fn.name, "; default " + self.seg(d) if d is not None else ""))
        return vals


def verdict(kinds):
    unresolved = sorted(k[2:] for k in kinds if k.startswith("u:"))
    strings = {k[2:] for k in kinds if k.startswith("s:")}
    nonstr = sorted(k[3:] for k in kinds if k.startswith("ns:"))
    bad_str = sorted(s for s in strings if s not in SUPPORTED and s != "<Tensor.dtype>")
    return unresolved, bad_str, nonstr


def classify_arg(fc, expr, scope):
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, str):
            return ("supported-literal" if expr.value in SUPPORTED else "unsupported-literal"), [expr.value]
        return "non-string-literal", [repr(expr.value)]
    if isinstance(expr, ast.Name) and expr.id in BUILTIN_TYPES and not fc.assigned(expr.id, scope):
        return "non-string-literal", ["type " + expr.id]  # a builtin type object written directly (e.g. dtype=float)
    direct = fc.classify(expr, scope) if isinstance(expr, (ast.Attribute, ast.Name, ast.Call)) else None
    if direct and len(direct) == 1 and next(iter(direct)).startswith("ns:") and not isinstance(expr, ast.Name):
        return "non-string-literal", [next(iter(direct))[3:]]
    kinds = fc.classify(expr, scope)
    unresolved, bad_str, nonstr = verdict(kinds)
    values = sorted(k for k in kinds)
    if nonstr:
        return "traced-non-string", values
    if bad_str:
        return "traced-unsupported", values
    if unresolved:
        return "unresolved", values
    return "traced-supported", values


def census_file(repo, path, src, extra_alias=None, line_offset=0, in_string=False):
    fc = FileCensus(src, extra_alias)
    rows = []
    # Code held in string constants (subprocess preambles, `python -c` texts): parsed and censused the same way when it
    # mentions tensor_cuda and parses; its rows carry in_string_code and the string's starting line (an approximation).
    for node in ast.walk(fc.tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "tensor_cuda" in node.value \
                and "import" in node.value and "\n" in node.value:
            try:
                _, inner = census_file(repo, path, textwrap.dedent(node.value), None, line_offset + node.lineno - 1, True)
            except (SyntaxError, ValueError, RecursionError):
                continue
            rows.extend(inner)
    for node in ast.walk(fc.tree):
        if not isinstance(node, ast.Call):
            continue
        q = fc.qualify(node.func)
        via_attr = False
        if q is None:
            q = fc.qualify_attr_alias(node.func)
            via_attr = q is not None
        spec = ENTRIES.get(q) if q else None
        method = None
        if spec is None and isinstance(node.func, ast.Attribute) and node.func.attr in METHODS and q is None:
            if node.func.attr == "lookup" and "RegisteredMargins" not in src and "apa_sp2" not in src:
                continue
            spec, method = METHODS[node.func.attr], node.func.attr
        if spec is None:
            continue
        entry, changed, kw, posi = spec
        arg = next((k.value for k in node.keywords if k.arg == kw), None)
        if arg is None and posi is not None and len(node.args) > posi and not any(isinstance(a, ast.Starred) for a in node.args[:posi + 1]):
            arg = node.args[posi]
        splat = any(k.arg is None for k in node.keywords) or (posi is not None and any(isinstance(a, ast.Starred) for a in node.args))
        if arg is None and not splat:
            continue  # no dtype passed: the entry's default
        scope = fc.enclosing(node)
        if arg is None:
            cls, vals = "unresolved", ["u:**kwargs/*args may carry " + kw]
        elif isinstance(arg, ast.Starred):
            cls, vals = "unresolved", ["u:starred"]
        else:
            cls, vals = classify_arg(fc, arg, scope)
        rows.append(dict(repo=Path(repo).name, file=path, line=node.lineno + line_offset, in_string_code=in_string,
                         role=role_of(path),
                         live=role_of(path) != "receipt", entry=entry, changed_by_eng2=changed,
                         callee=fc.seg(node.func) + ("  [alias via attribute]" if via_attr else ""),
                         receiver=fc.seg(node.func.value) if method else None,
                         arg=fc.seg(arg) if arg is not None else "**kwargs", cls=cls, values=vals[:12],
                         newly_refused=changed and cls in NEW_REFUSAL))
    return fc, rows


def module_of(path):
    mod = path[:-3].replace("/", ".")
    return mod[:-len(".__init__")] if mod.endswith(".__init__") else mod


def resolve_from(path, node):
    """Absolute module named by an ImportFrom in `path` (relative imports resolved against the file's package)."""
    if node.level == 0:
        return node.module
    pkg = module_of(path).split(".")
    if not path.endswith("__init__.py"):
        pkg = pkg[:-1]
    pkg = pkg[:len(pkg) - (node.level - 1)] if node.level > 1 else pkg
    return ".".join(pkg + ([node.module] if node.module else []))


def cross_file_aliases(repo, tracked):
    """Pass A (no trees kept): every tracked .py file's own tensor_cuda aliases and its `from <repo module> import n`
    statements; then a fixpoint: a name imported from a repo module that binds it to tensor_cuda (directly or through
    another repo module) is a tensor_cuda alias in the importing file (e.g. GraftRepository core/deepseek_v2_lite_tc.py
    `from core.mistral7b_tc import F, tc`). Star imports from repo modules are not followed (stated)."""
    own, froms, failed = {}, {}, []
    for path in tracked:
        try:
            src = (Path(repo) / path).read_text(encoding="utf-8")
            if "tensor_cuda" not in src and "import" not in src:
                continue
            fc = FileCensus(src)
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError, RecursionError) as exc:
            failed.append(dict(repo=Path(repo).name, file=path, error="%s: %s" % (type(exc).__name__, exc)))
            continue
        own[path] = {k: v for k, v in fc.alias.items()}
        froms[path] = [(resolve_from(path, n), a.name, a.asname or a.name) for n in ast.walk(fc.tree)
                       if isinstance(n, ast.ImportFrom) for a in n.names if a.name != "*"]
    exports = {module_of(p): dict(a) for p, a in own.items() if a}
    got = {p: {} for p in froms}
    changed = True
    while changed:
        changed = False
        for p, fl in froms.items():
            for mod, name, asname in fl:
                if mod in exports and name in exports[mod] and got[p].get(asname) != exports[mod][name] \
                        and asname not in own.get(p, {}):
                    got[p][asname] = exports[mod][name]
                    exports.setdefault(module_of(p), {})[asname] = exports[mod][name]
                    changed = True
    return {p: a for p, a in got.items() if a}, failed


def main(out_path):
    files_meta, rows, parse_fail, supp_meta = [], [], [], []
    for repo in REPOS:
        head = git(repo, "rev-parse", "HEAD").stdout.decode().strip()
        branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.decode().strip()
        tracked_py = [p for p in git(repo, "ls-files", "--", "*.py").stdout.decode().splitlines()]
        listed = git(repo, "grep", "-l", "-E", IMPORT_RE, "--", "*.py", ":!WorkTree").stdout.decode().splitlines()
        wt = [p for p in listed if "WorkTree/" in p or p.startswith("WorkTree")]
        listed = sorted(p for p in listed if p not in wt)
        xfile, xfail = cross_file_aliases(repo, [p for p in tracked_py if "WorkTree/" not in p and not p.startswith("WorkTree")])
        parse_fail.extend(dict(f, group="pass-A") for f in xfail)
        cross = sorted(p for p in xfile if p not in listed)
        supp = git(repo, "grep", "-l", "-E", SUPP_RE, "--", "*.py", ":!WorkTree").stdout.decode().splitlines()
        supp = sorted(p for p in supp if p not in listed and p not in cross and "WorkTree/" not in p)
        meta = dict(repo=repo, branch=branch, head=head, tracked_py=len(tracked_py),
                    tracked_py_under_worktree=sum("WorkTree/" in p for p in tracked_py),
                    listed=len(listed), dropped_worktree=len(wt), cross_file=len(cross),
                    listed_with_cross_file_aliases=sum(p in xfile for p in listed), supplementary=len(supp))
        for group, paths in (("import", listed), ("cross-file", cross), ("supplementary", supp)):
            for path in paths:
                full = Path(repo) / path
                try:
                    data = full.read_bytes()
                except OSError as exc:
                    parse_fail.append(dict(repo=Path(repo).name, file=path, error="unreadable: %s" % exc))
                    continue
                headblob = git(repo, "show", "HEAD:" + path, check=False)
                modified = headblob.returncode != 0 or headblob.stdout != data
                try:
                    src = data.decode("utf-8")
                    extra = dict(xfile.get(path, {}))
                    if group == "supplementary":
                        extra.update(tc="tensor_cuda")
                    fc, frows = census_file(repo, path, src, extra)
                except (SyntaxError, UnicodeDecodeError, ValueError, RecursionError) as exc:
                    parse_fail.append(dict(repo=Path(repo).name, file=path, group=group, role=role_of(path),
                                           error="%s: %s" % (type(exc).__name__, exc)))
                    continue
                imports_tc = any(isinstance(n, (ast.Import, ast.ImportFrom)) and (
                    any(a.name == "tensor_cuda" or a.name.startswith(T) for a in getattr(n, "names", []))
                    if isinstance(n, ast.Import) else (n.module or "").split(".")[0] == "tensor_cuda") for n in ast.walk(fc.tree))
                for r in frows:
                    r["group"] = group
                    r["modified_vs_HEAD"] = modified
                rows.extend(frows)
                files_meta.append(dict(repo=Path(repo).name, file=path, group=group, role=role_of(path),
                                       ast_imports_tensor_cuda=imports_tc, cross_file_aliases=xfile.get(path, {}),
                                       modified_vs_HEAD=modified, sites=len(frows)))
        supp_meta.append(meta)
    # summary tables
    counts = {}
    for r in rows:
        key = (r["repo"], r["group"], "changed" if r["changed_by_eng2"] else "unchanged", r["cls"])
        counts.setdefault(key, {"total": 0, "live": 0, "receipt": 0})
        counts[key]["total"] += 1
        counts[key]["live" if r["live"] else "receipt"] += 1
    by_role = {}
    for r in rows:
        k = (r["repo"], r["role"], "changed" if r["changed_by_eng2"] else "unchanged")
        by_role[k] = by_role.get(k, 0) + 1
    newly = [r for r in rows if r["newly_refused"]]
    unresolved_changed = [r for r in rows if r["changed_by_eng2"] and r["cls"] == "unresolved"]
    unresolved_unchanged = [r for r in rows if not r["changed_by_eng2"] and r["cls"] == "unresolved"]
    out = dict(order="orders/ENG2_A3_ENG1_ROWS_AND_CONSUMERS.md (R2)", repos=supp_meta, supported=sorted(SUPPORTED),
               import_pattern=IMPORT_RE, supplementary_pattern=SUPP_RE,
               skipped="paths with a WorkTree/ component; untracked files (git grep lists tracked files only)",
               files=files_meta, parse_failures=parse_fail,
               counts=[dict(repo=k[0], group=k[1], entry_kind=k[2], cls=k[3], **v) for k, v in sorted(counts.items())],
               by_role=[dict(repo=k[0], role=k[1], entry_kind=k[2], sites=v) for k, v in sorted(by_role.items())],
               newly_refused=newly, unresolved_changed=unresolved_changed, unresolved_unchanged=unresolved_unchanged,
               rows=rows)
    Path(out_path).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(supp_meta, indent=1))
    print("files analysed:", len(files_meta), "parse failures:", len(parse_fail), "sites:", len(rows))
    print("newly refused:", len(newly), "unresolved (changed entries):", len(unresolved_changed),
          "unresolved (unchanged entries):", len(unresolved_unchanged))


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main(sys.argv[1])
