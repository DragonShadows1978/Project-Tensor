"""FP64 references and registered operands for the BWD-R1 gradient checks."""

from dataclasses import dataclass, field
from functools import cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from . import voice_fp64

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/bwd_r1"
FIXTURE_ENV = "TC_BWD_KOKORO_FIXTURES"
BP1 = Path(os.environ.get(FIXTURE_ENV, "/mnt/ForgeRealm/Project-Voice/artifacts/bp1")).absolute()
MANIFEST = BP1 / "fixtures/manifest.json"
CAPTURE = BP1 / "reference_inputs.npz"
MANIFEST_SHA256 = "3822b6a61978e9f510f030039aba260ecc7a264c4525eb31fb8a72bea305efee"
CAPTURE_SHA256 = "6e9afdaf664f15e361e95d23ff179a6d469d667b689d2edd43d2802ff641c67e"
# Source path for the finite-difference audit's input census.
VOICE_OPS = Path(voice_fp64.__file__)
VOICE_REFERENCES = {
    "add": "add_vjp", "leaky": "leakyrelu_vjp", "snake": "snake_vjp",
    "div_constant": "div_constant_vjp",
}

CORE = ("matmul", "add", "mul", "div", "square", "sin", "cos", "exp", "sqrt",
        "tanh", "mean", "leaky", "snake", "reciprocal")
POSITIVE = {"sqrt", "reciprocal"}
UNARY = {"square", "sin", "cos", "exp", "sqrt", "tanh", "reciprocal"}


@dataclass(frozen=True)
class Case:
    op: str
    label: str
    operands: tuple
    seed: np.ndarray
    attrs: dict = field(default_factory=dict)
    source: str = ""


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# Prior art: pytest imperative skips (pytest, 2004 onward) and SHA-256
# (NIST FIPS 180-4, 2015). TensorCUDA skips only when the configured root
# is absent and checks the pinned manifest, individual NPZs and capture.
def require_fixture_root():
    if not BP1.exists():
        pytest.skip(f"Kokoro fixture root absent: {BP1} (set {FIXTURE_ENV} to its location)")


@cache
def fixture_index():
    require_fixture_root()
    data = MANIFEST.read_bytes()
    assert hashlib.sha256(data).hexdigest() == MANIFEST_SHA256, MANIFEST
    return {row["file"]: row for row in json.loads(data)["files"]}


@cache
def fixture(name):
    path = BP1 / "fixtures" / (name + "_kokoro.npz")
    info = fixture_index()[path.name]
    assert sha(path) == info["sha256"], path
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: np.ascontiguousarray(npz[k], dtype=np.float32) for k in npz.files
                  if k.startswith("arg_") or k == "upstream"}
    return arrays, info["attrs"], path


def f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def seed_with_zeros(seed):
    seed = f32(seed).copy()
    if seed.size > 1:
        seed.flat[0] = 0
        seed.flat[-1] = 0
    return seed


def small(op, kind="plain"):
    rng = np.random.default_rng(1718)
    x = rng.uniform(-0.9, 0.9, (2, 3, 7))
    b_shape = (1, 3, 1) if kind == "broadcast" else x.shape
    seed = rng.uniform(-1, 1, x.shape)
    attrs = {}
    if op == "matmul":
        prefix = (2, 2) if kind.startswith("batch") else ()
        trans = kind.endswith("trans")
        x = rng.uniform(-1, 1, (*prefix, 3, 5))
        wp = prefix if kind.startswith("batched") else ()
        w = rng.uniform(-1, 1, (*wp, *((4, 5) if trans else (5, 4))))
        operands = (x, w)
        seed = rng.uniform(-1, 1, (*prefix, 3, 4))
        attrs = {"trans_b": trans, "alpha": -0.625}
    elif op in {"add", "mul", "div"}:
        b = rng.uniform(0.55, 1.75, b_shape) if op == "div" else rng.normal(size=b_shape)
        operands = (x, b)
    elif op == "snake":
        operands = (x, rng.uniform(0.4, 1.7, (1, 3, 1)))
    elif op == "leaky":
        operands = (np.where(x < 0, x - .05, x + .05),)
        attrs = {"slope": float(np.float32(.2))}
    elif op == "mean":
        operands = (x,)
        attrs = {"axis": -1}
        seed = rng.normal(size=x.shape[:-1])
    elif op == "div_constant":
        operands = (x,)
        attrs = {"divisor": 3.}
    else:
        operands = (np.abs(x) + .5 if op in POSITIVE else x,)
    return Case(op, kind, tuple(map(f32, operands)), seed_with_zeros(seed), attrs, "small PCG64")


def kokoro(op, kind="plain"):
    attrs = {}
    if op == "matmul":
        data, _, path = fixture("adain")
        operands = (data["arg_s"], data["arg_W"])
        seed = data["upstream"][0, :, [0, -1]].reshape(1, -1)
        attrs = {"trans_b": True, "alpha": 1.}
    elif op == "leaky":
        data, attrs, path = fixture("leakyrelu_" + (kind if kind != "plain" else "0p2"))
        operands = (data["arg_x"],)
        seed = data["upstream"]
    elif op == "div_constant":
        data, attrs, path = fixture("div_constant")
        operands = (data["arg_x"],)
        seed = data["upstream"]
    elif op == "add" and kind == "plain":
        data, _, path = fixture("add")
        operands = (data["arg_x"], data["arg_y"])
        seed = data["upstream"]
    else:
        data, _, path = fixture("snake")
        x, a, seed = data["arg_x"], data["arg_alpha"], data["upstream"]
        if op == "snake":
            operands = (x, a)
        elif op in {"add", "mul", "div"}:
            b = a if kind == "broadcast" else np.abs(np.roll(x, 1, axis=-1)) + np.float32(.5)
            operands = (x, b)
        elif op == "mean":
            operands = (x,)
            seed = seed[..., 1]
            attrs = {"axis": -1}
        else:
            operands = (np.abs(x) + np.float32(.5) if op in POSITIVE else np.clip(x, -4, 4),)
    return Case(op, "kokoro_" + kind, tuple(map(f32, operands)), seed_with_zeros(seed), attrs, str(path))


def real_tail():
    fixture_index()
    assert sha(CAPTURE) == CAPTURE_SHA256, CAPTURE
    with np.load(CAPTURE, allow_pickle=False) as npz:
        x = f32(npz["wave_decoder"])
    assert x.shape == (1, 128, 25441) and x.size and np.ptp(x)
    seed = np.random.default_rng(1718).uniform(-1, 1, x.shape)
    return Case("leaky", "captured_tail", (x,), seed_with_zeros(seed),
                {"slope": float(np.float32(.01))}, str(CAPTURE))


def unbroadcast(terms, shape):
    padded = (1,) * (terms.ndim - len(shape)) + shape
    axes = tuple(i for i, (a, b) in enumerate(zip(terms.shape, padded)) if a != b)
    return terms.sum(axis=axes, keepdims=True).reshape(shape)


def transpose(x):
    return x.swapaxes(-2, -1)


# Prior art: Linnainmaa (1970) and Baydin et al. (2018) reverse mode and VJP
# identities. Ours: NumPy implementations for rows outside Voice's fp64 key
# and termwise magnitudes for the frozen tolerance table.
def forward(case, xs):
    op, attrs = case.op, case.attrs
    x = xs[0]
    if op == "matmul":
        w = transpose(xs[1]) if attrs["trans_b"] else xs[1]
        return attrs["alpha"] * (x @ w)
    if op in VOICE_REFERENCES:
        x = np.asarray(x, dtype=np.float64)
        if op == "add":
            return x + np.asarray(xs[1], dtype=np.float64)
        if op == "leaky":
            return np.where(x < 0, x * attrs["slope"], x)
        if op == "snake":
            alpha = np.asarray(xs[1], dtype=np.float64)
            return x + np.reciprocal(alpha) * np.square(np.sin(alpha * x))
        return x / attrs["divisor"]
    if op == "mul":
        return x * xs[1]
    if op == "div":
        return x / xs[1]
    if op == "mean":
        return x.mean(axis=attrs["axis"])
    return getattr(np, op)(x)


def vjp(case, xs, g):
    op, attrs = case.op, case.attrs
    x = xs[0]
    if op == "matmul":
        w = transpose(xs[1]) if attrs["trans_b"] else xs[1]
        alpha = attrs["alpha"]
        dx = alpha * (g @ transpose(w))
        dw = alpha * (transpose(x) @ g)
        sx = abs(alpha) * (abs(g) @ transpose(abs(w)))
        sw = abs(alpha) * (transpose(abs(x)) @ abs(g))
        if attrs["trans_b"]:
            dw, sw = transpose(dw), transpose(sw)
        return (dx, unbroadcast(dw, xs[1].shape)), (sx, unbroadcast(sw, xs[1].shape))
    if op in VOICE_REFERENCES:
        grads = getattr(voice_fp64, VOICE_REFERENCES[op])(*xs, g, **attrs)
        if op == "snake":
            a = xs[1]
            sine = np.sin(a * x)
            activation = g * (2 * sine * np.cos(a * x) * x / a)
            reciprocal = -g * np.square(sine) / np.square(a)
            return grads, (None, unbroadcast(abs(activation) + abs(reciprocal), a.shape))
        if op == "add":
            return grads, tuple(unbroadcast(abs(g), operand.shape) for operand in xs)
        return grads, (None,)
    if op == "mul":
        terms = (g * xs[1], g * x)
    elif op == "div":
        terms = (g / xs[1], -g * x / np.square(xs[1]))
    elif op == "mean":
        grad = np.expand_dims(g, attrs["axis"]) / x.shape[attrs["axis"]]
        return (np.broadcast_to(grad, x.shape),), (None,)
    else:
        derivative = {
            "square": lambda: 2 * x,
            "sin": lambda: np.cos(x),
            "cos": lambda: -np.sin(x),
            "exp": lambda: np.exp(x),
            "sqrt": lambda: .5 / np.sqrt(x),
            "tanh": lambda: 1 - np.square(np.tanh(x)),
            "reciprocal": lambda: -1 / np.square(x),
        }[op]()
        return (g * derivative,), (None,)
    return (tuple(unbroadcast(term, operand.shape) for term, operand in zip(terms, xs)),
            tuple(unbroadcast(abs(term), operand.shape) for term, operand in zip(terms, xs)))


def classify(case, slot):
    op = case.op
    if op == "matmul":
        return "matmul_conv"
    if op == "snake":
        return "unbroadcast" if slot == 1 else "snake_dx"
    if op in {"add", "mul", "div"} and case.operands[slot].shape != case.seed.shape:
        return "unbroadcast"
    if op in {"add", "mul", "square", "leaky", "mean"}:
        return "simple"
    return "unary_div"


@cache
def tolerances():
    path = ART / "TOLERANCES.json"
    seal = (ART / "TOLERANCES.sha256").read_text().split()
    assert sha(path) == seal[0], "tolerance seal changed"
    return json.loads(path.read_text())["classes"]


def allowed(case, slot, reference, terms):
    spec = tolerances()[classify(case, slot)]
    atol = np.maximum(spec["atol_floor"], terms * spec["atol_scale"]) if terms is not None else spec["atol_floor"]
    return atol + spec["rtol"] * abs(reference)


def numerical(got, expected):
    assert got.shape == expected.shape and got.size
    assert np.isfinite(got).all() and np.isfinite(expected).all()
    assert np.ptp(got) > 0 and np.ptp(expected) > 0


def compare(case, got, reference=None, terms=None):
    if reference is None:
        reference, terms = vjp(case, tuple(a.astype(np.float64) for a in case.operands),
                               case.seed.astype(np.float64))
    assert len(got) == len(reference) == len(terms)
    for i, (observed, want, s) in enumerate(zip(got, reference, terms)):
        numerical(observed, want)
        err = abs(observed.astype(np.float64) - want)
        rel = np.divide(err, abs(want), out=np.zeros_like(err), where=want != 0)
        ratio = err / allowed(case, i, want, s)
        loc = lambda a: tuple(map(int, np.unravel_index(a.argmax(), a.shape)))
        print(json.dumps(dict(op=case.op, cell=case.label, slot=i, shape=want.shape, source=case.source,
                              tolerance=classify(case, i), max_abs=float(err.max()), abs_at=loc(err),
                              max_rel=float(rel.max()), rel_at=loc(rel), max_bound_ratio=float(ratio.max()),
                              bound_at=loc(ratio), rel_l2=float(np.linalg.norm(err) / np.linalg.norm(want))),
                         sort_keys=True), flush=True)
        assert np.all(observed[want == 0] == 0), "nonzero gradient at exact reference zeros"
        assert np.all(ratio <= 1), (case.op, case.label, i, float(ratio.max()), loc(ratio))
