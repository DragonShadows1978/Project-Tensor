"""APA-P1-D: packed SP key storage as the shipped default, with its own margins.

Policy (lead rulings R-A to R-H of APA-P1-D):
- quantize_sp2_keys(k, bulk_bits, *, storage=None) chooses the representation:
  the explicit argument, then TC_APA_SP_STORAGE, then the shipped default
  "packed". Only "packed" and "dense" are accepted.
- Packed storage covers 4-bit bulk at D64/D128 on the public fp32 route. Under
  the default, an ineligible input takes the dense representation and records
  the reason; an explicit storage="packed" that cannot be honoured raises.
  Nothing changes 2 bits to 4.
- PackedSpKeys owns the 4-bit codes, the per-key norms and the fixed tables, and
  no K-shaped dense tensor. Its exact-K identity is the source object plus an
  append generation (and the source's storage revision).
- The public wrapper takes the ordered packed-r2 path for a PackedSpKeys,
  governed only by the packed margin table, with delta from the same
  upward-rounded rule as the native epsilon entry.

Prior art (Prior Art Directive):
- TurboQuant (Zandieh et al., 2025, arXiv 2504.19874): rotated Lloyd-Max 4-bit
  codes. Taken unchanged from quant.py / the SPQ1 packer; nothing new here.
- BLASST (Yuan et al., 2025, arXiv 2512.12087): the running-max refine
  criterion used by the packed and dense kernels; inherited, unchanged.
- APA-SP2 (Project-Tensor, 2026): the two-error margin argument
  delta = -log(epsilon) + 2*e_q and its upward rounding to fp32; the Python
  helper below re-states the native rule (apa_sp2.cuh:3-14) for the packed route.
- PyTorch-style owned storage handles / tensor version counters (PyTorch,
  Paszke et al., 2019): an object that owns its buffers and a revision used to
  detect in-place edits. Taken: the idea of checking a storage revision; ours:
  binding it, with an append generation, into the exact-K identity.
- ApaSpq1Workspace (Project-Tensor P1, 2026): the append-only packed buffers
  this object is built on.
- Policy precedence argument > environment > shipped default: ordinary
  configuration practice; no specific prior art known to me.
"""
import hashlib
import json
import math
import os
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

STORAGE_ENV = "TC_APA_SP_STORAGE"
SHIPPED_DEFAULT = "packed"
STORAGES = ("packed", "dense")
PACKED_BULK_BITS = 4
PACKED_DIMS = (64, 128)
PACKED_FORMAT = "tq4"
# Ordered packed-r2 score arithmetic with partition carry, the fp32 public route.
PACKED_BACKEND = "spq4_packed_r2_tq4_partition"
DENSE_BACKEND = "sp_epsilon_dense"
# Quantizer identity of the packed table: same TurboQuant tables as the dense
# quantizer, packed codes and norms, bulk scores from packed-r2 arithmetic.
PACKED_QUANTIZER = "turboquant_lloyd_max_rotated_fp32_packed_tq4_r2"
PACKED_STATISTIC = "maximum_over_all_registered_G2_valid_scores"
PACKED_MEASUREMENT_CLASSES = 32
# sha256 of artifacts/apa_p1d/MARGIN_REGISTRATION.json (registered before L5).
PACKED_REGISTRATION_SHA256 = "025b1a5116ceae3e0f272d5a8823970c31d0d65e709b785cd81e6a2cbbda1538"
# sha256 of the frozen packed table artifacts/apa_p1d/packed_e_q_table.json.
PACKED_TABLE_SHA256 = "1897fedd2a8d05945e34433c75a7afd3b33feb64e0a6b2ca410d611d18868e98"

_FLOAT32_MAX = float(np.finfo(np.float32).max)

# R-H: internal route receipts (thread-local; no attention output schema change).
_ROUTES = threading.local()


def _record_route(kind, **fields):
    setattr(_ROUTES, kind, dict(fields))


def last_route(kind="attention"):
    """Return the route of this thread's last quantize/attention call, or None.

    kind is "quantize" or "attention". Fields: storage, backend,
    fallback_reason, source (quantize) or l (attention).
    """
    if kind not in ("quantize", "attention"):
        raise ValueError("kind must be quantize or attention")
    route = getattr(_ROUTES, kind, None)
    return None if route is None else dict(route)


def resolve_storage(storage):
    """R-A: return (storage, source). Raises ValueError before any device work."""
    if storage is not None:
        if type(storage) is not str or storage not in STORAGES:
            raise ValueError("apa_sp2: storage must be 'packed' or 'dense'")
        return storage, "argument"
    value = os.environ.get(STORAGE_ENV)
    if value is not None:
        if value not in STORAGES:
            raise ValueError(f"apa_sp2: {STORAGE_ENV} must be 'packed' or 'dense'")
        return value, "environment"
    return SHIPPED_DEFAULT, "default"


def _packed_backend_present():
    from . import _C  # noqa: F401  (package must be importable)
    import importlib
    package = importlib.import_module(__package__)
    spo2 = getattr(package, "_spo2", None)
    return spo2 is not None and hasattr(spo2, "packed_run") and hasattr(spo2, "packed_append")


def packed_ineligible_reason(k, bulk_bits):
    """R-B: None if packed storage covers this input, else the fallback reason."""
    if bulk_bits != PACKED_BULK_BITS:
        return f"bulk_bits={bulk_bits}: packed storage covers 4-bit bulk only"
    if k.ndim != 4 or k.shape[-1] not in PACKED_DIMS:
        return "packed storage covers rank-4 keys at D64/D128 only"
    if k.dtype != "float32":
        return f"dtype={k.dtype}: packed storage covers the public fp32 route only"
    if not str(k.device).startswith("cuda"):
        return f"device={k.device}: packed storage needs a CUDA tensor"
    if not _packed_backend_present():
        return "this build has no packed backend (TC_APA_PACKED off)"
    return None


def derive_delta(epsilon, e_q):
    """Python statement of the native rule in apa_sp2.cuh:3-14 (apa_sp2_delta).

    delta = upward-rounded fp32(-log(epsilon) + 2*e_q), computed in double, cast
    round-to-nearest to fp32, then raised one fp32 ulp if the cast rounded down.
    Same domain refusals, raised as ValueError before any launch. This is the
    one helper the packed route uses; its results are checked against the
    native _C.apa_sp2_delta on registered boundary fixtures (L2).
    Prior art: APA-SP2 (2026) two-error margin; IEEE 754 directed rounding.
    """
    epsilon = float(epsilon)
    e_q = float(e_q)
    if (not math.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 1.0 or
            not math.isfinite(e_q) or e_q < 0.0):
        raise ValueError("apa_sp2: epsilon must be in (0,1]; e_q finite nonnegative")
    margin = -math.log(epsilon) + 2.0 * e_q
    if not math.isfinite(margin) or margin > _FLOAT32_MAX:
        raise ValueError("apa_sp2: derived delta exceeds float32 range")
    result = np.float32(margin)
    if float(result) < margin:
        result = np.nextafter(result, np.float32(np.inf))
    return float(result)


def _device_ordinal(device):
    device = str(device)
    if not device.startswith("cuda"):
        return None
    return int(device.split(":", 1)[1]) if ":" in device else 0


def _source_ref(k):
    try:
        return weakref.ref(k)
    except TypeError:
        return None


class PackedSpKeys:
    """Owned packed SP key state (R-C). Not a Tensor and not a tuple.

    Owns: 4-bit codes [B,KV,capacity,D/2] uint8 and per-key norms
    [B,KV,capacity] fp32 (68 B per key per KV head at D128, 36 at D64), plus
    the fixed TurboQuant tables (rotation [KV,D,D], codebook [16], boundaries
    [15]). It owns no K-shaped dense tensor and holds only a weak reference to
    its exact-K source.

    Exact-K identity: the source object (weak reference and id), its storage
    revision at binding, and an append generation. append() is the only way to
    extend; any other edit of the keys must build a new object (invalidate()
    marks this one stale).
    """

    __slots__ = ("_ws", "batch", "kv_heads", "dim", "capacity", "dtype",
                 "device", "device_ordinal", "bulk_bits", "quantizer",
                 "packed_format", "table_quantizer", "table_registration_sha256",
                 "generation", "_bound_generation", "_source", "_source_id",
                 "_source_revision", "__weakref__")

    def __init__(self, k, capacity=None):
        from . import ApaSpq1Workspace
        from .apa_sp2 import QUANTIZER
        if k.ndim != 4 or k.dtype != "float32" or k.shape[-1] not in PACKED_DIMS:
            raise ValueError("apa_sp2: packed keys need rank-4 fp32 K at D64/D128")
        ordinal = _device_ordinal(k.device)
        if ordinal is None:
            raise ValueError("apa_sp2: packed keys need a CUDA tensor")
        B, KV, S, D = k.shape
        capacity = S if capacity is None else capacity
        if type(capacity) is not int or capacity < S or capacity < 1:
            raise ValueError("apa_sp2: capacity must be an int >= visible rows")
        self.batch, self.kv_heads, self.dim = B, KV, D
        self.capacity, self.dtype = capacity, k.dtype
        self.device, self.device_ordinal = str(k.device), ordinal
        self.bulk_bits, self.quantizer = PACKED_BULK_BITS, QUANTIZER
        self.packed_format = PACKED_FORMAT
        self.table_quantizer = PACKED_QUANTIZER
        self.table_registration_sha256 = PACKED_REGISTRATION_SHA256
        self.generation = 0
        self._bound_generation = -1
        self._source = self._source_id = self._source_revision = None
        # The workspace allocates codes/norms, borrows the cached tables and
        # packs rows [0,S) of k; its prepare transients are released on return.
        self._ws = ApaSpq1Workspace(k, capacity, fmt=PACKED_FORMAT)
        self._bind(k)

    # -- buffers -------------------------------------------------------------
    @property
    def codes(self):
        return self._ws.codes

    @property
    def norms(self):
        return self._ws.scales

    @property
    def rotation(self):
        return self._ws.rotation

    @property
    def codebook(self):
        return self._ws.codebook

    @property
    def boundaries(self):
        return self._ws.boundaries

    @property
    def visible_rows(self):
        return self._ws.valid_rows

    def owned_buffers(self):
        """Per-key buffers this object allocated (codes, norms)."""
        return {"codes": self._ws.codes, "norms": self._ws.scales}

    def fixed_tables(self):
        """Shared TurboQuant tables (cached per geometry and device)."""
        return {"rotation": self._ws.rotation, "codebook": self._ws.codebook,
                "boundaries": self._ws.boundaries}

    def provenance(self):
        return dict(quantizer=self.quantizer, packed_format=self.packed_format,
                    bulk_bits=self.bulk_bits, backend=PACKED_BACKEND,
                    table_quantizer=self.table_quantizer,
                    table_registration_sha256=self.table_registration_sha256)

    # -- identity --------------------------------------------------------------
    def _bind(self, k):
        self._source = _source_ref(k)
        self._source_id = id(k)
        self._source_revision = k.revision
        self._bound_generation = self.generation

    def _is_source(self, k):
        if self._source is not None:
            return self._source() is k
        return self._source_id == id(k)

    def invalidate(self):
        """Mark stale after a non-append edit of the keys; build a new object."""
        self.generation += 1

    def append(self, k):
        """Explicit, validated append of rows [visible_rows, k.shape[2]) of k.

        k must be the extended exact K (rows below visible_rows unchanged; the
        caller owns that contract, as for ApaSpq1Workspace). Rebinds the exact-K
        identity to k and advances the generation.
        """
        if self._bound_generation != self.generation:
            raise ValueError("apa_sp2: stale packed keys (append generation); build a new PackedSpKeys")
        if (k.ndim != 4 or k.dtype != self.dtype or _device_ordinal(k.device) != self.device_ordinal or
                (k.shape[0], k.shape[1], k.shape[3]) != (self.batch, self.kv_heads, self.dim)):
            raise ValueError("apa_sp2: packed append geometry/dtype/device mismatch")
        S = k.shape[2]
        if S < self.visible_rows or S > self.capacity:
            raise ValueError("apa_sp2: packed append must extend rows within capacity")
        if S == self.visible_rows:
            if not (self._is_source(k) and k.revision == self._source_revision):
                raise ValueError("apa_sp2: append adds no rows; a different K needs a new PackedSpKeys")
            return self
        self.generation += 1
        self._ws.append(k)
        self._bind(k)
        return self

    def check_source(self, k):
        """R-D: raise ValueError unless k is this object's current exact K."""
        if self._bound_generation != self.generation:
            raise ValueError("apa_sp2: stale packed keys (append generation); build a new PackedSpKeys")
        if not self._is_source(k):
            raise ValueError("apa_sp2: k is not the exact-K source of these packed keys; append explicitly")
        if k.revision != self._source_revision:
            raise ValueError("apa_sp2: exact K was edited in place after packing; build a new PackedSpKeys")
        if (k.ndim != 4 or k.dtype != self.dtype or _device_ordinal(k.device) != self.device_ordinal or
                (k.shape[0], k.shape[1], k.shape[3]) != (self.batch, self.kv_heads, self.dim)):
            raise ValueError("apa_sp2: k geometry/dtype/device does not match the packed keys")
        if k.shape[2] != self.visible_rows:
            raise ValueError("apa_sp2: packed visible rows differ from k.shape[2]; append explicitly")

    def _packed_tuple(self):
        return self._ws.packed

    def _dense_reconstruction(self):
        """Transient dense fp32 [B,KV,S,D] from codes and norms (L3, R-E).

        Mirrors quant._quantize_keys' last step: matmul(centroids, R) * norms.
        The result is returned to the caller and never stored on this object.
        """
        import tensor_cuda as tc
        S, D, KV, B = self.visible_rows, self.dim, self.kv_heads, self.batch
        codes = self._ws.codes.numpy()[:, :, :S, :]
        index = np.empty((B, KV, S, D), np.int64)
        index[..., 0::2] = codes & 15
        index[..., 1::2] = codes >> 4
        centroids = self._ws.codebook.numpy()[index].astype(np.float32)
        norms = self._ws.scales.numpy()[:, :, :S].reshape(B, KV, S, 1)
        cent = tc.tensor(np.ascontiguousarray(centroids), device=self.device)
        norm = tc.tensor(np.ascontiguousarray(norms), device=self.device)
        Rb = self._ws.rotation.float().reshape([1, KV, D, D]).expand([B, KV, D, D])
        return (tc.matmul(cent, Rb) * norm).detach()

    def __repr__(self):
        return (f"PackedSpKeys(B={self.batch}, KV={self.kv_heads}, D={self.dim}, "
                f"visible={self.visible_rows}, capacity={self.capacity}, "
                f"device={self.device}, generation={self.generation})")


@dataclass(frozen=True)
class PackedRegisteredMargins:
    """Frozen packed-route margin table (R-F); not interchangeable with the dense one.

    The loader applies RegisteredMargins.load's checks to this table's own
    registration, quantizer identity and its two entries (4:64, 4:128).
    """
    sha256: str
    entries: object

    @classmethod
    def load(cls, path, *, expected_sha256=PACKED_TABLE_SHA256):
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected_sha256:
            raise ValueError("apa_sp2: packed margin table SHA256 mismatch")
        table = json.loads(data)
        if (table.get("registration_sha256") != PACKED_REGISTRATION_SHA256 or
                table.get("status") != "FROZEN" or
                table.get("quantizer") != PACKED_QUANTIZER or
                table.get("statistic") != PACKED_STATISTIC or
                table.get("measurement_classes") != PACKED_MEASUREMENT_CLASSES):
            raise ValueError("apa_sp2: unregistered or incomplete packed margin table")
        required = {f"{PACKED_BULK_BITS}:{dim}" for dim in PACKED_DIMS}
        if set(table["entries"]) != required:
            raise ValueError("apa_sp2: packed margin table must contain entries 4:64 and 4:128")
        entries = {}
        for key, entry in table["entries"].items():
            value = float(entry["e_q"])
            if (not math.isfinite(value) or value < 0 or entry["count"] <= 0 or
                    not 0 <= entry["mean"] <= entry["max"] <= value or
                    not 0 <= entry["p99_9"] <= entry["max"]):
                raise ValueError("apa_sp2: invalid measured packed margin")
            entries[key] = value
        return cls(digest, MappingProxyType(entries))

    def lookup(self, bulk_bits, dim, dtype, scale):
        if type(bulk_bits) is not int or bulk_bits != PACKED_BULK_BITS or dim not in PACKED_DIMS:
            raise ValueError("apa_sp2: packed table covers 4-bit bulk at D64/D128 only")
        if dtype != "float32" or not math.isfinite(scale) or np.float32(scale) != np.float32(1/math.sqrt(dim)):
            raise ValueError("apa_sp2: packed table covers fp32 at scale=1/sqrt(D) only")
        return self.entries[f"{bulk_bits}:{dim}"]


def packed_attention(q, k, kq, v, scale, epsilon, *, bulk_bits, margins,
                     is_causal=False, sinks=None, diagnostics=False):
    """R-D: the packed route of apa_selective_attention_sp. Every refusal is raised
    before any device work."""
    import importlib
    from .apa_sp2 import RegisteredMargins
    if os.environ.get("TC_APA_SP") != "1":
        raise RuntimeError("apa_sp: experimental entry requires TC_APA_SP=1 (default OFF)")
    if isinstance(margins, RegisteredMargins):
        raise ValueError("apa_sp2: dense margin table with packed keys; load the packed table")
    if not isinstance(margins, PackedRegisteredMargins):
        raise ValueError("apa_sp2: load a registered packed margin table before launching")
    if q.ndim != 4:
        raise ValueError("apa_sp2: query must be rank four")
    if type(bulk_bits) is not int or bulk_bits != kq.bulk_bits:
        raise ValueError("apa_sp2: bulk_bits does not match the packed keys")
    kq.check_source(k)
    for name, tensor in (("q", q), ("v", v)) + ((("sinks", sinks),) if sinks is not None else ()):
        if _device_ordinal(tensor.device) != kq.device_ordinal:
            raise ValueError(f"apa_sp2: {name} is not on the packed keys' device")
    e_q = margins.lookup(bulk_bits, q.shape[-1], q.dtype, scale)
    delta = derive_delta(epsilon, e_q)
    L = q.shape[2]
    if L != 1:
        # R-E, branch decided by L3 (artifacts/apa_p1d/gpu/L3/RESULT.json): the
        # transient dense reconstruction of packed keys is NOT bitwise equal to
        # quantize_sp2_keys(k, 4, storage="dense") (norms and rotation use
        # different fp32 arithmetic), so prefill on packed keys is refused, with
        # no persistent second representation. No prior art known to me.
        _record_route("attention", storage="packed", backend=None, l=L,
                      fallback_reason="refused: prefill (L > 1) on packed keys")
        raise ValueError("apa_sp2: prefill (L > 1) on packed keys is not supported; "
                         "quantize with storage='dense' for prefill")
    spo2 = importlib.import_module(__package__)._spo2
    result = spo2.packed_run(q, k, v, kq._packed_tuple(), float(scale), delta, is_causal,
                             None, diagnostics, fmt=PACKED_FORMAT, carry="partition",
                             gqa_path="r2")
    _record_route("attention", storage="packed", backend=PACKED_BACKEND, l=L,
                  fallback_reason=None, delta=delta, e_q=e_q)
    return result
