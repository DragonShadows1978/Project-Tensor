"""APA-SP2 epsilon governance for the opt-in single-pass inference kernels.

The table is a finite calibration receipt, not a bound for arbitrary tensors.
Callers must supply keys from quantize_sp2_keys and the calibrated input regime.
No epsilon default or per-shape fit. No device error-estimation prepass.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np

REGISTRATION_SHA256 = "3b8df8bfdd6e3d351395f96ca654bfe9070a26bfaed390d73cc620546a7ff027"
QUANTIZER = "turboquant_lloyd_max_rotated_fp32"


@dataclass(frozen=True)
class RegisteredMargins:
    """Load once, then reuse the checked, immutable four-entry table."""
    sha256: str
    entries: object

    @classmethod
    def load(cls, path, *, expected_sha256):
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected_sha256:
            raise ValueError("apa_sp2: margin table SHA256 mismatch")
        table = json.loads(data)
        if (table.get("registration_sha256") != REGISTRATION_SHA256 or
                table.get("status") != "FROZEN" or
                table.get("quantizer") != QUANTIZER or
                table.get("statistic") != "maximum_over_all_registered_G2_valid_scores" or
                table.get("measurement_classes") != 64):
            raise ValueError("apa_sp2: unregistered or incomplete margin table")
        required = {f"{bits}:{dim}" for bits in (2, 4) for dim in (64, 128)}
        if set(table["entries"]) != required:
            raise ValueError("apa_sp2: margin table must contain all four entries")
        entries = {}
        for key, entry in table["entries"].items():
            value = float(entry["e_q"])
            if (not math.isfinite(value) or value < 0 or entry["count"] <= 0 or
                    not 0 <= entry["mean"] <= entry["max"] <= value or
                    not 0 <= entry["p99_9"] <= entry["max"]):
                raise ValueError("apa_sp2: invalid measured margin")
            entries[key] = value
        return cls(digest, MappingProxyType(entries))

    def lookup(self, bulk_bits, dim, dtype, scale):
        if type(bulk_bits) is not int or bulk_bits not in (2, 4) or dim not in (64, 128):
            raise ValueError("apa_sp2: registered bulk bits are 2/4 and D is 64/128")
        if dtype != "float32" or not math.isfinite(scale) or np.float32(scale) != np.float32(1/math.sqrt(dim)):
            raise ValueError("apa_sp2: table covers fp32 at scale=1/sqrt(D) only")
        return self.entries[f"{bulk_bits}:{dim}"]


def quantize_sp2_keys(k, bulk_bits):
    """Existing TurboQuant implementation, one rotation per KV head, fp32 only."""
    from . import quant
    if type(bulk_bits) is not int or bulk_bits not in (2, 4):
        raise ValueError("apa_sp2: bulk_bits must be 2 or 4")
    if k.ndim != 4 or k.dtype != "float32" or k.shape[-1] not in (64, 128):
        raise ValueError("apa_sp2: quantizer requires registered fp32 key geometry")
    tables = quant._tables(k.shape[-1], bulk_bits, k.shape[1], True, k.device.split(":")[0])
    return quant._quantize_keys(k, *tables)


def apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *, bulk_bits,
                               margins, is_causal=False, sinks=None, diagnostics=False):
    """Production SP entry: explicit epsilon + registered (bits,D) error lookup.

    Inference only. TC_APA_SP must be exactly '1' (default OFF).
    kq must come from quantize_sp2_keys(k, bulk_bits). Calibration does not certify
    an unseen query/key distribution. The raw _C direct-delta API reproduces SP1.
    """
    from . import _C
    if os.environ.get("TC_APA_SP") != "1":
        raise RuntimeError("apa_sp: experimental entry requires TC_APA_SP=1 (default OFF)")
    if not isinstance(margins, RegisteredMargins):
        raise ValueError("apa_sp2: load a registered margin table before launching")
    if q.ndim != 4:
        raise ValueError("apa_sp2: query must be rank four")
    e_q = margins.lookup(bulk_bits, q.shape[-1], q.dtype, scale)
    return _C.apa_selective_attention_sp_epsilon(q, k, kq, v, float(scale),
        float(epsilon), e_q, is_causal, sinks, diagnostics)
