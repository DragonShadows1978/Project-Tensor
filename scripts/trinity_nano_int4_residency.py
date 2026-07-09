#!/usr/bin/env python3
"""All-resident Trinity Nano INT4 load receipt."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tensor_cuda"))

import tensor_cuda as tc  # noqa: E402
from core.trinity_nano_tc import TrinityNano_TC  # noqa: E402


def nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/mnt/ForgeRealm/models/trinity-nano")
    parser.add_argument(
        "--output",
        default="artifacts/trinity_nano/residency/int4_all_resident_load.json",
    )
    parser.add_argument("--no-lm-head", action="store_true")
    args = parser.parse_args()

    out = Path(args.output).expanduser().resolve()
    payload = {
        "schema": "trinity_nano_int4_residency_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(Path(args.model_dir).expanduser().resolve()),
        "weight_mode": "int4",
        "load_lm_head": not args.no_lm_head,
        "gpu_before": nvidia_smi(),
        "status": "loading",
    }
    write_json(out, payload)
    started = time.perf_counter()
    with tc.no_grad():
        model, info = TrinityNano_TC.from_pretrained(
            args.model_dir,
            weight_mode="int4",
            load_lm_head=not args.no_lm_head,
            progress=False,
        )
        tc.synchronize()
        payload.update(
            {
                "status": "loaded",
                "wall_seconds": time.perf_counter() - started,
                "gpu_after_load": nvidia_smi(),
                "loader_info": info,
                "layers_loaded": len(model.layers),
            }
        )
        write_json(out, payload)
        del model
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()
        tc.synchronize()
    payload["gpu_after_free"] = nvidia_smi()
    write_json(out, payload)
    print(json.dumps({"status": payload["status"], "artifact": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
