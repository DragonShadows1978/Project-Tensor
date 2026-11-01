"""D2 mechanical transport for inherited CPU controls, never product execution.

Prior art: Meszaros test stubs (2007); GNU timeout(1); Python argparse
allow_abbrev (PSF, 2015). A5 expresses valid scaffolding in the exact D2
grammar while preserving intentional hostile bounds and fixture payloads.
Installed-runner children use a private PATH python3 dispatcher, so the
runner actually receives the registered inner argv. Every transport is
recorded; no alternative bound is concealed in an accepted receipt.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

TOOL = Path(__file__).resolve().parent
REAL = shutil.which("forge-gpu-run")


def executable(body, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("transport_" + hashlib.sha256(body.encode()).hexdigest()[:20])
    if path.exists():
        assert path.read_text() == body
    else:
        path.write_text(body)
        path.chmod(0o755)
    return str(path)


def adapt(args, env):
    args, env = list(args), dict(env)
    indexes = [i for i, v in enumerate(args) if Path(v).name == "run_gpu_line.py"]
    if not indexes:
        return args, env
    i = indexes[0]
    label = args[i + 1]
    rows = json.loads((TOOL / "gpu_lines.json").read_text())
    line = next(r for r in rows if r["label"] == label)
    old = args[i + 4:]
    split = old.index("--")
    runner, inner = old[:split], old[split + 1:]
    options_at = next((j for j, v in enumerate(runner) if v.startswith("--")), len(runner))
    prefix, options = runner[:options_at], runner[options_at:]
    scratch = Path(env["TMPDIR"]) / "transports"
    if "--label" in options:
        options[options.index("--label") + 1] = label
    else:
        options += ["--label", label]
    # Missing options are inherited scaffolding, never hostile substitutions.
    for option, field in (("--max", "maximum"), ("--min", "minimum")):
        if option not in options:
            options[0:0] = [option, str(line[field])]
    if prefix[0] == "forge-gpu-run":
        margin = None
        if "--margin" in options:
            j = options.index("--margin")
            margin = options[j + 1]
            options = options[:j] + options[j + 2:]
        # Real runner gets original test margin, but registered min/max and
        # inner argv. Margin is not a D2 bound and is disclosed here.
        prefix = [REAL] + (["--margin", margin] if margin is not None else [])
        if inner and inner[0] == "timeout":
            j = 1
            while inner[j].startswith("-"):
                j += 1
            duration = inner[j]
            payload = inner[j + 1:-3]
        else:
            duration, payload = f'{line["inner"]}s', inner[:-3]
        if duration == "10s":
            duration = f'{line["inner"]}s'
        env["GN1_CPU_PAYLOAD"] = json.dumps(payload)
        bindir = scratch / "bin"
        bindir.mkdir(parents=True, exist_ok=True)
        dispatcher = bindir / "python3"
        body = (f"#!{sys.executable}\nimport os,sys,json\n"
                "payload=json.loads(os.environ.get('GN1_CPU_PAYLOAD','[]'))\n"
                "argv=payload+sys.argv[1:] if sys.argv[1:2]==['run_gpu_tests.py'] else "
                f"[{sys.executable!r}]+sys.argv[1:]\n"
                "os.execv(argv[0],argv)\n")
        if dispatcher.exists():
            assert dispatcher.read_text() == body
        else:
            dispatcher.write_text(body)
            dispatcher.chmod(0o755)
        env["PATH"] = str(bindir) + os.pathsep + env["PATH"]
        inner = ["timeout", "--kill-after=10s", duration, "python3", "run_gpu_tests.py", line["arm"], line["suite"]]
    else:
        # These fixtures write raw malformed receipts; their old inner argv
        # was unused scaffolding. The payload and mode remain unchanged.
        inner = ["timeout", "--kill-after=10s", f'{line["inner"]}s', "python3", "run_gpu_tests.py", line["arm"], line["suite"]]
    body = (f"#!{sys.executable}\nimport os,sys\nprefix={prefix!r}\n"
            "os.execv(prefix[0],prefix+sys.argv[1:])\n")
    exe = executable(body, scratch)
    # Preserve explicit hostile min/max values, use canonical ordering.
    options = ["--min", options[options.index("--min") + 1], "--max", options[options.index("--max") + 1], "--label", label]
    args[i + 4:] = [exe, *options, "--", *inner]
    with (TOOL / "control_transports.jsonl").open("a") as log:
        log.write(json.dumps(dict(label=label, original=old, exact=args[i+4:], runner_prefix=prefix,
                                  payload=env.get("GN1_CPU_PAYLOAD"), RUNS=env.get("RUNS"))) + "\n")
    return args, env
