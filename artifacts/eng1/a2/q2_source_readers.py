#!/usr/bin/env python3
"""ENG1-REV2 Q2: my own check of the 16 excluded source-path readers, by dynamic attribution on CPU.

Runs the given regression test files with CUDA hidden and the candidate arm preloaded and bound exactly as the staged
child does, under a Python audit hook (PEP 578) that (1) records every file READ inside this worktree, attributed to
the running test ID and the opening code's file, and (2) REFUSES every write/rename/remove/mkdir outside
/tmp/eng1_rev2_work and every subprocess launch, so the run cannot touch the worktree (L46). Tests that need the GPU
fail at their first device call; reads after that point are not observed (named residual).
Usage: python3 -B q2_source_readers.py TAG FILE...   (paths relative to the worktree root)
Prior art: PEP 578 runtime audit hooks (Dower, 2019); pytest runtest hooks. Taken: standard instruments; ours: the
read attribution and the write fence.
"""
import json, os, sys, threading
from pathlib import Path

REV2 = Path(__file__).resolve().parent
ROOT = REV2.parents[2]
TOOL = ROOT / "artifacts/eng1/tool"
SCRATCH = "/tmp/eng1_rev2_work"
tag, files = sys.argv[1], sys.argv[2:]
package = TOOL / "arms/candidate/tensor_cuda"
os.environ["PYTHONPATH"] = str(package.resolve())
os.environ["ENG1_CENSUS_DIR"] = SCRATCH + "/census_" + tag
Path(os.environ["ENG1_CENSUS_DIR"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(package))
sys.path.insert(1, str(ROOT))
import tensor_cuda as tc  # preload the candidate arm (process-local), as run_gpu_tests.py does
assert Path(tc.__file__).resolve() == (package / "tensor_cuda/__init__.py").resolve()

state = dict(test="<collection>", fence=True)
reads, blocked = [], []
ROOTS = str(ROOT) + os.sep
ALLOWED_WRITE = (SCRATCH + os.sep, "/dev/")
WRITE_EVENTS = {"os.rename", "os.replace", "os.remove", "os.rmdir", "os.mkdir", "shutil.rmtree", "os.truncate",
                "os.chmod", "os.utime", "os.symlink", "os.link", "shutil.copyfile", "shutil.move", "os.chown"}


def kind_of_read():
    """import (an importlib frame is on the stack), traceback (tokenize/linecache render a failure), tree (shutil
    walks a directory by fd-relative names), else data (a by-path read by test or helper code)."""
    f = sys._getframe(2)
    while f is not None:
        n = f.f_code.co_filename
        if "importlib" in n or n.startswith("<frozen importlib"):
            return "import"
        if n.endswith(("tokenize.py", "linecache.py", "traceback.py", "inspect.py")):
            return "traceback"
        if n.endswith("shutil.py"):
            return "tree"
        f = f.f_back
    return "data"


def caller():
    f = sys._getframe(2)
    while f is not None:
        name = f.f_code.co_filename
        if "importlib" not in name and "<frozen" not in name and not name.endswith(("pathlib.py", "pathlib/_local.py",
                                                                                    "pathlib/_abc.py", "io.py")):
            return name, f.f_lineno
        f = f.f_back
    return None, None


def hook(event, args):
    if not state["fence"]:
        return
    if event == "open":
        path, mode = args[0], args[1]
        if isinstance(path, int) or path is None:
            return
        p = os.path.abspath(os.fsdecode(path))
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            mode is None and isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing:
            if not p.startswith(ALLOWED_WRITE):
                blocked.append(dict(test=state["test"], event=event, path=p))
                raise PermissionError("ENG1-REV2 write fence: " + p)
            return
        if p.startswith(ROOTS):
            fn, ln = caller()
            reads.append(dict(test=state["test"], path=os.path.relpath(p, ROOT), opener=fn and os.path.relpath(fn, ROOT)
                              if fn and fn.startswith(ROOTS) else fn, line=ln, kind=kind_of_read()))
    elif event in WRITE_EVENTS:
        target = args[0] if args else None
        if isinstance(target, (str, bytes, os.PathLike)):
            target = os.fsdecode(target)
            # Tree-removal helpers pass names relative to a directory fd: resolve them against that fd, not the cwd.
            dir_fd = {"os.remove": 1, "os.rmdir": 1, "os.mkdir": 2, "os.rename": 2, "os.replace": 2, "os.chmod": 2,
                      "os.utime": 3, "os.chown": 3}.get(event)
            fd = args[dir_fd] if dir_fd is not None and len(args) > dir_fd else None
            if isinstance(fd, int) and fd >= 0 and not os.path.isabs(target):
                target = os.path.join(os.readlink("/proc/self/fd/%d" % fd), target)
            p = os.path.abspath(target)
            if not p.startswith(ALLOWED_WRITE):
                blocked.append(dict(test=state["test"], event=event, path=p))
                raise PermissionError("ENG1-REV2 write fence: " + event + " " + p)
    elif event == "ctypes.dlopen":
        name = args[0] if args else None
        if isinstance(name, (str, bytes)) and os.path.abspath(os.fsdecode(name)).startswith(ROOTS):
            reads.append(dict(test=state["test"], path=os.path.relpath(os.path.abspath(os.fsdecode(name)), ROOT),
                              opener=None, line=None, kind="dlopen"))
    elif event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork"):
        blocked.append(dict(test=state["test"], event=event, path=str(args[:2])[:300]))
        raise PermissionError("ENG1-REV2 subprocess fence: " + event)


class Attribution:
    def pytest_runtest_setup(self, item):
        state["test"] = item.nodeid

    def pytest_runtest_teardown(self, item, nextitem):
        state["test"] = item.nodeid

    def pytest_runtest_logfinish(self, nodeid, location):
        state["test"] = "<between>"

    outcomes = {}

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.outcome != "passed":
            self.outcomes[report.nodeid + "|" + report.when] = report.outcome


import pytest  # noqa: E402

plugin = Attribution()
sys.addaudithook(hook)
args = ["-q", "-p", "no:cacheprovider", "--import-mode=prepend", "--rootdir=" + str(ROOT),
        "--basetemp=" + SCRATCH + "/bt_" + tag, "-o", "faulthandler_timeout=120"] + [str(ROOT / f) for f in files]
rc = int(pytest.main(args, plugins=[plugin]))
state["fence"] = False  # pytest finished: only this script's own rev2 receipt is written below
by_test = {}
for r in reads:
    by_test.setdefault(r["test"], set()).add((r["path"], r["opener"], r["kind"]))
out = dict(tag=tag, pytest_exit=rc, files=files, reads_by_test={k: sorted(map(list, v)) for k, v in by_test.items()},
           blocked=blocked, outcomes=plugin.outcomes)
(REV2 / f"{tag}.json").write_text(json.dumps(out, indent=1) + "\n")
print("pytest exit", rc, "tests with worktree reads:", len(by_test), "blocked:", len(blocked))
