"""PF2-F4 worktree scope and bounded source diff receipts."""
import argparse
import ast
import difflib
from hashlib import sha256
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/apa_pf2f4"
OLD = "test_prefill_on_packed_keys_is_refused_naming_dense"
NEW = "test_prefill_on_packed_keys_routes_to_packed_prefill"
RETAINED = "test_r1_pinned_route_refusals_still_precede_any_device_call"
P1D = Path("tensor_cuda/tests/test_apa_p1d_policy.py")
P1F2 = Path("tensor_cuda/tests/test_apa_p1f2_cpu.py")


def snapshot():
    result = {}
    # Prior art: SHA-256 scope manifests from MAINCHK3 (Project-Tensor, 2026).
    # PF2-F4 excludes its authorized evidence paths from this comparison.
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        path = Path(base)
        dirs[:] = sorted(
            d for d in dirs
            if d not in {".git", ".agents", ".aws", ".codex", "__pycache__", ".pytest_cache"}
            and path / d != ART
        )
        links = [d for d in dirs if (path / d).is_symlink()]
        for name in sorted([*files, *links]):
            full = path / name
            relative = full.relative_to(ROOT).as_posix()
            if relative.startswith(("scripts/apa_pf2f4_", "logs/apa_pf2f4_")) or name.endswith(".pyc"):
                continue
            result[relative] = (
                {"link": os.readlink(full)} if full.is_symlink()
                else {"sha256": sha256(full.read_bytes()).hexdigest(), "mode": full.stat().st_mode & 0o777}
            )
    return result


def save(name, data):
    with (ART / name).open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def source_without(source, names):
    # AST source spans define the edit boundary. Compare the remaining bytes.
    lines = source.splitlines(keepends=True)
    selected = [node for node in ast.walk(ast.parse(source))
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in selected} == names and len(selected) == len(names)
    for node in sorted(selected, key=lambda node: node.lineno, reverse=True):
        lines[node.lineno - 1:node.end_lineno] = [f"# {node.name}\n"]
    return "".join(lines)


def check():
    before = json.loads((ART / "SCOPE_BEFORE.json").read_text())
    after = snapshot()
    save("SCOPE_AFTER.json", after)
    changes = [p for p in sorted(before.keys() | after.keys()) if before.get(p) != after.get(p)]
    assert changes == [str(P1D), str(P1F2)], changes
    archive = ROOT.parent / "tensor_fork_apa_wt_mainchk3/artifacts/mainchk3/trees/PF2"
    chunks = []
    for path, old_names, new_names in [
        (P1D, {OLD}, {NEW}),
        (P1F2, {RETAINED, "FakeSpo2"}, {RETAINED, "FakeSpo2"}),
    ]:
        original = (archive / path).read_text()
        current = (ROOT / path).read_text()
        assert sha256(original.encode()).hexdigest() == before[str(path)]["sha256"], path
        assert source_without(original, old_names).replace(f"# {OLD}\n", f"# {NEW}\n") == source_without(
            current, new_names
        ), path
        if path == P1F2:
            # The only change permitted within the existing fake is the added method.
            assert source_without(current, {"packed_prefill"}).replace(
                '\n\n# packed_prefill\n', '\n'
            ).split("    class FakeSpo2:", 1)[1].split("    class FakeC:", 1)[0] == (
                original.split("    class FakeSpo2:", 1)[1].split("    class FakeC:", 1)[0]
            )
        chunks.extend(difflib.unified_diff(
            original.splitlines(keepends=True), current.splitlines(keepends=True),
            f"a/{path}", f"b/{path}",
        ))
    with (ART / "TESTS.diff").open("x") as stream:
        stream.writelines(chunks)
    print(json.dumps({"worktree_paths": len(before), "changed_paths": changes}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("before", "check"))
    if parser.parse_args().action == "before":
        save("SCOPE_BEFORE.json", snapshot())
    else:
        check()
