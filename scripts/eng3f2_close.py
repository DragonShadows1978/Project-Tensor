"""Seal ENG3-F2 evidence and enumerate every retained written file.
Prior art: ENG3-F1 metadata census (2026), NIST SHA-256 (2001),
Python difflib (PSF; year unverified -- lead to check). Taken: scope inventory,
content hashes and unified diff; ours: this report/receipt assembly.
No prior art known to me for the exact report layout.
"""
from pathlib import Path
import hashlib
import json
import os
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/eng3f2"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))

def state():
    result = {}
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if not (Path(base)/d).is_symlink()
                   and str((Path(base)/d).relative_to(ROOT)) != "artifacts/eng3f2"]
        for name in files:
            p = Path(base)/name
            st = p.lstat()
            result[str(p.relative_to(ROOT))] = [st.st_size, st.st_mtime_ns, st.st_mode]
    return result

def main():
    assert sha(OUT/"REGISTRATION.md") == (OUT/"REGISTRATION.sha256").read_text().split()[0]
    before = json.loads((OUT/"SCOPE_BEFORE.json").read_text())
    after = state()
    changed = sorted(k for k in set(before)|set(after) if before.get(k) != after.get(k))
    deleted = sorted(k for k in before if k not in after)
    product = ["tensor_cuda/src/kernels.cu", "tensor_cuda/src/bindings.cpp",
               "tensor_cuda/tests/test_eng3_host.py"]
    def allowed(k):
        return k in product or k.startswith(("tensor_cuda/build-eng3f2/", "logs/eng3f2_")) or (
            k.startswith("scripts/eng3f2_") and k.endswith(".py") and k not in before)
    unexpected = [k for k in changed if not allowed(k)]
    assert not unexpected and not deleted, (unexpected, deleted)
    assert (ROOT/product[2]).read_text().startswith((OUT/"before/test_eng3_host.py").read_text())
    dump(OUT/"SCOPE_AFTER.json", after)
    dump(OUT/"G4_SCOPE.json", {"pass": True, "changed_outside_artifact_root": changed,
         "unexpected_writes": unexpected, "deleted": deleted,
         "git_pointer_unchanged": before.get(".git") == after.get(".git"),
         "git_commands": 0, "gpu_runs": 0, "subagents": 0,
         "test_changes_append_only": True,
         "source_diff": "artifacts/eng3f2/source.diff",
         "scope_evidence_limit": "Workspace file size/mtime/mode census plus command/source review; no recursive scan of other worktrees or git backing store."})
    g1 = json.loads((OUT/"host/G1/RESULT.json").read_text())
    g3 = json.loads((OUT/"MUTANT_RESULTS.json").read_text())
    g2 = json.loads((OUT/"staged/g3-cpu/COMPARE.json").read_text())
    hc = json.loads((OUT/"G2_HOST_COMPARE.json").read_text())
    assert g1["exit"] == 0 and g1["counts"] == {"passed": 254}
    assert g3["pass"] and g2["pass"] and hc["pass"]
    modules = json.loads((OUT/"MODULES.json").read_text())
    for info in modules.values():
        assert sha(ROOT/info["module"]) == info["sha256"]
    gates = {"G1": "PASS", "G2_CPU": "PASS", "G2_GPU": "STAGED_NOT_RUN",
             "G3": "PASS", "G4": "PASS", "acceptance": "PENDING_LEAD_GPU_LINES"}
    dump(OUT/"GATES.json", gates)
    with (OUT/"LEDGER.md").open("a") as f:
        f.write("- Final scope PASS: only authorized writes, no missing files, .git pointer unchanged; no git commands, no subagents, no GPU execution. Test edits strictly append-only; source.diff reviewed at the two authorized message sites. Registration hash rechecked unchanged. No full acceptance claimed before lead GPU parity. Final evidence enumerated in FILES_WRITTEN.txt and FILES.sha256.\n")
    # External dispatcher log remains live. Hash a captured immutable snapshot,
    # retain the live path in the inventory and identify the observation limit.
    volatile = [k for k in changed if k.startswith("logs/eng3f2_")]
    captures = {}
    for rel in volatile:
        snap = OUT/"runtime_logs"/Path(rel).name
        snap.parent.mkdir(exist_ok=True)
        snap.write_bytes((ROOT/rel).read_bytes())
        captures[rel] = {"snapshot": str(snap.relative_to(ROOT)),
                         "snapshot_sha256": sha(snap),
                         "note": "External dispatcher appends this live session log; digest covers captured snapshot only."}
    dump(OUT/"VOLATILE_FILES.json", captures)
    planned = {"artifacts/eng3f2/REPORT.md", "artifacts/eng3f2/FILES_WRITTEN.txt",
               "artifacts/eng3f2/FILES.sha256", "artifacts/eng3f2/SCRATCH_WRITTEN.txt"}
    files = sorted(set(changed) | {str(p.relative_to(ROOT)) for p in OUT.rglob("*") if p.is_file()} | planned)
    # Include retained scratch files and symlinks; deleted pytest temporaries under
    # /tmp are exempt and no agent delete command was used.
    scratch = sorted(str(p) for p in Path("/tmp/eng3f2_scratch").rglob("*") if p.is_file() or p.is_symlink())
    (OUT/"SCRATCH_WRITTEN.txt").write_text("\n".join(scratch) + "\n")
    (OUT/"FILES_WRITTEN.txt").write_text("\n".join(files) + "\n")
    module_lines = "\n".join(f"- {arm}: `{info['sha256']}` (`{info['module']}`)." for arm,info in modules.items())
    source_diff = (OUT/"source.diff").read_text()
    added = sum(line.startswith("+") and not line.startswith("+++") for line in source_diff.splitlines())
    removed = sum(line.startswith("-") and not line.startswith("---") for line in source_diff.splitlines())
    report = f"""# ENG3-F2 report

G1 PASS. G3 PASS. G4 PASS. G2 CPU PASS.
G2 GPU STAGED, NOT RUN. Registered acceptance remains pending the lead's GPU lines; no further diagnostic review is requested by this order.

## Changes and unit-test evidence

- ENG3R2-F01 fixed at tensor_cuda/src/kernels.cu:551: the existing byte-bound throw now names the entire shape and canonical dtype. The comparison, limit and allocation path are unchanged.
- ENG3R2-F02 fixed in core_device's shared refusal helper at tensor_cuda/src/bindings.cpp:2661: every byte outside printable ASCII 0x20..0x7e becomes lowercase `\\xNN` before exception translation. NUL and following bytes remain visible. The three device-taking hooks use this helper; _core_clone has no string operand.
- ENG3R2-F03 remains closed with no change; no device-index parser or numeric behavior edits.
- Diff: three existing files, {added} inserted lines, {removed} replaced lines; test_eng3_host.py only receives appended tests. See source.diff and before/ copies. All engine source hashes match Build F except the two authorized message files (build/BUILD_G.json versus the immutable F receipt).

G1 unit-test/suite evidence: host/G1_first/RESULT.json has 48 passing overflow cases; host/G1/RESULT.json has 254 passing CPU cases (77 inherited, 48 overflow, 129 device-message cases). Overflow cases cover all six dtypes, both creation hooks, bare cpu/cuda descriptors and nonempty/zero-extent shapes without device allocation. Device cases cover all three string-taking hooks, embedded NUL and suffixes, every ASCII control byte and DEL, UTF-8 high bytes and printable malformed inputs. CUDA_VISIBLE_DEVICES was empty in every CPU run. G1.log, G1_first.log and per-case JSONL preserve the actual messages.

G2 CPU regression evidence: staged/g3-cpu/COMPARE.json compares exactly 3,368 test IDs and the collection-error file sets against Build F, with zero differences and zero test writes. Each inherited suite exits 1: 815 passed, 258 skipped, 1,946 failed, 349 error outcomes. These are identical inherited outcomes, not a clean-engine claim. Collection-error details remain in g3/g3-cpu_F/outcomes.json and g3/g3-cpu_G/outcomes.json; no log/timing equality was used. G2_HOST_COMPARE.json separately records exact identity for all 77 inherited host CPU test IDs. Added F2 tests are validated against the order, not expected to pass unchanged Build F.

G3 mutant evidence: f2m01_message_unchanged was fully built from an isolated copy of G with only the F01 throw text reverted to F1's. The first G1 test executes that throw and exits 1 as expected, with 48 failures. Every captured assertion contains the exact old message, proving mutated-line execution. Example failure, verbatim:

```
E       AssertionError: NDArray: byte size overflows int64 (nonzero extents count next to zero)
E       assert '[2305843009213693952]' in 'NDArray: byte size overflows int64 (nonzero extents count next to zero)'
```

See MUTANT_RESULTS.json, host/G3/RESULT.json and G3.log. G remains unchanged.

## Builds and module sha256s

Both full CMake builds configure and compile successfully, sm_89 matching Build F, disconnected pybind11, at most -j4 under nice -n 19 and /usr/bin/time -v, bounded by timeout. Build receipts contain command arrays and SHA256 maps of every copied engine source plus CMakeLists.txt. The monitor enforces MemAvailable >=12 GiB and disk free >=40 GiB throughout. See build/BUILD_G.json, build/BUILD_f2m01_message_unchanged.json and their configure/build logs (including maximum resident set size). No build/GPU performance claim.

{module_lines}

## Staged GPU lines and expected exits

From `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3`, the lead runs exactly:

```bash
timeout 1850s env FORGE_GPU_DRY_RUN=0 bash artifacts/eng3f2/staged/launch-g3-g1.sh
timeout 1850s env FORGE_GPU_DRY_RUN=0 bash artifacts/eng3f2/staged/launch-g3-g2.sh
```

Each launcher sets its package, thread, temp and bytecode environment explicitly and sets cwd, then checks BOTH module SHA256 pins and executes both packages' real host prefix before requesting the lease. Its GPU worker starts with the same prefix and stops there in prefix mode, immediately before tc.tensor(src, device=cuda). Each runs the unchanged F1 split-file census/options against F and G. The launcher bodies contain exactly these forge forms:

```bash
forge-gpu-run --min 30 --max 1800 --label eng3f2-g3-g1 -- timeout 1750s nice -n 19 python3 -B -P scripts/eng3f2_stage.py run g3-g1 > artifacts/eng3f2/staged/g3-g1/GPU.log 2>&1
forge-gpu-run --min 30 --max 1800 --label eng3f2-g3-g2 -- timeout 1750s nice -n 19 python3 -B -P scripts/eng3f2_stage.py run g3-g2 > artifacts/eng3f2/staged/g3-g2/GPU.log 2>&1
```

Expected for each real launcher: forge exit 0, inner comparison exit 0, launcher exit 0; individual pytest exits/outcomes are retained and compared per ID. The launcher's finish step records forge_exit, inner_exit and GPU.log SHA256 in staged/<line>/RESULT.json. Lead retains those receipts (L45). A module mismatch refuses before the lease. Scope excludes promotion.

This seat ran only `FORGE_GPU_DRY_RUN=1` launchers: both F/G CPU prefixes passed; both forge/launcher exits were 78, inner_exit null, gpu_run false. The dry-run branch is before lock/lease acquisition. See STAGED_LINES.json, staged/<line>/CPU_PREFIX.json and DRY_RESULT.json, dry_<line>.log and the hashed GPU.log dry-run output. GPU parity is pending, not inferred from these CPU prefixes.

## Scope and deviations

G4 PASS (metadata census plus source/command review): only the authorized files/new roots were written; no files deleted; no git commands/writes, no subagents and no GPU runs. Existing test content is byte-for-byte retained at its prefix. Earlier artifacts/builds, other worktrees and production were read-only. G4_SCOPE.json and SCOPE_BEFORE/AFTER.json hold the workspace evidence; source.diff limits product changes to the two message sites. No claims about arbitrary external filesystem activity are inferred from the workspace census.

Deviations/tool slips (no gate or threshold changes):

- Initial read-only inspection commands did not all set TMPDIR/CUDA_VISIBLE_DEVICES; no gates, builds or Python data processing ran in those commands.
- One ls included absent scripts/eng3f1* and tensor_cuda/tests/test_eng3f1* (exit 2); actual helpers are artifacts/eng3f1/tools. A later read-only cat probe found no conftest.py or artifacts/eng3/tools/suite.py (exit 1).
- Oversized inspection output was truncated; relevant files were reread in bounded batches.
- Own-script repair 1/3: pre-execution review restored an omitted out = BUILD/arm assignment while adapting the inherited build helper; no script exception or gate run preceded it. Own-script repair 2/3: final hash verification detected the external dispatcher appending logs/eng3f2_r1.log; the sealing helper now snapshots and labels live logs rather than claiming their hashes are stable. No gate/source/threshold changed.
- Expected mutant exit 1 and dry-run exits 78 are registered results, not deviations.

No deletes or moves to DELETE_ME. Retained scratch: /tmp/eng3f2_scratch, both build trees and all receipts. SCRATCH_WRITTEN.txt enumerates every retained scratch file/symlink; removed internal pytest temporaries under /tmp are exempt from the no-delete rule.

## Prior art

Existing P1T contextual shape/dtype diagnostics (2026); Python repr and C++ hexadecimal escape conventions (ISO C++17, 2017). Taken: operand context and printable byte escaping. Ours: placement at these two shared diagnostic sites and the registered test census. Bibliography unverified -- lead to check: Python repr nonprintable byte escapes, C++17 hexadecimal escape sequences. No new numeric algorithm, selection rule, data structure or optimization.

Harness prior art: inherited ENG3-F1 suite/split/pin/prefix protocol (2026); pytest hooks/parametrization (2004 onward); NIST SHA-256 (2001); DeMillo, Lipton and Sayward mutation testing (1978); CMake out-of-tree builds (Kitware, 2000); /proc/meminfo and time(1); Python difflib (PSF, year unverified -- lead to check). Taken: per-ID phase folding, inherited corpus/options, hashes, controlled mutation, isolated builds and resource rails. Ours: F2 receipt/report assembly; no prior art known to me for the exact report layout. Code-site comments and LEDGER.md carry these attributions.

## Every file written

FILES_WRITTEN.txt is the complete worktree inventory ({len(files)} files, including this report and itself); FILES.sha256 hashes every listed file except the hash manifest itself and the live dispatcher logs identified by VOLATILE_FILES.json. The immutable runtime_logs snapshots are hashed. It covers the three edited files, all scripts/eng3f2_*.py helpers, every generated file in tensor_cuda/build-eng3f2 and every artifacts/eng3f2 receipt. SCRATCH_WRITTEN.txt separately enumerates retained /tmp files. The dispatcher appended logs/eng3f2_r1.log during this session; it is retained in the inventory, with a captured snapshot and its observation limit in VOLATILE_FILES.json. Agent commands created no logs/eng3f2_* file.

The complete worktree file list follows, verbatim:

```text
""" + "\n".join(files) + "\n```\n"
    (OUT/"REPORT.md").write_text(report)
    (OUT/"FILES.sha256").write_text("".join(sha(ROOT/rel) + "  " + rel + "\n"
        for rel in files if rel != "artifacts/eng3f2/FILES.sha256" and rel not in volatile))
    print(json.dumps({"gates": gates, "worktree_files_written": len(files),
                      "scratch_entries": len(scratch), "report": str(OUT/"REPORT.md")}))

if __name__ == "__main__":
    main()
