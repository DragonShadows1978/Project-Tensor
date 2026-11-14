"""Format completed review receipts. Prior art: ordinary Markdown/JSON report
formatting; no new algorithm or optimization. Review methods and their known
prior art are recorded in the report and at their implementation sites.
"""
import hashlib
import json
from pathlib import Path

OUT=Path('artifacts/eng2/rev3')


def read(name):
    return json.loads((OUT/name).read_text())


def main():
    q1=read('Q1_ids.json'); q2=read('Q2_mutants.json'); q3=read('Q3_consumer_final.json')
    q4=read('Q4_seal.json'); l60=read('Q3_L60_parity.json'); spots=read('Q3_spot_checks.json')
    old=read('findings.json')['findings'][0]
    old['line']=556
    old['verbatim']='E2R3-F01 — MINOR — artifacts/eng2/a3/consumer_census.py:556: File selection omits 16 changed-entry call sites, including GRAPA grapa/tripwire.py:655 with unresolved self.dtype(t). The independent census finds 498 sites and 49 unresolved, versus A3’s 482 and 48. Reproduction: run consumer_scan.py and finish_consumer.py; omitted_by_a3 in Q3_consumer_final.json lists all 16. Suggested fix: census every tracked Python file, resolve field/namespace receivers, and update the site lists and counts. No newly refused call site was confirmed.'
    findings=dict(review='ENG2-REV3',status='complete',scope='A3 only; promotion and GPU reruns excluded',
                  verdict='one_minor_census_completeness_finding',findings=[old],
                  questions={'Q1':'checked_sound','Q2':'checked_sound','Q3':'zero_confirmed_newly_refused_but_census_incomplete','Q4':'checked_sound'})
    (OUT/'findings.json').write_text(json.dumps(findings,indent=2,ensure_ascii=False)+'\n')
    findings_sha=hashlib.sha256((OUT/'findings.json').read_bytes()).hexdigest()
    (OUT/'findings.sha256').write_text(findings_sha+'  findings.json\n')
    text=[]
    def add(s=''):
        text.append(s)
    add('# ENG2-REV3 — A3 narrow cross-model review')
    add()
    add('**Completed: one MINOR census-completeness finding; no BLOCKER or MAJOR finding.** The test edit, retained mutant kills, staging seal and recorded dry-run routing check sound. No confirmed newly refused consumer call site was found, but A3’s consumer inventory is incomplete. This is a CPU/static/recorded-receipt review; it does not establish GPU acceptance or authorize promotion.')
    add()
    add('Reviewed before `e2f166b8`, after `10738fe3`; mechanism: `orders/ENG2_A3_ENG1_ROWS_AND_CONSUMERS.md`; review order: `orders/ENG2_REV3_A3_REVIEW.md`. Registration is in `REGISTRATION.md`; work and repair history in `LEDGER.md`. All deliverables are inside `artifacts/eng2/rev3/`; retained scratch is `/tmp/eng2_rev3_20261114/`. No product/consumer edits, GPU use, git writes, subagents or deletes.')
    add()
    add('## Finding (verbatim)')
    add()
    add(old['verbatim'])
    add()
    add('Evidence class: independent static census plus source reasoning. The import/cross-file/bare-`tc` file selection at `consumer_census.py:556–568` excludes files whose only engine access uses fields or namespaces. For the new unresolved site, `EngineDevice.__init__` stores the constructor’s `tc` at `grapa/tripwire.py:630`; `grapa/train.py:758` imports tensor_cuda and `:1195` passes it into `EngineDevice`. `tripwire.py:643` returns `str(t.dtype)`, which supports the expected engine-tensor path but remains unresolved for arbitrary injected objects. The other 15 omitted sites pass supported literals. Both extra Voice files are byte-identical at A3’s recorded HEAD, so these are historical omissions, not checkout drift. Finding details and suggested fix: `findings.json`.')
    add()
    add('All 16 omitted sites (the explicit dtype is shown):')
    add()
    add('| Repository | File:line | Dtype argument | Role |')
    add('|---|---|---|---|')
    for r in sorted(q3['omitted_by_a3'],key=lambda x:(x['repo'],x['file'],x['line'])):
        add(f"| {r['repo']} | `{r['file']}:{r['line']}` | `{r['argument']}` | {r['role']} |")
    add()
    add('## Q1 — the edit: checked sound')
    add()
    add('Ran `git diff e2f166b8 10738fe3 -- tensor_cuda` and `git diff --name-only` independently (`source.diff`, `commands.jsonl`). Only `tensor_cuda/tests/test_eng1.py` changed: lines 180, 186, 187 and 266, removing float64 from E2_DTYPES, its mapping and the bits-test parameter list, and changing the explanatory comment. No other tensor_cuda file changed.')
    add()
    add('`review_checks.py q1` used `git show <commit>:tensor_cuda/tests/test_eng1.py` into isolated scratch and ran four fresh `python3 -m pytest --collect-only -q -p no:cacheprovider --confcutdir=... --rootdir=...` collections: both reviewed commits against both staged package arms, with CUDA hidden and plugin autoload disabled. Exact normalized-ID comparisons exclude logs/timing.')
    add()
    add('| Package arm | Before | After | Removed | Added |')
    add('|---|---:|---:|---:|---:|')
    for arm in ['baseline','candidate']:
        add(f"| {arm} | {q1[arm+'_e2f166b8']['count']} | {q1[arm+'_10738fe3']['count']} | 10 | 0 |")
    add()
    add('The removed sets are identical and equal exactly to the old L38 expected_changes keys:')
    add()
    for id_ in q1['baseline_diff']['removed']:
        add('- `'+id_+'`')
    add()
    add('`supplement_checks.py` also checked the stored staging censuses at both commits: regnorm and regnorm_candidate each 578→568, losing only those ten IDs; regdtype (1779), eng2 (394) and audit (253) byte-identical. All 17 sealed arms carry the after-commit test-file hash, and current L38 expected_changes is empty. Receipts: `Q1_ids.json`, all four `collect_*.txt` and `ids_*.txt`, `Q1_staged_censuses.json`, `Q1_run.txt`, `supplement_run_repaired.txt`. Evidence class: CPU collection and exact file/ID comparison, not test execution.')
    add()
    add('## Q2 — mutation coverage: checked sound')
    add()
    add('Ran independent `review_checks.py q2`, without executing the A3 mutation-coverage script or re-executing any mutant. ENG1 registry has 14 original records plus the two registered amendments in `artifacts/eng1/a1/MUTANTS.json`; ENG2 has 15 records. The matcher treats pytest parameter brackets literally and only `*` as a wildcard. Positive controls against actual retired IDs match 5, 1 and 10 IDs. Every pattern has zero removed-ID matches; every mutant retains registered killers that failed on its own admitted G5 line. Child call outcomes and JUnit agree by test ID; parent actual_killers are among the retained failing registered tests.')
    add()
    add('Before parsing each G5 input, verified its SHA-256 against that run’s canonical manifest. The manifests are stored under the lead receipt directories, not inside the run roots:')
    add()
    for side,run in q2['runs'].items():
        add(f"- {side}: `{run['manifest']}`; SHA-256 `{run['manifest_sha256']}`; {run['manifest_entries']} manifest entries; {len(run['verified'])} verified inputs. Run root: `{run['root']}`.")
    add()
    add('93 files verified in total: parent_result.json, child_result.json and junit.xml for each of 31 mutants. Full per-file expected/actual hashes and the complete registered patterns, matches and retained failed IDs are in `Q2_mutants.json`.')
    add()
    add('| Side | Mutant | Patterns | Removed matches | Retained failed killers | Example retained failed killer |')
    add('|---|---|---:|---:|---:|---|')
    for r in q2['rows']:
        add(f"| {r['side']} | {r['mutant']} | {len(r['patterns'])} | 0 | {len(r['remaining_failed_killers'])} | `{r['remaining_failed_killers'][0]}` |")
    add()
    add('Result: 31/31 retain a recorded G5 kill. Receipts: `Q2_mutants.json`, `Q2_run_repaired.txt`; initial schema-slip transcript is retained in `Q2_run.txt`. Evidence class: manifest-bound recorded G5 results plus independent set matching. Fresh GPU kills under the new seal remain the lead’s task.')
    add()
    add('## Q3 — consumer census: zero confirmed refusals; inventory correction needed')
    add()
    add('Ran own `consumer_scan.py` over every tracked Python file listed by `git ls-files`, loading each pinned HEAD file through `git show`, parsing with Python ast, resolving imports/re-exports/assignment aliases, and conservatively tracing explicit dtype values in lexical scopes. Retained all unresolved calls and unbound receivers. `finish_consumer.py` then resolved 55 field/namespace/runtime-dictionary receiver candidates by source reading and compared the results to A3. It also used `git grep -n -I -e ^ -- *.py` to confirm all current tracked Python text matches the pinned HEAD sources (no source differences in any repository). No consumer module was imported or executed. Parse failures: zero. All seven entry types were considered, including those with zero calls.')
    add()
    add('| Repository | Pinned HEAD | Tracked Python | Explicit dtype sites | Supported | Unresolved | Confirmed newly refused |')
    add('|---|---|---:|---:|---:|---:|---:|')
    for repo in q3['summary']:
        add(f"| {repo['repo']} | `{repo['head']}` | {repo['tracked_python']} | {repo['dtype_sites']} | {repo['counts'].get('supported',0)} | {repo['counts'].get('unresolved',0)} | 0 |")
    add()
    add('Thus “0 newly refused call sites” is supported as **zero statically confirmed bad string/non-string dtype arguments**, with 49 unresolved explicit-dtype sites retained; it is not proof that arbitrary runtime values or external callers cannot be refused. A3 reports 482 explicit-dtype changed-entry sites and 48 unresolved; the corrected inventory is 498 and 49. The Voice HEAD advanced from A3’s 17846484 to caeb64d3, with no tracked-Python working-copy differences; the omitted Voice operands were separately checked at the historical revision. GRAPA and GRM HEADs match A3. GRM remains on the A3-reviewed lc1-wip snapshot; no claim about a different main snapshot is made.')
    add()
    add('Role counts (all roles receive the same classification; paths under p1t_tests are counted as tests here):')
    add()
    add('| Repository | Role | Supported | Unresolved | Sites |')
    add('|---|---|---:|---:|---:|')
    for repo in q3['summary']:
        for role,counts in repo['roles'].items():
            add(f"| {repo['repo']} | {role} | {counts.get('supported',0)} | {counts.get('unresolved',0)} | {sum(counts.values())} |")
    add()
    add('Entry counts:')
    add()
    add('| Repository | tensor | zeros | ones | randn | rand | nn.parameter | functional._causal_mask |')
    add('|---|---:|---:|---:|---:|---:|---:|---:|')
    for repo in q3['summary']:
        e=repo['entries']
        add(f"| {repo['repo']} | {e['tensor']} | {e['zeros']} | {e['ones']} | {e['randn']} | {e['rand']} | {e['parameter']} | {e['_causal_mask']} |")
    add()
    add('**Original unresolved-site spot checks: 11 checked, across every repository, including live and receipt code.** Each supports the seat’s source reading with the qualifications below; none demonstrates a newly refused call. `Q3_spot_checks.json` contains exact source excerpts, file hashes and source-chain references for each. `Q3_engine_dtype_source.json` binds the Tensor.dtype reading to reviewed-commit C++ source: the property returns dtype_name, whose six named cases are supported strings.')
    add()
    add('| Repository / file:line | Independent source reading |')
    add('|---|---|')
    for s in spots:
        add(f"| {s['repo']} `{s['file']}:{s['line']}` | {s['seat_reading']} |")
    add()
    add('**Unchanged-entry L60 parity: checked sound within the recorded probes.** `Q3_L60_parity.json` compares all recorded result/exception fields, excluding source-site metadata and normalizing only process-specific Tensor object addresses in pybind TypeError messages. The order’s top-level L60 files contain 54 unchanged-entry rows: tc.empty, Tensor.astype and three dequant out_dtype entries. They match behaviorally; the two raw differences are object addresses. Those top-level files do not contain the affine helper probes. The expanded A2 L60 files cited by A3 contain those helpers and RegisteredMargins.lookup: all 162 unchanged-entry rows match behaviorally; six raw differences are object addresses. This verifies the stated finite census, not all possible arguments.')
    add()
    add('All 49 unresolved explicit-dtype changed-entry sites (including the newly found tripwire site):')
    add()
    add('| Repository | File:line | Dtype expression |')
    add('|---|---|---|')
    for r in sorted(q3['unresolved'],key=lambda x:(x['repo'],x['file'],x['line'])):
        add(f"| {r['repo']} | `{r['file']}:{r['line']}` | `{r['argument']}` |")
    add()
    add('Receipts: `Q3_consumer_final.json` (all 498 sites, full unresolved and omission lists), `Q3_consumer_scan.json` (initial inventory, default-only calls, all other candidate receivers and per-file hashes), `Q3_git_commands.json`, `Q3_scan_run.txt`, `Q3_finish_run.txt`, `Q3_spot_checks.json`, `Q3_extra_sources.txt`, `Q3_spot_sources.txt`, `Q3_engine_dtype_source.json`, and `Q3_L60_parity.json`. Source copies remain in scratch. Limits: untracked files and WorkTree paths are excluded; arbitrary reflection/eval, dynamic import strings, external calls and mutable-global values cannot be exhaustively proven by this static pass.')
    add()
    add('## Q4 — seal and staging: checked sound')
    add()
    add('Ran `review_checks.py q4`, importing the copied tool’s original `evidence.verify_seal()` and calling its original `verify_arm(arm, seal)` on every arm. This recomputed order/tool/registration digests, exact arm file maps, symlink absence and engine pins. All 17 arms passed, arms_pending is empty. Seal SHA-256: `'+q4['seal_sha256']+'`. The copied evidence module has no --version implementation: asking --version exits zero with no output; its source SHA-256 is recorded as its identifier in `Q4_seal.json`.')
    add()
    add('Independently compared `a3/G4_DRY_dry_4.json` with the sealed gpu_lines.json: exactly 21 unique registered labels, matching arm, suite, inner timeout, governor label, min/max and margin. `supplement_checks.py` corroborated original dry_4 root parent_result.json and stdout.log: 4 roots reached dry governor exit 78 without starting a child; 17 dependents had their registered dependency refusals and no attempt. Each of all 21 S1 argv routes returned 78. This is recorded dry-run routing evidence, not GPU acceptance. No staged line was executed by this review. Receipts: `Q4_seal.json`, `Q4_run.txt`, `Q4_dry_corroborated.json`, `supplement_run_repaired.txt`.')
    add()
    add('## Commands and execution controls')
    add()
    add('Every independent check used this foreground prefix (timeouts 60/120 seconds except the 540-second full consumer pass):')
    add()
    add('```sh\nTMPDIR=/tmp/eng2_rev3_20261114 CUDA_VISIBLE_DEVICES= \\\nOMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\\nPYTHONDONTWRITEBYTECODE=1 timeout <seconds>s nice -n 10 python3 <script> [argument]\n```')
    add()
    add('Independent invocations: `review_checks.py q1`, `review_checks.py q2` (one repaired schema lookup), `consumer_scan.py`, `finish_consumer.py`, `review_checks.py l60` (label correction), `review_checks.py q4`, and `supplement_checks.py` (one repaired source needle). Instrument receipts: `preflight_and_final_checks.txt` (Python 3.12.3, pytest 9.0.2, NumPy 2.4.2, git 2.43.0). Q1 exact subprocess commands and their full outputs are in `commands.jsonl`; the full all-file git command list and output digests are in `Q3_git_commands.json`; source outputs are retained in scratch and source hashes/call excerpts in the census. Resource-heavy execution was not needed; the consumer scan streams one AST at a time and no job was designed to exceed 1GB. At most two CPU-busy processes; no background command launched. When the exec tool yielded a running command, waited again with yield_time_ms=300000 without intervening status messages.')
    add()
    add('## Deviations and repairs')
    add()
    add('- D01: initial house-rule/memory reads used TMPDIR=/tmp and omitted nice/thread controls, before any check. Repeated the house-rule read under required controls (`house_rules_read.txt`); all independent checks used the required environment. Memory search had no relevant hit and supplied no review facts.')
    add('- D02: exploratory reads assumed MANIFEST.sha256 lived in each G5 run root; neither path exists. Located the canonical lead manifests and used only those, checking every G5 input before parsing.')
    add('- R01: review_checks.py q2 initially requested report[nodeid]; real receipt schema uses report[id]. One schema repair, then 31/31 passed; first transcript retained.')
    add('- R02: initial L60 label set omitted the exact recorded tc.nn.parameter and CUDA tensor labels, causing changed rows to appear in the unchanged comparison. Corrected the labels, preserved Q3_L60.initial.json, then retained raw object-address differences and separately checked behavioral parity. No product or historical receipt was edited.')
    add('- D03: spot-source inline script requested nonexistent grapa/engine_native.py. Removed that operand from the review script and reran successfully; no consumer content executed.')
    add('- R03: supplement_checks.py searched for std::string dtype_name; the declaration is const char*. Corrected that source needle once and reran successfully. Both transcripts retained. No script exception required more than one repair; review_checks.py also had the separately recorded L60 classification correction.')
    add('- Instrument note: NumPy version was read from numpy.__version__; the copied evidence module accepts --version as an ignored argument and emits no version. The exact tool source digest is recorded. No gate was run.')
    add()
    add('## Prior art')
    add()
    add('- Pytest collection: pytest project (Krekel and contributors, 2004 onward), unverified — lead to check “pytest history collection”. Taken: fresh test-ID collection. Review-specific: exact before/after/arm comparison and L38 binding.')
    add('- Mutation kill-set review: DeMillo, Lipton and Sayward (1978), unverified — lead to check “mutation testing 1978”; Python standard-library shell-pattern semantics. Taken: a registered failing test witnesses a mutant kill, with literal pytest brackets and star wildcards. Review-specific: retired-ID exclusion and independent manifest binding for all 31 mutants. No new mutation algorithm.')
    add('- Static census: Python project ast module; Kildall (1973) data-flow tradition, unverified — lead to check “Kildall 1973 data flow analysis”. Taken: syntax traversal, conservative constant/value and alias propagation. Review-specific: all-file seven-entry boundary census, receiver follow-up, role counts and omission comparison. Proposed fix uses the same standard AST inventory method; no novelty claimed.')
    add('- Integrity: SHA-256, NIST FIPS 180-4 (2015), unverified — lead to check “NIST FIPS 180-4 SHA-256”. Taken: content digests and manifest matching. Review-specific: binding each consumed G5 receipt and staged arm to its registered bytes.')
    add('- Seal verification: existing ENG2 evidence.py reused unchanged; no new technique. Source spot checks and address-normalized behavioral comparison are ordinary review methods; no specific additional prior art known to me.')
    add()
    add('## Deliverable integrity')
    add()
    add('`findings.json` SHA-256: `'+findings_sha+'`. Final status: complete, one open MINOR census finding. No NIT raised; no additional review round requested. No files moved to DELETE_ME; nothing deleted. Retained scratch: `/tmp/eng2_rev3_20261114/`. Promotion and a fresh GPU G5 run under the new seal are outside this review.')
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
    with (OUT/'LEDGER.md').open('a') as f:
        f.write('\n- Completed 11 source spot checks across all three repositories; supported the seat readings with mutable/global/external-caller qualifications. Expanded L60 parity: top-level 54 unchanged rows and A2 162 unchanged rows agree after only pybind object-address normalization.\n- Staged censuses, current empty pins and after-commit test hash across all 17 arms verified. Dry_4 corroborated: 4 governor-78 roots, 17 dependency refusals, all 21 S1 argv/bounds match registration.\n- Prior art in final report: pytest (2004+), DeMillo/Lipton/Sayward (1978), Python ast/Kildall (1973), SHA-256/FIPS 180-4 (2015), all unverified literature leads; standard methods reused and review-specific bindings only.\n- Final REPORT.md and findings.json written, one MINOR E2R3-F01. findings.json sha256 '+findings_sha+'. No product changes or deletes.\n')
    print('findings.json sha256',findings_sha)
    print('REPORT.md bytes',(OUT/'REPORT.md').stat().st_size)
    print('Findings:',old['verbatim'])


if __name__=='__main__':
    main()
