"""ENG1-REV2 Q2 probe (run only under q2_source_readers.py's write fence): why the source-reading tests fail on the
candidate worktree, read-only. Prints the refusal each helper raises; asserts nothing about the product."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))


def test_probe_apa_sp3_verify_sources(capsys):
    import apa_sp3_common as common
    try:
        common.verify_sources()
        msg = "verify_sources: no refusal"
    except Exception as e:  # report, do not judge
        msg = "verify_sources refused: %s: %s" % (type(e).__name__, str(e)[:300])
    try:
        common.load_runtime()
        msg2 = "load_runtime: no refusal"
    except Exception as e:
        msg2 = "load_runtime refused: %s: %s" % (type(e).__name__, str(e)[:300])
    with capsys.disabled():
        print("\nPROBE " + msg + "\nPROBE " + msg2)
