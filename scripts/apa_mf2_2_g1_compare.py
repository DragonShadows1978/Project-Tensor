"""APA-MF2.2 G1: FAILED/ERROR id set vs the fd32b6d baseline (52 ids). Mine."""
import re, sys
base_log, now_log = sys.argv[1], sys.argv[2]
def ids(p):
    out = set()
    for line in open(p, errors="replace"):
        m = re.match(r"^(FAILED|ERROR) (\S+)", line)
        if m:
            out.add(m.group(2))
    return out
b, n = ids(base_log), ids(now_log)
summary = [l.strip() for l in open(now_log, errors="replace") if re.search(r"\d+ (passed|failed)", l)][-1]
print(f"G1 compare (APA-MF2.2): baseline = {base_log}; now = {now_log}")
print("summary:", summary)
print("baseline ids", len(b), "now ids", len(n))
print("baseline ids that no longer fail:", len(b - n), sorted(b - n))
print("new failing ids:", len(n - b), sorted(n - b))
touched = [i for i in n if "test_apa_mf2_2_" in i]
print("new MF2.2 test files with any FAILED/ERROR:", len(touched))
print("G1", "PASS" if n == b and not touched else "FAIL")
