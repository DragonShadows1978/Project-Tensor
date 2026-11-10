#!/usr/bin/env python3
"""ENG1-REV2 Q5: spot-check 10 of A1's CALLER_CENSUS.md entries against the live source, plus an independent token
count of '.reshape(' occurrences (Python tokenize: NAME 'reshape' preceded by OP '.' and followed by OP '('), which
skips comments and strings. Prior art: Python tokenize module (PSF). No new algorithm."""
import io, json, re, tokenize
from pathlib import Path
REV2 = Path(__file__).resolve().parent
ROOT = REV2.parents[2]
rows = [l for l in (ROOT / "artifacts/eng1/a1/CALLER_CENSUS.md").read_text().splitlines() if l.startswith("| tensor_cuda/")]
entries = []
for l in rows:
    loc, call, check = [c.strip() for c in l.strip("|").split("|")][:3]
    f, ln, col = loc.rsplit(":", 2)
    entries.append(dict(file=f, line=int(ln), col=int(col), call=call.strip("`"), check=check,
                        numpy=check.startswith("NumPy")))
# Independent token census over every .py under tensor_cuda/tensor_cuda/.
tok = []
for p in sorted((ROOT / "tensor_cuda/tensor_cuda").rglob("*.py")):
    toks = list(tokenize.generate_tokens(io.StringIO(p.read_text()).readline))
    for i in range(1, len(toks) - 1):
        if toks[i].type == tokenize.NAME and toks[i].string == "reshape" and toks[i - 1].string == "." and toks[i + 1].string == "(":
            tok.append((str(p.relative_to(ROOT)), toks[i].start[0]))
census_lines = sorted({(e["file"], e["line"]) for e in entries})
# Map each census entry to the line that holds its 'reshape' token (an expression may start one line earlier).
spot_ids = ["tensor_cuda/tensor_cuda/__init__.py:553", "tensor_cuda/tensor_cuda/functional.py:189",
            "tensor_cuda/tensor_cuda/functional.py:190", "tensor_cuda/tensor_cuda/nn.py:304",
            "tensor_cuda/tensor_cuda/nn.py:498", "tensor_cuda/tensor_cuda/nn.py:576",
            "tensor_cuda/tensor_cuda/quant.py:124", "tensor_cuda/tensor_cuda/quant.py:186",
            "tensor_cuda/tensor_cuda/quantization/affine.py:156"]
spot = []
for e in entries:
    if f"{e['file']}:{e['line']}" in spot_ids:
        src = (ROOT / e["file"]).read_text().splitlines()
        text = src[e["line"] - 1]
        window = " ".join(s.strip() for s in src[e["line"] - 1:e["line"] + 1])
        first = re.sub(r"\s+", " ", e["call"])[:25]
        spot.append(dict(entry=f"{e['file']}:{e['line']}:{e['col']}", call=e["call"], census_says_numpy=e["numpy"],
                         source_line=text.strip(), call_text_found=first in re.sub(r"\s+", " ", window)
                         or first.replace(" ", "") in window.replace(" ", ""),
                         col_points_at=text[e["col"]:e["col"] + 30]))
res = dict(census_rows=len(entries), census_engine=sum(not e["numpy"] for e in entries),
           census_numpy=sum(e["numpy"] for e in entries),
           census_engine_lines=len({(e["file"], e["line"]) for e in entries if not e["numpy"]}),
           token_reshape_calls=len(tok), token_lines=len(set(tok)), spot=spot)
(REV2 / "q5_census_spot.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "spot"}))
for s in spot:
    print(json.dumps(s))
