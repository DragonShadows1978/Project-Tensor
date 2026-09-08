#!/usr/bin/env python3
"""APA-SP5 PROTOCOL-O token stream. Prior art: SP3 PROTOCOL-2 / SP4G
PROTOCOL-G (this repo, 2026) -- same construction, this model's own
harmony/o200k tokenizer, NO chat template. wikitext-2-raw-v1 test split
from the offline HF cache (Merity et al., 2016). Nothing new here.
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path('/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5')
A = ROOT / 'artifacts/apa_sp5'
SNAP = ('/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/'
        'snapshots/6cee5e81ee83917806bbde320786a8fb61efebee')


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(ds['text'])
    tok = AutoTokenizer.from_pretrained(SNAP)
    ids = np.asarray(tok(text, add_special_tokens=False).input_ids, dtype=np.int64)
    A.mkdir(parents=True, exist_ok=True)
    p = A / 'tokens.npy'
    with p.open('xb') as f:
        np.save(f, ids, allow_pickle=False)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    meta = dict(
        dataset='wikitext/wikitext-2-raw-v1', split='test',
        joiner='\\n\\n', tokenizer=SNAP, chat_template=False,
        add_special_tokens=False,
        token_count=int(ids.size), dtype=str(ids.dtype),
        vocab_size=int(tok.vocab_size),
        token_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        tokens_file_sha256=h.hexdigest(),
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        text_chars=len(text),
        min_id=int(ids.min()), max_id=int(ids.max()))
    print(json.dumps(meta, indent=2))
    (A / 'tokens_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
