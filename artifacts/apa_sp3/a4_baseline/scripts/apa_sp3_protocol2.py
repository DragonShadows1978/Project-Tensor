#!/usr/bin/env python3
"""Create-only PROTOCOL-2 stream and amendment; no model/CUDA execution.

Prior art: GraftRepository/tests/minicpm3_bulkbits_floor.py (2026), get_text
and window_nll: reuse newline join and next-token NLL/window layout. The lead
amendment mandates 512 targets (reference loop has 511), fp64, full prefills.
Content pinning and offline Arrow reads are standard reproducibility methods.
SP3 adds the immutable manifest; no new tokenizer, corpus or scoring algorithm.
"""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import numpy as np
from apa_sp3_common import (ART, FEEDING, PROTOCOL2_ORDER, PROTOCOL2_ORDER_SHA,
                           PROTOCOL2_STATUS, REG_SHA, Red, publish, registration, sha)

CORPUS = Path('/home/vader/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3')
FLOOR = Path('/mnt/ForgeRealm/GraftRepository/tests/minicpm3_bulkbits_floor.py')


def main():
    if (ART/'protocol_amendment.json').exists() or (ART/'protocol2_tokens.npy').exists():
        raise Red('create-only protocol registration; existing artifacts must not be overwritten')
    r = registration()
    if sha(PROTOCOL2_ORDER) != PROTOCOL2_ORDER_SHA:
        raise Red('amendment order changed before registration')
    os.environ.update(HF_DATASETS_OFFLINE='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    from datasets import Dataset
    from transformers import AutoTokenizer
    # Directly open the specified read-only cached test Arrow: same rows as
    # load_dataset's offline snapshot; no cache lock/write or fallback corpus.
    ds = Dataset.from_file(str(CORPUS/'wikitext-test.arrow'))
    text = '\n'.join(row['text'] for row in ds)
    snapshot = Path(r['model']['snapshot'])
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    ids = np.asarray(tok(text, return_tensors='np').input_ids[0], dtype='<i8')
    plain = np.asarray(tok(text, add_special_tokens=False, return_tensors='np').input_ids[0], dtype='<i8')
    if not (len(ids) == len(plain)+1 and ids[0] == tok.bos_token_id == 1 and np.array_equal(ids[1:],plain)):
        raise Red('unexpected default tokenizer additions; register finding before any gate')
    if ids.ndim != 1 or len(ids) <= 32800 or np.any(ids<0) or np.any(ids>=73448):
        raise Red('invalid canonical stream')
    tokens = ART/'protocol2_tokens.npy'
    with tokens.open('xb') as f:
        np.save(f, ids, allow_pickle=False)
    inputs = [CORPUS/'wikitext-test.arrow', CORPUS/'dataset_info.json']
    inputs += [p for p in snapshot.iterdir() if p.name in
               ('tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','config.json','added_tokens.json')]
    j = dict(schema='apa_sp3_protocol_amendment_v2',status=PROTOCOL2_STATUS,immutable=True,
             registration_sha256=REG_SHA,amendment_order_path=str(PROTOCOL2_ORDER),
             amendment_order_sha256=PROTOCOL2_ORDER_SHA,
             source_path=str(PROTOCOL2_ORDER),source_sha256=sha(PROTOCOL2_ORDER),
             scoring_source_path=str(FLOOR),scoring_source_sha256=sha(FLOOR),
             corpus=dict(name='wikitext',config='wikitext-2-raw-v1',split='test',snapshot=str(CORPUS),
                         loader='datasets.Dataset.from_file of cached test Arrow; read-only, no fallback',
                         join='\n',rows=len(ds),characters=len(text),text_utf8_sha256=hashlib.sha256(text.encode()).hexdigest()),
             tokenizer=dict(snapshot=str(snapshot),class_name=type(tok).__name__,
                            call='AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False); tok(text, return_tensors="np")',
                            special_tokens='one BOS <s> id 1 at start of entire stream; no EOS, no chat template, no per-window BOS',
                            default_additions_verified_against_no_specials=True,bos_token_id=tok.bos_token_id,eos_token_id=tok.eos_token_id),
             input_sha256={str(p):sha(p) for p in sorted(inputs)},
             packages={p:importlib.metadata.version(p) for p in ('datasets','transformers','tokenizers','numpy','pyarrow')},
             tokens_path=str(tokens),tokens_sha256=sha(tokens),token_count=len(ids),
             token_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),token_dtype='<i8',
             scoring='last_512_targets_within_input',window=1024,scored=512,n_windows=6,
             window_starts=list(range(0,6144,1024)),nll_dtype='float64',feeding=FEEDING,
             target_indices='within each window: logits[S-513:S-1] predict ids[S-512:S]; all 512',
             aggregation='exp(sum(window total_nll) / 3072); never average PPL',
             long_rows='prefix at offset 0, full prefill; last 512 within input; decode uses same stream',
             diagnostics='C calibration, B/C/E G2 and fraction pins retain single prefix starting at 0, S=1024/8192; separate from six-window scored PPL',
             reference_difference='floor window_nll scores 511 at float32 in cached 64-token steps; lead amendment explicitly requires 512 at fp64 and permits full prefill',
             model=r['model'],adapter_compute_dtype='default bfloat16 from MiniCPM3_TC.from_pretrained; matches registration',
             g0=dict(kind='fresh_process_determinism',arms=['A','B'],repeats=2,absolute_tolerance=.001,
                     order=['g0_A_1','g0_B_1','g0_A_2','g0_B_2','g0'],baseline='repeat 1 after both repeats pass',
                     per_model_process_guard='A/B six-window scores vs fresh repeat-1 baselines at <=0.001 before SP; preserve r1 in-process safeguard'),
             added_prediction=dict(name='B_minus_A_bulk4',interval=[-.3,.3],is_gate=False),
             historical_context=dict(A=20.065,B=19.817,device='RTX 3070 8 GB',targets=False,recoverable=False),
             predictions='lead P1-P5 and seat S1-S5 retained verbatim in registration; P2 uses fresh B; S1 historical-protocol premise retired, no post-result replacement',
             prior_art='GraftRepository minicpm3_bulkbits_floor.py (2026) join/layout/scorer reference; standard NLL, immutable hashes, repeated-process comparisons. No new algorithm.')
    publish(ART/'protocol_amendment.json', j)
    print(__import__('json').dumps(dict(amendment_sha256=sha(ART/'protocol_amendment.json'),
          token_count=len(ids),tokens_file_sha256=sha(tokens),token_sha256=j['token_sha256'],special_tokens=j['tokenizer']['special_tokens']),indent=2))


if __name__ == '__main__':
    main()
