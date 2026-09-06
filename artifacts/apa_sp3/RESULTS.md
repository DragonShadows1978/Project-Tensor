# APA-SP3 — model comparison measured; remaining RED/blocked rows listed

Evidence class: **model perplexity** for scored rows; **kernel sweep** for activation margins and decode timing.
**G2/G3 rows establish nothing about model quality by themselves.**

Registration SHA256: `d9b6511702a894f72174795141c72b2097b1bbe6110e810a1d4d8bfc3cd3498c` (immutable).
Model: openbmb/MiniCPM3-4B; INT4 affine groups of 128; BF16 compute; 62 layers, 40 heads, composite D=96 and zero-padded V=96.
Execution seat: gpt-6-astra, reasoning effort xhigh, confirmed by logs/apa_sp3_r2.log. No GPU model execution by this seat.

## G0 PROTOCOL-2 determinism

Historical context only: A=20.065 / B=19.817 came from an unrecoverable single guide window on an 8 GB RTX 3070. They are not targets. No further recovery attempted.
Authorized amendment: `orders/APA_SP3_AMENDMENT_2.md`; immutable implementation JSON `artifacts/apa_sp3/protocol_amendment.json`, SHA256 `db95e3ecb958ef6c31a105ee77be19c455ae8b210ed8a8082194389294843224`. Base registration is unchanged.
Offline wikitext-2-raw-v1 test: 4358 rows joined by newline, 1289979 characters, 333337 tokens. Stream file SHA256 `5684e72cbab28236391ebcac9113aeca5b318ce6eeed69c410f1094950269a28`; canonical little-endian int64 bytes SHA256 `d3c64882a9cbd4264a21f60ac239680d3a7d0631ab06d0d8a5d4fac8286d268a`.
Tokenizer: snapshot AutoTokenizer defaults; one BOS <s> id 1 at the beginning of the entire stream; no EOS, chat template or per-window BOS. Direct read-only cached test Arrow; no fallback corpus or cache writes.
Reference: `/mnt/ForgeRealm/GraftRepository/tests/minicpm3_bulkbits_floor.py::get_text` and `window_nll`. Reuse newline join and six consecutive disjoint windows. The reference loop scores 511 targets; amendment 2 explicitly requires all 512, which this scorer implements.
Feeding: one full prefill per window, fresh KV cache each time, six windows of 1024 at offsets 0,1024,2048,3072,4096,5120. Logits 511:1023 predict tokens 512:1024, fp64 log-softmax; PPL = exp(total NLL / 3072). Same feeding for every arm. Long rows use the prefix at token 0 and the last 512 in-input targets.
G0: four fresh processes A1,B1,A2,B2; repeats of each arm must agree within 0.001 PPL. Process identities and target hashes are checked. Repeat 1 supplies each baseline after both repeats pass. B-A within +/-0.3 PPL is a prediction, never a stop gate. The preserved in-process A/B safeguard compares six-window PPL to these fresh baselines at 0.001 before later model arms.
G0 miss stops model arms; D must still refine all eligible keys and satisfy |D-A|<=0.005. C calibration and G2 retain the registered single-prefix scope; their diagnostic fractions are labeled separately from six-window PPL.
G0 status: PASS; repeat differences: {"A": 0.0, "B": 0.0}; B-A: 0.12297422620243914; prediction met: True.

## Perplexity and prefill table — model perplexity

| Bits | S | Arm | Status | PPL last-512 | ms wall | Peak resident MiB* | Refined fraction | Layer min..max | δ |
|---|---:|---|---|---:|---:|---:|---:|---|---:|
| 4 | 1024 | A standard | PASS | 8.65583 | 8723.41 | 3185.45 | — | —..— | — |
| 4 | 1024 | B apa_two_pass | PASS | 8.7788 | 11966.5 | 3300.45 | 0.103298 | 0.0869153..0.114088 | — |
| 4 | 1024 | C apa_sp_matched | PASS | 8.66112 | 10821.7 | 3102.26 | 0.104146 | 0.016127..0.59053 | 4.28125 |
| 4 | 1024 | D apa_sp_refine_all | PASS | 8.65535 | 12019.3 | 3114.01 | 1 | 1..1 | 3.40282e+38 |
| 4 | 1024 | E apa_sp_provable | BLOCKED / unrun | — | — | — | — | —..— | — |
| 4 | 8192 | A standard | STALE | — | — | — | — | —..— | — |
| 4 | 8192 | B apa_two_pass | PASS | 10.4036 | 46184.2 | 4446.09 | — | —..— | — |
| 4 | 8192 | C apa_sp_matched | PASS | 10.0424 | 29275.9 | 4435.78 | 0.0406538 | 0.00208322..0.387479 | 4.28125 |
| 4 | 8192 | D apa_sp_refine_all | BLOCKED / unrun | — | — | — | — | —..— | — |
| 4 | 8192 | E apa_sp_provable | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 1024 | A standard | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 1024 | B apa_two_pass | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 1024 | C apa_sp_matched | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 1024 | D apa_sp_refine_all | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 8192 | A standard | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 8192 | B apa_two_pass | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 8192 | C apa_sp_matched | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 8192 | D apa_sp_refine_all | BLOCKED / unrun | — | — | — | — | —..— | — |

*Peak is explicitly an estimate: exact intercepted cudaMalloc allocation high-water plus the pre-call device/context offset. Internal driver transient allocations can be missed. Raw pooling is OFF. No background sampling thread. Diagnostic/capture timings never fill this table.
8192 rows request full prefill plus last-512 logits in one forward. A deadline or OOM is RED, including if logits do not fit. No chunking or precision fallback. Bulk8 is optional secondary; no E8 is registered.

## Empirical margin table — kernel sweep on model activations

| Bits | Arm | S | Layer | Error mean | p99 | p99.9 | max | Unrefined mass mean | p99 | max | Max skipped w/w* | Fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | B | 1024 | 0 | 2.02558 | 13.9285 | 26.9108 | 105.659 | 0.011841 | 0.999773 | 1 | 1 | 0.0869153 |
| 4 | B | 1024 | 1 | 0.347742 | 1.49699 | 2.54029 | 7.25032 | 0.62946 | 1 | 1 | 1 | 0.114088 |
| 4 | B | 1024 | 2 | 0.547352 | 2.08564 | 3.07899 | 7.22334 | 0.664892 | 1 | 1 | 1 | 0.113577 |
| 4 | B | 1024 | 3 | 0.303575 | 1.11815 | 1.61649 | 3.99111 | 0.489227 | 1 | 1 | 1 | 0.112606 |
| 4 | B | 1024 | 4 | 0.3849 | 1.44325 | 2.18344 | 5.47669 | 0.815962 | 1 | 1 | 1 | 0.108097 |
| 4 | B | 1024 | 5 | 0.492026 | 1.72182 | 2.35851 | 4.92093 | 0.646704 | 1 | 1 | 1 | 0.111775 |
| 4 | B | 1024 | 6 | 0.22073 | 0.786902 | 1.09143 | 2.53487 | 0.323743 | 0.999668 | 1 | 1 | 0.101704 |
| 4 | B | 1024 | 7 | 0.381548 | 1.3414 | 1.86934 | 3.96169 | 0.507788 | 0.999995 | 1 | 1 | 0.105397 |
| 4 | B | 1024 | 8 | 0.228296 | 0.785705 | 1.05971 | 2.4931 | 0.437299 | 0.999881 | 1 | 1 | 0.104435 |
| 4 | B | 1024 | 9 | 0.20077 | 0.694365 | 0.949089 | 2.28879 | 0.294419 | 0.999866 | 1 | 1 | 0.107787 |
| 4 | B | 1024 | 10 | 0.360747 | 1.2274 | 1.65926 | 3.3458 | 0.39811 | 0.999991 | 1 | 1 | 0.102311 |
| 4 | B | 1024 | 11 | 0.401656 | 1.37001 | 1.84531 | 4.1336 | 0.816795 | 1 | 1 | 1 | 0.104305 |
| 4 | B | 1024 | 12 | 0.425292 | 1.55281 | 2.20312 | 5.20272 | 0.941667 | 1 | 1 | 1 | 0.106172 |
| 4 | B | 1024 | 13 | 0.213009 | 0.763429 | 1.08128 | 2.77007 | 0.735253 | 1 | 1 | 1 | 0.10705 |
| 4 | B | 1024 | 14 | 0.253824 | 0.926494 | 1.29121 | 3.09585 | 0.284059 | 1 | 1 | 1 | 0.109375 |
| 4 | B | 1024 | 15 | 0.238404 | 0.822439 | 1.12361 | 2.56833 | 0.970656 | 0.999998 | 1 | 1 | 0.0979169 |
| 4 | B | 1024 | 16 | 0.247798 | 0.926775 | 1.38728 | 4.07266 | 0.936307 | 1 | 1 | 1 | 0.100737 |
| 4 | B | 1024 | 17 | 0.214084 | 0.759193 | 1.06034 | 2.41904 | 0.869479 | 0.999999 | 1 | 1 | 0.103042 |
| 4 | B | 1024 | 18 | 0.209932 | 0.73989 | 1.04327 | 2.90517 | 0.92668 | 0.999996 | 1 | 1 | 0.102189 |
| 4 | B | 1024 | 19 | 0.222928 | 0.795657 | 1.13347 | 2.37238 | 0.876329 | 1 | 1 | 1 | 0.10651 |
| 4 | B | 1024 | 20 | 0.194143 | 0.701374 | 0.98592 | 2.60638 | 0.793317 | 0.999996 | 1 | 1 | 0.104684 |
| 4 | B | 1024 | 21 | 0.191281 | 0.681806 | 0.980914 | 2.38162 | 0.939271 | 0.999984 | 1 | 1 | 0.100128 |
| 4 | B | 1024 | 22 | 0.304186 | 1.08444 | 1.53292 | 3.25045 | 0.990423 | 1 | 1 | 1 | 0.100415 |
| 4 | B | 1024 | 23 | 0.283029 | 0.983741 | 1.36438 | 4.28315 | 0.990986 | 1 | 1 | 1 | 0.0994992 |
| 4 | B | 1024 | 24 | 0.255701 | 0.913006 | 1.2773 | 2.43719 | 0.968098 | 1 | 1 | 1 | 0.102245 |
| 4 | B | 1024 | 25 | 0.225098 | 0.776891 | 1.06079 | 2.33508 | 0.977346 | 0.999999 | 1 | 1 | 0.1005 |
| 4 | B | 1024 | 26 | 0.269259 | 0.927197 | 1.26924 | 2.34911 | 0.986927 | 1 | 1 | 1 | 0.10348 |
| 4 | B | 1024 | 27 | 0.243306 | 0.876798 | 1.26173 | 3.19654 | 0.976985 | 0.999999 | 1 | 1 | 0.102672 |
| 4 | B | 1024 | 28 | 0.215024 | 0.741496 | 1.00864 | 2.26298 | 0.940317 | 0.999994 | 1 | 1 | 0.105234 |
| 4 | B | 1024 | 29 | 0.290332 | 1.0637 | 1.59061 | 3.54386 | 0.988945 | 1 | 1 | 1 | 0.0985408 |
| 4 | B | 1024 | 30 | 0.326384 | 1.2245 | 1.81708 | 5.42582 | 0.953922 | 1 | 1 | 1 | 0.100435 |
| 4 | B | 1024 | 31 | 0.236303 | 0.863492 | 1.21151 | 2.62914 | 0.755448 | 1 | 1 | 1 | 0.109728 |
| 4 | B | 1024 | 32 | 0.225935 | 0.795784 | 1.10793 | 2.57405 | 0.92289 | 1 | 1 | 1 | 0.10574 |
| 4 | B | 1024 | 33 | 0.230959 | 0.822116 | 1.1571 | 2.85581 | 0.649724 | 0.999998 | 1 | 1 | 0.113216 |
| 4 | B | 1024 | 34 | 0.222147 | 0.760071 | 1.02536 | 2.50383 | 0.926102 | 0.999995 | 1 | 1 | 0.105769 |
| 4 | B | 1024 | 35 | 0.272943 | 0.980098 | 1.38404 | 2.71695 | 0.978917 | 1 | 1 | 1 | 0.100801 |
| 4 | B | 1024 | 36 | 0.195057 | 0.683396 | 0.945231 | 2.0205 | 0.633696 | 0.999993 | 1 | 1 | 0.112473 |
| 4 | B | 1024 | 37 | 0.24456 | 0.83091 | 1.10988 | 2.07155 | 0.986982 | 0.999999 | 1 | 1 | 0.0975576 |
| 4 | B | 1024 | 38 | 0.290067 | 1.01061 | 1.41533 | 3.29681 | 0.993057 | 1 | 1 | 1 | 0.0949914 |
| 4 | B | 1024 | 39 | 0.258642 | 0.923015 | 1.28324 | 2.85751 | 0.83972 | 1 | 1 | 1 | 0.107129 |
| 4 | B | 1024 | 40 | 0.238465 | 0.848332 | 1.22342 | 3.20443 | 0.908491 | 0.999998 | 1 | 1 | 0.103689 |
| 4 | B | 1024 | 41 | 0.241669 | 0.902022 | 1.4486 | 2.85586 | 0.93976 | 0.999999 | 1 | 1 | 0.104346 |
| 4 | B | 1024 | 42 | 0.319438 | 1.1504 | 1.69521 | 4.13484 | 0.989846 | 1 | 1 | 1 | 0.096859 |
| 4 | B | 1024 | 43 | 0.212011 | 0.761959 | 1.12282 | 2.86257 | 0.366834 | 0.999988 | 1 | 1 | 0.11221 |
| 4 | B | 1024 | 44 | 0.268515 | 0.932514 | 1.27815 | 3.36455 | 0.954428 | 0.999999 | 1 | 1 | 0.101779 |
| 4 | B | 1024 | 45 | 0.264056 | 0.97283 | 1.4263 | 3.19745 | 0.989879 | 1 | 1 | 1 | 0.0971435 |
| 4 | B | 1024 | 46 | 0.205994 | 0.723526 | 0.996171 | 2.14522 | 0.411047 | 0.999991 | 1 | 1 | 0.108544 |
| 4 | B | 1024 | 47 | 0.252097 | 0.868171 | 1.18545 | 3.02301 | 0.959186 | 0.999996 | 1 | 1 | 0.0960775 |
| 4 | B | 1024 | 48 | 0.310284 | 1.07303 | 1.48341 | 2.89385 | 0.99762 | 1 | 1 | 1 | 0.0970381 |
| 4 | B | 1024 | 49 | 0.209167 | 0.728317 | 1.00763 | 2.3138 | 0.844932 | 0.999984 | 1 | 1 | 0.104608 |
| 4 | B | 1024 | 50 | 0.253514 | 0.902229 | 1.26351 | 2.57224 | 0.956782 | 1 | 1 | 1 | 0.101326 |
| 4 | B | 1024 | 51 | 0.274844 | 0.934474 | 1.26576 | 2.65846 | 0.996453 | 0.999998 | 1 | 1 | 0.0954603 |
| 4 | B | 1024 | 52 | 0.222232 | 0.787297 | 1.11441 | 2.41416 | 0.875159 | 0.999992 | 1 | 1 | 0.0998191 |
| 4 | B | 1024 | 53 | 0.358685 | 1.32003 | 2.03093 | 4.68268 | 0.972185 | 1 | 1 | 1 | 0.0982883 |
| 4 | B | 1024 | 54 | 0.373579 | 1.33758 | 1.90529 | 4.11667 | 0.981971 | 1 | 1 | 1 | 0.0998886 |
| 4 | B | 1024 | 55 | 0.238614 | 0.808435 | 1.08845 | 2.20153 | 0.919267 | 0.999992 | 1 | 1 | 0.0977832 |
| 4 | B | 1024 | 56 | 0.241344 | 0.847778 | 1.16425 | 2.85629 | 0.930499 | 0.999994 | 1 | 1 | 0.102504 |
| 4 | B | 1024 | 57 | 0.33029 | 1.14107 | 1.56095 | 3.54357 | 0.993576 | 1 | 1 | 1 | 0.0969934 |
| 4 | B | 1024 | 58 | 0.200682 | 0.695338 | 0.94862 | 2.45786 | 0.42402 | 0.999991 | 1 | 1 | 0.109903 |
| 4 | B | 1024 | 59 | 0.191739 | 0.671209 | 0.94401 | 2.25218 | 0.625991 | 0.999988 | 1 | 1 | 0.104425 |
| 4 | B | 1024 | 60 | 0.326302 | 1.20494 | 1.84381 | 4.34037 | 0.810977 | 1 | 1 | 1 | 0.103483 |
| 4 | B | 1024 | 61 | 0.266937 | 0.949006 | 1.33883 | 3.18249 | 0.91456 | 1 | 1 | 1 | 0.101079 |
| 4 | C | 1024 | 0 | 2.01424 | 13.8122 | 26.6696 | 101.973 | 0.223297 | 0.999572 | 1 | 1 | 0.0169695 |
| 4 | C | 1024 | 1 | 0.347692 | 1.49682 | 2.5445 | 7.23057 | 0.0363616 | 0.44193 | 0.785215 | 0.236979 | 0.514349 |
| 4 | C | 1024 | 2 | 0.546458 | 2.08159 | 3.06833 | 6.99337 | 0.0277764 | 0.140094 | 0.317202 | 0.679556 | 0.222378 |
| 4 | C | 1024 | 3 | 0.302566 | 1.1178 | 1.61885 | 3.87255 | 0.057526 | 0.42266 | 0.702639 | 0.363062 | 0.483682 |
| 4 | C | 1024 | 4 | 0.384653 | 1.44189 | 2.17879 | 5.32724 | 0.0197809 | 0.213074 | 0.620081 | 0.250069 | 0.59053 |
| 4 | C | 1024 | 5 | 0.490336 | 1.7181 | 2.35796 | 4.80159 | 0.0487912 | 0.198585 | 0.405486 | 0.432105 | 0.25205 |
| 4 | C | 1024 | 6 | 0.221011 | 0.78622 | 1.08969 | 2.45201 | 0.144011 | 0.546126 | 0.697512 | 0.106488 | 0.205521 |
| 4 | C | 1024 | 7 | 0.382184 | 1.34814 | 1.88564 | 4.12139 | 0.14897 | 0.508689 | 0.683696 | 0.317754 | 0.151151 |
| 4 | C | 1024 | 8 | 0.228569 | 0.783683 | 1.05443 | 2.12206 | 0.17627 | 0.498981 | 0.655038 | 0.093873 | 0.129715 |
| 4 | C | 1024 | 9 | 0.201039 | 0.697605 | 0.949164 | 2.13994 | 0.160451 | 0.486797 | 0.652058 | 0.0786948 | 0.0974016 |
| 4 | C | 1024 | 10 | 0.359874 | 1.22472 | 1.65428 | 3.29613 | 0.118152 | 0.445849 | 0.673006 | 0.127704 | 0.0490302 |
| 4 | C | 1024 | 11 | 0.399589 | 1.36061 | 1.82838 | 3.98922 | 0.0826538 | 0.324914 | 0.557925 | 0.139576 | 0.0464734 |
| 4 | C | 1024 | 12 | 0.423092 | 1.54154 | 2.17953 | 4.74974 | 0.0627854 | 0.322261 | 0.575892 | 0.246469 | 0.0639781 |
| 4 | C | 1024 | 13 | 0.21179 | 0.761707 | 1.0897 | 2.81311 | 0.15841 | 0.609039 | 0.743122 | 0.0863228 | 0.108368 |
| 4 | C | 1024 | 14 | 0.252658 | 0.921393 | 1.28744 | 2.66411 | 0.116037 | 0.44111 | 0.627278 | 0.111567 | 0.0731924 |
| 4 | C | 1024 | 15 | 0.238151 | 0.821707 | 1.12539 | 2.5118 | 0.168901 | 0.503057 | 0.675902 | 0.069586 | 0.169266 |
| 4 | C | 1024 | 16 | 0.24682 | 0.923434 | 1.37889 | 3.29003 | 0.12885 | 0.531911 | 0.693561 | 0.0633428 | 0.101432 |
| 4 | C | 1024 | 17 | 0.21363 | 0.756584 | 1.05899 | 2.43044 | 0.142531 | 0.478275 | 0.673904 | 0.109204 | 0.0744391 |
| 4 | C | 1024 | 18 | 0.210622 | 0.741427 | 1.04413 | 2.66128 | 0.181374 | 0.506714 | 0.645298 | 0.0833201 | 0.0816184 |
| 4 | C | 1024 | 19 | 0.222044 | 0.793721 | 1.12677 | 2.6737 | 0.109005 | 0.48231 | 0.663097 | 0.0657164 | 0.0499319 |
| 4 | C | 1024 | 20 | 0.193327 | 0.698332 | 0.979233 | 2.92054 | 0.170983 | 0.528782 | 0.693899 | 0.0565603 | 0.104154 |
| 4 | C | 1024 | 21 | 0.19134 | 0.676837 | 0.967596 | 2.1648 | 0.208926 | 0.543375 | 0.690769 | 0.0452448 | 0.127471 |
| 4 | C | 1024 | 22 | 0.302571 | 1.07598 | 1.51999 | 3.34985 | 0.0874925 | 0.537992 | 0.766659 | 0.104126 | 0.0603904 |
| 4 | C | 1024 | 23 | 0.282488 | 0.982599 | 1.36605 | 3.75875 | 0.126587 | 0.421204 | 0.615867 | 0.0863231 | 0.0493818 |
| 4 | C | 1024 | 24 | 0.254293 | 0.906136 | 1.26666 | 2.46171 | 0.126032 | 0.426337 | 0.611017 | 0.0893283 | 0.0944836 |
| 4 | C | 1024 | 25 | 0.223364 | 0.773432 | 1.05539 | 2.56822 | 0.131024 | 0.409308 | 0.66616 | 0.0686659 | 0.0931926 |
| 4 | C | 1024 | 26 | 0.266294 | 0.916227 | 1.25007 | 2.51327 | 0.0796555 | 0.290005 | 0.514205 | 0.0744798 | 0.0375643 |
| 4 | C | 1024 | 27 | 0.240711 | 0.868984 | 1.24675 | 2.67214 | 0.142659 | 0.427394 | 0.623323 | 0.0817476 | 0.0828404 |
| 4 | C | 1024 | 28 | 0.214727 | 0.738598 | 1.00127 | 2.17597 | 0.157616 | 0.415615 | 0.563907 | 0.0711299 | 0.0884694 |
| 4 | C | 1024 | 29 | 0.288326 | 1.05001 | 1.57479 | 3.67974 | 0.0774449 | 0.328176 | 0.576119 | 0.0770431 | 0.0674039 |
| 4 | C | 1024 | 30 | 0.32361 | 1.20974 | 1.79779 | 5.09747 | 0.0863942 | 0.468194 | 0.720573 | 0.12033 | 0.0781088 |
| 4 | C | 1024 | 31 | 0.235278 | 0.85768 | 1.20482 | 2.72039 | 0.110143 | 0.39298 | 0.627633 | 0.104402 | 0.101878 |
| 4 | C | 1024 | 32 | 0.224554 | 0.792797 | 1.11101 | 2.5585 | 0.115274 | 0.45532 | 0.661546 | 0.0698093 | 0.169434 |
| 4 | C | 1024 | 33 | 0.229368 | 0.819324 | 1.16242 | 2.79176 | 0.12985 | 0.384971 | 0.538164 | 0.0700786 | 0.100854 |
| 4 | C | 1024 | 34 | 0.219965 | 0.7544 | 1.02035 | 2.59545 | 0.151799 | 0.44854 | 0.656906 | 0.115362 | 0.107661 |
| 4 | C | 1024 | 35 | 0.270711 | 0.970043 | 1.35786 | 2.62119 | 0.101291 | 0.35951 | 0.572901 | 0.150892 | 0.103712 |
| 4 | C | 1024 | 36 | 0.193561 | 0.677228 | 0.931953 | 2.11224 | 0.130077 | 0.387616 | 0.545627 | 0.0961978 | 0.132635 |
| 4 | C | 1024 | 37 | 0.243251 | 0.827321 | 1.10739 | 2.42735 | 0.151699 | 0.428181 | 0.597981 | 0.145715 | 0.0794147 |
| 4 | C | 1024 | 38 | 0.288772 | 1.0057 | 1.41047 | 3.5456 | 0.100345 | 0.336841 | 0.523285 | 0.0808964 | 0.045005 |
| 4 | C | 1024 | 39 | 0.255934 | 0.910705 | 1.26837 | 2.77895 | 0.116596 | 0.540474 | 0.752861 | 0.134044 | 0.104548 |
| 4 | C | 1024 | 40 | 0.237622 | 0.843186 | 1.21525 | 3.07989 | 0.144548 | 0.481452 | 0.692183 | 0.0785374 | 0.0853361 |
| 4 | C | 1024 | 41 | 0.24106 | 0.902628 | 1.44532 | 2.861 | 0.143465 | 0.414274 | 0.667781 | 0.0648543 | 0.0605252 |
| 4 | C | 1024 | 42 | 0.318035 | 1.14366 | 1.68109 | 3.59958 | 0.0893209 | 0.3812 | 0.681485 | 0.0736658 | 0.0436244 |
| 4 | C | 1024 | 43 | 0.211474 | 0.760441 | 1.11954 | 3.09659 | 0.135454 | 0.428154 | 0.68428 | 0.0981801 | 0.0684772 |
| 4 | C | 1024 | 44 | 0.267163 | 0.927864 | 1.27869 | 3.04159 | 0.121203 | 0.40574 | 0.600795 | 0.0706207 | 0.049379 |
| 4 | C | 1024 | 45 | 0.26245 | 0.96458 | 1.41363 | 3.70525 | 0.16575 | 0.552769 | 0.692505 | 0.0928745 | 0.0463425 |
| 4 | C | 1024 | 46 | 0.205197 | 0.718257 | 0.985659 | 2.35297 | 0.140596 | 0.420716 | 0.664362 | 0.048198 | 0.0379015 |
| 4 | C | 1024 | 47 | 0.251391 | 0.866238 | 1.17925 | 3.03285 | 0.14057 | 0.434452 | 0.644984 | 0.0686684 | 0.0267495 |
| 4 | C | 1024 | 48 | 0.309066 | 1.06482 | 1.47415 | 3.13904 | 0.087913 | 0.318973 | 0.588622 | 0.0570552 | 0.0194647 |
| 4 | C | 1024 | 49 | 0.208105 | 0.725072 | 0.999359 | 2.09034 | 0.186228 | 0.489542 | 0.694087 | 0.0635748 | 0.0405316 |
| 4 | C | 1024 | 50 | 0.252511 | 0.898253 | 1.25024 | 2.64286 | 0.119861 | 0.511907 | 0.692545 | 0.0738177 | 0.0264988 |
| 4 | C | 1024 | 51 | 0.272856 | 0.928101 | 1.26366 | 2.74549 | 0.111441 | 0.383925 | 0.600808 | 0.0964478 | 0.0162123 |
| 4 | C | 1024 | 52 | 0.222168 | 0.788927 | 1.12604 | 2.64515 | 0.164337 | 0.453514 | 0.652081 | 0.0693455 | 0.0262677 |
| 4 | C | 1024 | 53 | 0.354926 | 1.30424 | 2.01562 | 4.5101 | 0.071256 | 0.351645 | 0.608744 | 0.151847 | 0.0275241 |
| 4 | C | 1024 | 54 | 0.371431 | 1.32711 | 1.89982 | 4.24587 | 0.0753798 | 0.369555 | 0.601061 | 0.144329 | 0.0382993 |
| 4 | C | 1024 | 55 | 0.238291 | 0.810539 | 1.08998 | 2.14667 | 0.137668 | 0.411427 | 0.649139 | 0.0462245 | 0.016127 |
| 4 | C | 1024 | 56 | 0.24121 | 0.850787 | 1.16905 | 2.87713 | 0.173934 | 0.515618 | 0.703844 | 0.0662545 | 0.0840591 |
| 4 | C | 1024 | 57 | 0.329198 | 1.13725 | 1.55829 | 3.26029 | 0.0729486 | 0.414026 | 0.673346 | 0.121397 | 0.027222 |
| 4 | C | 1024 | 58 | 0.200749 | 0.698729 | 0.953497 | 2.39953 | 0.158245 | 0.419321 | 0.672617 | 0.0477095 | 0.0308789 |
| 4 | C | 1024 | 59 | 0.19065 | 0.670448 | 0.948712 | 2.44851 | 0.175614 | 0.489022 | 0.716265 | 0.0512151 | 0.0900103 |
| 4 | C | 1024 | 60 | 0.324973 | 1.19986 | 1.82067 | 4.28272 | 0.097612 | 0.460344 | 0.676912 | 0.181247 | 0.228925 |
| 4 | C | 1024 | 61 | 0.266708 | 0.946812 | 1.3399 | 3.16612 | 0.136922 | 0.468589 | 0.630722 | 0.0877432 | 0.0566162 |

124/248 primary B/C layer receipts available. Every row requires all eligible heads, queries and keys; future masked keys are excluded. Bulk scores come from a native SP-order FP32 CUDA score probe; exact scores are float64 Q.K dots at the actual float32 scale (amendment_002_native_bulk.json). B blend error replays its exact native bulk chunks and requires the captured byte SHA; B/C also expose sp_error in JSON for the E margin (amendment_003_B_native_scores.json). Error percentiles are exact nearest ranks after float32 storage; mean/max use float64 errors. Actual selection masks come from native SP diagnostics or literal instrumented B copies, checked bit-identical against B output on the same inputs.

Observed maximum unrefined exact-softmax mass is 1; maximum skipped-key weight relative to the row maximum is 1. These quantify attention importance of keys kept at bulk precision. Coverage is limited to the completed rows above; they supply no distributional bound beyond those activations.

C−B PPL is -0.117686 over the six registered last-512 windows. Interpret it only alongside C’s actual matched fraction. This six-window model-perplexity evidence does not establish cross-corpus or cross-length quality.

## Decode table — kernel sweep / in-model timing

| Bits | Starting S | Arm | Status | tokens/s | Steps | Prefill seconds |
|---|---:|---|---|---:|---:|---:|
| 4 | 2048 | A | BLOCKED / unrun | — | — | — |
| 4 | 2048 | B | BLOCKED / unrun | — | — | — |
| 4 | 2048 | C | BLOCKED / unrun | — | — | — |
| 4 | 8192 | A | BLOCKED / unrun | — | — | — |
| 4 | 8192 | B | BLOCKED / unrun | — | — | — |
| 4 | 8192 | C | BLOCKED / unrun | — | — | — |
| 4 | 32768 | A | BLOCKED / unrun | — | — | — |
| 4 | 32768 | B | BLOCKED / unrun | — | — | — |
| 4 | 32768 | C | BLOCKED / unrun | — | — | — |
| 8 | 2048 | A | BLOCKED / unrun | — | — | — |
| 8 | 2048 | B | BLOCKED / unrun | — | — | — |
| 8 | 2048 | C | BLOCKED / unrun | — | — | — |
| 8 | 8192 | A | BLOCKED / unrun | — | — | — |
| 8 | 8192 | B | BLOCKED / unrun | — | — | — |
| 8 | 8192 | C | BLOCKED / unrun | — | — | — |
| 8 | 32768 | A | BLOCKED / unrun | — | — | — |
| 8 | 32768 | B | BLOCKED / unrun | — | — | — |
| 8 | 32768 | C | BLOCKED / unrun | — | — | — |

Identical teacher-forced continuations, 32 measured steps, CUDA-synchronized wall time per token; expanded MLA for A/B/C, absorbed decode OFF. Cache prefill is excluded from tokens/s but included in the 480s worker ceiling. At starting S=32768, measured attention lengths are 32769–32800: explicitly beyond the trained window, with no quality claim.
**G2/G3 rows establish nothing about model quality by themselves.**

## Registry, predictions and gates

C uses one global δ per bitwidth, matched over all layers at S=1024 to B’s actual diagnostic fraction ±0.01. Fixed grid on B activations initializes at the smallest tied δ; at most eight actual C fraction-only trials, bounded bisection on [0,32]. First match freezes δ for long prefill/decode; no PPL-based tuning. Report per-layer min/max/std.
D uses finite float32 max δ, requires every eligible pair selected and |D−A|≤0.005 PPL. E uses upward_float32(ln(1000)+2·eq), eq the upward float32 global SP-arithmetic error maximum across B/C, both lengths, all layers. Missing layer blocks E. E’s own captures separately check the finite-envelope transfer.

| Owner | ID | Registered prediction | Status |
|---|---|---|---|
| Lead | P1 | D equals A within 0.005 ppl | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Lead | P2 | C within 0.05 ppl of B at matched fraction | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Lead | P3 | E fraction >=0.95, real eq >=0.8, E ppl within 0.01 of A | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Lead | P4 | bulk4 p99 error <0.5*max; C skipped weight ratio >0.1 on some layer while ppl unmoved (P2 tolerance) | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Lead | P5 | C/B decode tokens/s at S=32768 >=2 | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Seat | S1 | G0 likely RED on current engine even after tokens recovered; documented matmul/softmax drift exceeds 0.01 | HISTORICAL PREMISE RETIRED by lead amendment 2; retained verbatim |
| Seat | S2 | D likely differs from A by >0.005 ppl: BF16 cuBLAS/softmax rounding differs from fused FP32 accumulation | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Seat | S3 | E fraction >=0.95 at bulk4; conditional finite bound will not establish general low-precision usefulness | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Seat | S4 | C within 0.05 of B if realised fraction matches; skipped weight ratio >0.1 likely | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Seat | S5 | C/B full-model decode at S=32768 <2 because latent expansion, re-quantization and INT4 projections remain in both arms | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Lead amendment 2 | B-A | Fresh B-A within +/-0.3 at bulk4, prediction only | True |

CPU gates: {"D96_pins": ["test_d96_prefill_contract_and_dense_pin", "test_d96_splitk_contract_and_dense_pin"], "GPU": "BLOCKED: cudaGetDeviceCount=100, device_count=0, no CUDA-capable device is detected; 0 GPU jobs", "adapter_cpu_import": "PASS; six modules pinned; no model instantiation; host_protocol2.json", "amendment_guards": {"names": ["test_protocol2_forged_amendment_red", "test_protocol2_stale_amendment_red", "test_protocol2_wrong_stream_sha_red"], "receipt": "artifacts/apa_sp3/guard_rejections_protocol2.json", "status": "PASS_ALL_REJECTED"}, "blind_verification": "lead-owned, unrun", "dry_run_cells": 692, "evidence_class": "unit test / host compile / code inspection", "fingerprint": {"artifacts/apa_sp3/adapter_import_cpu.json": "89f6369bbf700eff2209abc07b337775a59259c7abd465fa19be5072e9e6a490", "artifacts/apa_sp3/amendment_001_execution.json": "c680b376073a9790c7fb995c73d8ce9bbbc46512381133b0fae6e8861f9897b5", "artifacts/apa_sp3/amendment_002_native_bulk.json": "622b55b4b66a580fe38dd6f25e2471da5abb8dc7b76c590176a1b5618ffd1ba4", "artifacts/apa_sp3/amendment_003_B_native_scores.json": "b2689d07f8f70481d1f0740de256975a7be93cd902a4eb07831f94fd43bf00d5", "artifacts/apa_sp3/build/manifest.json": "cc66365b0733abd6c17e21bd2ed32420003a038d3440b33c82f49e5ecbc063e1", "artifacts/apa_sp3/registration.json": "d9b6511702a894f72174795141c72b2097b1bbe6110e810a1d4d8bfc3cd3498c", "artifacts/apa_sp3/weight_identity.json": "16d121d485ad7a21fdeb8cf2b881b8ecc1bb1532185e6fd425a8cbcdc8356ead", "scripts/apa_sp3_build.sh": "c9b28b49872d515deff42ba6ecf5c840f5ffab6c516533025f435208e0054c66", "scripts/apa_sp3_common.py": "2365412261effca3a0dbe4d80ee1b9b730510a3d50b52c826375a0c65cb261d5", "scripts/apa_sp3_control.py": "485cd771a747ec07b5f258be7b22c3c25608a2242d4b212542c5350e468ff734", "scripts/apa_sp3_diag_bindings.cpp": "e3612f50d4cc9456e79ab063889bc62071b2f3fa365e0890902bebe232d24f17", "scripts/apa_sp3_gpu.py": "9a680340294b9d0dcac3ca9fc731cb91304ae7f9253f597e43649707bdd33a09", "scripts/apa_sp3_lead_gpu.sh": "e5f970e7a29bee103f82c82bc0acf4318fd6087c76979ead4266ef60209fed86", "scripts/apa_sp3_make_diag.py": "2b01cb3c58434d88cc08f51797eb89fa101e527a1b093b3867cb26f99fd010ef", "scripts/apa_sp3_metrics.py": "26e9b9e13f15baae7069e981dea064939c8b232f8b43e3517625f44792c178a3", "scripts/apa_sp3_model.py": "e95f2ba8ee4407a46792c1c332cd7a31d62981775429ec6becf275f471507cf2", "scripts/apa_sp3_mutations.py": "affc907439e27b2c71d334093d9be819fb78d4eb4add0e0a2982f8a0b1cfb2c0", "scripts/apa_sp3_peak.cpp": "fb439af92e0e968ff2d09b7a53f58ddb69a32e442953ca455c89506615acb7bc", "scripts/apa_sp3_protocol2.py": "f585a942aae501f2efeac8469531ee315a3c09bee98853433aa1de7335c93899", "scripts/apa_sp3_report.py": "5c5e475cf62c9c4348b340488cc7c066ddc09062ce0676c9c773a8a1fb1b9b50"}, "g0_preflight": "PASS for g0_A_1; no lease/device/model needed", "host_build": {"command": "timeout --kill-after=5s 575s env PYTHONDONTWRITEBYTECODE=1 bash scripts/apa_sp3_build.sh", "log": "artifacts/apa_sp3/build_protocol2.log", "log_sha256": "3ef7b7a60d88ed51464b8f312cf59a875eeaec3cbf47c5c473ee6736d61d4f39", "manifest_sha256": "cc66365b0733abd6c17e21bd2ed32420003a038d3440b33c82f49e5ecbc063e1", "status": "PASS"}, "mutation": {"killed": 6, "manifest": "artifacts/apa_sp3/mutation_manifest_protocol2.json", "nonerror": 6, "rate": 1.0, "receipt": "artifacts/apa_sp3/mutation_results_protocol2.json", "source_pins_current": true, "threshold": 0.8}, "observer_host_gate": "PASS native no-device failure preserved; GPU allocation accounting remains untested", "pins": {"amendment_sha256": "db95e3ecb958ef6c31a105ee77be19c455ae8b210ed8a8082194389294843224", "preexisting_kernel_bodies": 107, "preexisting_source_files": 24, "registration_sha256": "d9b6511702a894f72174795141c72b2097b1bbe6110e810a1d4d8bfc3cd3498c", "unchanged": true}, "protocol": {"feeding": "one_full_prefill_per_window_no_cache_between_windows", "status": "REGISTERED_PROTOCOL_2", "token_count": 333337, "token_sha256": "d3c64882a9cbd4264a21f60ac239680d3a7d0631ab06d0d8a5d4fac8286d268a", "tokens_file_sha256": "5684e72cbab28236391ebcac9113aeca5b318ce6eeed69c410f1094950269a28"}, "pytest": {"failed": 0, "log": "artifacts/apa_sp3/cpu_protocol2_delivery.log", "log_sha256": "bc597605a824565d6cc6b7a4d38324edf67198280b8360727e56107cfb2c1075", "passed": 57, "skipped": 0}, "status": "PASS_CPU_ONLY", "supersedes": "CPU_GATES_FINAL.json (r1 retained unchanged)"}
D=96 tests: `test_d96_prefill_contract_and_dense_pin`, `test_d96_splitk_contract_and_dense_pin`, and compiled flag/device/scalar guards for both geometries. Native GPU numerical pins are delivered in job `kernel96`, unrun here.
Author tests and mutations are baseline evidence only. Independent blind verification under House Rules §8 is lead-owned and unrun; no subagents were launched.

## Lead commands and bounds

```bash
cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3
timeout 30s bash scripts/apa_sp3_lead_gpu.sh list
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run kernel96
# Four separately leased fresh processes, then aggregate determinism:
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run g0_A_1
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run g0_B_1
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run g0_A_2
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run g0_B_2
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run g0
# Each resume invocation runs at most ONE cell; stops on RED/stale receipts:
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh resume 4
timeout 30s bash scripts/apa_sp3_lead_gpu.sh summary
```
Complete commands: `artifacts/apa_sp3/lead_commands.txt`; every cell and per-job estimate: `dry_run.json`. Optional secondary uses `resume 8` after primary prerequisites. No multi-hour automatic batch is launched.
Each leased operation: flock --wait 20, worker timeout 480s plus 5s own-child termination grace, foreground 30s cooldown; outer guard TERM at 585s plus 3s grace. Device/PID inspection fails closed. Never signal a foreign PID. Operator keeps right of way; advisory-lock cooperation is required.
Planning estimates for workers, not timings: each fresh baseline 60–480s; G0 aggregation 1–10s; model PPL/capture/match/decode 140–480s including twelve control prefills, or RED timeout/OOM; per-layer G2 3–60s at 1024, 30–480s at 8192. The 480s worker cap is unchanged; six-window controls increase deadline risk. Add 30s cooldown plus up to 20s lease / 40s setup-receipt overhead; outer bound 590s. Actual upper-bound compliance is enforced, not predicted.
Memory reasoning: a single BF16 40-head S² score tensor needs 5.0 GiB at 8192 and 80 GiB at 32768, before intermediates and ~2.9 GB model residency. Standard full prefill at 32768 cannot fit on 12 GB; its decode setup will record OOM. 8192 standard fit and B/SP full-prefill deadlines remain unverified. No alternate cache-building scheme is silently substituted.

## Prior art

| Work | What is reused / what SP3 adds |
|---|---|
| GraftRepository MiniCPM3 floor protocol (2026) | get_text newline join and six-window teacher-forced NLL reference; lead amendment fixes 512 targets, chooses full prefills and fp64. New immutable token manifest and fresh-process gate wiring; no new scoring algorithm. |
| [BLASST](https://arxiv.org/abs/2512.12087) | Yuan et al. 2025/2026, arXiv:2512.12087: running-max log threshold; APA promotes individual keys instead of omitting blocks |
| [ThriftAttention](https://arxiv.org/abs/2605.23081) | Sharratt 2026, arXiv:2605.23081: selective precision and weight-sensitive error motivation; no FP4 tensor-core implementation port |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Dao 2023, arXiv:2307.08691 and Milakov/Gimelshein 2018 online normalizer: stable online softmax/work partitioning; existing SP implementation reused, no FA2 port |
| [TurboQuant](https://arxiv.org/abs/2504.19874) | Zandieh et al. 2025, arXiv:2504.19874: rotation and scalar codebook via existing quant.py; reconstructed BF16 keys, no QJL residual or packed FP4 arithmetic |
| APA | David Perry 2026, docs/APA_PAPER_DRAFT.md: all-key denominator, |bulk| z selection and precision promotion |
| SP2 | existing conditional log(1/epsilon)+2eq proof; here model finite max measurement, no new universal theorem |
| SP3 | new experiment wiring, global model calibration, captured-activation tail audit and fail-closed receipts; empirical percentiles and matched-control experiments are standard, no novelty claimed |

The APA draft is David Perry (2026), `docs/APA_PAPER_DRAFT.md`; treated as local prior art, not independently verified theorem. Arm A uses standard scaled dot-product attention (Vaswani et al. 2017, https://arxiv.org/abs/1706.03762), not an external FA2 package. Standard NLL (Shannon 1948), nearest-rank order statistics, directed rounding, memory maps, ELF allocation interposition, content hashes and leases are not new algorithms. Mutation-testing historical attribution is an unverified lead: DeMillo/Lipton/Sayward 1978, search “Hints on Test Data Selection”. Code sites and ledger contain the same provenance distinctions.
Bulk4/8 here means a TurboQuant codebook reconstructed to BF16 Kq. Existing SP bulk dots execute floating-point instructions; this experiment cannot establish packed FP4 speed or compressed KV residency. No source/kernels outside the authorized new harness/test paths were edited.

## Deviations, RED and residuals

- Original registration token SHA remains null and original S1 is retained as historical; lead amendment 2 supplies the new protocol and token pins. This seat did not run fresh A/B or GPU gates; their receipt statuses are shown above. Current blocked receipt: `artifacts/apa_sp3/GPU_BLOCKED_PROTOCOL2.json`. Historical values are not targets.
- Floor reference deviations explicitly authorized by amendment: full-prefill feeding, 512 instead of 511 scored targets, fp64 instead of fp32 NLL. Default BF16 matches the adapter; no compute dtype deviation. Read-only Arrow load uses exactly the specified cached test split.
- Missing local SP2/SPD1 artifacts were read from sibling worktrees, read-only. SP1/SP1.1 registration hashes are corroborated by the local ledger and SP2 parent registration; their original JSONs are absent here.
- Native GPU behavior, diagnostic bit parity, BF16 D≈A, long-context fit/time and actual model quality remain untested. Kernel-body hash equality proves source preservation, not GPU correctness.
- Resident peak is an explicitly qualified estimate. Cold/warm effects and single-call prefill variability remain; decode measures 32 steps and includes expansion/quantization overhead common to arms.
- Full long-context captures use tens of GB of disk and native diagnostic masks add quadratic transient memory. Disk/OOM/time failures remain RED; no sampling reduction.
- No git, subagents, shell background jobs, live-service changes or foreign-process termination. Only explicitly bounded own child processes may be terminated by timeout. Host build/tests complete; lead GPU and blind verification pending.

Files: `scripts/apa_sp3_*`, `tensor_cuda/tests/test_apa_sp3.py`, `docs/APA_SP3_LEDGER.md`, and `artifacts/apa_sp3/` (registration, source pins, build, CPU/mutation receipts, exact commands, blocked JSON and this renderer output).
