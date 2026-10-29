# TX3-X4P — ACCEPT; all four real modes KILLED

CPU finite computation on stored Qwen3 operands. X4P-G1 PASS (8 hand tests); X4P-G2 PASS (98 cells, 2,824 rows, zero nonfallback mask mismatches, every table recomputed). The registered probe is accepted; no predictor met D5. No kernel, speed, model-quality or adoption claim.

Receipts: `g1.log`, `g1_receipt.json`; CLI `probe.log`, `cli_receipt.json` (exit 0); check `check_final.log`, `check_final_receipt.json`, `g2_receipt.json`. First independent check preserved as `check.log`, `check_receipt.json`. Seeds: real 4282026, synthetic 4282027. All CPU/BLAS thread limits 1. No scratch files, deletions, git commands, GPU runs or product-source edits.

## D5 verdicts

|model|mode|verdict|operating point c=1|operating point c=4|lowest fallback (κ)|
|---|---|---|---|---|---|
|Qwen3-1.7B|decode|KILLED|none|none|966/1024 = 94.3359% (0.05)|
|Qwen3-1.7B|prefill|KILLED|none|none|836/1024 = 81.6406% (0.05)|
|Qwen3-4B|decode|KILLED|none|none|956/1024 = 93.3594% (0.05)|
|Qwen3-4B|prefill|KILLED|none|none|813/1024 = 79.3945% (0.05)|

The best fallback rates were 94.3359%/93.3594% for decode (1.7B/4B) and 81.6406%/79.3945% for prefill. Narrow brackets missed the true threshold; wider brackets overflowed the 2% band. At prefill κ≥0.2, ≥95% of thresholds were bracketed on both models, but all 1,024 rows overflowed. These measurements kill these fixed predictors/brackets on this draw, not every conceivable predictor.

## Predictions

P22: **NOT HELD**. Decode did not survive at κ≤0.05 on either model; prefill was killed on both, with the predicted coverage/overflow conflict.

P23: **NOT HELD**. Its house-tolerance component HELD: zero A/B/C tolerance failures across all cells, including every nonfallback B. Its 10× order-control component did not hold: 427 real-main nonfallback row×mode×κ observations exceeded 10× that same row's C max-abs error. These are repeated observations across κ, not 427 distinct input rows. All-zero nonfallback cells supply no numerical evidence.

The cell-maximum interpretation also yields NOT HELD on matched nonfallback rows:

|model|mode|κ|N|B max-abs max|C max-abs max|B≤10C|
|---|---|---:|---:|---:|---:|---|
|Qwen3-1.7B|decode|0.01|19|0.000112805|6.26585e-06|False|
|Qwen3-1.7B|decode|0.02|41|0.000110898|8.22857e-06|False|
|Qwen3-1.7B|decode|0.05|58|0.000127853|6.26585e-06|False|
|Qwen3-1.7B|prefill|0.01|88|0.000280803|1.56714e-05|False|
|Qwen3-1.7B|prefill|0.02|186|0.000343622|1.56714e-05|False|
|Qwen3-1.7B|prefill|0.05|188|0.000343622|1.19327e-05|False|
|Qwen3-4B|decode|0.01|26|1.63204e-05|4.26568e-06|True|
|Qwen3-4B|decode|0.02|50|3.15307e-05|4.26568e-06|True|
|Qwen3-4B|decode|0.05|68|3.48434e-05|4.26568e-06|True|
|Qwen3-4B|decode|0.1|1|2.85433e-07|2.7491e-08|False|
|Qwen3-4B|prefill|0.01|108|5.77738e-05|1.70323e-06|False|
|Qwen3-4B|prefill|0.02|208|5.77738e-05|4.26568e-06|False|
|Qwen3-4B|prefill|0.05|211|2.74497e-05|4.26568e-06|True|
|Qwen3-4B|prefill|0.1|1|1.35654e-07|2.7491e-08|True|

## D4 complete tables

All fractions are shares. Triples are mean/p99/max. Low/high/overflow/missing counts overlap. Full-spec rate = 1 − fallback. Extra/n and extra/r measure extra exact dots divided by row length and two-pass refine count. Costs are total weighted reads divided by total baseline reads, c=1/c=4. A and C are invariant across mode and κ and appear once per class below. Every B numerical table includes fallback rows (B=A); failures give all/nonfallback with nonfallback N. `row_metrics.npz` explicitly reports each row×mode×κ max-abs and rel-L2 for A/B/C. `run/tables.json` gives full precision and cost read totals, predictor-read sensitivity and band-count correction upper bounds. Empty nonfallback cells are never numerical passes.

|model/class|A max-abs μ/p99/max|A rel-L2 μ/p99/max|C max-abs μ/p99/max|C rel-L2 μ/p99/max|A/C failures|
|---|---|---|---|---|---|
|Qwen3-1.7B/edge|3.0382e-05/0.00011243/0.00011575|2.4356e-06/5.456e-06/5.5355e-06|3.0804e-06/7.0969e-06/7.1573e-06|2.3428e-07/4.2043e-07/4.2352e-07|0/0|
|Qwen3-1.7B/main|1.4735e-05/0.00012697/0.00034362|2.9432e-06/2.8385e-05/5.7966e-05|1.5367e-06/8.1968e-06/1.5671e-05|2.2984e-07/1.1307e-06/2.1523e-06|0/0|
|Qwen3-1.7B/short|4.2352e-06/2.9897e-05/4.2856e-05|6.4744e-07/4.6948e-06/7.6892e-06|1.2201e-06/6.1795e-06/6.9743e-06|1.6613e-07/5.1401e-07/6.7243e-07|0/0|
|Qwen3-4B/edge|2.9364e-06/7.2085e-06/7.3006e-06|1.8571e-06/2.6855e-06/2.6857e-06|4.6212e-07/1.2342e-06/1.2548e-06|2.1286e-07/3.3861e-07/3.4217e-07|0/0|
|Qwen3-4B/main|3.2867e-06/3.3126e-05/7.7704e-05|4.1141e-06/4.06e-05/6.671e-05|3.211e-07/2.4027e-06/6.9168e-06|2.7754e-07/1.5784e-06/2.0975e-06|0/0|
|Qwen3-4B/short|5.9717e-07/4.8244e-06/8.7755e-06|6.6406e-07/7.0096e-06/9.718e-06|2.1532e-07/1.5319e-06/2.2432e-06|1.8347e-07/6.7619e-07/7.1831e-07|0/0|
|synthetic-heavy-tail/synthetic|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|4.9482e-08/3.2624e-07/4.1787e-07|1.3308e-08/7.4176e-08/1.1345e-07|0/0|

### Qwen3-1.7B / edge

|mode|κ|rows|fallback / full-spec|low/high/overflow/missing|mask mismatch|band μ/p99/max|extra/n μ/p99/max|extra/r μ/p99/max|cost c1/c4|
|---|---:|---:|---|---|---:|---|---|---|---|
|decode|0.01|4|1/0|2/2/0/0|0|0.002533/0.0043762/0.0043945|0.015594/0.04995/0.051147|0.094851/0.29783/0.30473|1.5168/1.555|
|decode|0.02|4|1/0|2/2/0/0|0|0.0049133/0.0085083/0.0085449|0.016449/0.052347/0.053589|0.10011/0.31213/0.31927|1.5174/1.5569|
|decode|0.05|4|0.75/0.25|2/1/1/0|0|0.013245/0.022803/0.022949|0.019501/0.059598/0.060913|0.11914/0.35543/0.36291|1.4205/1.4652|
|decode|0.1|4|1/0|2/1/3/0|0|0.027374/0.045828/0.046143|0.024963/0.071614/0.072998|0.15337/0.42722/0.43491|1.5229/1.5751|
|decode|0.2|4|1/0|2/1/3/0|0|0.056091/0.090796/0.091309|0.036682/0.095051/0.096436|0.22724/0.56734/0.57455|1.53/1.5985|
|decode|0.5|4|1/0|2/0/4/0|0|0.14111/0.22884/0.23035|0.074554/0.17774/0.17957|0.46469/1.0614/1.0698|1.556/1.6837|
|decode|1.0|4|1/0|0/0/4/0|0|0.3063/0.44242/0.44434|0.17941/0.32949/0.33118|1.139/1.9687/1.9731|1.618/1.887|
|prefill|0.01|4|1/0|3/1/0/0|0|0.0037537/0.0053381/0.0053711|0.0058594/0.022734/0.023438|0.034909/0.13545/0.13964|1.54/1.6311|
|prefill|0.02|4|0.75/0.25|2/1/0/0|0|0.0069275/0.0096887/0.0097656|0.0065308/0.025225/0.026001|0.038926/0.15029/0.15491|1.1418/1.2369|
|prefill|0.05|4|0.5/0.5|1/1/1/0|0|0.017944/0.022211/0.022339|0.0095215/0.031765/0.032593|0.05753/0.18933/0.19418|1.0458/1.1479|
|prefill|0.1|4|1/0|0/1/4/0|0|0.035889/0.044481/0.0448|0.018097/0.043859/0.0448|0.11254/0.26153/0.26691|1.5481/1.6577|
|prefill|0.2|4|1/0|0/0/4/0|0|0.071381/0.087955/0.088501|0.038208/0.06874/0.069824|0.24189/0.41011/0.416|1.5581/1.6906|
|prefill|0.5|4|1/0|0/0/4/0|0|0.1843/0.22273/0.224|0.112/0.14879/0.15002|0.71873/0.88832/0.89382|1.5934/1.8063|
|prefill|1.0|4|1/0|0/0/4/0|0|0.38696/0.4336/0.4353|0.27124/0.30443/0.30566|1.7517/1.8191/1.8211|1.6662/2.045|

|mode|κ|B max-abs μ/p99/max|B rel-L2 μ/p99/max|B failures all/nonfallback (N)|B>10C NF rows|
|---|---:|---|---|---|---:|
|decode|0.01|3.0382e-05/0.00011243/0.00011575|2.4356e-06/5.456e-06/5.5355e-06|0/0 (0)|0|
|decode|0.02|3.0382e-05/0.00011243/0.00011575|2.4356e-06/5.456e-06/5.5355e-06|0/0 (0)|0|
|decode|0.05|3.2766e-05/0.00011271/0.00011575|2.5716e-06/5.456e-06/5.5355e-06|0/0 (1)|0|
|decode|0.1|3.0382e-05/0.00011243/0.00011575|2.4356e-06/5.456e-06/5.5355e-06|0/0 (0)|0|
|decode|0.2|3.0382e-05/0.00011243/0.00011575|2.4356e-06/5.456e-06/5.5355e-06|0/0 (0)|0|
|decode|0.5|3.0382e-05/0.00011243/0.00011575|2.4356e-06/5.456e-06/5.5355e-06|0/0 (0)|0|
|decode|1.0|3.0382e-05/0.00011243/0.00011575|2.4356e-06/5.456e-06/5.5355e-06|0/0 (0)|0|
|prefill|0.01|3.0382e-05/0.00011243/0.00011575|2.4356e-06/5.456e-06/5.5355e-06|0/0 (0)|0|
|prefill|0.02|3.027e-05/0.00011243/0.00011575|1.9624e-06/5.4029e-06/5.5355e-06|0/0 (1)|1|
|prefill|0.05|3.0308e-05/0.00011243/0.00011575|2.0702e-06/5.4116e-06/5.5355e-06|0/0 (2)|2|
|prefill|0.1|3.0382e-05/0.00011243/0.00011575|2.4356e-06/5.456e-06/5.5355e-06|0/0 (0)|0|
|prefill|0.2|3.0382e-05/0.00011243/0.00011575|2.4356e-06/5.456e-06/5.5355e-06|0/0 (0)|0|
|prefill|0.5|3.0382e-05/0.00011243/0.00011575|2.4356e-06/5.456e-06/5.5355e-06|0/0 (0)|0|
|prefill|1.0|3.0382e-05/0.00011243/0.00011575|2.4356e-06/5.456e-06/5.5355e-06|0/0 (0)|0|

### Qwen3-1.7B / main

|mode|κ|rows|fallback / full-spec|low/high/overflow/missing|mask mismatch|band μ/p99/max|extra/n μ/p99/max|extra/r μ/p99/max|cost c1/c4|
|---|---:|---:|---|---|---:|---|---|---|---|
|decode|0.01|1024|0.98145/0.018555|491/514/0/0|0|0.0038432/0.0077628/0.0092323|0.047412/0.36993/0.60751|0.30523/2.4637/3.897|1.5279/1.6315|
|decode|0.02|1024|0.95996/0.040039|479/504/0/0|0|0.0076892/0.014645/0.017152|0.048657/0.37179/0.60895|0.31327/2.4766/3.9113|1.5067/1.6123|
|decode|0.05|1024|0.94336/0.056641|453/470/465/0|0|0.019226/0.034377/0.039065|0.052572/0.38138/0.61487|0.33855/2.5293/3.9485|1.4904/1.6025|
|decode|0.1|1024|1/0|399/427/931/0|0|0.038396/0.068457/0.0761|0.059639/0.39714/0.62426|0.38419/2.6374/3.9971|1.5537/1.6767|
|decode|0.2|1024|1/0|324/341/996/0|0|0.077172/0.13477/0.15392|0.075913/0.43021/0.64274|0.48927/2.848/4.1144|1.5639/1.7101|
|decode|0.5|1024|1/0|135/151/1020/0|0|0.19411/0.32921/0.37854|0.1405/0.52218/0.68866|0.90593/3.4699/4.4177|1.5988/1.8251|
|decode|1.0|1024|1/0|23/46/1023/0|0|0.39424/0.60906/0.68776|0.2891/0.65989/0.74762|1.8652/4.3704/4.8598|1.6707/2.0616|
|prefill|0.01|1024|0.91406/0.085938|578/358/0/0|0|0.0040326/0.0068989/0.0081486|0.0054349/0.038477/0.05077|0.035055/0.24885/0.34968|1.4431/1.5234|
|prefill|0.02|1024|0.81836/0.18164|529/309/0/0|0|0.0080305/0.012411/0.016547|0.0064122/0.040843/0.052931|0.041337/0.26004/0.36076|1.3529/1.4352|
|prefill|0.05|1024|0.81641/0.18359|363/183/506/0|0|0.020101/0.029141/0.031078|0.010071/0.047588/0.05881|0.064844/0.30513/0.40506|1.3631/1.4525|
|prefill|0.1|1024|1/0|153/74/1024/0|0|0.040101/0.055524/0.059531|0.018254/0.05952/0.070986|0.11753/0.37841/0.48892|1.544/1.6446|
|prefill|0.2|1024|1/0|9/9/1024/0|0|0.080318/0.1087/0.12527|0.040069/0.086758/0.096945|0.25816/0.55844/0.66772|1.5546/1.6797|
|prefill|0.5|1024|1/0|0/0/1024/0|0|0.20211/0.2641/0.29794|0.12002/0.18205/0.20299|0.77403/1.1943/1.4601|1.5919/1.8022|
|prefill|1.0|1024|1/0|0/0/1024/0|0|0.41004/0.49953/0.53132|0.28647/0.3707/0.39755|1.8498/2.4855/2.8595|1.6693/2.0568|

|mode|κ|B max-abs μ/p99/max|B rel-L2 μ/p99/max|B failures all/nonfallback (N)|B>10C NF rows|
|---|---:|---|---|---|---:|
|decode|0.01|1.4725e-05/0.00012697/0.00034362|2.9432e-06/2.8385e-05/5.7966e-05|0/0 (19)|3|
|decode|0.02|1.4755e-05/0.00012697/0.00034362|2.9444e-06/2.8385e-05/5.7966e-05|0/0 (41)|14|
|decode|0.05|1.477e-05/0.00012769/0.00034362|2.9499e-06/2.8385e-05/5.7966e-05|0/0 (58)|24|
|decode|0.1|1.4735e-05/0.00012697/0.00034362|2.9432e-06/2.8385e-05/5.7966e-05|0/0 (0)|0|
|decode|0.2|1.4735e-05/0.00012697/0.00034362|2.9432e-06/2.8385e-05/5.7966e-05|0/0 (0)|0|
|decode|0.5|1.4735e-05/0.00012697/0.00034362|2.9432e-06/2.8385e-05/5.7966e-05|0/0 (0)|0|
|decode|1.0|1.4735e-05/0.00012697/0.00034362|2.9432e-06/2.8385e-05/5.7966e-05|0/0 (0)|0|
|prefill|0.01|1.4731e-05/0.00012697/0.00034362|2.9422e-06/2.8385e-05/5.7966e-05|0/0 (88)|31|
|prefill|0.02|1.4755e-05/0.00012697/0.00034362|2.9444e-06/2.8385e-05/5.7966e-05|0/0 (186)|56|
|prefill|0.05|1.4844e-05/0.00012697/0.00034362|2.943e-06/2.8385e-05/5.7966e-05|0/0 (188)|65|
|prefill|0.1|1.4735e-05/0.00012697/0.00034362|2.9432e-06/2.8385e-05/5.7966e-05|0/0 (0)|0|
|prefill|0.2|1.4735e-05/0.00012697/0.00034362|2.9432e-06/2.8385e-05/5.7966e-05|0/0 (0)|0|
|prefill|0.5|1.4735e-05/0.00012697/0.00034362|2.9432e-06/2.8385e-05/5.7966e-05|0/0 (0)|0|
|prefill|1.0|1.4735e-05/0.00012697/0.00034362|2.9432e-06/2.8385e-05/5.7966e-05|0/0 (0)|0|

### Qwen3-1.7B / short

|mode|κ|rows|fallback / full-spec|low/high/overflow/missing|mask mismatch|band μ/p99/max|extra/n μ/p99/max|extra/r μ/p99/max|cost c1/c4|
|---|---:|---:|---|---|---:|---|---|---|---|
|decode|0.01|256|0.99219/0.0078125|128/126/0/0|0|0.0036068/0.010828/0.01626|0.050104/0.46061/0.58989|0.33619/3.348/4.1245|1.5355/1.6332|
|decode|0.02|256|0.97656/0.023438|126/124/7/0|0|0.0076218/0.027506/0.03252|0.05154/0.46697/0.59219|0.34548/3.3869/4.1406|1.5165/1.6161|
|decode|0.05|256|0.93359/0.066406|115/116/102/0|0|0.018457/0.042051/0.070175|0.055193/0.48548/0.60138|0.36966/3.5189/4.2048|1.479/1.5844|
|decode|0.1|256|0.99609/0.0039062|102/106/223/0|0|0.036899/0.083985/0.10163|0.06227/0.49755/0.61057|0.41673/3.6322/4.2691|1.55/1.6656|
|decode|0.2|256|1/0|89/87/242/0|0|0.073757/0.15422/0.1626|0.077375/0.52687/0.63469|0.51748/3.9783/4.4378|1.5597/1.6967|
|decode|0.5|256|1/0|32/41/254/0|0|0.18754/0.35103/0.4187|0.13872/0.6071/0.70592|0.92777/4.4679/5.4615|1.5932/1.8069|
|decode|1.0|256|1/0|8/15/256/0|0|0.38479/0.63119/0.72269|0.28365/0.74883/0.77771|1.8922/5.307/6.7692|1.6633/2.0379|
|prefill|0.01|256|0.78906/0.21094|137/63/2/0|0|0.0040936/0.01415/0.028571|0.0042654/0.03209/0.044568|0.028039/0.22872/0.29091|1.4398/1.5169|
|prefill|0.02|256|0.73047/0.26953|126/56/5/0|0|0.0082343/0.026897/0.057143|0.0055798/0.037604/0.057143|0.03705/0.24697/0.5|1.3522/1.4312|
|prefill|0.05|256|0.76953/0.23047|97/37/117/0|0|0.020208/0.042775/0.085714|0.0090655/0.045667/0.057143|0.060136/0.28545/0.5|1.3309/1.416|
|prefill|0.1|256|0.98828/0.011719|43/14/252/0|0|0.039979/0.069531/0.085714|0.016835/0.058323/0.061547|0.11148/0.3769/0.5|1.5416/1.6384|
|prefill|0.2|256|0.99609/0.0039062|1/1/255/0|0|0.079154/0.12189/0.33333|0.036912/0.078906/0.088901|0.24431/0.57776/0.75|1.5524/1.6727|
|prefill|0.5|256|0.99609/0.0039062|0/0/255/0|0|0.19719/0.29408/0.33333|0.11509/0.18534/0.23415|0.76483/1.4921/2.5263|1.5888/1.7925|
|prefill|1.0|256|1/0|0/0/256/0|0|0.40197/0.50752/0.58537|0.28131/0.39364/0.49756|1.8704/3.363/5.3684|1.6657/2.0458|

|mode|κ|B max-abs μ/p99/max|B rel-L2 μ/p99/max|B failures all/nonfallback (N)|B>10C NF rows|
|---|---:|---|---|---|---:|
|decode|0.01|4.2352e-06/2.9897e-05/4.2856e-05|6.4746e-07/4.6948e-06/7.6892e-06|0/0 (2)|0|
|decode|0.02|4.2374e-06/2.9897e-05/4.2856e-05|6.4653e-07/4.6948e-06/7.6892e-06|0/0 (6)|0|
|decode|0.05|4.2449e-06/2.9897e-05/4.2856e-05|6.4663e-07/4.6948e-06/7.6892e-06|0/0 (17)|3|
|decode|0.1|4.2371e-06/2.9897e-05/4.2856e-05|6.4745e-07/4.6948e-06/7.6892e-06|0/0 (1)|0|
|decode|0.2|4.2352e-06/2.9897e-05/4.2856e-05|6.4744e-07/4.6948e-06/7.6892e-06|0/0 (0)|0|
|decode|0.5|4.2352e-06/2.9897e-05/4.2856e-05|6.4744e-07/4.6948e-06/7.6892e-06|0/0 (0)|0|
|decode|1.0|4.2352e-06/2.9897e-05/4.2856e-05|6.4744e-07/4.6948e-06/7.6892e-06|0/0 (0)|0|
|prefill|0.01|4.2331e-06/2.9897e-05/4.2856e-05|6.489e-07/4.6948e-06/7.6892e-06|0/0 (54)|3|
|prefill|0.02|4.2725e-06/3.0159e-05/4.2856e-05|6.5113e-07/4.6948e-06/7.6892e-06|0/0 (69)|5|
|prefill|0.05|4.4445e-06/3.6594e-05/4.2856e-05|6.6017e-07/4.6948e-06/7.6892e-06|0/0 (59)|8|
|prefill|0.1|4.2352e-06/2.9897e-05/4.2856e-05|6.4724e-07/4.6948e-06/7.6892e-06|0/0 (3)|0|
|prefill|0.2|4.2352e-06/2.9897e-05/4.2856e-05|6.4744e-07/4.6948e-06/7.6892e-06|0/0 (1)|0|
|prefill|0.5|4.2352e-06/2.9897e-05/4.2856e-05|6.4744e-07/4.6948e-06/7.6892e-06|0/0 (1)|0|
|prefill|1.0|4.2352e-06/2.9897e-05/4.2856e-05|6.4744e-07/4.6948e-06/7.6892e-06|0/0 (0)|0|

### Qwen3-4B / edge

|mode|κ|rows|fallback / full-spec|low/high/overflow/missing|mask mismatch|band μ/p99/max|extra/n μ/p99/max|extra/r μ/p99/max|cost c1/c4|
|---|---:|---:|---|---|---:|---|---|---|---|
|decode|0.01|4|1/0|1/3/0/0|0|0.004303/0.0053564/0.0053711|0.04303/0.11624/0.11865|0.29991/0.8126/0.82935|1.5332/1.6098|
|decode|0.02|4|1/0|1/3/0/0|0|0.0073242/0.0087598/0.0087891|0.044281/0.11677/0.11914|0.30838/0.81632/0.83276|1.534/1.6125|
|decode|0.05|4|1/0|1/3/2/0|0|0.018402/0.020996/0.020996|0.049377/0.12224/0.12451|0.34345/0.85461/0.87031|1.5366/1.6209|
|decode|0.1|4|1/0|1/2/4/0|0|0.039276/0.047734/0.047852|0.059052/0.13637/0.13867|0.40998/0.95343/0.96928|1.5409/1.6353|
|decode|0.2|4|1/0|1/1/4/0|0|0.078186/0.092994/0.093262|0.07782/0.15901/0.16113|0.53885/1.1118/1.1263|1.5511/1.669|
|decode|0.5|4|1/0|0/1/4/0|0|0.1937/0.23038/0.23096|0.14236/0.24923/0.25195|0.98009/1.7426/1.7611|1.5833/1.7754|
|decode|1.0|4|1/0|0/0/4/0|0|0.39294/0.43649/0.43652|0.29913/0.39831/0.40039|2.0462/2.7853/2.7986|1.6555/2.0141|
|prefill|0.01|4|0.75/0.25|2/1/0/0|0|0.0032959/0.0043579/0.0043945|0.0078735/0.027795/0.028564|0.052488/0.1844/0.18947|1.4393/1.5296|
|prefill|0.02|4|0.75/0.25|2/1/0/0|0|0.0061951/0.0069543/0.006958|0.0085144/0.029364/0.030151|0.056826/0.19482/0.2|1.4403/1.5327|
|prefill|0.05|4|0.75/0.25|1/1/1/0|0|0.018616/0.021733/0.021851|0.012665/0.036274/0.037109|0.084656/0.24071/0.24615|1.4437/1.5433|
|prefill|0.1|4|1/0|0/1/4/0|0|0.035583/0.039301/0.039429|0.019989/0.045686/0.046631|0.13405/0.30321/0.30931|1.547/1.6554|
|prefill|0.2|4|1/0|0/0/4/0|0|0.071472/0.079902/0.080078|0.042145/0.064817/0.065674|0.28483/0.43034/0.43563|1.5571/1.6887|
|prefill|0.5|4|1/0|0/0/4/0|0|0.18311/0.20974/0.21033|0.11697/0.13003/0.13025|0.79331/0.8619/0.86397|1.5921/1.8043|
|prefill|1.0|4|1/0|0/0/4/0|0|0.38489/0.43494/0.43652|0.27939/0.30665/0.30762|1.897/1.988/1.989|1.6673/2.0529|

|mode|κ|B max-abs μ/p99/max|B rel-L2 μ/p99/max|B failures all/nonfallback (N)|B>10C NF rows|
|---|---:|---|---|---|---:|
|decode|0.01|2.9364e-06/7.2085e-06/7.3006e-06|1.8571e-06/2.6855e-06/2.6857e-06|0/0 (0)|0|
|decode|0.02|2.9364e-06/7.2085e-06/7.3006e-06|1.8571e-06/2.6855e-06/2.6857e-06|0/0 (0)|0|
|decode|0.05|2.9364e-06/7.2085e-06/7.3006e-06|1.8571e-06/2.6855e-06/2.6857e-06|0/0 (0)|0|
|decode|0.1|2.9364e-06/7.2085e-06/7.3006e-06|1.8571e-06/2.6855e-06/2.6857e-06|0/0 (0)|0|
|decode|0.2|2.9364e-06/7.2085e-06/7.3006e-06|1.8571e-06/2.6855e-06/2.6857e-06|0/0 (0)|0|
|decode|0.5|2.9364e-06/7.2085e-06/7.3006e-06|1.8571e-06/2.6855e-06/2.6857e-06|0/0 (0)|0|
|decode|1.0|2.9364e-06/7.2085e-06/7.3006e-06|1.8571e-06/2.6855e-06/2.6857e-06|0/0 (0)|0|
|prefill|0.01|2.9429e-06/7.2085e-06/7.3006e-06|1.9485e-06/2.6855e-06/2.6857e-06|0/0 (1)|0|
|prefill|0.02|2.9457e-06/7.2085e-06/7.3006e-06|1.9583e-06/2.6855e-06/2.6857e-06|0/0 (1)|0|
|prefill|0.05|2.9503e-06/7.2085e-06/7.3006e-06|2.0077e-06/2.6855e-06/2.6857e-06|0/0 (1)|1|
|prefill|0.1|2.9364e-06/7.2085e-06/7.3006e-06|1.8571e-06/2.6855e-06/2.6857e-06|0/0 (0)|0|
|prefill|0.2|2.9364e-06/7.2085e-06/7.3006e-06|1.8571e-06/2.6855e-06/2.6857e-06|0/0 (0)|0|
|prefill|0.5|2.9364e-06/7.2085e-06/7.3006e-06|1.8571e-06/2.6855e-06/2.6857e-06|0/0 (0)|0|
|prefill|1.0|2.9364e-06/7.2085e-06/7.3006e-06|1.8571e-06/2.6855e-06/2.6857e-06|0/0 (0)|0|

### Qwen3-4B / main

|mode|κ|rows|fallback / full-spec|low/high/overflow/missing|mask mismatch|band μ/p99/max|extra/n μ/p99/max|extra/r μ/p99/max|cost c1/c4|
|---|---:|---:|---|---|---:|---|---|---|---|
|decode|0.01|1024|0.97461/0.025391|479/519/0/0|0|0.0038124/0.0078281/0.0087489|0.04336/0.34717/0.60424|0.2776/2.3231/3.7991|1.5195/1.6201|
|decode|0.02|1024|0.95117/0.048828|473/501/0/0|0|0.0076404/0.014615/0.017935|0.044585/0.35073/0.60658|0.28548/2.3386/3.8138|1.4963/1.5989|
|decode|0.05|1024|0.93359/0.066406|436/464/455/0|0|0.019076/0.03483/0.040536|0.04845/0.35771/0.60961|0.31033/2.4082/3.8328|1.4777/1.5868|
|decode|0.1|1024|0.99902/0.00097656|379/410/946/0|0|0.038149/0.068858/0.081399|0.055488/0.37632/0.62127|0.35561/2.5194/3.9062|1.5513/1.671|
|decode|0.2|1024|1/0|275/320/995/0|0|0.076205/0.13515/0.15986|0.07184/0.40714/0.64412|0.46091/2.6456/4.0499|1.5622/1.7046|
|decode|0.5|1024|1/0|104/139/1015/0|0|0.1914/0.32839/0.38902|0.13739/0.48964/0.70056|0.88319/3.074/4.4047|1.5965/1.8172|
|decode|1.0|1024|1/0|30/47/1022/0|0|0.38945/0.60035/0.67898|0.28546/0.61379/0.77006|1.838/3.8961/4.8909|1.6674/2.0506|
|prefill|0.01|1024|0.89453/0.10547|610/306/0/0|0|0.0039731/0.006948/0.0079566|0.0045534/0.034112/0.05997|0.029112/0.21777/0.37578|1.431/1.5095|
|prefill|0.02|1024|0.79688/0.20312|555/261/0/0|0|0.00789/0.012597/0.014481|0.005443/0.035665/0.06377|0.034801/0.23717/0.39959|1.3397/1.4201|
|prefill|0.05|1024|0.79395/0.20605|390/151/485/0|0|0.019777/0.027825/0.033896|0.0088871/0.041878/0.070544|0.056804/0.27951/0.44203|1.3385/1.4261|
|prefill|0.1|1024|0.99902/0.00097656|184/61/1023/0|0|0.039431/0.055657/0.063654|0.01678/0.055598/0.08343|0.10739/0.35448/0.52277|1.542/1.6404|
|prefill|0.2|1024|1/0|15/3/1024/0|0|0.078947/0.1102/0.12333|0.037834/0.08185/0.11432|0.24271/0.5417/0.71636|1.5534/1.6756|
|prefill|0.5|1024|1/0|0/0/1024/0|0|0.19814/0.26328/0.30792|0.11593/0.1799/0.21404|0.74579/1.1809/1.4263|1.5897/1.7949|
|prefill|1.0|1024|1/0|0/0/1024/0|0|0.403/0.49796/0.54233|0.27959/0.3671/0.40309|1.8022/2.4736/2.6861|1.6656/2.0446|

|mode|κ|B max-abs μ/p99/max|B rel-L2 μ/p99/max|B failures all/nonfallback (N)|B>10C NF rows|
|---|---:|---|---|---|---:|
|decode|0.01|3.2862e-06/3.3126e-05/7.7704e-05|4.114e-06/4.06e-05/6.671e-05|0/0 (26)|8|
|decode|0.02|3.2891e-06/3.3126e-05/7.7704e-05|4.1176e-06/4.06e-05/6.671e-05|0/0 (50)|18|
|decode|0.05|3.2859e-06/3.3126e-05/7.7704e-05|4.114e-06/4.06e-05/6.671e-05|0/0 (68)|21|
|decode|0.1|3.2869e-06/3.3126e-05/7.7704e-05|4.1147e-06/4.06e-05/6.671e-05|0/0 (1)|1|
|decode|0.2|3.2867e-06/3.3126e-05/7.7704e-05|4.1141e-06/4.06e-05/6.671e-05|0/0 (0)|0|
|decode|0.5|3.2867e-06/3.3126e-05/7.7704e-05|4.1141e-06/4.06e-05/6.671e-05|0/0 (0)|0|
|decode|1.0|3.2867e-06/3.3126e-05/7.7704e-05|4.1141e-06/4.06e-05/6.671e-05|0/0 (0)|0|
|prefill|0.01|3.2911e-06/3.3126e-05/7.7704e-05|4.1163e-06/4.06e-05/6.671e-05|0/0 (108)|40|
|prefill|0.02|3.3117e-06/3.3126e-05/7.7704e-05|4.1286e-06/4.06e-05/6.671e-05|0/0 (208)|80|
|prefill|0.05|3.3061e-06/3.3126e-05/7.7704e-05|4.1164e-06/4.06e-05/6.671e-05|0/0 (211)|66|
|prefill|0.1|3.2867e-06/3.3126e-05/7.7704e-05|4.1141e-06/4.06e-05/6.671e-05|0/0 (1)|0|
|prefill|0.2|3.2867e-06/3.3126e-05/7.7704e-05|4.1141e-06/4.06e-05/6.671e-05|0/0 (0)|0|
|prefill|0.5|3.2867e-06/3.3126e-05/7.7704e-05|4.1141e-06/4.06e-05/6.671e-05|0/0 (0)|0|
|prefill|1.0|3.2867e-06/3.3126e-05/7.7704e-05|4.1141e-06/4.06e-05/6.671e-05|0/0 (0)|0|

### Qwen3-4B / short

|mode|κ|rows|fallback / full-spec|low/high/overflow/missing|mask mismatch|band μ/p99/max|extra/n μ/p99/max|extra/r μ/p99/max|cost c1/c4|
|---|---:|---:|---|---|---:|---|---|---|---|
|decode|0.01|256|0.97266/0.027344|107/142/1/0|0|0.0039595/0.015545/0.022727|0.06558/0.65856/0.81246|0.42861/4.2002/5.555|1.5255/1.6484|
|decode|0.02|256|0.96094/0.039062|105/141/8/0|0|0.0076572/0.02572/0.058824|0.066982/0.65856/0.81456|0.43821/4.2002/5.5694|1.5092/1.6343|
|decode|0.05|256|0.95312/0.046875|102/131/106/0|0|0.019062/0.056617/0.078431|0.071301/0.66751/0.81596|0.46687/4.2582/5.5789|1.4987/1.6298|
|decode|0.1|256|0.99219/0.0078125|87/115/221/0|0|0.036664/0.088114/0.13725|0.078038/0.67178/0.81805|0.51195/4.2819/5.5933|1.5556/1.6969|
|decode|0.2|256|0.99609/0.0039062|74/92/248/0|0|0.07376/0.16503/0.18182|0.094106/0.68859/0.82225|0.61808/4.3766/5.622|1.5709/1.734|
|decode|0.5|256|1/0|30/50/253/0|0|0.18907/0.36911/0.39673|0.1578/0.74467/0.83135|1.0428/4.7019/5.6842|1.6042/1.8435|
|decode|1.0|256|1/0|5/23/254/0|0|0.38419/0.64612/0.68182|0.30304/0.79594/0.83975|2.0119/5.2032/5.7416|1.6733/2.0715|
|prefill|0.01|256|0.77344/0.22656|130/67/1/0|0|0.0040629/0.014033/0.066667|0.004521/0.039215/0.066667|0.02954/0.26157/0.33333|1.4162/1.4945|
|prefill|0.02|256|0.70312/0.29688|121/56/5/0|0|0.0079939/0.021806/0.066667|0.0055235/0.040792/0.066667|0.036291/0.26577/0.33333|1.3242/1.4045|
|prefill|0.05|256|0.72656/0.27344|85/39/111/0|0|0.019207/0.044563/0.066667|0.0091475/0.047185/0.066667|0.060267/0.30865/0.35849|1.3074/1.3944|
|prefill|0.1|256|0.96484/0.035156|35/16/247/0|0|0.038089/0.077638/0.13333|0.017028/0.058224/0.067159|0.11256/0.34199/0.42925|1.527/1.6248|
|prefill|0.2|256|1/0|2/1/256/0|0|0.076366/0.1219/0.13333|0.037544/0.078545/0.090037|0.24906/0.56223/0.58824|1.5527/1.6739|
|prefill|0.5|256|1/0|0/0/256/0|0|0.19231/0.26586/0.27841|0.1142/0.181/0.18713|0.76076/1.3259/1.5882|1.5883/1.791|
|prefill|1.0|256|1/0|0/0/256/0|0|0.39294/0.49941/0.52857|0.27583/0.37162/0.40714|1.8424/3.0265/3.4706|1.6634/2.0389|

|mode|κ|B max-abs μ/p99/max|B rel-L2 μ/p99/max|B failures all/nonfallback (N)|B>10C NF rows|
|---|---:|---|---|---|---:|
|decode|0.01|5.9643e-07/4.8244e-06/8.7755e-06|6.639e-07/7.0096e-06/9.718e-06|0/0 (7)|1|
|decode|0.02|5.9822e-07/4.8244e-06/8.7755e-06|6.6538e-07/7.0096e-06/9.718e-06|0/0 (10)|1|
|decode|0.05|5.947e-07/4.8244e-06/8.7755e-06|6.656e-07/7.0096e-06/9.718e-06|0/0 (12)|0|
|decode|0.1|5.9722e-07/4.8244e-06/8.7755e-06|6.6433e-07/7.0096e-06/9.718e-06|0/0 (2)|0|
|decode|0.2|5.9717e-07/4.8244e-06/8.7755e-06|6.6406e-07/7.0096e-06/9.718e-06|0/0 (1)|0|
|decode|0.5|5.9717e-07/4.8244e-06/8.7755e-06|6.6406e-07/7.0096e-06/9.718e-06|0/0 (0)|0|
|decode|1.0|5.9717e-07/4.8244e-06/8.7755e-06|6.6406e-07/7.0096e-06/9.718e-06|0/0 (0)|0|
|prefill|0.01|5.9672e-07/4.8244e-06/8.7755e-06|6.6328e-07/7.0096e-06/9.718e-06|0/0 (58)|1|
|prefill|0.02|5.9449e-07/4.8244e-06/8.7755e-06|6.6508e-07/7.0096e-06/9.718e-06|0/0 (76)|2|
|prefill|0.05|6.0708e-07/4.602e-06/8.7755e-06|6.6599e-07/7.0096e-06/9.718e-06|0/0 (70)|2|
|prefill|0.1|5.9699e-07/4.8244e-06/8.7755e-06|6.6359e-07/7.0096e-06/9.718e-06|0/0 (9)|0|
|prefill|0.2|5.9717e-07/4.8244e-06/8.7755e-06|6.6406e-07/7.0096e-06/9.718e-06|0/0 (0)|0|
|prefill|0.5|5.9717e-07/4.8244e-06/8.7755e-06|6.6406e-07/7.0096e-06/9.718e-06|0/0 (0)|0|
|prefill|1.0|5.9717e-07/4.8244e-06/8.7755e-06|6.6406e-07/7.0096e-06/9.718e-06|0/0 (0)|0|

### synthetic-heavy-tail / synthetic

|mode|κ|rows|fallback / full-spec|low/high/overflow/missing|mask mismatch|band μ/p99/max|extra/n μ/p99/max|extra/r μ/p99/max|cost c1/c4|
|---|---:|---:|---|---|---:|---|---|---|---|
|decode|0.01|256|0.93359/0.066406|119/120/0/0|0|0.00088739/0.0018311/0.0020752|0.003727/0.020062/0.022949|0.085278/0.51113/0.56081|1.4456/1.4785|
|decode|0.02|256|0.86719/0.13281|106/116/0/0|0|0.0018053/0.0030518/0.00354|0.0040045/0.020874/0.023926|0.09139/0.5266/0.5777|1.3794/1.4129|
|decode|0.05|256|0.78125/0.21875|97/103/0/0|0|0.0045004/0.0067139/0.0068359|0.0049415/0.022247/0.025635|0.112/0.5662/0.625|1.2944/1.3296|
|decode|0.1|256|0.60547/0.39453|75/80/0/0|0|0.0089869/0.012195/0.012573|0.0067105/0.025311/0.027832|0.15083/0.62728/0.70126|1.1206/1.1593|
|decode|0.2|256|0.39062/0.60938|26/37/60/0|0|0.018265/0.023114/0.024536|0.0115/0.032422/0.036987|0.25487/0.81868/0.88514|0.91028/0.95651|
|decode|0.5|256|1/0|0/5/256/0|0|0.051235/0.063373/0.065674|0.035625/0.060535/0.064697|0.77305/1.5274/1.6318|1.5289/1.6081|
|decode|1.0|256|1/0|0/0/256/0|0|0.17189/0.20098/0.20728|0.14733/0.18406/0.18921|3.17/4.5708/5.0878|1.5835/1.8123|
|prefill|0.01|256|0.97656/0.023438|8/242/0/0|0|0.0010343/0.0020081/0.0020752|0.017313/0.046332/0.05542|0.37934/1.1396/1.3547|1.4964/1.5508|
|prefill|0.02|256|0.96094/0.039062|8/238/0/0|0|0.0020347/0.0032837/0.0037842|0.017834/0.047211/0.056641|0.3906/1.1662/1.3716|1.481/1.5361|
|prefill|0.05|256|0.94531/0.054688|7/235/0/0|0|0.0050745/0.0073242/0.0074463|0.01939/0.04903/0.059326|0.42413/1.1993/1.4392|1.4662/1.5234|
|prefill|0.1|256|0.91406/0.085938|4/230/0/0|0|0.010075/0.013745/0.014893|0.022079/0.05213/0.063843|0.48186/1.2743/1.5101|1.4365/1.4973|
|prefill|0.2|256|0.81641/0.18359|2/200/137/0|0|0.020363/0.026666/0.030273|0.028102/0.059662/0.07312|0.6113/1.4585/1.723|1.3426/1.4113|
|prefill|0.5|256|1/0|0/92/256/0|0|0.055666/0.069098/0.07312|0.053312/0.088422/0.10303|1.1527/2.1816/2.4527|1.5375/1.6404|
|prefill|1.0|256|1/0|0/11/256/0|0|0.15887/0.18539/0.19104|0.14502/0.18738/0.20166|3.1212/4.6868/5.223|1.5823/1.8081|

|mode|κ|B max-abs μ/p99/max|B rel-L2 μ/p99/max|B failures all/nonfallback (N)|B>10C NF rows|
|---|---:|---|---|---|---:|
|decode|0.01|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|0/0 (17)|0|
|decode|0.02|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|0/0 (34)|0|
|decode|0.05|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|0/0 (56)|0|
|decode|0.1|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|0/0 (101)|0|
|decode|0.2|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|0/0 (156)|0|
|decode|0.5|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|0/0 (0)|0|
|decode|1.0|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|0/0 (0)|0|
|prefill|0.01|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|0/0 (6)|0|
|prefill|0.02|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|0/0 (10)|0|
|prefill|0.05|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|0/0 (14)|0|
|prefill|0.1|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|0/0 (22)|0|
|prefill|0.2|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|0/0 (47)|0|
|prefill|0.5|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|0/0 (0)|0|
|prefill|1.0|5.2602e-08/3.9997e-07/5.7431e-07|1.3807e-08/1.0031e-07/1.3654e-07|0/0 (0)|0|

## Cost interpretation and numerical scope

Baseline = 2n + c·r. Single attempt = n + c·r_lo; successful correction rereads V once for each switched band key; fallback adds the full baseline. Reported real-main cost ratios all exceed 1; these are key-read accounting, not measured runtime. Prefill predictor reads are excluded from the requested primary cost model and included separately in `run/tables.json`. The speculative failed attempt is fully charged; no early-abort saving assumed.

R64 is the fp64 softmax of the identical fp32 mixed scores and captured fp32-upcast V; it is not dense exact attention. A/B use index-order fp32 online states; C uses 32 strided states merged in lane order. Correction updates numerator and denominator at the initial walk maximum, in band-key order. Fallback output is exactly A. CPU matmul/reduction orders and non-FMA NumPy operations do not certify bit identity to a CUDA kernel. The bf16 capture bits are exactly upcast; production-style bulk quantization is applied to fp32-upcast K and reconstructed in fp32 (per order/TX4), without a second bf16 rounding.

## Operand and capture census

Every captured Q/K/V file was read, shape/dtype/finite/nonzero/nonconstant checked, hashed and compared with its manifest. Check step rehashed all 38 inputs (36 Q/K/V arrays and two manifests). Q: (16,8192,128) for 1.7B and (32,8192,128) for 4B; K/V: (8,8192,128), all uint16 raw bf16. Source root:

`/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures/`

|model|captured layers|main/short/edge rows|shortest main (L,h,i;n)|longest main (L,h,i;n)|
|---|---|---|---|---|
|Qwen3-1.7B|0,5,11,16,22,27|1024/256/4|(11, 8, 2055); 2056|(22, 4, 8190); 8191|
|Qwen3-4B|0,7,14,21,28,35|1024/256/4|(14, 10, 2052); 2053|(7, 27, 8173); 8174|

Edges per model: (first layer,head0,n2048), (first layer,head0,n8192), (last layer,last head,n2048), (last layer,last head,n8192). Every drawn row was used without replacement within its class. Synthetic: 256 Gaussian Q/V rows at S8192,D128 with per-key lognormal-scaled K; independent Gaussian predecessor queries, labelled separately. Its operands are in `run/synthetic_inputs.npz`.

|model/relative path|shape|sha256|
|---|---|---|
|Qwen3-1.7B/manifest.json|manifest|7b9ba739504e49352934a6b7545c9a772a42457826745338be51b62d93452feb|
|Qwen3-1.7B/L00_q.npy|[16, 8192, 128]|59cfa3765dc58c821da6b1afeb05dc79d9b303426bc3c65ccf44356a2e7cf2a1|
|Qwen3-1.7B/L00_k.npy|[8, 8192, 128]|1a1015083d85c9a1ee0f3de3ff373ba60ee90983a2859a766ee8ecb701aa8c59|
|Qwen3-1.7B/L00_v.npy|[8, 8192, 128]|0d1b6333802e7d5e8c8bd6db9cc2189e7569203d6c5bf62287cbcc4caf67db6d|
|Qwen3-1.7B/L05_q.npy|[16, 8192, 128]|fbcd67e76932f9ff7da79982be0d61d18cd0dc6d8cd8a2c6ba6b2822da0f7fea|
|Qwen3-1.7B/L05_k.npy|[8, 8192, 128]|744cbc753deec884c30cb534270c99317b31e7fd1e43564851a5b4d60b075122|
|Qwen3-1.7B/L05_v.npy|[8, 8192, 128]|b73975a6bdc6e773d207ba33ec5c3dcbfe158453e28153cb45422999e444e164|
|Qwen3-1.7B/L11_q.npy|[16, 8192, 128]|b0f6caa26214344dd4c5ab294dc68facf0029c3b01f5165ffd771f301a5f6827|
|Qwen3-1.7B/L11_k.npy|[8, 8192, 128]|b2efc9e581dbfeb9dd747eb047e2bfbbcf315e09c805537d67548a77335eb0f3|
|Qwen3-1.7B/L11_v.npy|[8, 8192, 128]|3db445969a1c839ee2e25d9bf8ab978c07b2d9d7c1c50e51169d9d1f73b87d98|
|Qwen3-1.7B/L16_q.npy|[16, 8192, 128]|a8bc8caf454fd98b6904866e8563d1083aea2d491bf80da8543ef1cab768cce1|
|Qwen3-1.7B/L16_k.npy|[8, 8192, 128]|64225e3412cfb2bba729ee9cb8223252b381e564ca0b38f936ca8ef4feacde0b|
|Qwen3-1.7B/L16_v.npy|[8, 8192, 128]|3bd645d828e1d4e583d66ed154408ed81fe6e8fb5e40ec8bfe7c785ac8c23ba7|
|Qwen3-1.7B/L22_q.npy|[16, 8192, 128]|56acbfd24b4be0368ced84bb525ca548bb76ba1c9d184e359487c904bc84ffa8|
|Qwen3-1.7B/L22_k.npy|[8, 8192, 128]|f5ac6a3a8609d530b16d79fd014d6c298ffd3b9f037705577f093404bb40e173|
|Qwen3-1.7B/L22_v.npy|[8, 8192, 128]|f02501d9146d2d98cfb2932ee6b93e08bdb7c0552b3dcb403410ada93f5fd96d|
|Qwen3-1.7B/L27_q.npy|[16, 8192, 128]|0265d9dd5e5fec878183abc0d944f314f1789916de91e58d99fe800f6296054e|
|Qwen3-1.7B/L27_k.npy|[8, 8192, 128]|9347cc6836c1b73028d980f2b9d4531bdf9addc248112d935f1efdc9f723c404|
|Qwen3-1.7B/L27_v.npy|[8, 8192, 128]|2751977b555df0c1c7eaa2af54631fafb13d2303ee1c6143866f1ab4771444b4|
|Qwen3-4B/manifest.json|manifest|a6114aafe86948718f745c8f333d38c3a15ce704d4aed5a3e8f39ffeacabe93d|
|Qwen3-4B/L00_q.npy|[32, 8192, 128]|0474e2e34b63eacdb93e9c7b41e55b8dd14976172d6d21e66bfef687e175297b|
|Qwen3-4B/L00_k.npy|[8, 8192, 128]|0bb37985f5807f0d86b7d4e2fabe78215a3f4b8c424ea4bc59b90500a3121f37|
|Qwen3-4B/L00_v.npy|[8, 8192, 128]|4ee60750efb70dedad9e8c27521da236313418d703e7e7077a126ef878f0a3ce|
|Qwen3-4B/L07_q.npy|[32, 8192, 128]|7e740fd15b8f8e57516bd98a7a7183220169a667fa017aa96a1414ec383a6088|
|Qwen3-4B/L07_k.npy|[8, 8192, 128]|984189a6c68e0e527ac9a2fbadb98f27ca1466d3c43576b228a74fcb84de23c4|
|Qwen3-4B/L07_v.npy|[8, 8192, 128]|fd28bacd27fae8be6eaab721189fe84a2617ad1a16155a1fbd0130e2fbd17e36|
|Qwen3-4B/L14_q.npy|[32, 8192, 128]|e34fc30a1f23823d5254313de729245506d967cd01ee2b7401250fbc0afd4a53|
|Qwen3-4B/L14_k.npy|[8, 8192, 128]|0895394d1360080c7c9a71a1967d3e7eec5dc8f7e143909206412c8a591bb1a4|
|Qwen3-4B/L14_v.npy|[8, 8192, 128]|51cf5dbdd8197425ee26a9dbd924b4f6b5b3616d23d2e2ad3b966e68ee096dff|
|Qwen3-4B/L21_q.npy|[32, 8192, 128]|b82ab6014fda53b3e1610b56754ffd6d2fa03ca90ce52a6057ff1708e59539ef|
|Qwen3-4B/L21_k.npy|[8, 8192, 128]|c676ae9776c8da323525e48040ab28584aeedc3da2d13139c35a76162b4ed865|
|Qwen3-4B/L21_v.npy|[8, 8192, 128]|d021678192516a6c91503d5b79b70661f789a7154480faf325e117768d1e78a9|
|Qwen3-4B/L28_q.npy|[32, 8192, 128]|6af576270c86cf7d9e26dcefc856850cc037354f3a881a8852302d36de7aa8fb|
|Qwen3-4B/L28_k.npy|[8, 8192, 128]|50cb7944f42d03a689f12cb417365d81a116d42deedcf9e1c95fd11ea2776aea|
|Qwen3-4B/L28_v.npy|[8, 8192, 128]|fcadacbb92356879a3c5cd9e17fc9800e6b221361ce8ae1f2fe042757784ff7b|
|Qwen3-4B/L35_q.npy|[32, 8192, 128]|b8aee8ac20e393a1990eaebc7f9fff43a010c3a007221c3a0bcdb66bd7fe395e|
|Qwen3-4B/L35_k.npy|[8, 8192, 128]|a05e46f2ae02f0c4765f893a99c84613e82c8a84c4230f2a2948201bf7a0cccd|
|Qwen3-4B/L35_v.npy|[8, 8192, 128]|164245f2cfa42fce6a4ff5ace79f215085407dc6854dc3c4e94b82c356104feb|

Census witnesses below use row (layer,query-head,zero-based query); detailed thresholds/brackets/band caps and receipt paths are in `run/operand_census.json`.

|model/mode/event|row|n|κ|receipt|
|---|---|---:|---:|---|
|Qwen3-1.7B/decode/miss_low|(0, 1, 2137)|2138|0.01|Qwen3-1.7B_main_L00_kv0.npz|
|Qwen3-1.7B/decode/overflow|(0, 1, 2137)|2138|0.1|Qwen3-1.7B_main_L00_kv0.npz|
|Qwen3-1.7B/prefill/miss_high|(0, 1, 2137)|2138|0.01|Qwen3-1.7B_main_L00_kv0.npz|
|Qwen3-1.7B/prefill/overflow|(0, 1, 2137)|2138|0.05|Qwen3-1.7B_main_L00_kv0.npz|
|Qwen3-1.7B/prefill/miss_low|(0, 1, 2809)|2810|0.01|Qwen3-1.7B_main_L00_kv0.npz|
|Qwen3-1.7B/decode/miss_high|(0, 1, 5629)|5630|0.01|Qwen3-1.7B_main_L00_kv0.npz|
|Qwen3-1.7B/decode/zero_band|(0, 0, 1468)|1469|0.01|Qwen3-1.7B_short_L00_kv0.npz|
|Qwen3-1.7B/prefill/zero_band|(0, 2, 382)|383|0.01|Qwen3-1.7B_short_L00_kv1.npz|
|Qwen3-4B/decode/miss_low|(0, 2, 7021)|7022|0.01|Qwen3-4B_main_L00_kv0.npz|
|Qwen3-4B/decode/overflow|(0, 2, 7021)|7022|0.1|Qwen3-4B_main_L00_kv0.npz|
|Qwen3-4B/prefill/miss_high|(0, 2, 7021)|7022|0.01|Qwen3-4B_main_L00_kv0.npz|
|Qwen3-4B/prefill/overflow|(0, 2, 7021)|7022|0.1|Qwen3-4B_main_L00_kv0.npz|
|Qwen3-4B/decode/miss_high|(0, 2, 2319)|2320|0.01|Qwen3-4B_main_L00_kv0.npz|
|Qwen3-4B/prefill/miss_low|(0, 0, 3575)|3576|0.01|Qwen3-4B_main_L00_kv0.npz|
|Qwen3-4B/decode/zero_band|(0, 4, 3279)|3280|0.01|Qwen3-4B_main_L00_kv1.npz|
|Qwen3-4B/prefill/zero_band|(0, 5, 713)|714|0.01|Qwen3-4B_short_L00_kv1.npz|
|synthetic-heavy-tail/decode/miss_high|(0, 0, 8191)|8192|0.01|synthetic_heavytail_000.npz|
|synthetic-heavy-tail/decode/overflow|(0, 0, 8191)|8192|0.5|synthetic_heavytail_000.npz|
|synthetic-heavy-tail/prefill/miss_high|(0, 0, 8191)|8192|0.01|synthetic_heavytail_000.npz|
|synthetic-heavy-tail/prefill/overflow|(0, 0, 8191)|8192|0.2|synthetic_heavytail_000.npz|
|synthetic-heavy-tail/decode/miss_low|(0, 3, 8191)|8192|0.01|synthetic_heavytail_000.npz|
|synthetic-heavy-tail/prefill/miss_low|(0, 35, 8191)|8192|0.01|synthetic_heavytail_032.npz|
|synthetic-heavy-tail/decode/zero_band|(0, 106, 8191)|8192|0.01|synthetic_heavytail_096.npz|
|synthetic-heavy-tail/prefill/zero_band|(0, 168, 8191)|8192|0.01|synthetic_heavytail_160.npz|

Ties at thr and θ_lo and an integer-score correction-multiset identity were exercised before capture reads in `test_bracket_ties_and_miss_low_high_overflow_zero_band` and `test_correction_integer_score_multiset_and_two_sides`. Zero-band, low/high misses and overflow witnesses occur in both real models and in the synthetic family. No failed array loads and no replacement rows.

## Prior art

Bracket-and-band: Floyd and Rivest, SELECT (1975); GPU bucket select, Alabi et al. (2012). Unverified — lead to check those author/year/title search terms. Taken: threshold bracketing and bounded exceptional set; this seat implements the order’s CPU probe, with no novelty claim.

Online softmax/state merging: Milakov and Gimelshein (2018), FlashAttention, Dao et al. (2022). Unverified — lead to check. Taken: running maximum and rescaled numerator/denominator states. APA two-pass selector: house `apa_selective_kernel`, read locally. Subtract-and-add correction: no prior art known to me; identity supplied by X4 order.

Quantizer: production `_quantize_keys` and CPU table definitions; adapted from TX4-T1A `apa_monitor_probe.py`, declared commit 66818210 in the order. Lloyd (1982), Max (1960), TurboQuant (authors/year unverified — lead to check), QR rotation Mezzadri (2006/07), all unverified literature leads. Taken: fixed quantize/reconstruct and rotation formulas. NumPy PCG64 (2019 system): uniform draw without replacement. Lognormal norm fixture: no prior art known to me for this particular fixture. Standard population moments, norms and descriptive summaries are not original claims. Code-site and ledger annotations accompany these entries. No literature was checked over the network; commit identifiers are supplied by the order, not verified with git.

## Every file written

All files below are relative to the authorized worktree. No other paths were written, no /tmp scratch used, and no files deleted. `FILES_WRITTEN.txt` is the exhaustive inventory; `checked_hashes.json` hashes the run receipts.

```text
artifacts/x4p/AMENDMENTS.md
artifacts/x4p/FILES_WRITTEN.txt
artifacts/x4p/FINAL_REPORT.md
artifacts/x4p/LEDGER.md
artifacts/x4p/REGISTRATION.md
artifacts/x4p/check.log
artifacts/x4p/check_final.log
artifacts/x4p/check_final_receipt.json
artifacts/x4p/check_receipt.json
artifacts/x4p/check_x4p.py
artifacts/x4p/checked_hashes.json
artifacts/x4p/cli_receipt.json
artifacts/x4p/g1.log
artifacts/x4p/g1_receipt.json
artifacts/x4p/g2_receipt.json
artifacts/x4p/probe.log
artifacts/x4p/report_x4p.py
artifacts/x4p/row_metrics.npz
artifacts/x4p/run/D4_TABLES.md
artifacts/x4p/run/Qwen3-1.7B_edge_L00_kv0.npz
artifacts/x4p/run/Qwen3-1.7B_edge_L27_kv7.npz
artifacts/x4p/run/Qwen3-1.7B_main_L00_kv0.npz
artifacts/x4p/run/Qwen3-1.7B_main_L00_kv1.npz
artifacts/x4p/run/Qwen3-1.7B_main_L00_kv2.npz
artifacts/x4p/run/Qwen3-1.7B_main_L00_kv3.npz
artifacts/x4p/run/Qwen3-1.7B_main_L00_kv4.npz
artifacts/x4p/run/Qwen3-1.7B_main_L00_kv5.npz
artifacts/x4p/run/Qwen3-1.7B_main_L00_kv6.npz
artifacts/x4p/run/Qwen3-1.7B_main_L00_kv7.npz
artifacts/x4p/run/Qwen3-1.7B_main_L05_kv0.npz
artifacts/x4p/run/Qwen3-1.7B_main_L05_kv1.npz
artifacts/x4p/run/Qwen3-1.7B_main_L05_kv2.npz
artifacts/x4p/run/Qwen3-1.7B_main_L05_kv3.npz
artifacts/x4p/run/Qwen3-1.7B_main_L05_kv4.npz
artifacts/x4p/run/Qwen3-1.7B_main_L05_kv5.npz
artifacts/x4p/run/Qwen3-1.7B_main_L05_kv6.npz
artifacts/x4p/run/Qwen3-1.7B_main_L05_kv7.npz
artifacts/x4p/run/Qwen3-1.7B_main_L11_kv0.npz
artifacts/x4p/run/Qwen3-1.7B_main_L11_kv1.npz
artifacts/x4p/run/Qwen3-1.7B_main_L11_kv2.npz
artifacts/x4p/run/Qwen3-1.7B_main_L11_kv3.npz
artifacts/x4p/run/Qwen3-1.7B_main_L11_kv4.npz
artifacts/x4p/run/Qwen3-1.7B_main_L11_kv5.npz
artifacts/x4p/run/Qwen3-1.7B_main_L11_kv6.npz
artifacts/x4p/run/Qwen3-1.7B_main_L11_kv7.npz
artifacts/x4p/run/Qwen3-1.7B_main_L16_kv0.npz
artifacts/x4p/run/Qwen3-1.7B_main_L16_kv1.npz
artifacts/x4p/run/Qwen3-1.7B_main_L16_kv2.npz
artifacts/x4p/run/Qwen3-1.7B_main_L16_kv3.npz
artifacts/x4p/run/Qwen3-1.7B_main_L16_kv4.npz
artifacts/x4p/run/Qwen3-1.7B_main_L16_kv5.npz
artifacts/x4p/run/Qwen3-1.7B_main_L16_kv6.npz
artifacts/x4p/run/Qwen3-1.7B_main_L16_kv7.npz
artifacts/x4p/run/Qwen3-1.7B_main_L22_kv0.npz
artifacts/x4p/run/Qwen3-1.7B_main_L22_kv1.npz
artifacts/x4p/run/Qwen3-1.7B_main_L22_kv2.npz
artifacts/x4p/run/Qwen3-1.7B_main_L22_kv3.npz
artifacts/x4p/run/Qwen3-1.7B_main_L22_kv4.npz
artifacts/x4p/run/Qwen3-1.7B_main_L22_kv5.npz
artifacts/x4p/run/Qwen3-1.7B_main_L22_kv6.npz
artifacts/x4p/run/Qwen3-1.7B_main_L22_kv7.npz
artifacts/x4p/run/Qwen3-1.7B_main_L27_kv0.npz
artifacts/x4p/run/Qwen3-1.7B_main_L27_kv1.npz
artifacts/x4p/run/Qwen3-1.7B_main_L27_kv2.npz
artifacts/x4p/run/Qwen3-1.7B_main_L27_kv3.npz
artifacts/x4p/run/Qwen3-1.7B_main_L27_kv4.npz
artifacts/x4p/run/Qwen3-1.7B_main_L27_kv5.npz
artifacts/x4p/run/Qwen3-1.7B_main_L27_kv6.npz
artifacts/x4p/run/Qwen3-1.7B_main_L27_kv7.npz
artifacts/x4p/run/Qwen3-1.7B_short_L00_kv0.npz
artifacts/x4p/run/Qwen3-1.7B_short_L00_kv1.npz
artifacts/x4p/run/Qwen3-1.7B_short_L00_kv2.npz
artifacts/x4p/run/Qwen3-1.7B_short_L00_kv3.npz
artifacts/x4p/run/Qwen3-1.7B_short_L00_kv4.npz
artifacts/x4p/run/Qwen3-1.7B_short_L00_kv5.npz
artifacts/x4p/run/Qwen3-1.7B_short_L00_kv6.npz
artifacts/x4p/run/Qwen3-1.7B_short_L00_kv7.npz
artifacts/x4p/run/Qwen3-1.7B_short_L05_kv0.npz
artifacts/x4p/run/Qwen3-1.7B_short_L05_kv1.npz
artifacts/x4p/run/Qwen3-1.7B_short_L05_kv2.npz
artifacts/x4p/run/Qwen3-1.7B_short_L05_kv3.npz
artifacts/x4p/run/Qwen3-1.7B_short_L05_kv4.npz
artifacts/x4p/run/Qwen3-1.7B_short_L05_kv6.npz
artifacts/x4p/run/Qwen3-1.7B_short_L05_kv7.npz
artifacts/x4p/run/Qwen3-1.7B_short_L11_kv0.npz
artifacts/x4p/run/Qwen3-1.7B_short_L11_kv1.npz
artifacts/x4p/run/Qwen3-1.7B_short_L11_kv2.npz
artifacts/x4p/run/Qwen3-1.7B_short_L11_kv3.npz
artifacts/x4p/run/Qwen3-1.7B_short_L11_kv4.npz
artifacts/x4p/run/Qwen3-1.7B_short_L11_kv5.npz
artifacts/x4p/run/Qwen3-1.7B_short_L11_kv6.npz
artifacts/x4p/run/Qwen3-1.7B_short_L11_kv7.npz
artifacts/x4p/run/Qwen3-1.7B_short_L16_kv0.npz
artifacts/x4p/run/Qwen3-1.7B_short_L16_kv1.npz
artifacts/x4p/run/Qwen3-1.7B_short_L16_kv2.npz
artifacts/x4p/run/Qwen3-1.7B_short_L16_kv3.npz
artifacts/x4p/run/Qwen3-1.7B_short_L16_kv4.npz
artifacts/x4p/run/Qwen3-1.7B_short_L16_kv5.npz
artifacts/x4p/run/Qwen3-1.7B_short_L16_kv6.npz
artifacts/x4p/run/Qwen3-1.7B_short_L16_kv7.npz
artifacts/x4p/run/Qwen3-1.7B_short_L22_kv0.npz
artifacts/x4p/run/Qwen3-1.7B_short_L22_kv1.npz
artifacts/x4p/run/Qwen3-1.7B_short_L22_kv2.npz
artifacts/x4p/run/Qwen3-1.7B_short_L22_kv3.npz
artifacts/x4p/run/Qwen3-1.7B_short_L22_kv4.npz
artifacts/x4p/run/Qwen3-1.7B_short_L22_kv5.npz
artifacts/x4p/run/Qwen3-1.7B_short_L22_kv6.npz
artifacts/x4p/run/Qwen3-1.7B_short_L22_kv7.npz
artifacts/x4p/run/Qwen3-1.7B_short_L27_kv0.npz
artifacts/x4p/run/Qwen3-1.7B_short_L27_kv1.npz
artifacts/x4p/run/Qwen3-1.7B_short_L27_kv2.npz
artifacts/x4p/run/Qwen3-1.7B_short_L27_kv3.npz
artifacts/x4p/run/Qwen3-1.7B_short_L27_kv4.npz
artifacts/x4p/run/Qwen3-1.7B_short_L27_kv5.npz
artifacts/x4p/run/Qwen3-1.7B_short_L27_kv6.npz
artifacts/x4p/run/Qwen3-1.7B_short_L27_kv7.npz
artifacts/x4p/run/Qwen3-4B_edge_L00_kv0.npz
artifacts/x4p/run/Qwen3-4B_edge_L35_kv7.npz
artifacts/x4p/run/Qwen3-4B_main_L00_kv0.npz
artifacts/x4p/run/Qwen3-4B_main_L00_kv1.npz
artifacts/x4p/run/Qwen3-4B_main_L00_kv2.npz
artifacts/x4p/run/Qwen3-4B_main_L00_kv3.npz
artifacts/x4p/run/Qwen3-4B_main_L00_kv4.npz
artifacts/x4p/run/Qwen3-4B_main_L00_kv5.npz
artifacts/x4p/run/Qwen3-4B_main_L00_kv6.npz
artifacts/x4p/run/Qwen3-4B_main_L00_kv7.npz
artifacts/x4p/run/Qwen3-4B_main_L07_kv0.npz
artifacts/x4p/run/Qwen3-4B_main_L07_kv1.npz
artifacts/x4p/run/Qwen3-4B_main_L07_kv2.npz
artifacts/x4p/run/Qwen3-4B_main_L07_kv3.npz
artifacts/x4p/run/Qwen3-4B_main_L07_kv4.npz
artifacts/x4p/run/Qwen3-4B_main_L07_kv5.npz
artifacts/x4p/run/Qwen3-4B_main_L07_kv6.npz
artifacts/x4p/run/Qwen3-4B_main_L07_kv7.npz
artifacts/x4p/run/Qwen3-4B_main_L14_kv0.npz
artifacts/x4p/run/Qwen3-4B_main_L14_kv1.npz
artifacts/x4p/run/Qwen3-4B_main_L14_kv2.npz
artifacts/x4p/run/Qwen3-4B_main_L14_kv3.npz
artifacts/x4p/run/Qwen3-4B_main_L14_kv4.npz
artifacts/x4p/run/Qwen3-4B_main_L14_kv5.npz
artifacts/x4p/run/Qwen3-4B_main_L14_kv6.npz
artifacts/x4p/run/Qwen3-4B_main_L14_kv7.npz
artifacts/x4p/run/Qwen3-4B_main_L21_kv0.npz
artifacts/x4p/run/Qwen3-4B_main_L21_kv1.npz
artifacts/x4p/run/Qwen3-4B_main_L21_kv2.npz
artifacts/x4p/run/Qwen3-4B_main_L21_kv3.npz
artifacts/x4p/run/Qwen3-4B_main_L21_kv4.npz
artifacts/x4p/run/Qwen3-4B_main_L21_kv5.npz
artifacts/x4p/run/Qwen3-4B_main_L21_kv6.npz
artifacts/x4p/run/Qwen3-4B_main_L21_kv7.npz
artifacts/x4p/run/Qwen3-4B_main_L28_kv0.npz
artifacts/x4p/run/Qwen3-4B_main_L28_kv1.npz
artifacts/x4p/run/Qwen3-4B_main_L28_kv2.npz
artifacts/x4p/run/Qwen3-4B_main_L28_kv3.npz
artifacts/x4p/run/Qwen3-4B_main_L28_kv4.npz
artifacts/x4p/run/Qwen3-4B_main_L28_kv5.npz
artifacts/x4p/run/Qwen3-4B_main_L28_kv6.npz
artifacts/x4p/run/Qwen3-4B_main_L28_kv7.npz
artifacts/x4p/run/Qwen3-4B_main_L35_kv0.npz
artifacts/x4p/run/Qwen3-4B_main_L35_kv1.npz
artifacts/x4p/run/Qwen3-4B_main_L35_kv2.npz
artifacts/x4p/run/Qwen3-4B_main_L35_kv3.npz
artifacts/x4p/run/Qwen3-4B_main_L35_kv4.npz
artifacts/x4p/run/Qwen3-4B_main_L35_kv5.npz
artifacts/x4p/run/Qwen3-4B_main_L35_kv6.npz
artifacts/x4p/run/Qwen3-4B_main_L35_kv7.npz
artifacts/x4p/run/Qwen3-4B_short_L00_kv0.npz
artifacts/x4p/run/Qwen3-4B_short_L00_kv1.npz
artifacts/x4p/run/Qwen3-4B_short_L00_kv2.npz
artifacts/x4p/run/Qwen3-4B_short_L00_kv3.npz
artifacts/x4p/run/Qwen3-4B_short_L00_kv4.npz
artifacts/x4p/run/Qwen3-4B_short_L00_kv5.npz
artifacts/x4p/run/Qwen3-4B_short_L00_kv6.npz
artifacts/x4p/run/Qwen3-4B_short_L00_kv7.npz
artifacts/x4p/run/Qwen3-4B_short_L07_kv0.npz
artifacts/x4p/run/Qwen3-4B_short_L07_kv1.npz
artifacts/x4p/run/Qwen3-4B_short_L07_kv2.npz
artifacts/x4p/run/Qwen3-4B_short_L07_kv3.npz
artifacts/x4p/run/Qwen3-4B_short_L07_kv4.npz
artifacts/x4p/run/Qwen3-4B_short_L07_kv5.npz
artifacts/x4p/run/Qwen3-4B_short_L07_kv6.npz
artifacts/x4p/run/Qwen3-4B_short_L07_kv7.npz
artifacts/x4p/run/Qwen3-4B_short_L14_kv0.npz
artifacts/x4p/run/Qwen3-4B_short_L14_kv1.npz
artifacts/x4p/run/Qwen3-4B_short_L14_kv2.npz
artifacts/x4p/run/Qwen3-4B_short_L14_kv3.npz
artifacts/x4p/run/Qwen3-4B_short_L14_kv4.npz
artifacts/x4p/run/Qwen3-4B_short_L14_kv5.npz
artifacts/x4p/run/Qwen3-4B_short_L14_kv6.npz
artifacts/x4p/run/Qwen3-4B_short_L14_kv7.npz
artifacts/x4p/run/Qwen3-4B_short_L21_kv0.npz
artifacts/x4p/run/Qwen3-4B_short_L21_kv1.npz
artifacts/x4p/run/Qwen3-4B_short_L21_kv2.npz
artifacts/x4p/run/Qwen3-4B_short_L21_kv3.npz
artifacts/x4p/run/Qwen3-4B_short_L21_kv4.npz
artifacts/x4p/run/Qwen3-4B_short_L21_kv5.npz
artifacts/x4p/run/Qwen3-4B_short_L21_kv6.npz
artifacts/x4p/run/Qwen3-4B_short_L21_kv7.npz
artifacts/x4p/run/Qwen3-4B_short_L28_kv0.npz
artifacts/x4p/run/Qwen3-4B_short_L28_kv1.npz
artifacts/x4p/run/Qwen3-4B_short_L28_kv2.npz
artifacts/x4p/run/Qwen3-4B_short_L28_kv3.npz
artifacts/x4p/run/Qwen3-4B_short_L28_kv4.npz
artifacts/x4p/run/Qwen3-4B_short_L28_kv5.npz
artifacts/x4p/run/Qwen3-4B_short_L28_kv6.npz
artifacts/x4p/run/Qwen3-4B_short_L28_kv7.npz
artifacts/x4p/run/Qwen3-4B_short_L35_kv0.npz
artifacts/x4p/run/Qwen3-4B_short_L35_kv1.npz
artifacts/x4p/run/Qwen3-4B_short_L35_kv2.npz
artifacts/x4p/run/Qwen3-4B_short_L35_kv3.npz
artifacts/x4p/run/Qwen3-4B_short_L35_kv4.npz
artifacts/x4p/run/Qwen3-4B_short_L35_kv5.npz
artifacts/x4p/run/Qwen3-4B_short_L35_kv6.npz
artifacts/x4p/run/Qwen3-4B_short_L35_kv7.npz
artifacts/x4p/run/capture_census.json
artifacts/x4p/run/draws.json
artifacts/x4p/run/groups.json
artifacts/x4p/run/operand_census.json
artifacts/x4p/run/predictions.json
artifacts/x4p/run/quantizer_tables.npz
artifacts/x4p/run/run.json
artifacts/x4p/run/synthetic_heavytail_000.npz
artifacts/x4p/run/synthetic_heavytail_032.npz
artifacts/x4p/run/synthetic_heavytail_064.npz
artifacts/x4p/run/synthetic_heavytail_096.npz
artifacts/x4p/run/synthetic_heavytail_128.npz
artifacts/x4p/run/synthetic_heavytail_160.npz
artifacts/x4p/run/synthetic_heavytail_192.npz
artifacts/x4p/run/synthetic_heavytail_224.npz
artifacts/x4p/run/synthetic_inputs.npz
artifacts/x4p/run/tables.json
artifacts/x4p/run/verdicts.json
artifacts/x4p/test_x4p.py
artifacts/x4p/x4_probe.py
```
