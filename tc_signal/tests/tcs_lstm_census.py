"""tc_signal.lstm own-test census and gates (P1T-M3). House tunable law: every number the LSTM tests use lives here.

Gate (order P1T-M3, G1, registered in docs/P1T_M3_LEDGER.md 20:33 before any run): for EACH output Y, Y_h, Y_c,
normwise N = ||y - o||_inf / ||o||_inf <= 2e-5 against the float64 NumPy oracle (tcs_lstm_oracle.lstm_onnx_f64); every
element finite (outputs and scratch come from a NaN-prefilled allocator); two calls bitwise equal (the second call's
allocator pre-fills a different value, so the result cannot depend on the scratch's prior contents).

Cells:
 * Edge cells (order, Operand census): forward, reverse and bidirectional; T 1 and 2; batch 2 and 3; non-zero
   initial_h and initial_c (short T, where the initial state dominates the output); B None; H not a multiple of 32
   (20) and H 1; input size 1; large pre-activations (|z| up to ~60) for sigmoid/tanh saturation.
 * Kokoro-shape synthetic cells: H 256, bidirectional, batch 1, I 640 at T 82 and 1193, I 512 at T 505 (the real
   nodes' shapes; random weights with the real weights' scale, ledger 20:33).
 * Real fixture cells: the 12 T4 cases of Voice's fixtures (6 nodes x {short_07, max_1}), real x, W, R, B, against MY
   oracle (Voice's runner, G2, applies Voice's oracle to the same data).

Synthetic scales, matched to the real Kokoro LSTMs (ledger 20:33, fixture census): W std 0.082-0.097 (~2.1/sqrt(640)),
R std 0.104-0.146 (~1.7/sqrt(256)), B std 0.066-0.092, x std 0.32-1.34; real max |z| 12-28, real max |c| up to 367.

Prior art: none needed for a constants module; the census itself is ours (no prior art known to me).
"""

GATE_N = 2e-5                 # order G1 (= tolerances_r2.json T4 gate)
ORACLE_XCHECK_N = 1e-12       # own NumPy oracle vs torch fp64 nn.LSTM (gate order converted), normwise per output
ORACLE_BLAS_THREADS = 1       # BLAS threads while the NumPy oracle runs (threadpoolctl). Its per-step [N, I] x [I, 4H]
                              # products are tiny; a 12-thread OpenBLAS pool on a loaded host took 7.8 ms per product
                              # vs 82 us single-threaded (ledger 20:47). Speed only: fp64 values are unaffected.
SEED = 20261012               # NumPy Generator seed base for every cell's data (plus a crc32 of the cell tag)
NAN_FILL = float('nan')       # first call: the allocator pre-fills outputs and scratch with NaN (a skipped write shows)
SECOND_FILL = 12345.0         # second call: a different pre-fill, so determinism also proves scratch independence

# Synthetic data scales (normal distributions).
W_GAIN = 2.0                  # W ~ N(0, 1) * W_GAIN / sqrt(I)
R_GAIN = 1.7                  # R ~ N(0, 1) * R_GAIN / sqrt(H)
B_STD = 0.08                  # B ~ N(0, B_STD)
X_STD = 0.5                   # x ~ N(0, X_STD)
STATE_STD = 1.0               # initial_h, initial_c ~ N(0, STATE_STD) for cells with an initial state

# Large pre-activation cells ('large_z'): bigger weights and every LARGE_Z_PIN_EVERY-th bias entry pinned at
# +-LARGE_Z_BIAS (alternating sign), so gate pre-activations reach |z| ~ 60.
LARGE_W_GAIN = 15.0
LARGE_R_GAIN = 5.0
LARGE_B_STD = 5.0
LARGE_Z_BIAS = 58.0
LARGE_Z_PIN_EVERY = 7
LARGE_Z_MIN_REACHED = 50.0    # census property: the fp64 oracle's max |z| must reach this, else the cell is not testing
                              # saturation (asserted per large_z cell)

# cell: (name, direction, T, N, I, H, bias, init_state, kind)   kind: 'normal' or 'large_z'
EDGE_CELLS = [
    ('fwd_T1_N1',            'forward',       1,  1,  8, 16, True,  False, 'normal'),
    ('rev_T1_N2',            'reverse',       1,  2,  8, 16, True,  False, 'normal'),
    ('bi_T1_N3',             'bidirectional', 1,  3,  8, 16, True,  False, 'normal'),
    ('fwd_T2_N2',            'forward',       2,  2,  5, 12, True,  False, 'normal'),
    ('rev_T2_N3',            'reverse',       2,  3,  5, 12, True,  False, 'normal'),
    ('bi_T2_N2',             'bidirectional', 2,  2,  5, 12, True,  False, 'normal'),
    ('fwd_T17_N2',           'forward',      17,  2, 10, 32, True,  False, 'normal'),
    ('rev_T17_N3',           'reverse',      17,  3, 10, 32, True,  False, 'normal'),
    ('bi_T33_N2',            'bidirectional', 33, 2, 24, 48, True,  False, 'normal'),
    ('fwd_init_T1_N2',       'forward',       1,  2,  6, 16, True,  True,  'normal'),
    ('rev_init_T3_N3',       'reverse',       3,  3,  6, 16, True,  True,  'normal'),
    ('bi_init_T4_N2',        'bidirectional', 4,  2,  6, 16, True,  True,  'normal'),
    ('bi_init_T2_N3_H20',    'bidirectional', 2,  3,  6, 20, True,  True,  'normal'),
    ('bi_init_T40_N3_H20',   'bidirectional', 40, 3, 12, 20, True,  True,  'normal'),
    ('fwd_noB_T12_N2',       'forward',      12,  2,  8, 16, False, False, 'normal'),
    ('bi_noB_init_T9_N3',    'bidirectional', 9,  3,  8, 16, False, True,  'normal'),
    ('bi_H20_T25_N2',        'bidirectional', 25, 2, 12, 20, True,  False, 'normal'),
    ('rev_H20_init_T25_N3',  'reverse',      25,  3, 12, 20, True,  True,  'normal'),
    ('fwd_I1_T30',           'forward',      30,  1,  1, 16, True,  False, 'normal'),
    ('bi_I1_H20_init_T30_N3', 'bidirectional', 30, 3, 1, 20, True,  True,  'normal'),
    ('bi_H1_init_T6_N2',     'bidirectional', 6,  2,  3,  1, True,  True,  'normal'),
    ('bi_large_z_T12_N2',    'bidirectional', 12, 2, 16, 24, True,  False, 'large_z'),
    ('fwd_large_z_init_T3_N3', 'forward',     3,  3, 16, 20, True,  True,  'large_z'),
    ('rev_large_z_T20_N1',   'reverse',      20,  1,  8, 40, True,  False, 'large_z'),
]

KOKORO_SHAPE_CELLS = [
    ('kokoro_I640_T82',      'bidirectional', 82,   1, 640, 256, True, False, 'normal'),   # /shared, short_07 length
    ('kokoro_I640_T1193',    'bidirectional', 1193, 1, 640, 256, True, False, 'normal'),   # /shared, max_1 length
    ('kokoro_I512_T505',     'bidirectional', 505,  1, 512, 256, True, False, 'normal'),   # text_encoder_1 lstm
]

# Cells the oracle cross-check (NumPy fp64 vs torch fp64 nn.LSTM) runs on: every edge cell plus the T 82 Kokoro shape.
XCHECK_KOKORO = ('kokoro_I640_T82',)

# Voice's fixtures (read-only), for the real-data cells. The manifest hash is the one Voice's runner checks.
FIXTURES_DIR = '/mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures'
FIXTURES_MANIFEST_SHA256 = '89499fa537d109a1b3a616f88dfec1423fcb586ac26f0f94687fb4b8cea8639f'
FIXTURES_T4_CASES = 12        # 6 LSTM nodes x {short_07, max_1}

# Transcendental probe (my addition; registered in the ledger 20:33 before any run). One call, T 1, N 1, I 1, x = 0,
# W = 0, R = 0, initial state 0, so z = Wb exactly and Y_c = sigmoid(z_i) * tanh(z_c), Y_h = sigmoid(z_o) * tanh(Y_c).
PROBE_SIGMOID_Z = (-60.0, 60.0, 481)      # (lo, hi, points): sigmoid sweep on z_i (z_c = PROBE_TANH_ONE_Z)
PROBE_TANH_Z = (-12.0, 12.0, 481)         # tanh sweep on z_c (z_i = PROBE_SIGMOID_ONE_Z)
PROBE_SIGMOID_ONE_Z = 60.0                # sigmoid(60) rounds to exactly 1.0f
PROBE_TANH_ONE_Z = 20.0                   # tanh(20) rounds to exactly 1.0f
PROBE_SIGMOID_MAX_ULP = 8.0               # expf 2 ulp + add/divide 0.5 ulp each -> <= 6 ulp worst case (honest)
PROBE_TANH_MAX_ULP = 4.0                  # tanhf 2 ulp (CUDA Math API documented maximum)

# Refusal / contract cell (small, cheap): direction bidirectional, T, N, I, H.
CONTRACT_SHAPE = (5, 2, 3, 4)
