"""tc_signal P1T-M4 own-test census and gates (T5 norms, T6 small ops). House tunable law: every number the M4 tests
use lives here.

Gates (order P1T-M4, G1; registered by the order before any run):
 * norms: normwise N = ||y - o||_inf / ||o||_inf <= 1e-5 against a float64 CPU oracle computed in the same process
   (numpy two-pass), on every cell; every element written (NaN-prefilled allocator, outputs must be finite);
 * exact ops (upsample, reflection pad, rint): fp32 bit patterns equal the oracle's (uint32 view), NaN equal to NaN
   (the two gathers are additionally checked payload-exact: they copy 32-bit words);
 * determinism: two calls bitwise equal; every refusal raises the documented error.

Cells:
 * Kokoro (fixtures manifest /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures/manifest.json, read 2026-10-12): the
   real shapes of every InstanceNormalization node (short_07 and max_1 lengths; noise_res.1 at its full generator
   length 143,161, which the fixtures crop to 65,536), every LayerNormalization node (axis -1, eps fp32(1e-12) for
   bert/albert, fp32(1e-5) for text_encoder), the synthetic GroupNorm set ([1, 256, 3000], groups 1, 8, 256), every
   Resize (x2, x300), the reflection pad (1, 0) and the Round shapes. Data are random with activation-like
   per-row offsets and scales (the real data run in G2, Voice's runner on the fixtures).
 * Edge cells (order, Operand census): T = 1 instance norm (variance 0, output = bias); constant rows; large-offset rows
   (mean 1e4, std 1e-2); LayerNorm over two axes (axis -2); scale and bias None; rint at +-0.5, +-1.5, +-2.5, -0.4,
   +-Inf, NaN, the 2^23 + 0.5 neighbours and subnormals; pads (0, 0), (2, 3), (T-1, T-1); upsample x1 and x7; N = 2
   everywhere. Plus: eps-sensitive LayerNorm rows (std 1e-4, eps 1e-12), positive axes, rank 1/4, L = 1 rows.

Prior art: none needed for a constants module; the census itself is ours (no prior art known to me).
"""
import math

GATE_N = 1e-5                  # order G1 (= Voice tolerances.json gates.fp32_paths.max)
SEED = 20261012                # numpy default_rng base seed (per-cell seed = SEED + crc32(cell name) % SEED_MOD)
SEED_MOD = 1_000_003
# Oracle self-check (numpy two-pass fp64 vs torch fp64 norms). Not a gate on tc_signal. 1e-9, not 1e-12: on the
# large-offset rows (mean 1e4, std 1e-2) fp64 itself carries ~eps64 * |mean| / std ~ 2e-10 normwise (seen: 6.9e-11).
ORACLE_CROSSCHECK_N = 1e-9
NAN_FILL = float('nan')        # norm outputs are pre-filled with NaN: a skipped write shows up as non-finite
EXACT_FILL = -1.2345678e38     # exact-op outputs are pre-filled with a finite sentinel never present in the inputs
                               # (a NaN fill could hide a skipped write under NaN == NaN)

# Activation-like data: x[row] = offset_row + spread_row * N(0, 1), offset ~ N(0, OFFSET_SD), log spread ~ N(0, LOG_SD).
OFFSET_SD = 2.0
LOG_SPREAD_SD = 1.0
# Large-offset rows (order): mean 1e4, std 1e-2.
BIG_OFFSET = 1e4
BIG_OFFSET_STD = 1e-2
# eps-sensitive LayerNorm rows: std 1e-4 around 0.5, normalized with eps 1e-12 (an eps of 1e-5 would dominate var).
SMALL_VAR_MEAN = 0.5
SMALL_VAR_STD = 1e-4

EPS_IN = 9.999999747378752e-06             # fp32(1e-5), exactly as the fixtures manifest records Kokoro's eps
EPS_LN_TEXT = EPS_IN                       # text_encoder LayerNormalization
EPS_LN_BERT = 9.999999960041972e-13        # fp32(1e-12), bert/albert LayerNormalization (manifest value)

# ------------------------------------------------------------------------------------------------ norms
# norm cell: (name, kind, op, x_shape, arg, eps, data, affine)
#   op: 'instance' | 'layer' | 'group'; arg: None | axis | num_groups
#   data: 'act' | 'const' | 'offset' | 'neg_offset' | 'small_var'
#   affine: 'both' | 'none' | 'scale' | 'bias'
NORM_KOKORO = [
    ('in_F0.0_short',          'instance', (1, 512, 82), None, EPS_IN, 'act', 'both'),
    ('in_F0.0_long',           'instance', (1, 512, 1193), None, EPS_IN, 'act', 'both'),
    ('in_F0.1_short',          'instance', (1, 256, 164), None, EPS_IN, 'act', 'both'),
    ('in_F0.1_long',           'instance', (1, 256, 2386), None, EPS_IN, 'act', 'both'),
    ('in_encode1_short',       'instance', (1, 514, 82), None, EPS_IN, 'act', 'both'),
    ('in_encode1_long',        'instance', (1, 514, 1193), None, EPS_IN, 'act', 'both'),
    ('in_encode2_short',       'instance', (1, 1024, 82), None, EPS_IN, 'act', 'both'),
    ('in_encode2_long',        'instance', (1, 1024, 1193), None, EPS_IN, 'act', 'both'),
    ('in_decode0_short',       'instance', (1, 1090, 82), None, EPS_IN, 'act', 'both'),
    ('in_decode0_long',        'instance', (1, 1090, 1193), None, EPS_IN, 'act', 'both'),
    ('in_decode3_short',       'instance', (1, 512, 164), None, EPS_IN, 'act', 'both'),
    ('in_decode3_long',        'instance', (1, 512, 2386), None, EPS_IN, 'act', 'both'),
    ('in_noise_res0_short',    'instance', (1, 256, 1640), None, EPS_IN, 'act', 'both'),
    ('in_noise_res0_long',     'instance', (1, 256, 23860), None, EPS_IN, 'act', 'both'),
    ('in_noise_res1_short',    'instance', (1, 128, 9841), None, EPS_IN, 'act', 'both'),
    ('in_noise_res1_full',     'instance', (1, 128, 143161), None, EPS_IN, 'act', 'both'),
    ('ln_bert_emb_short',      'layer', (1, 30, 128), -1, EPS_LN_BERT, 'act', 'both'),
    ('ln_bert_emb_long',       'layer', (1, 505, 128), -1, EPS_LN_BERT, 'act', 'both'),
    ('ln_albert_attn_short',   'layer', (1, 30, 768), -1, EPS_LN_BERT, 'act', 'both'),
    ('ln_albert_attn_long',    'layer', (1, 505, 768), -1, EPS_LN_BERT, 'act', 'both'),
    ('ln_text_enc_short',      'layer', (1, 30, 512), -1, EPS_LN_TEXT, 'act', 'both'),
    ('ln_text_enc_long',       'layer', (1, 505, 512), -1, EPS_LN_TEXT, 'act', 'both'),
    ('gn_g1',                  'group', (1, 256, 3000), 1, 1e-5, 'act', 'both'),
    ('gn_g8',                  'group', (1, 256, 3000), 8, 1e-5, 'act', 'both'),
    ('gn_g256',                'group', (1, 256, 3000), 256, 1e-5, 'act', 'both'),
]

NORM_EDGE = [
    ('in_T1_N2',               'instance', (2, 5, 1), None, EPS_IN, 'act', 'both'),
    ('in_T1_no_bias',          'instance', (2, 5, 1), None, EPS_IN, 'act', 'scale'),
    ('in_const_rows_N2',       'instance', (2, 8, 100), None, EPS_IN, 'const', 'both'),
    ('in_offset_N2',           'instance', (2, 8, 1193), None, EPS_IN, 'offset', 'both'),
    ('in_offset_long',         'instance', (1, 4, 65536), None, EPS_IN, 'offset', 'both'),
    ('in_neg_offset_N2',       'instance', (2, 4, 3000), None, EPS_IN, 'neg_offset', 'both'),
    ('in_no_affine_N2',        'instance', (2, 16, 300), None, EPS_IN, 'act', 'none'),
    ('in_scale_only_N2',       'instance', (2, 16, 300), None, EPS_IN, 'act', 'scale'),
    ('in_bias_only_N2',        'instance', (2, 16, 300), None, EPS_IN, 'act', 'bias'),
    ('in_N2_C33_T777',         'instance', (2, 33, 777), None, EPS_IN, 'act', 'both'),
    ('ln_axis-2_N2',           'layer', (2, 30, 64), -2, EPS_LN_TEXT, 'act', 'both'),
    ('ln_axis1_pos_N2',        'layer', (2, 30, 64), 1, EPS_LN_TEXT, 'act', 'both'),
    ('ln_axis0_all',           'layer', (3, 4, 50), 0, EPS_LN_TEXT, 'act', 'both'),
    ('ln_rank1',               'layer', (768,), -1, EPS_LN_BERT, 'act', 'both'),
    ('ln_rank4_axis2_N2',      'layer', (2, 3, 4, 50), 2, EPS_LN_TEXT, 'act', 'both'),
    ('ln_no_affine_N2',        'layer', (2, 30, 512), -1, EPS_LN_TEXT, 'act', 'none'),
    ('ln_bias_none_N2',        'layer', (2, 30, 512), -1, EPS_LN_TEXT, 'act', 'scale'),
    ('ln_offset_N2',           'layer', (2, 30, 768), -1, EPS_LN_BERT, 'offset', 'both'),
    ('ln_const_rows_N2',       'layer', (2, 30, 768), -1, EPS_LN_BERT, 'const', 'both'),
    ('ln_small_var_eps1e-12',  'layer', (2, 30, 768), -1, EPS_LN_BERT, 'small_var', 'both'),
    ('ln_L1_N2',               'layer', (2, 7, 1), -1, EPS_LN_TEXT, 'act', 'both'),
    ('gn_g4_N2',               'group', (2, 64, 500), 4, 1e-5, 'act', 'both'),
    ('gn_offset_g8_N2',        'group', (2, 32, 500), 8, 1e-5, 'offset', 'both'),
    ('gn_const_g4_N2',         'group', (2, 16, 50), 4, 1e-5, 'const', 'both'),
    ('gn_no_affine_g2_N2',     'group', (2, 16, 200), 2, 1e-5, 'act', 'none'),
    ('gn_gC_T1_N2',            'group', (2, 8, 1), 8, 1e-5, 'act', 'both'),
    ('gn_g3_C9_N2',            'group', (2, 9, 123), 3, 1e-5, 'act', 'both'),
]

# Cells whose output must equal the bias EXACTLY (bitwise): every row has zero variance, so (x - K) - d = 0.
NORM_BIAS_EXACT = {'in_T1_N2', 'in_T1_no_bias', 'in_const_rows_N2', 'ln_const_rows_N2', 'gn_const_g4_N2', 'ln_L1_N2',
                   'gn_gC_T1_N2'}

# ------------------------------------------------------------------------------------------------ exact ops
# upsample cell: (name, kind, x_shape, scale, data); data: 'act' | 'special' (NaN payloads, +-0, +-Inf, subnormals)
UPSAMPLE_CELLS = [
    ('F0.1_x2_short',          'kokoro', (1, 512, 82), 2, 'act'),
    ('F0.1_x2_long',           'kokoro', (1, 512, 1193), 2, 'act'),
    ('decode3_x2_long',        'kokoro', (1, 1090, 961), 2, 'act'),
    ('f0_upsamp_x300_short',   'kokoro', (1, 1, 164), 300, 'act'),
    ('f0_upsamp_x300_long',    'kokoro', (1, 1, 2386), 300, 'act'),
    ('x1_N2',                  'edge', (2, 3, 17), 1, 'special'),
    ('x7_N2',                  'edge', (2, 3, 13), 7, 'special'),
    ('x2_N2_special',          'edge', (2, 4, 9), 2, 'special'),
    ('x5_T1_N2',               'edge', (2, 3, 1), 5, 'special'),
    ('x3_rank1',               'edge', (10,), 3, 'special'),
    ('x2_rank4_N2',            'edge', (2, 2, 2, 5), 2, 'special'),
    ('x300_N2',                'edge', (2, 1, 37), 300, 'special'),
]

# pad cell: (name, kind, x_shape, (pad_begin, pad_end), data); 'T-1' entries are resolved against the last axis.
PAD_CELLS = [
    ('gen_reflection_pad_short', 'kokoro', (1, 128, 9840), (1, 0), 'act'),
    ('gen_reflection_pad_long',  'kokoro', (1, 128, 16383), (1, 0), 'act'),
    ('p00_N2',                 'edge', (2, 4, 10), (0, 0), 'special'),
    ('p23_N2',                 'edge', (2, 4, 10), (2, 3), 'special'),
    ('pTm1_Tm1_N2',            'edge', (2, 4, 10), ('T-1', 'T-1'), 'special'),
    ('pTm1_0_N2',              'edge', (2, 4, 10), ('T-1', 0), 'special'),
    ('p0_Tm1_N2',              'edge', (2, 4, 10), (0, 'T-1'), 'special'),
    ('p10_N2_special',         'edge', (2, 4, 9), (1, 0), 'special'),
    ('T1_p00_N2',              'edge', (2, 3, 1), (0, 0), 'special'),
    ('T2_p11_N2',              'edge', (2, 3, 2), (1, 1), 'special'),
    ('rank1_p34',              'edge', (10,), (3, 4), 'special'),
    ('rank4_p21_N2',           'edge', (2, 2, 3, 6), (2, 1), 'special'),
]

# rint cell: (name, kind, x_shape, data); data: 'durations' | 'order_vector' | 'halves' | 'wide' | 'special'
RINT_CELLS = [
    ('durations_short',        'kokoro', (1, 30), 'durations'),
    ('durations_long',         'kokoro', (1, 505), 'durations'),
    ('order_vector_N2',        'edge', (2, -1), 'order_vector'),
    ('halves_-64_64_N2',       'edge', (2, -1), 'halves'),
    ('wide_exponents_N2',      'edge', (2, 3, 1000), 'wide'),
    ('special_N2',             'edge', (2, 4, 9), 'special'),
]
DURATION_MEAN = 5.0            # duration-like values: |N(mean, sd)| with some exact .5 ties
DURATION_SD = 3.0
DURATION_TIE_FRACTION = 0.25
HALVES_LIMIT = 64              # every k + 0.5 for k in [-LIMIT, LIMIT)
WIDE_LOG2_RANGE = (-30, 30)    # 'wide' magnitudes 2^e * N(0, 1)
SUBNORMAL_MIN = float.fromhex('0x0.000002p-126')     # smallest positive fp32 subnormal
SUBNORMAL_MAX = float.fromhex('0x0.fffffep-126')     # largest positive fp32 subnormal
TWO23 = 8388608.0
# The order's rint vector (+-0.5, +-1.5, +-2.5, -0.4, +-Inf, NaN, the 2^23 + 0.5 neighbours, subnormals) plus near ties.
RINT_ORDER_VECTOR = [0.5, -0.5, 1.5, -1.5, 2.5, -2.5, -0.4, 0.4, math.inf, -math.inf, math.nan, -math.nan,
                     TWO23 - 0.5, TWO23 - 1.5, TWO23, TWO23 + 1.0, TWO23 + 2.0, -(TWO23 - 0.5), -(TWO23 - 1.5),
                     4194304.5, 4194305.5, -4194304.5, float.fromhex('0x1.fffffcp+22'), float.fromhex('0x1.000002p+23'),
                     SUBNORMAL_MIN, -SUBNORMAL_MIN, SUBNORMAL_MAX, -SUBNORMAL_MAX, 0.0, -0.0,
                     float.fromhex('0x1.fffffep-2'), float.fromhex('0x1.000002p-1'), float.fromhex('-0x1.fffffep-2'),
                     float.fromhex('0x1.7ffffep+0'), float.fromhex('0x1.800002p+0'), 3.4028234663852886e38,
                     -3.4028234663852886e38, 1e-38]
# Strided sweep over fp32 bit patterns for rint (every RINT_SWEEP_STRIDE-th pattern of the 2^32), against numpy rint.
RINT_SWEEP_STRIDE = 257

# Special 32-bit words mixed into 'special' data (NaN payloads incl. signalling and negative NaNs, +-0, +-Inf, subnormals).
SPECIAL_WORDS = [0x7FC00000, 0x7FC00001, 0xFFC12345, 0x7F800001, 0xFF800ABC, 0x00000000, 0x80000000, 0x7F800000,
                 0xFF800000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF]
SPECIAL_FRACTION = 0.3         # share of elements replaced by special words

# Stream-ordering probe (torch writes x after a long sleep on the shared legacy stream; tc_signal reads it unsynced).
SLEEP_CYCLES = 50_000_000
ORDER_PROBE_TRIALS = 3
ORDER_PROBE_SHAPE = (1, 64, 4096)
