"""tc_signal own-test census and gates (P1T-M2). House tunable law: every number the tests use lives here.

Gate (order P1T-M2, G1; registered by the order before any run): normwise N = ||y - o||_inf / ||o||_inf <= 1e-5 against
a torch float64 CPU oracle computed in the same process; two calls bitwise equal.

Cells:
 * Kokoro attribute sets: every (K, stride, pads, dilation, groups) set of the shipped graph's Conv / ConvTranspose nodes
   (fixtures manifest /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures/manifest.json, read 2026-10-12) with the
   real channel counts and a shortened time axis (the real lengths run in G2, Voice's runner on the real fixtures).
 * Edge cells (order, Operand census): asymmetric pads (pb != pe); groups in {2, 4, C}; T < K; stride > K;
   dilation*(K-1) >= T; N = 2; output length 1; C_out not a multiple of 32. Plus: K = 1 with pads or stride (the
   im2col path, not the direct GEMM), N = 2 on the direct path, no bias, output_padding with dilation > stride.

Prior art: none needed for a constants module; the census itself is ours (no prior art known to me).
"""

GATE_N = 1e-5              # order G1 (= tolerances.json gates.fp32_paths.max)
SEED = 20261012            # torch.Generator seed for every cell's data
WEIGHT_SCALE = 1.0         # weights ~ N(0, 1) / sqrt(fan_in) * WEIGHT_SCALE (normwise N is scale-free)
NAN_FILL = float('nan')    # census allocator pre-fills outputs and scratch with NaN: a skipped write shows up

# conv1d cell: (name, N, C_in, C_out, T, K, stride, (pad_begin, pad_end), dilation, groups, bias)
CONV1D_KOKORO = [
    ('k3_s1_p11_512to512',        1, 512, 512, 96, 3, 1, (1, 1), 1, 1, True),     # F0.0 conv1, text-encoder-like
    ('k3_s1_p11_1090to1024',      1, 1090, 1024, 48, 3, 1, (1, 1), 1, 1, True),   # decode.0 conv1
    ('k3_s1_p11_514to1024',       1, 514, 1024, 40, 3, 1, (1, 1), 1, 1, True),    # encode conv1
    ('k3_s1_p11_1090to512',       1, 1090, 512, 48, 3, 1, (1, 1), 1, 1, True),    # decode.3 conv1
    ('k1_s1_p00_512to256',        1, 512, 256, 96, 1, 1, (0, 0), 1, 1, True),     # F0.1 conv1x1 (direct GEMM path)
    ('k1_s1_p00_256to1',          1, 256, 1, 200, 1, 1, (0, 0), 1, 1, True),      # F0_proj
    ('k1_s1_p00_22to128',         1, 22, 128, 300, 1, 1, (0, 0), 1, 1, True),     # noise_convs.1
    ('k1_s1_p00_1090to1024',      1, 1090, 1024, 40, 1, 1, (0, 0), 1, 1, False),  # decode.0 conv1x1 (no bias)
    ('k5_s1_p22_512to512',        1, 512, 512, 64, 5, 1, (2, 2), 1, 1, True),     # text_encoder cnn
    ('k3_s2_p11_1to1',            1, 1, 1, 401, 3, 2, (1, 1), 1, 1, True),        # decoder F0_conv
    ('k20_s5_p00_1to11',          1, 1, 11, 1000, 20, 5, (0, 0), 1, 1, True),     # generator stft conv
    ('k12_s6_p33_22to256',        1, 22, 256, 600, 12, 6, (3, 3), 1, 1, True),    # noise_convs.0
    ('k7_s1_p33_256to256',        1, 256, 256, 128, 7, 1, (3, 3), 1, 1, True),    # noise_res.0 convs1.0
    ('k7_s1_p33_128to22',         1, 128, 22, 128, 7, 1, (3, 3), 1, 1, True),     # conv_post
    ('k7_d3_p99_128',             1, 128, 128, 128, 7, 1, (9, 9), 3, 1, True),    # resblocks.4 convs1.1
    ('k7_d5_p1515_256',           1, 256, 256, 96, 7, 1, (15, 15), 5, 1, True),   # noise_res.0 convs1.2
    ('k3_d3_p33_256',             1, 256, 256, 96, 3, 1, (3, 3), 3, 1, True),     # resblocks.0 convs1.1
    ('k3_d5_p55_128',             1, 128, 128, 128, 3, 1, (5, 5), 5, 1, True),    # resblocks.3 convs1.2
    ('k11_d1_p55_256',            1, 256, 256, 96, 11, 1, (5, 5), 1, 1, True),    # resblocks.2 convs1.0
    ('k11_d3_p1515_128',          1, 128, 128, 160, 11, 1, (15, 15), 3, 1, True), # noise_res.1 convs1.1
    ('k11_d5_p2525_128',          1, 128, 128, 160, 11, 1, (25, 25), 5, 1, True), # noise_res.1 convs1.2
]

CONV1D_EDGE = [
    ('asym_p20_k3',               1, 16, 24, 50, 3, 1, (2, 0), 1, 1, True),
    ('asym_p03_k5_d2',            1, 16, 24, 50, 5, 1, (0, 3), 2, 1, True),
    ('asym_p41_k3_s2',            1, 16, 24, 51, 3, 2, (4, 1), 1, 1, True),
    ('groups2',                   1, 8, 6, 40, 3, 1, (1, 2), 1, 2, True),
    ('groups4_k5_d2',             1, 16, 12, 40, 5, 1, (4, 4), 2, 4, True),
    ('groups_C_depthwise',        1, 32, 32, 40, 3, 1, (1, 1), 1, 32, True),
    ('groups_C_multiplier2',      1, 16, 32, 40, 3, 1, (1, 1), 1, 16, True),
    ('T_lt_K',                    1, 8, 8, 2, 5, 1, (2, 2), 1, 1, True),
    ('stride_gt_K',               1, 8, 8, 23, 2, 5, (0, 0), 1, 1, True),
    ('stride_gt_K_asym',          1, 8, 8, 23, 2, 5, (1, 0), 1, 1, True),
    ('dil_span_ge_T',             1, 8, 8, 4, 3, 1, (3, 3), 3, 1, True),
    ('N2_Cout33',                 2, 8, 33, 40, 3, 1, (1, 1), 1, 1, True),
    ('N2_direct_k1_Cout5',        2, 8, 5, 30, 1, 1, (0, 0), 1, 1, True),
    ('N2_groups2_asym',           2, 8, 6, 30, 3, 2, (2, 1), 2, 2, True),
    ('out_len1_T_eq_K',           1, 8, 11, 7, 7, 1, (0, 0), 1, 1, True),
    ('out_len1_T1_p11',           1, 8, 11, 1, 3, 1, (1, 1), 1, 1, True),
    ('Cout11_no_bias',            1, 8, 11, 40, 3, 1, (1, 1), 1, 1, False),
    ('k1_pad10_im2col_path',      1, 8, 12, 30, 1, 1, (1, 0), 1, 1, True),
    ('k1_s2_im2col_path',         1, 8, 12, 30, 1, 2, (0, 0), 1, 1, True),
]

# conv_transpose1d cell: (name, N, C_in, C_out, T, K, stride, (pad_begin, pad_end), output_padding, dilation, groups,
# bias). w is [C_in, C_out/groups, K] (ONNX layout).
CONVT_KOKORO = [
    ('k20_s10_p55_512to256',      1, 512, 256, 24, 20, 10, (5, 5), 0, 1, 1, True),    # generator ups.0
    ('k12_s6_p33_256to128',       1, 256, 128, 64, 12, 6, (3, 3), 0, 1, 1, True),     # generator ups.1
    ('dw512_k3_s2_p11_op1',       1, 512, 512, 82, 3, 2, (1, 1), 1, 1, 512, True),    # F0.1 / N.1 pool
    ('dw1090_k3_s2_p11_op1',      1, 1090, 1090, 50, 3, 2, (1, 1), 1, 1, 1090, True), # decode.3 pool
    ('k20_s5_p00_11to1',          1, 11, 1, 400, 20, 5, (0, 0), 0, 1, 1, True),       # generator istft-side ConvTranspose
]

CONVT_EDGE = [
    ('asym_p20_k4_s2',            1, 16, 12, 30, 4, 2, (2, 0), 0, 1, 1, True),
    ('asym_p03_k5_s3_op1',        1, 16, 12, 30, 5, 3, (0, 3), 1, 1, 1, True),
    ('asym_p31_k3_d2',            1, 16, 12, 30, 3, 1, (3, 1), 0, 2, 1, True),
    ('groups2',                   1, 8, 6, 30, 4, 2, (1, 1), 0, 1, 2, True),
    ('groups4_s3_op2',            1, 16, 8, 30, 3, 3, (0, 2), 2, 1, 4, True),
    ('groups_C_depthwise',        1, 32, 32, 30, 3, 2, (1, 1), 1, 1, 32, True),
    ('groups_C_multiplier2',      1, 16, 32, 30, 3, 2, (1, 1), 1, 1, 16, True),
    ('T_lt_K',                    1, 8, 8, 2, 7, 2, (1, 1), 0, 1, 1, True),
    ('stride_gt_K',               1, 8, 8, 9, 2, 5, (0, 0), 0, 1, 1, True),
    ('stride_gt_K_asym_op',       1, 8, 8, 9, 2, 5, (2, 0), 3, 1, 1, True),
    ('dil_span_ge_T',             1, 8, 8, 3, 4, 1, (0, 0), 0, 2, 1, True),
    ('N2_Cout33',                 2, 8, 33, 20, 4, 2, (1, 1), 0, 1, 1, True),
    ('N2_groups2_asym',           2, 8, 6, 20, 3, 2, (2, 1), 1, 1, 2, True),
    ('out_len1',                  1, 8, 11, 1, 3, 1, (1, 1), 0, 1, 1, True),
    ('op_dil_gt_stride',          1, 8, 8, 20, 3, 1, (1, 1), 2, 3, 1, True),
    ('Cout11_no_bias',            1, 8, 11, 20, 4, 2, (1, 1), 0, 1, 1, False),
]

# Stream-ordering probes against torch (both queue on the legacy default stream).
SLEEP_CYCLES = 50_000_000                  # torch.cuda._sleep before the producer's write (tens of ms)
ORDER_PROBE_CONV = (1, 1024, 1024, 2048, 11)   # (N, C_in, C_out, T, K): a conv of ~ms, so an unordered read races
ORDER_PROBE_TRIALS = 3

# Fast-math canary: number of random fp32 division pairs compared bitwise with host IEEE division.
DIVIDE_PROBE_PAIRS = 4096
