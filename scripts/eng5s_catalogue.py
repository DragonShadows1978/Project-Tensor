"""ENG5-S copy of the eng5 branch's scripts/eng5_catalogue.py (read-only there), unchanged: ENG5's catalogue of
every public callable (Build B classes) used here for G1's CPU rows on Builds B and S. ENG5 text follows.

ENG5 H1 catalogue: every public callable of both binding modules (_tensor_cuda as tc._C, with the Tensor and
ApaInt4Workspace classes; _tensor_cuda_spo2 as tc._spo2) and the package's Python aliases assigned onto tc._C, each
with its H1 class, its tensor operands (the names the guard reports) and call builders.

A call builder takes a factory f(name, shape, dtype) and returns the call's result. The factory decides the device
of each operand (host for G1, all CUDA but one for G2) and whether its extents are kept (G1 nonzero rows) or emptied
(the empty-operand rows: see EMPTY below). Shapes are the smallest the seat could give for a dtype the entry accepts
(order G1); where the seat did not derive an entry's full geometry rule, Build B's behaviour on the same call is what
BOUNDARY.json records (a validation error on Build B is recorded as such, never hidden).

Used by scripts/eng5_census.py, scripts/eng5_boundary.py, tensor_cuda/tests/test_eng5_host_guard.py and the G2 line.
Prior art: no prior art known to me for this table; the classes are ENG3's (artifacts/eng3/HOST_CENSUS.json).
"""
import numpy as np

F32, F16, BF16, I64, U8, BOOL = "float32", "float16", "bfloat16", "int64", "uint8", "bool"

NO, HOST, REF, COMP = ("no_device_work_on_host_storage", "host_path", "refuses_host_already", "compute")

# Empty-operand rule names (H2): the condition under which Build B's zero-element path is taken. "any": the guard is
# skipped when any tensor operand has zero elements; "all": when every operand has zero elements; "matmul": the
# forward's own zero-output / K = 0 returns; "never": no zero-element return exists ahead of the first CUDA call.
ROWS = {}


def e(key, cls, ops=(), call=None, empty=None, note="", empty_rule=None, alias_of=None, guard=None):
    assert key not in ROWS, key
    ROWS[key] = dict(key=key, cls=cls, ops=list(ops), call=call, empty=empty, note=note, empty_rule=empty_rule,
                     alias_of=alias_of, guard=guard)


def T(f, n="a", s=(2, 3), d=F32):
    return f(n, s, d)


# ------------------------------------------------------------------------------------------------ Tensor class
META_T = ["shape", "ndim", "dtype", "device", "requires_grad", "grad", "data_ptr", "mark_modified", "revision",
          "zero_grad", "detach"]
for n in META_T:
    e("T." + n, NO, ["self"], note="metadata or graph state; no device work")
e("T.__init__", NO, [], note="pybind11 no-constructor __init__: raises TypeError; no device work")
e("T.reshape", NO, ["a"], call=lambda f: T(f).reshape([3, 2]), note="ops::reshape -> NDArray::reshape (metadata)")
e("T.squeeze", NO, ["a"], call=lambda f: T(f, s=(1, 3)).squeeze(0), note="ops::squeeze -> ops::reshape (metadata)")
e("T.unsqueeze", NO, ["a"], call=lambda f: T(f).unsqueeze(0), note="ops::unsqueeze -> ops::reshape (metadata)")
e("T.flatten", NO, ["a"], call=lambda f: T(f).flatten(), note="ops::flatten -> ops::reshape (metadata)")
e("T.__dlpack__", REF, ["self"], call=lambda f: T(f).__dlpack__(),
  note="bindings.cpp tensor_dlpack: BufferError for a non-CUDA tensor")
e("T.__dlpack_device__", REF, ["self"], call=lambda f: T(f).__dlpack_device__(),
  note="bindings.cpp tensor_dlpack_device: BufferError on CPU")
e("T.fill_", REF, ["a"], call=lambda f: T(f).fill_(1.0), note="kernels.cu fill_: 'fill_: CUDA arrays only'")
e("T.numpy", HOST, ["self"], call=lambda f: T(f).numpy(), note="tensor_to_numpy -> NDArray::to_host (ENG3 E2)")
e("T.astype", HOST, ["a"], call=lambda f: T(f).astype("float16"), note="ops::cast -> NDArray::astype (ENG3 E2/E3)")
e("T.half", HOST, ["a"], call=lambda f: T(f).half(), note="ops::cast -> NDArray::astype(Float16)")
e("T.float", HOST, ["a"], call=lambda f: T(f, d=F16).float(), note="ops::cast -> NDArray::astype(Float32)")
e("T.backward", HOST, ["self"], note="Tensor::backward: seed NDArray::full (E2), first accumulation reduce_to -> "
  "NDArray::clone (E2); a second accumulation (ew_binary) or a broadcast reduction (reduce_sum) is compute and is "
  "guarded inside Variable::accumulate_grad / reduce_to (entry 'backward')")

UNARY = ["relu", "gelu", "gelu_exact", "silu", "abs", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh",
         "log2", "log10", "sign", "floor", "ceil", "round", "isnan", "isinf", "isfinite", "reciprocal", "sigmoid",
         "tanh", "exp", "log", "sqrt", "transpose_last", "argmax_last_axis", "__neg__"]
for n in UNARY:
    e("T." + n, COMP, ["a"], call=(lambda n: lambda f: getattr(T(f), n)())(n), empty_rule="any",
      guard="neg" if n == "__neg__" else n)
e("T.nan_to_num", COMP, ["a"], call=lambda f: T(f).nan_to_num(), empty_rule="any")
e("T.clamp", COMP, ["a"], call=lambda f: T(f).clamp(-1.0, 1.0), empty_rule="any")
e("T.pow", COMP, ["a"], call=lambda f: T(f).pow(2.0), empty_rule="any", guard="pow_scalar")
e("T.__pow__", COMP, ["a"], call=lambda f: T(f) ** 2.0, empty_rule="any", guard="pow_scalar")
for n in ["sum", "mean", "max", "min", "var", "std", "prod"]:
    e("T." + n, COMP, ["a"], call=(lambda n: lambda f: getattr(T(f), n)([1]))(n), empty_rule="any")
for n in ["argmax", "argmin", "cumsum", "softmax", "log_softmax"]:
    e("T." + n, COMP, ["a"], call=(lambda n: lambda f: getattr(T(f), n)(-1))(n), empty_rule="any")
e("T.slice", COMP, ["a"], call=lambda f: T(f).slice(1, 0, 2), empty_rule="any")
e("T.__getitem__", COMP, ["t"], call=lambda f: T(f)[0], empty_rule="any", guard="getitem")
e("T.flip", COMP, ["a"], call=lambda f: T(f).flip([1]), empty_rule="any")
e("T.topk", COMP, ["a"], call=lambda f: T(f).topk(1), empty_rule="any")
e("T.sort", COMP, ["a"], call=lambda f: T(f).sort(), empty_rule="any", guard="sort")
e("T.permute", COMP, ["a"], call=lambda f: T(f).permute([1, 0]), empty_rule="any")
e("T.transpose", COMP, ["a"], call=lambda f: T(f).transpose(0, 1), empty_rule="any")
e("T.expand", COMP, ["a"], call=lambda f: T(f, s=(1, 3)).expand([2, 3]), empty_rule="any")
e("T.gather", COMP, ["a", "index"], call=lambda f: T(f).gather(1, f("index", (2, 1), I64)), empty_rule="any")
e("T.masked_fill", COMP, ["a", "mask"], call=lambda f: T(f).masked_fill(f("mask", (2, 3), F32), 0.0),
  empty_rule="any")
for n, g in [("maximum", "maximum"), ("minimum", "minimum"), ("gt", "gt"), ("ge", "ge"), ("lt", "lt"), ("le", "le"),
             ("eq", "eq"), ("__add__", "add"), ("__sub__", "sub"), ("__mul__", "mul"), ("__truediv__", "div"),
             ("__matmul__", "matmul")]:
    if n == "__matmul__":
        e("T." + n, COMP, ["a", "b"], call=lambda f: T(f) @ f("b", (3, 2), F32), empty_rule="matmul", guard="matmul")
    else:
        e("T." + n, COMP, ["a", "b"], call=(lambda n: lambda f: getattr(T(f), n)(f("b", (2, 3), F32)))(n),
          empty_rule="any", guard=g)
for n, g in [("__gt__", "gt"), ("__ge__", "ge"), ("__lt__", "lt"), ("__le__", "le"), ("__radd__", "add_scalar"),
             ("__rmul__", "mul_scalar")]:
    e("T." + n, COMP, ["a"], call=(lambda n: lambda f: getattr(T(f), n)(2.0))(n), empty_rule="any", guard=g)
# The scalar overloads of __add__/__sub__/__mul__/__truediv__ share the name with the tensor overloads; G1 calls both.
SCALAR_OVERLOADS = {"__add__": "add_scalar", "__sub__": "add_scalar", "__mul__": "mul_scalar",
                    "__truediv__": "mul_scalar"}

# ------------------------------------------------------------------------------------------ ApaInt4Workspace class
e("W.__init__", NO, [], note="pybind11 no-constructor __init__: raises TypeError; no device work")
for n in ["capacity", "valid_rows", "nbytes", "_rewind"]:
    e("W." + n, NO, ["self"], note="workspace metadata; no device work")
e("W.reset", COMP, ["self"], note="apa_int4_workspace_reset: needs a workspace, which only apa_int4_workspace "
  "(compute, guarded) creates; host workspaces cannot be built, so no host operand reaches it")

# ---------------------------------------------------------------------------------------------- module _C
NO_C = ["is_grad_enabled", "set_grad_enabled", "synchronize", "empty_cache", "set_alloc_pooling",
        "_dlpack_release_stats", "_dlpack_raise_unraisable", "_dlpack_begin_shutdown", "_dlpack_exit_hook_runner",
        "_dlpack_exit_report_written", "_dlpack_test_fail_exit_report", "bp_source_pin", "bp_wall_begin",
        "bp_wall_end", "bp_configure", "bp_begin", "bp_end", "bp_rows", "bp_kernel_4_set_variant",
        "bp_kernel_4_get_variant", "bp_kernel_2_set_variant", "bp_kernel_2_get_variant", "bp_kernel_3_profile",
        "bp_kernel_3_halves", "apa_refine_stats", "apa_gemm_selective_stats", "_apa_int4_decode_plan",
        "apa_sp2_delta", "checkpoint", "attach_vjp", "_bwd_native_selftest"]
for n in NO_C:
    e("C." + n, NO, [], note="takes no engine tensor operand, or only builds graph/configuration state")
e("C.Tensor", NO, [], note="class object")
e("C.ApaInt4Workspace", NO, [], note="class object")
e("C._DLPACK_SHUTDOWN_DIAG_DEFAULT_S", NO, [], note="constant")
e("C.tensor", HOST, [], note="tensor_from_numpy -> NDArray::from_host (E2)")
e("C.empty", HOST, [], note="tensor_empty -> NDArray::empty (host: malloc only)")
e("C._core_full", HOST, [], note="ENG3 test hook: NDArray::full (E2)")
e("C._core_zeros", HOST, [], note="ENG3 test hook: NDArray::zeros (E2)")
e("C._core_clone", HOST, ["tensor"], call=lambda f: __import__("tensor_cuda")._C._core_clone(T(f, "tensor")),
  note="ENG3 test hook: NDArray::clone (E2)")
e("C._core_to", HOST, ["tensor"], note="ENG3 test hook: NDArray::to (E2); cuda target queries cudaGetDevice")
e("C._from_dlpack_capsule", REF, [], note="refuses non-CUDA capsules (BufferError); takes no engine tensor")


def NG(fn):
    """Run fn with grad mode off (the cache-row entries are inference-only and refuse under grad mode)."""
    import tensor_cuda
    prev = tensor_cuda._C.is_grad_enabled()
    tensor_cuda._C.set_grad_enabled(False)
    try:
        return fn()
    finally:
        tensor_cuda._C.set_grad_enabled(prev)


def C(name):
    import tensor_cuda
    return getattr(tensor_cuda._C, name)


def S(name):
    import tensor_cuda
    return getattr(tensor_cuda._spo2, name)


e("C.matmul", COMP, ["a", "b"], call=lambda f: C("matmul")(T(f), f("b", (3, 2), F32)), empty_rule="matmul")
e("C.add", COMP, ["a", "b"], call=lambda f: C("add")(T(f), f("b", (2, 3), F32)), empty_rule="any")
e("C.mul", COMP, ["a", "b"], call=lambda f: C("mul")(T(f), f("b", (2, 3), F32)), empty_rule="any")
e("C.where", COMP, ["cond", "x", "y"], call=lambda f: C("where")(f("cond", (2, 3), F32), f("x", (2, 3), F32),
                                                                   f("y", (2, 3), F32)), empty_rule="any")
e("C.mse_loss", COMP, ["pred", "target"], call=lambda f: C("mse_loss")(f("pred", (2, 3), F32),
                                                                      f("target", (2, 3), F32)), empty_rule="never")
e("C.cross_entropy", COMP, ["logits", "onehot"],
  call=lambda f: C("cross_entropy")(f("logits", (2, 3), F32), f("onehot", (2, 3), F32)), empty_rule="never")
e("C.cat", COMP, ["ts[0]", "ts[1]"], call=lambda f: C("cat")([f("ts[0]", (2, 3), F32), f("ts[1]", (2, 3), F32)], 0),
  empty_rule="all")
e("C.stack", COMP, ["ts[0]", "ts[1]"],
  call=lambda f: C("stack")([f("ts[0]", (2, 3), F32), f("ts[1]", (2, 3), F32)], 0), empty_rule="all")
e("C.embedding", COMP, ["weight", "idx"], call=lambda f: C("embedding")(f("weight", (4, 3), F32),
                                                                       f("idx", (2,), I64)), empty_rule="all")
e("C.argmax_last_axis", COMP, ["a"], call=lambda f: C("argmax_last_axis")(T(f)), empty_rule="any")
e("C.causal_softmax", COMP, ["scores"], call=lambda f: C("causal_softmax")(f("scores", (1, 2, 2), F32)),
  empty_rule="any")
e("C.rms_norm", COMP, ["x", "w"], call=lambda f: C("rms_norm")(f("x", (2, 4), F32), f("w", (4,), F32), 1e-6),
  empty_rule="all")
e("C.rope_apply", COMP, ["x", "cs", "sn"],
  call=lambda f: NG(lambda: C("rope_apply")(f("x", (1, 1, 2, 4), F32), f("cs", (8, 4), F32), f("sn", (8, 4), F32), 0)),
  empty_rule="all")
e("C.write_rows", COMP, ["buf", "src"], call=lambda f: NG(lambda: C("write_rows")(f("buf", (1, 1, 4, 2), F32),
                                                                      f("src", (1, 1, 1, 2), F32), 0)),
  empty_rule="all")
e("C.export_rows", COMP, ["cache"], call=lambda f: NG(lambda: C("export_rows")(f("cache", (1, 1, 4, 2), F32), 2, 0, 2)),
  empty_rule="all")
e("C.export_rope_rows", COMP, ["cache", "cs", "sn"],
  call=lambda f: NG(lambda: C("export_rope_rows")(f("cache", (1, 1, 4, 4), F32), f("cs", (8, 4), F32), f("sn", (8, 4), F32),
                                       2, 0, 2, 0)),
  empty_rule="all")
e("C.export_row_pair", COMP, ["raw_cache", "rope_cache", "cs", "sn"],
  call=lambda f: NG(lambda: C("export_row_pair")(f("raw_cache", (1, 1, 4, 4), F32), f("rope_cache", (1, 1, 4, 4), F32),
                                      f("cs", (8, 4), F32), f("sn", (8, 4), F32), 2, 2, 0, 0, 2, 0)),
  empty_rule="all")
e("C.export_row_pairs", COMP, ["raw_caches[0]", "rope_caches[0]", "cs", "sn"],
  call=lambda f: NG(lambda: C("export_row_pairs")([f("raw_caches[0]", (1, 1, 4, 4), F32)],
                                       [f("rope_caches[0]", (1, 1, 4, 4), F32)],
                                       f("cs", (8, 4), F32), f("sn", (8, 4), F32), 2, 2, [0], [0], 2, 0)),
  empty_rule="all")
e("C.swap_row_pairs_with_rope", COMP, ["raw_caches[0]", "rope_caches[0]", "raw_inserts[0]", "rope_inserts[0]",
                                       "cs", "sn"],
  call=lambda f: NG(lambda: C("swap_row_pairs_with_rope")([f("raw_caches[0]", (1, 1, 4, 4), F32)],
                                               [f("rope_caches[0]", (1, 1, 4, 4), F32)],
                                               [f("raw_inserts[0]", (1, 1, 1, 4), F32)],
                                               [f("rope_inserts[0]", (1, 1, 1, 4), F32)],
                                               f("cs", (8, 4), F32), f("sn", (8, 4), F32), 2, 2, 1, 2, 0)),
  empty_rule="all")
e("C.evict_row_pairs", COMP, ["raw_caches[0]", "rope_caches[0]"],
  call=lambda f: NG(lambda: C("evict_row_pairs")([f("raw_caches[0]", (1, 1, 4, 4), F32)],
                                      [f("rope_caches[0]", (1, 1, 4, 4), F32)], 2, 2, 1, 1)),
  empty_rule="all")
e("C.arena_row_pair_transaction", COMP, ["raw_caches[0]", "rope_caches[0]", "raw_inserts[0]", "rope_inserts[0]",
                                         "cs", "sn"],
  call=lambda f: NG(lambda: C("arena_row_pair_transaction")([f("raw_caches[0]", (1, 1, 4, 4), F32)],
                                                 [f("rope_caches[0]", (1, 1, 4, 4), F32)],
                                                 [f("raw_inserts[0]", (1, 1, 1, 4), F32)],
                                                 [f("rope_inserts[0]", (1, 1, 1, 4), F32)],
                                                 f("cs", (8, 4), F32), f("sn", (8, 4), F32), 2, 2, 1, 2, 4)),
  empty_rule="all")
e("C.splice_rows", COMP, ["old_cache", "insert"],
  call=lambda f: NG(lambda: C("splice_rows")(f("old_cache", (1, 1, 4, 2), F32), f("insert", (1, 1, 1, 2), F32), 2, 1, 2)),
  empty_rule="all")
e("C.evict_rows", COMP, ["old_cache"], call=lambda f: NG(lambda: C("evict_rows")(f("old_cache", (1, 1, 4, 2), F32), 2, 1, 1)),
  empty_rule="all")
e("C.dda_raycast", REF, ["grid", "origins", "directions"],
  call=lambda f: C("dda_raycast")(f("grid", (2, 2, 2), U8), f("origins", (1, 3), F32), f("directions", (1, 3), F32),
                                  4), note="dda.cu:229 refuses host operands")
for n, ops, call in [
    ("raster_winner_scatter_min", ["pixel_indices", "depth_keys", "face_ids"],
     lambda f: C("raster_winner_scatter_min")(f("pixel_indices", (1,), I64), f("depth_keys", (1,), I64),
                                              f("face_ids", (1,), I64), 1)),
    ("raster_triangle_winners", ["clip_positions", "faces"],
     lambda f: C("raster_triangle_winners")(f("clip_positions", (3, 4), F32), f("faces", (1, 3), I64), 1, 1)),
    ("raster_winner_resolve", ["clip_positions", "faces", "winner_face_ids"],
     lambda f: C("raster_winner_resolve")(f("clip_positions", (3, 4), F32), f("faces", (1, 3), I64),
                                          f("winner_face_ids", (1, 1), I64))),
    ("rasterize_clip", ["clip_positions", "faces"],
     lambda f: C("rasterize_clip")(f("clip_positions", (3, 4), F32), f("faces", (1, 3), I64), 1, 1)),
    ("bake_back_project", ["atlas_positions_h", "view", "view_depth", "view_reliable", "view_cosine",
                           "world_to_camera", "image_projection"],
     lambda f: C("bake_back_project")(f("atlas_positions_h", (1, 4), F32), f("view", (1, 1, 3), F32),
                                      f("view_depth", (1, 1), F32), f("view_reliable", (1, 1), U8),
                                      f("view_cosine", (1, 1), F32), f("world_to_camera", (4, 4), F32),
                                      f("image_projection", (3, 4), F32), 0.1)),
    ("bake_cosine_blend", ["view_colors", "view_cosine", "view_valid", "view_weights", "view_enabled"],
     lambda f: C("bake_cosine_blend")(f("view_colors", (1, 1, 3), F32), f("view_cosine", (1, 1), F32),
                                      f("view_valid", (1, 1), U8), f("view_weights", (1,), F32),
                                      f("view_enabled", (1,), U8))),
    ("inpaint_island_passes", ["positions", "vertex_colors", "vertex_mask", "neighbor_offsets", "neighbors",
                               "island_offsets", "island_occurrences"],
     lambda f: C("inpaint_island_passes")(f("positions", (1, 3), F32), f("vertex_colors", (1, 3), F32),
                                          f("vertex_mask", (1,), F32), f("neighbor_offsets", (2,), I64),
                                          f("neighbors", (1,), I64), f("island_offsets", (2,), I64),
                                          f("island_occurrences", (1,), I64), 1)),
    ("terrain_render", ["materials", "palette"],
     lambda f: C("terrain_render")(f("materials", (2, 2, 2), U8), f("palette", (2, 3), U8), [0, 0, 0], [1, 0, 0],
                                   [0, 1, 0], [0, 0, 1], 0.5, 0.5, 1, 1, [0, 0, -1], 4)),
]:
    e("C." + n, REF, ops, call=call, note="paint_*.cu / terrain.cu refuse host operands (pre-existing)")
e("C.fused_sdpa_noncausal", COMP, ["q", "k", "v"],
  call=lambda f: C("fused_sdpa_noncausal")(f("q", (1, 1, 1, 64), F16), f("k", (1, 1, 2, 64), F16),
                                           f("v", (1, 1, 2, 64), F16), 0.125), empty_rule="never")
e("C.apa_int4_sdpa_noncausal", COMP, ["q", "k", "v"],
  call=lambda f: C("apa_int4_sdpa_noncausal")(f("q", (1, 1, 1, 64), F16), f("k", (1, 1, 2, 64), F16),
                                              f("v", (1, 1, 2, 64), F16), 0.125, 1.0), empty_rule="never")
e("C.apa_quantize_gather", COMP, ["rotated", "boundaries", "centroids"],
  call=lambda f: C("apa_quantize_gather")(f("rotated", (1, 1, 2, 64), F32), f("boundaries", (15,), F32),
                                          f("centroids", (16,), F32)), empty_rule="never")
e("C.int4_linear", COMP, ["x", "packed", "scales", "zeros"],
  call=lambda f: C("int4_linear")(f("x", (1, 128), F16), f("packed", (2, 64), U8), f("scales", (2, 1), F16),
                                  f("zeros", (2, 1), F16)), empty_rule="never")
e("C.int4_linear_fused", COMP, ["x", "packed", "scales", "zeros"],
  call=lambda f: C("int4_linear_fused")(f("x", (1, 128), F16), f("packed", (2, 64), U8), f("scales", (2, 1), F16),
                                        f("zeros", (2, 1), F16)), empty_rule="never")
e("C.w8a16_matmul", COMP, ["x", "codes", "scales"],
  call=lambda f: C("w8a16_matmul")(f("x", (1, 32), F16), f("codes", (2, 32), U8), f("scales", (2, 1), F16)),
  empty_rule="never")
e("C.intn_linear", COMP, ["x", "packed", "scales", "zeros"],
  call=lambda f: C("intn_linear")(f("x", (1, 128), F16), f("packed", (2, 64), U8), f("scales", (2, 1), F16),
                                  f("zeros", (2, 1), F16), 2, 128), empty_rule="never")
e("C.intn_linear_fused", COMP, ["x", "packed", "scales", "zeros"],
  call=lambda f: C("intn_linear_fused")(f("x", (1, 128), F16), f("packed", (2, 64), U8), f("scales", (2, 1), F16),
                                        f("zeros", (2, 1), F16), 2, 128), empty_rule="never")
e("C.mxfp4_linear", COMP, ["x", "blocks", "scales"],
  call=lambda f: C("mxfp4_linear")(f("x", (1, 32), F32), f("blocks", (2, 1, 16), U8), f("scales", (2, 1), U8)),
  empty_rule="never")
e("C.mxfp4_linear_expert", COMP, ["x", "blocks", "scales"],
  call=lambda f: C("mxfp4_linear_expert")(f("x", (1, 32), F32), f("blocks", (1, 2, 1, 16), U8),
                                          f("scales", (1, 2, 1), U8), 0), empty_rule="never")
QKV = lambda f, L=1, S=2, D=64, d=F16: (f("q", (1, 1, L, D), d), f("k", (1, 1, S, D), d))
e("C.apa_selective_train", COMP, ["q", "k", "kq", "v"],
  call=lambda f: C("apa_selective_train")(*QKV(f), f("kq", (1, 1, 2, 64), F16), f("v", (1, 1, 2, 64), F16), 0.125,
                                          1.0), empty_rule="never")
e("C.gated_delta_step", COMP, ["q", "k", "v", "a", "b", "A_neg", "dt_bias", "state"],
  call=lambda f: C("gated_delta_step")(f("q", (1, 1, 128), F32), f("k", (1, 1, 128), F32),
                                       f("v", (1, 1, 128), F32), f("a", (1, 1), F32), f("b", (1, 1), F32),
                                       f("A_neg", (1,), F32), f("dt_bias", (1,), F32),
                                       f("state", (1, 1, 128, 128), F32)), empty_rule="never")
e("C.int4_dequant", COMP, ["packed", "scales", "zeros"],
  call=lambda f: C("int4_dequant")(f("packed", (2, 64), U8), f("scales", (2, 1), F16), f("zeros", (2, 1), F16)),
  empty_rule="never")
e("C.intn_dequant", COMP, ["packed", "scales", "zeros"],
  call=lambda f: C("intn_dequant")(f("packed", (2, 32), U8), f("scales", (2, 1), F16), f("zeros", (2, 1), F16), 2,
                                   128), empty_rule="never")
e("C.kv_int4_pack", COMP, ["x"], call=lambda f: C("kv_int4_pack")(f("x", (1, 1, 2, 64), BF16)), empty_rule="never")
e("C.kv_int4_unpack", COMP, ["packed", "scales"],
  call=lambda f: C("kv_int4_unpack")(f("packed", (1, 1, 2, 32), U8), f("scales", (1, 1, 2, 2), F32), 32, 0, 2),
  empty_rule="never")
SP = ["q", "k", "kq", "v"]
e("C.apa_selective_attention_sp", COMP, SP + ["sinks"],
  call=lambda f: C("apa_selective_attention_sp")(*QKV(f, d=BF16), f("kq", (1, 1, 2, 64), BF16),
                                                 f("v", (1, 1, 2, 64), BF16), 0.125, 1.0), empty_rule="never")
e("C.apa_sp1_1_baseline_diagnostics", COMP, ["q", "kq"],
  call=lambda f: C("apa_sp1_1_baseline_diagnostics")(f("q", (1, 1, 1, 64), BF16), f("kq", (1, 1, 2, 64), BF16)),
  empty_rule="never")
e("C.apa_selective_attention_sp_epsilon", COMP, SP + ["sinks"],
  call=lambda f: C("apa_selective_attention_sp_epsilon")(*QKV(f, d=BF16), f("kq", (1, 1, 2, 64), BF16),
                                                         f("v", (1, 1, 2, 64), BF16), 0.125, 0.01, 0.01),
  empty_rule="never")
e("C.apa_sp2_scores", COMP, ["q", "k", "kq"],
  call=lambda f: C("apa_sp2_scores")(*QKV(f, d=BF16), f("kq", (1, 1, 2, 64), BF16), 0.125, 0, 1, False),
  empty_rule="never")
e("C.apa_sp2_baseline_thresholds", COMP, ["q", "kq"],
  call=lambda f: C("apa_sp2_baseline_thresholds")(f("q", (1, 1, 1, 64), BF16), f("kq", (1, 1, 2, 64), BF16), 0.125,
                                                  1.0, False), empty_rule="never")
e("C.apa_selective_attention", COMP, SP,
  call=lambda f: C("apa_selective_attention")(*QKV(f), f("kq", (1, 1, 2, 64), F16), f("v", (1, 1, 2, 64), F16),
                                              0.125, 1.0), empty_rule="never")
e("C.apa_selective_attention_int4", COMP, ["q", "k", "v"],
  call=lambda f: C("apa_selective_attention_int4")(*QKV(f), f("v", (1, 1, 2, 64), F16), 0.125, 1.0),
  empty_rule="never")
e("C.apa_int4_workspace", COMP, ["k"], call=lambda f: C("apa_int4_workspace")(f("k", (1, 1, 2, 64), F16), 4),
  empty_rule="never")
e("C.apa_selective_attention_int4_workspace", COMP, ["q", "k", "v"],
  note="needs an ApaInt4Workspace, which only the guarded apa_int4_workspace creates; G1 calls it with a host q/k/v "
       "and workspace=None is refused by the binding first, so the row uses the same operands as "
       "apa_selective_attention_int4 through _apa_int4_profile_stages' workspace-free twin; the binding guards q/k/v "
       "ahead of its workspace check (bindings.cpp), so G1 passes workspace=None",
  call=lambda f: C("apa_selective_attention_int4_workspace")(*QKV(f), f("v", (1, 1, 2, 64), F16), None, 0.125, 1.0),
  empty_rule="never")
e("C._apa_int4_profile_stages", COMP, ["q", "k", "v"],
  call=lambda f: C("_apa_int4_profile_stages")(*QKV(f), f("v", (1, 1, 2, 64), F16), 0.125, 1.0, False, 0, 1),
  empty_rule="never")
e("C.apa_gemm_selective_attention", COMP, ["q", "k", "v"],
  call=lambda f: C("apa_gemm_selective_attention")(*QKV(f, D=128, d=BF16), f("v", (1, 1, 2, 128), BF16), 0.125, 1.0),
  empty_rule="never")
e("C.apa_gemm_selective_attention_cached", COMP, ["q", "k", "v", "k_codes", "k_scales"],
  call=lambda f: C("apa_gemm_selective_attention_cached")(*QKV(f, D=128, d=BF16), f("v", (1, 1, 2, 128), BF16),
                                                          f("k_codes", (1, 1, 2, 64), U8),
                                                          f("k_scales", (1, 1, 2, 4), F32), 0.125, 1.0),
  empty_rule="never")
e("C.apa_gemm_selective_quantize_k", COMP, ["k"],
  call=lambda f: C("apa_gemm_selective_quantize_k")(f("k", (1, 1, 2, 128), BF16)), empty_rule="never")
e("C.apa_gemm_selective_debug", COMP, ["q", "k"],
  call=lambda f: C("apa_gemm_selective_debug")(*QKV(f, D=128, d=BF16), 0.125, 1.0), empty_rule="never")
e("C.apa_selective_attention_sink", COMP, SP + ["sinks"],
  call=lambda f: C("apa_selective_attention_sink")(*QKV(f), f("kq", (1, 1, 2, 64), F16), f("v", (1, 1, 2, 64), F16),
                                                   f("sinks", (1,), F16), 0.125, 1.0), empty_rule="never")
e("C.apa_selective_fwd_train", COMP, SP,
  call=lambda f: C("apa_selective_fwd_train")(*QKV(f), f("kq", (1, 1, 2, 64), F16), f("v", (1, 1, 2, 64), F16),
                                              0.125, 1.0), empty_rule="never")
e("C.apa_selective_bwd", COMP, SP + ["dO", "lse", "thr"],
  call=lambda f: C("apa_selective_bwd")(*QKV(f), f("kq", (1, 1, 2, 64), F16), f("v", (1, 1, 2, 64), F16),
                                        f("dO", (1, 1, 1, 64), F16), f("lse", (1, 1, 1), F32),
                                        f("thr", (1, 1, 1), F32), 0.125), empty_rule="never")
e("C.apa_selective_bwd_variant", COMP, SP + ["dO", "lse", "thr", "out"],
  call=lambda f: C("apa_selective_bwd_variant")(*QKV(f), f("kq", (1, 1, 2, 64), F16), f("v", (1, 1, 2, 64), F16),
                                                f("dO", (1, 1, 1, 64), F16), f("lse", (1, 1, 1), F32),
                                                f("thr", (1, 1, 1), F32), f("out", (1, 1, 1, 64), F16), 0.125, False,
                                                "a"), empty_rule="never")
e("C.apa_selective_fwd_train_variant", COMP, SP,
  call=lambda f: C("apa_selective_fwd_train_variant")(*QKV(f), f("kq", (1, 1, 2, 64), F16),
                                                      f("v", (1, 1, 2, 64), F16), 0.125, 1.0, False, "a"),
  empty_rule="never")
e("C.bp_kernel_4_diagnostic", COMP, SP,
  call=lambda f: C("bp_kernel_4_diagnostic")(*QKV(f), f("kq", (1, 1, 2, 64), F16), f("v", (1, 1, 2, 64), F16),
                                             0.125, 1.0, False, "a"), empty_rule="never")
e("C.apa_blend_softmax", COMP, ["bulk", "rank"],
  call=lambda f: C("apa_blend_softmax")(f("bulk", (1, 1, 2, 3), F32), f("rank", (1, 1, 2, 3), F32), 0.0),
  empty_rule="all")
e("C.apa_blend_softmax_sink", COMP, ["bulk", "rank", "sinks"],
  call=lambda f: C("apa_blend_softmax_sink")(f("bulk", (1, 1, 2, 3), F32), f("rank", (1, 1, 2, 3), F32),
                                             f("sinks", (1,), F32), 0.0), empty_rule="all")
e("C.im2col", COMP, ["a"], call=lambda f: C("im2col")(f("a", (1, 1, 2, 2), F32), 1, 1, 1, 1, 0, 0),
  empty_rule="any")
e("C.col2im", COMP, ["cols"], call=lambda f: C("col2im")(f("cols", (1, 1, 4), F32), 1, 1, 2, 2, 1, 1, 1, 1, 0, 0),
  empty_rule="any")
e("C.avg_pool2d", COMP, ["a"], call=lambda f: C("avg_pool2d")(f("a", (1, 1, 2, 2), F32), 1, 1, 1, 1, 0, 0),
  empty_rule="any")
e("C.max_pool2d", COMP, ["a"], call=lambda f: C("max_pool2d")(f("a", (1, 1, 2, 2), F32), 1, 1, 1, 1, 0, 0),
  empty_rule="any")
OPT = lambda f, names: [f(n, (2, 3), F32) for n in names]
e("C.sgd_step", COMP, ["param", "grad", "buf"],
  call=lambda f: C("sgd_step")(*OPT(f, ["param", "grad", "buf"]), 0.1, 0.9, 0.0), empty_rule="never")
e("C.adam_step", COMP, ["param", "grad", "m", "v"],
  call=lambda f: C("adam_step")(*OPT(f, ["param", "grad", "m", "v"]), 0.1, 0.9, 0.99, 1e-8, 1, 0.0, False),
  empty_rule="never")
e("C.rmsprop_step", COMP, ["param", "grad", "sq"],
  call=lambda f: C("rmsprop_step")(*OPT(f, ["param", "grad", "sq"]), 0.1, 0.9, 1e-8, 0.0), empty_rule="never")
e("C.lion_step", COMP, ["param", "grad", "m"],
  call=lambda f: C("lion_step")(*OPT(f, ["param", "grad", "m"]), 0.1, 0.9, 0.99, 0.0), empty_rule="never")
e("C.radam_step", COMP, ["param", "grad", "m", "v"],
  call=lambda f: C("radam_step")(*OPT(f, ["param", "grad", "m", "v"]), 0.1, 0.9, 0.99, 1e-8, 0.5, 0.5, 1.0, True,
                                 0.0, False), empty_rule="never")
e("C.adagrad_step", COMP, ["param", "grad", "acc"],
  call=lambda f: C("adagrad_step")(*OPT(f, ["param", "grad", "acc"]), 0.1, 1e-8, 0.0), empty_rule="never")
e("C.scale_", COMP, ["param"], call=lambda f: C("scale_")(f("param", (2, 3), F32), 2.0), empty_rule="never")
e("C.axpy_", COMP, ["param", "other"],
  call=lambda f: C("axpy_")(f("param", (2, 3), F32), f("other", (2, 3), F32), 2.0), empty_rule="never")

# ------------------------------------------------------------------------------------- module _spo2 (packed APA)
PK = lambda f, D=64: (f("codes", (1, 1, 2, D // 2), U8), f("scales", (1, 1, 2), F32), f("rotation", (D, D), F32),
                      f("codebook", (16,), F32))
e("S.run", COMP, ["q", "k", "kq", "v", "sinks"],
  call=lambda f: S("run")(f("q", (1, 1, 1, 64), BF16), f("k", (1, 1, 2, 64), BF16), f("kq", (1, 1, 2, 64), BF16),
                          f("v", (1, 1, 2, 64), BF16), 0.125, 1.0), empty_rule="never")
e("S.packed_prepare", COMP, ["k", "r"],
  call=lambda f: S("packed_prepare")(f("k", (1, 1, 2, 64), BF16), f("r", (64, 64), F32), 0), empty_rule="never")
e("S.packed_rotate", COMP, ["q", "r"],
  call=lambda f: S("packed_rotate")(f("q", (1, 1, 1, 64), BF16), f("r", (64, 64), F32), 1, True),
  empty_rule="never")
e("S.packed_append", COMP, ["k", "codes", "sc", "rot", "norms", "bnd"],
  call=lambda f: S("packed_append")(f("k", (1, 1, 2, 64), BF16), f("codes", (1, 1, 4, 32), U8),
                                    f("sc", (1, 1, 4), F32), f("rot", (64, 64), F32), f("norms", (1, 1, 4), F32),
                                    f("bnd", (15,), F32), 0, True), empty_rule="never")
PACKED = ["codes", "scales", "rotation", "codebook"]
e("S.packed_run", COMP, ["q", "k", "v"] + PACKED + ["sinks"],
  call=lambda f: S("packed_run")(f("q", (1, 1, 1, 64), BF16), f("k", (1, 1, 2, 64), BF16),
                                 f("v", (1, 1, 2, 64), BF16), PK(f), 0.125, 1.0, fmt="tq4"), empty_rule="never")
e("S.int4_stats", COMP, ["q", "codes", "scales"],
  call=lambda f: S("int4_stats")(f("q", (1, 1, 1, 64), BF16), f("codes", (1, 1, 2, 32), U8),
                                 f("scales", (1, 1, 2), F32), 2, 0.125, 1.0), empty_rule="never")
e("S.packed_prefill", COMP, ["q", "k", "v"] + PACKED + ["sinks"],
  call=lambda f: S("packed_prefill")(f("q", (1, 1, 2, 64), BF16), f("k", (1, 1, 2, 64), BF16),
                                     f("v", (1, 1, 2, 64), BF16), PK(f), 0.125, 1.0, fmt="tq4"), empty_rule="never")
e("S.eb2b_quantize", COMP, ["v"], call=lambda f: S("eb2b_quantize")(f("v", (1, 1, 2, 128), BF16)),
  empty_rule="never")
e("S.eb2b_values", COMP, ["v", "vi", "vs", "flags"],
  call=lambda f: S("eb2b_values")(f("v", (1, 1, 2, 128), BF16), f("vi", (1, 1, 2, 64), U8), f("vs", (1, 1, 2, 4), F32),
                                  f("flags", (1, 1, 2), U8)), empty_rule="never")
e("S.eb2b_run", COMP, ["q", "k", "v", "vi", "vs"] + PACKED,
  call=lambda f: S("eb2b_run")(f("q", (1, 1, 1, 64), BF16), f("k", (1, 1, 2, 64), BF16), f("v", (1, 1, 2, 64), BF16),
                               f("vi", (1, 1, 2, 32), U8), f("vs", (1, 1, 2), F32), PK(f), 0.125, 0.5, delta=1.0,
                               carry="partition"), empty_rule="never")
e("S.eb3_run", COMP, ["q", "k", "v"] + PACKED + ["sinks"],
  call=lambda f: S("eb3_run")(f("q", (1, 1, 1, 64), BF16), f("k", (1, 1, 2, 64), BF16), f("v", (1, 1, 2, 64), BF16),
                              PK(f), 0.125, 1.0), empty_rule="never")
e("S.eb3_resources", NO, [], note="kernel attribute table; no tensor operand")
e("S.eb4_run", COMP, ["q", "k", "v", "vi", "vs"] + PACKED + ["sinks", "inject"],
  call=lambda f: S("eb4_run")(f("q", (1, 1, 1, 64), BF16), f("k", (1, 1, 2, 64), BF16), f("v", (1, 1, 2, 64), BF16),
                              f("vi", (1, 1, 2, 32), U8), f("vs", (1, 1, 2), F32), PK(f), 0.125, 1.0, 0.5),
  empty_rule="never")
e("S.eb4_resources", NO, [], note="kernel attribute table; no tensor operand")

# ------------------------------------------------------------------- Python aliases assigned onto _C (__init__.py)
e("C.apa_selective_attention_sp_d1", COMP, ["q", "k", "kq", "v", "sinks"], alias_of="S.run",
  note="__init__.py:1656 _apa_spo2_entry -> _spo2.run (guarded there)",
  call=lambda f: C("apa_selective_attention_sp_d1")(f("q", (1, 1, 1, 64), BF16), f("k", (1, 1, 2, 64), BF16),
                                                    f("kq", (1, 1, 2, 64), BF16), f("v", (1, 1, 2, 64), BF16), 0.125,
                                                    1.0), empty_rule="never")
e("C.apa_spo2_dump", COMP, ["q", "k", "kq", "v", "sinks"], alias_of="S.run", note="__init__.py:1657 = _spo2.run",
  call=lambda f: C("apa_spo2_dump")(f("q", (1, 1, 1, 64), BF16), f("k", (1, 1, 2, 64), BF16),
                                    f("kq", (1, 1, 2, 64), BF16), f("v", (1, 1, 2, 64), BF16), 0.125, 1.0),
  empty_rule="never")
e("C.apa_spq1_dump", COMP, ["q", "k", "v"] + PACKED + ["sinks"], alias_of="S.packed_run",
  note="__init__.py:1746 = _spo2.packed_run",
  call=lambda f: C("apa_spq1_dump")(f("q", (1, 1, 1, 64), BF16), f("k", (1, 1, 2, 64), BF16),
                                    f("v", (1, 1, 2, 64), BF16), PK(f), 0.125, 1.0, fmt="tq4"), empty_rule="never")
e("C.apa_spq1_rotate", COMP, ["q", "r"], alias_of="S.packed_rotate", note="__init__.py:1747 = _spo2.packed_rotate",
  call=lambda f: C("apa_spq1_rotate")(f("q", (1, 1, 1, 64), BF16), f("r", (64, 64), F32), 1, True),
  empty_rule="never")
e("C.apa_selective_attention_sp_d1_packed", COMP, ["q", "k", "v"] + PACKED, alias_of="S.packed_run",
  note="__init__.py:1757 _apa_spq4_entry -> _spo2.packed_run (guarded there); a Python wrapper that may check first",
  call=lambda f: C("apa_selective_attention_sp_d1_packed")(f("q", (1, 1, 1, 64), BF16), f("k", (1, 1, 2, 64), BF16),
                                                           f("v", (1, 1, 2, 64), BF16), PK(f), 0.125, 1.0, fmt="tq4"),
  empty_rule="never")


def keys_by_class():
    out = {}
    for k, r in ROWS.items():
        out.setdefault(r["cls"], []).append(k)
    return out


# Entries whose pre-existing check refuses every host tensor operand before any CUDA call (seat reading at 54efb92ce,
# confirmed on CPU: Build B raises with 0 CUDA calls for the all-host row). They keep their own message (order H2).
REF_CITE = {'C.apa_gemm_selective_attention': 'gemm_apa.cu:83-86 validate(): q, k, v on one CUDA device', 'C.apa_gemm_selective_attention_cached': "gemm_apa.cu:83-86 (q, k, v) and :164-169 validate_cached_k (codes, scales on K's device)", 'C.apa_gemm_selective_debug': 'gemm_apa.cu:83-86 validate(): q, k on one CUDA device', 'C.apa_gemm_selective_quantize_k': 'gemm_apa.cu:674-677: CUDA BF16 K required', 'C.apa_int4_workspace': "kernels.cu apa_int4_workspace_create: 'apa_int4_workspace requires CUDA K'", 'C.apa_selective_attention_sp': 'apa_spo/launcher.cuh:164 (q, k, kq, v) and :182 (sinks): one CUDA device', 'C.apa_selective_attention_sp_epsilon': 'apa_spo/launcher.cuh:164/:182 through the SP2 epsilon wrapper', 'C.apa_sp1_1_baseline_diagnostics': 'apa_sp2.cuh:61-62 / apa_sp1_1.cuh: fp32 CUDA operands on one device', 'C.apa_sp2_scores': 'apa_sp2.cuh:76-77: q, k, kq CUDA on one device', 'C.apa_sp2_baseline_thresholds': 'apa_sp2.cuh:61-62: q, kq CUDA on one device', 'C.apa_selective_bwd_variant': "ops.cpp:1386-1387 (Build B line): all eight operands on q's CUDA device", 'C.bp_kernel_4_diagnostic': 'kernels.cu:8961 (Build B line): q, k, kq, v CUDA BF16 on one device', 'C.w8a16_matmul': 'kernels.cu:4717-4719 (Build B line): x, codes, scales on one CUDA device', 'C.export_rows': "kernels.cu export_rows: 'export_rows: CUDA tensor required' (single operand)", 'C.splice_rows': 'kernels.cu splice_rows: old_cache/insert device match + CUDA', 'C.evict_rows': "kernels.cu evict_rows: 'evict_rows: CUDA tensor required' (single operand)", 'S.run': 'apa_spo/launcher.cuh:164/:182', 'S.packed_prepare': 'apa_spo/packed_launcher.cuh:40/:51', 'S.packed_rotate': 'apa_spo/packed_launcher.cuh:30/:32', 'S.packed_append': 'apa_spo/packed_launcher.cuh:9/:12/:83', 'S.packed_run': 'apa_spo/packed_launcher.cuh:9/:12 (k, codes, scales), :97 (codebook), spq core (q, k, v, rotation, sinks)', 'S.int4_stats': 'apa_spo/packed_launcher.cuh:118-119', 'S.packed_prefill': 'apa_spo/packed_prefill.cuh:205/:217/:229/:234', 'S.eb2b_quantize': 'apa_spo/eb2b.cuh:129', 'S.eb2b_values': 'apa_spo/eb2b.cuh:108', 'S.eb2b_run': 'apa_spo/eb2b.cuh:173-174', 'S.eb3_run': 'apa_spo/eb3.cuh:924/:939', 'S.eb4_run': 'apa_spo/eb4.cuh:694/:709/:711', 'C.apa_selective_attention_sp_d1': 'alias of S.run (launcher.cuh:164/:182)', 'C.apa_spo2_dump': 'alias of S.run', 'C.apa_spq1_dump': 'alias of S.packed_run', 'C.apa_spq1_rotate': 'alias of S.packed_rotate', 'C.apa_selective_attention_sp_d1_packed': 'Python wrapper of S.packed_run'}
for _k, _cite in REF_CITE.items():
    ROWS[_k]["cls"] = REF
    ROWS[_k]["note"] = "pre-existing refusal: " + _cite
ROWS["W.reset"]["cls"] = NO
ROWS["W.reset"]["note"] = "apa_int4_workspace_reset only sets valid_rows = 0 (kernels.cu); no device work"
