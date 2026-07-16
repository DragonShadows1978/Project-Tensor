"""Project Tensor — standalone CUDA tensor library (no PyTorch, no CuPy).

Phase 1 (the core engine) exposes a NumPy-friendly Tensor with reverse-mode
autograd backed entirely by hand-written CUDA kernels + cuBLAS. The C++ does the
work; this module is the thin Python ergonomics layer (factory helpers, dtype
plumbing, no_grad). See ROADMAP.md for the layers still being ported.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
import operator

import numpy as np

try:
    from . import _tensor_cuda as _C
except ImportError:  # pragma: no cover
    import _tensor_cuda as _C  # fallback when the .so sits on PYTHONPATH

Tensor = _C.Tensor

_NP_DTYPE = {
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
    "bool": np.bool_,
    "uint8": np.uint8,
}

# bfloat16 has no NumPy equivalent: build the host array as fp32 and cast the
# device tensor to bf16 afterward. Listing it here so callers passing
# dtype="bfloat16" no longer fall through _NP_DTYPE.get(..., np.float32) and get
# a SILENT fp32 downcast (which corrupts bf16 constants for OLMoE/Qwen-class
# models). dtype_from_string in the C++ already supports "bfloat16".
_DEVICE_CAST_DTYPE = {"bfloat16"}


def tensor(data, *, device="cuda", dtype="float32", requires_grad=False):
    """Create a Tensor from array-like data."""
    if dtype in _DEVICE_CAST_DTYPE:
        arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
        return _C.tensor(arr, device, requires_grad).astype(dtype)
    arr = np.asarray(data, dtype=_NP_DTYPE.get(dtype, np.float32))
    arr = np.ascontiguousarray(arr)
    return _C.tensor(arr, device, requires_grad)


def _factory(np_fn):
    def make(*shape, device="cuda", dtype="float32", requires_grad=False):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if dtype in _DEVICE_CAST_DTYPE:
            arr = np.ascontiguousarray(np_fn(shape).astype(np.float32))
            return _C.tensor(arr, device, requires_grad).astype(dtype)
        arr = np_fn(shape).astype(_NP_DTYPE.get(dtype, np.float32))
        return _C.tensor(np.ascontiguousarray(arr), device, requires_grad)
    return make


zeros = _factory(np.zeros)
ones = _factory(np.ones)
randn = _factory(lambda s: np.random.randn(*s))
rand = _factory(lambda s: np.random.rand(*s))


def from_numpy(arr, *, device="cuda", requires_grad=False):
    return _C.tensor(np.ascontiguousarray(arr), device, requires_grad)


def matmul(a, b, alpha=1.0, trans_b=False):
    # trans_b reads b as (..., N, K) row-major via cuBLAS OP_T — no transpose
    # copy. alpha is applied in the fp32 accumulator before the 16-bit store.
    return _C.matmul(a, b, alpha, trans_b)


def causal_softmax(scores):
    """Fused bottom-right causal softmax over (..., L, S) scores, S >= L.
    Inference-only (backward raises). Equivalent to adding the -1e4 causal
    bias then softmax, with masked entries exactly zero."""
    return _C.causal_softmax(scores)


def fused_sdpa_noncausal(q, k, v, scale):
    """Streaming non-causal SDPA for matching ``[B,H,L,D]`` tensors.

    Inference-only primitive used by ``functional.scaled_dot_product_attention``
    for HY3D-sized unmasked attention. It keeps only per-row online-softmax
    state and never materializes an ``Lq x Lk`` score matrix.
    """
    return _C.fused_sdpa_noncausal(q, k, v, float(scale))


def apa_int4_sdpa_noncausal(q, k, v, scale, zthr, refine_all=False):
    """Fused non-causal APA attention with packed symmetric INT4 bulk K.

    Inference-only, opt-in (EXP-APA-2). ``[B,H,L,D]`` q/k/v, same fp16/fp32
    dtype, even ``D <= 128``. Bulk scores come from an in-kernel packed-INT4
    K (symmetric-7 grid, one group per key vector — the EXP-APA-1
    convention); keys with ``|bulk| >= mean + zthr*std`` are rescored at
    full precision; the rest keep their bulk score (nothing is dropped) and
    the softmax denominator is exact over all keys, V full precision.
    ``refine_all=True`` (the r >= 1.0 case) skips the bulk pass entirely and
    is exact streaming SDPA. No ``Lq x Lk`` tensor is ever materialized.
    """
    return _C.apa_int4_sdpa_noncausal(q, k, v, float(scale), float(zthr),
                                      bool(refine_all))


def apa_refine_stats(reset=False):
    """EXP-APA-4 (K2) realized-refine-fraction counters.

    Returns ``(refined_pairs, total_pairs)`` accumulated by the Q-tile APA
    kernel (r < 1) across calls since the last reset, when ``TC_APA_FRAC=1``
    was set for those calls; both are 0 if the instrumentation never engaged.
    ``reset=True`` clears both counters after reading. Realized refine
    fraction = refined / total. The legacy streaming path (TC_ATTN_QTILE=0)
    is the frozen EXP-APA-2 instrument and is not instrumented.
    """
    return _C.apa_refine_stats(bool(reset))


def dda_raycast(grid_u8, origins_f32, directions_f32, max_steps):
    """First-hit voxel DDA over a resident 3D uint8 grid.

    Returns ``(hit_u8, material_u8, voxel_i64, face_axis_i64,
    face_sign_i64, distance_f32)`` with one output row per input ray.
    The operation is non-differentiable and all outputs are detached.
    """
    return _C.dda_raycast(
        grid_u8, origins_f32, directions_f32, max_steps
    )


def raster_winner_scatter_min(
    pixel_indices_i64,
    depth_keys_i64,
    face_ids_i64,
    pixel_count,
    *,
    threads=256,
):
    """Deterministic duplicate-aware winner scatter-min.

    Returns ``(winner_depth_i64, winner_face_i64)`` of length ``pixel_count``.
    Winner order is signed-int32 depth followed by lower positive one-based
    face ID.  Packed uint64 keys stay internal; background is
    ``(INT32_MAX, 0)``.  Detached, CUDA-only, and order-independent.
    """
    return _C.raster_winner_scatter_min(
        pixel_indices_i64,
        depth_keys_i64,
        face_ids_i64,
        pixel_count,
        threads,
    )


def raster_triangle_winners(
    clip_positions_f32,
    faces_i64,
    height,
    width,
    *,
    face_threads=256,
):
    """Raster-produce deterministic depth/face winners from clip triangles.

    Returns ``(winner_depth_i64[H,W], winner_face_i64[H,W])``.  Coverage is
    the frozen exact barycentric conjunction for both windings, without an
    epsilon or culling.  ``face_threads`` changes launch shape only.
    """
    return _C.raster_triangle_winners(
        clip_positions_f32, faces_i64, height, width, face_threads
    )


def raster_winner_resolve(
    clip_positions_f32,
    faces_i64,
    winner_face_i64,
    *,
    pixel_threads=256,
):
    """Resolve one won pixel per thread from the frozen fp32 operation order.

    Returns ``(resolved_depth_i64[H,W], barycentric_f32[H,W,3])`` and writes
    ``(INT32_MAX, [0,0,0])`` for background pixels.
    """
    return _C.raster_winner_resolve(
        clip_positions_f32, faces_i64, winner_face_i64, pixel_threads
    )


def rasterize_clip(
    clip_positions_f32,
    faces_i64,
    height,
    width,
    *,
    face_threads=256,
    pixel_threads=256,
):
    """Two-pass deterministic clip rasterizer.

    Returns ``(face_ids_i64, barycentric_f32, depth_keys_i64)`` in the same
    field order as the frozen paint oracle.  Winner selection uses internal
    packed-key atomic minimum; attributes are recomputed after selection.
    """
    return _C.rasterize_clip(
        clip_positions_f32,
        faces_i64,
        height,
        width,
        face_threads,
        pixel_threads,
    )


def bake_back_project(
    atlas_positions_h_f32,
    view_f32,
    view_depth_f32,
    view_reliable_u8,
    view_cosine_f32,
    world_to_camera_f32,
    image_projection_f32,
    depth_threshold,
    *,
    threads=256,
):
    """Project atlas samples into one square view and bilinearly gather it.

    Returns ``(valid_u8[N], colors_f32[N,C], cosine_f32[N],
    depth_delta_f32[N])``.  Projection and four-tap interpolation preserve the
    frozen paint oracle's fp32 operation order.  Invalid color/cosine rows are
    zero; depth deltas remain reported for visibility diagnostics.
    """
    return _C.bake_back_project(
        atlas_positions_h_f32,
        view_f32,
        view_depth_f32,
        view_reliable_u8,
        view_cosine_f32,
        world_to_camera_f32,
        image_projection_f32,
        float(depth_threshold),
        threads,
    )


def bake_cosine_blend(
    view_colors_f32,
    view_cosine_f32,
    view_valid_u8,
    view_weights_f32,
    view_enabled_u8,
    *,
    threads=256,
):
    """Blend per-view atlas samples in frozen dimension-0 order.

    Returns ``(texture_f32[N,C], trust_f32[N], valid_u8[N])``.  Cosine power,
    positive-weight comparison, accumulation, and normalization are fused; no
    view-sized weight tensor or cross-view atomic is created.  ``view_enabled``
    carries the caller's deterministic overlap/skip decisions.
    """
    return _C.bake_cosine_blend(
        view_colors_f32,
        view_cosine_f32,
        view_valid_u8,
        view_weights_f32,
        view_enabled_u8,
        threads,
    )


def build_inpaint_island_csr(
    faces_i64,
    vertex_islands_i64,
    uncolored_occurrences_i64,
):
    """Build the PAINT-CUDA-3 stable CSR and segmented-island metadata on CPU.

    Returns ``(neighbor_offsets_i64, neighbors_i64, active_island_ids_i64,
    island_offsets_i64, island_occurrences_i64)`` as contiguous NumPy arrays.
    Face/corner neighbor order and occurrence order within each island are
    stable, matching the frozen mesh-inpaint oracle.
    """
    faces = np.ascontiguousarray(np.asarray(faces_i64, dtype=np.int64))
    vertex_islands = np.ascontiguousarray(
        np.asarray(vertex_islands_i64, dtype=np.int64)
    )
    occurrences = np.ascontiguousarray(
        np.asarray(uncolored_occurrences_i64, dtype=np.int64)
    )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces_i64 must have shape (F, 3)")
    if vertex_islands.ndim != 1:
        raise ValueError("vertex_islands_i64 must have shape (V,)")
    if occurrences.ndim != 1:
        raise ValueError("uncolored_occurrences_i64 must have shape (O,)")
    vertex_count = len(vertex_islands)
    if np.any(vertex_islands < 0):
        raise ValueError("vertex_islands_i64 must contain non-negative labels")
    if np.any(faces < 0) or np.any(faces >= vertex_count):
        raise ValueError("faces_i64 contains a vertex outside vertex_islands_i64")
    if np.any(occurrences < 0) or np.any(occurrences >= vertex_count):
        raise ValueError(
            "uncolored_occurrences_i64 contains a vertex outside vertex_islands_i64"
        )

    sources = np.ascontiguousarray(faces.reshape(-1), dtype=np.int64)
    targets = np.ascontiguousarray(
        np.roll(faces, -1, axis=1).reshape(-1), dtype=np.int64
    )
    if len(sources) and np.any(vertex_islands[sources] != vertex_islands[targets]):
        raise ValueError("vertex_islands_i64 splits an edge across island labels")
    source_order = np.argsort(sources, kind="stable")
    counts = np.bincount(sources, minlength=vertex_count).astype(
        np.int64, copy=False
    )
    neighbor_offsets = np.empty(vertex_count + 1, dtype=np.int64)
    neighbor_offsets[0] = 0
    np.cumsum(counts, out=neighbor_offsets[1:])
    neighbors = np.ascontiguousarray(targets[source_order], dtype=np.int64)

    if len(occurrences):
        occurrence_islands = vertex_islands[occurrences]
        active_islands, inverse = np.unique(occurrence_islands, return_inverse=True)
        island_order = np.argsort(inverse, kind="stable")
        island_occurrences = np.ascontiguousarray(
            occurrences[island_order], dtype=np.int64
        )
        island_counts = np.bincount(
            inverse, minlength=len(active_islands)
        ).astype(np.int64, copy=False)
    else:
        active_islands = np.empty(0, dtype=np.int64)
        island_occurrences = np.empty(0, dtype=np.int64)
        island_counts = np.empty(0, dtype=np.int64)
    island_offsets = np.empty(len(active_islands) + 1, dtype=np.int64)
    island_offsets[0] = 0
    np.cumsum(island_counts, out=island_offsets[1:])
    return (
        np.ascontiguousarray(neighbor_offsets),
        neighbors,
        np.ascontiguousarray(active_islands, dtype=np.int64),
        np.ascontiguousarray(island_offsets),
        island_occurrences,
    )


def inpaint_island_passes(
    positions_f32,
    vertex_colors_f32,
    vertex_mask_f32,
    neighbor_offsets_i64,
    neighbors_i64,
    island_offsets_i64,
    island_occurrences_i64,
    pass_count_cap,
    *,
    threads=128,
):
    """Execute bounded reference-order mesh smoothing on segmented islands.

    One warp owns one island and only lane zero visits its occurrences, so
    disconnected islands run concurrently without reordering work inside an
    island.  The call executes exactly ``pass_count_cap`` complete passes and
    returns cloned updated colors/mask plus final uncolored counts by active
    island.  Exact convergence/deadline policy remains host-side and normally
    drives this primitive one pass per launch.
    """
    return _C.inpaint_island_passes(
        positions_f32,
        vertex_colors_f32,
        vertex_mask_f32,
        neighbor_offsets_i64,
        neighbors_i64,
        island_offsets_i64,
        island_occurrences_i64,
        operator.index(pass_count_cap),
        operator.index(threads),
    )


_TERRAIN_RENDER_FROZEN_CONSTANTS = {
    "ambient_floor": np.float32(0.35),
    "ao_strength": np.float32(0.60),
    "ao_floor": np.float32(0.40),
    "value_jitter": np.float32(0.08),
    "channel_mix": np.float32(0.04),
}


def _terrain_lookup(source, names, label):
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
    else:
        for name in names:
            if hasattr(source, name):
                return getattr(source, name)
    raise ValueError(f"terrain_render: {label} is required")


def _terrain_vector(value, label):
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"terrain_render: {label} must be a finite three-vector"
        )
    return np.ascontiguousarray(vector)


def _terrain_camera_terms(cam):
    """Build frozen Scorch camera terms without constructing any ray array."""
    if isinstance(cam, (tuple, list)) and len(cam) == 6:
        position, look_at, world_up, fov, width, height = cam
    else:
        position = _terrain_lookup(cam, ("position",), "cam.position")
        look_at = _terrain_lookup(cam, ("look_at", "target"), "cam.look_at")
        world_up = _terrain_lookup(cam, ("world_up", "up"), "cam.world_up")
        fov = _terrain_lookup(
            cam,
            ("vertical_fov_degrees", "fov_degrees", "fov"),
            "cam.vertical_fov_degrees",
        )
        width = _terrain_lookup(cam, ("width",), "cam.width")
        height = _terrain_lookup(cam, ("height",), "cam.height")

    position = _terrain_vector(position, "cam.position")
    look_at = _terrain_vector(look_at, "cam.look_at")
    world_up = _terrain_vector(world_up, "cam.world_up")
    try:
        fov = float(fov)
        width = int(width)
        height = int(height)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "terrain_render: camera fov, width, and height must be numeric"
        ) from exc
    if not np.isfinite(fov) or not 0.0 < fov < 180.0:
        raise ValueError(
            "terrain_render: vertical_fov_degrees must lie between 0 and 180"
        )
    if width <= 0 or height <= 0:
        raise ValueError("terrain_render: image dimensions must be positive")

    # Keep the float32 operation order identical to Project-Scorch's
    # render.camera.camera_rays.  This is O(1) camera setup, not CPU ray work.
    forward = look_at - position
    forward_length = float(np.linalg.norm(forward))
    if forward_length == 0.0:
        raise ValueError("terrain_render: cam.position and cam.look_at must differ")
    forward /= np.float32(forward_length)
    right = np.cross(forward, world_up)
    right_length = float(np.linalg.norm(right))
    if right_length < 1.0e-7:
        raise ValueError(
            "terrain_render: cam.world_up must not be parallel to view direction"
        )
    right /= np.float32(right_length)
    camera_up = np.cross(right, forward).astype(np.float32)
    half_height = np.float32(np.tan(np.deg2rad(fov) * 0.5))
    half_width = half_height * np.float32(width / height)

    return (
        position,
        np.ascontiguousarray(forward),
        np.ascontiguousarray(right),
        np.ascontiguousarray(camera_up),
        float(half_width),
        float(half_height),
        width,
        height,
    )


def _terrain_light_direction(light):
    if isinstance(light, Mapping) or hasattr(light, "direction"):
        direction = _terrain_lookup(
            light,
            ("direction", "light_dir", "light_direction"),
            "light.direction",
        )
    else:
        direction = light
    direction = _terrain_vector(direction, "light.direction")
    length = float(np.linalg.norm(direction))
    if length < 1.0e-7:
        raise ValueError("terrain_render: light.direction must be non-zero")
    return np.ascontiguousarray(direction / np.float32(length))


def _terrain_max_steps(consts, materials_shape, object_shapes=()):
    default_steps = int(
        max(
            (sum(materials_shape), *(sum(shape) for shape in object_shapes))
        )
        + 3
    )
    if consts is None:
        return default_steps
    if isinstance(consts, Mapping):
        for name, frozen in _TERRAIN_RENDER_FROZEN_CONSTANTS.items():
            if name in consts and not np.isclose(
                np.float32(consts[name]), frozen, rtol=0.0, atol=1.0e-7
            ):
                raise ValueError(
                    f"terrain_render: {name} is frozen at {float(frozen)}"
                )
        value = consts.get("max_steps", default_steps)
    else:
        value = consts
    try:
        max_steps = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("terrain_render: consts.max_steps must be an integer") from exc
    if max_steps <= 0:
        raise ValueError("terrain_render: consts.max_steps must be positive")
    return max_steps


def _terrain_surface_mode(surface_mode):
    if not isinstance(surface_mode, str):
        raise TypeError("terrain_render: surface_mode must be 'blocky' or 'smooth'")
    if surface_mode not in ("blocky", "smooth"):
        raise ValueError("terrain_render: surface_mode must be 'blocky' or 'smooth'")
    return surface_mode


def _terrain_density_filter(density_filter):
    if isinstance(density_filter, (bool, np.bool_)):
        raise TypeError(
            "terrain_render: density_filter must be an integer 0, 1, or 2"
        )
    try:
        parsed = operator.index(density_filter)
    except TypeError as exc:
        raise TypeError(
            "terrain_render: density_filter must be an integer 0, 1, or 2"
        ) from exc
    if parsed not in (0, 1, 2):
        raise ValueError("terrain_render: density_filter must be 0, 1, or 2")
    return parsed


def _terrain_detail(detail):
    if isinstance(detail, (bool, np.bool_)):
        raise TypeError("terrain_render: detail must be an integer 0 or 1")
    try:
        parsed = operator.index(detail)
    except TypeError as exc:
        raise TypeError(
            "terrain_render: detail must be an integer 0 or 1"
        ) from exc
    if parsed not in (0, 1):
        raise ValueError("terrain_render: detail must be 0 or 1")
    return parsed


def _terrain_grounding(grounding):
    if isinstance(grounding, (bool, np.bool_)):
        raise TypeError("terrain_render: grounding must be an integer 0 or 1")
    try:
        parsed = operator.index(grounding)
    except TypeError as exc:
        raise TypeError(
            "terrain_render: grounding must be an integer 0 or 1"
        ) from exc
    if parsed not in (0, 1):
        raise ValueError("terrain_render: grounding must be 0 or 1")
    return parsed


def _terrain_grounding_parameters(z_horizon, fog_start, fog_full):
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(
                (z_horizon, fog_start, fog_full), dtype=np.float32
            )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "terrain_render: grounding parameters must be numeric"
        ) from exc
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("terrain_render: grounding parameters must be finite")
    if values[1] < np.float32(0.0) or values[2] <= values[1]:
        raise ValueError(
            "terrain_render: fog range must satisfy 0 <= fog_start < fog_full"
        )
    return tuple(float(value) for value in values)


def _terrain_objects(objects, materials_u8_device):
    """Validate and normalize the small host-side object descriptor list."""
    if objects is None:
        return None
    if not isinstance(objects, list):
        raise TypeError("terrain_render: objects must be a list or None")
    if len(objects) > 16:
        raise ValueError("terrain_render: objects supports at most 16 entries")

    normalized = []
    for index, descriptor in enumerate(objects):
        label = f"terrain_render: objects[{index}]"
        if not isinstance(descriptor, (tuple, list)) or len(descriptor) != 3:
            raise TypeError(f"{label} must be (grid, origin, palette)")
        grid, origin, palette = descriptor
        if not isinstance(grid, Tensor):
            raise TypeError(f"{label} grid must be a Tensor")
        if grid.dtype != "uint8" or grid.ndim != 3 or any(
            extent <= 0 for extent in grid.shape
        ):
            raise ValueError(f"{label} grid must be non-empty uint8 [X,Y,Z]")
        if grid.device != materials_u8_device.device:
            raise ValueError(f"{label} grid must share the terrain device")
        origin = _terrain_vector(origin, f"objects[{index}].origin")
        if not isinstance(palette, Tensor):
            raise TypeError(f"{label} palette must be a Tensor")
        if (
            palette.dtype != "uint8"
            or palette.ndim != 2
            or palette.shape[0] <= 0
            or palette.shape[1] != 3
        ):
            raise ValueError(f"{label} palette must be non-empty uint8 [N,3]")
        if palette.device != materials_u8_device.device:
            raise ValueError(f"{label} palette must share the terrain device")
        normalized.append((grid, origin.tolist(), palette))
    return normalized


def terrain_render(
    materials_u8_device,
    cam,
    light,
    palette,
    consts=None,
    surface_mode="blocky",
    density_filter=0,
    detail=0,
    objects=None,
    grounding=0,
    z_horizon=0.0,
    fog_start=600.0,
    fog_full=2400.0,
):
    """Render a resident voxel terrain entirely on the GPU.

    ``materials_u8_device`` is a CUDA ``uint8`` tensor shaped ``(X, Y, Z)``;
    ``palette`` is a CUDA ``uint8`` tensor shaped ``(N, 3)``.  ``cam`` is a
    mapping/object with ``position``, ``look_at``, ``world_up``,
    ``vertical_fov_degrees`` (``fov_degrees`` is accepted), ``width``, and
    ``height``; a six-item tuple in that order is also accepted.  ``light`` is
    a 3-vector or has a ``direction`` field.  Its direction points from light
    to terrain, so the shader uses ``dot(normal, -light_dir)``.

    ``consts`` may be ``None``, a positive integer max-step count, or a mapping
    containing ``max_steps`` plus any frozen constants at their specified
    values.  The return is ``(rgb_uint8[H,W,3], depth_float32[H,W])`` with
    black / ``-1`` for a miss.  ``surface_mode="blocky"`` is the unchanged
    WO-7B DDA path; ``surface_mode="smooth"`` uses a render-only trilinear
    occupancy isosurface.  Smooth ``density_filter=1`` samples a cached
    centered 3-tap box density and level 2 samples a cached centered 9-tap
    binomial/Gaussian density; zero preserves WO-8A byte-for-byte.  Cached
    fields are normalized u8 and are invalidated by source-storage revision.
    ``detail=1`` adds deterministic world-anchored procedural surface detail;
    the default `detail=0` retains the literal pre-WO-9B kernel paths.
    ``objects`` is ``None`` or a list of at most 16
    ``(grid_u8_device[X,Y,Z], origin_f32x3, palette_u8_device[N,3])``
    descriptors.  Object grids are axis-aligned in world space and use blocky
    entry-face shading plus object-local AO; terrain surface/detail selectors do
    not alter them.  ``grounding=1`` replaces terrain hits that enter solid
    directly through an x/y domain wall or the z=0 underside with flat basalt,
    and fills grid misses whose forward ray crosses ``z_horizon`` outside the
    x/y domain.  Plane color linearly fogs to black from ``fog_start`` through
    ``fog_full`` while retaining its true depth.  This operation is
    non-differentiable.
    """
    if not isinstance(materials_u8_device, Tensor):
        raise TypeError("terrain_render: materials_u8_device must be a Tensor")
    if not isinstance(palette, Tensor):
        raise TypeError("terrain_render: palette must be a device Tensor")
    (
        position,
        forward,
        right,
        up,
        half_width,
        half_height,
        width,
        height,
    ) = _terrain_camera_terms(cam)
    light_direction = _terrain_light_direction(light)
    objects = _terrain_objects(objects, materials_u8_device)
    object_shapes = () if not objects else tuple(obj[0].shape for obj in objects)
    max_steps = _terrain_max_steps(
        consts, materials_u8_device.shape, object_shapes
    )
    surface_mode = _terrain_surface_mode(surface_mode)
    density_filter = _terrain_density_filter(density_filter)
    detail = _terrain_detail(detail)
    grounding = _terrain_grounding(grounding)
    z_horizon, fog_start, fog_full = _terrain_grounding_parameters(
        z_horizon, fog_start, fog_full
    )
    if surface_mode == "blocky" and density_filter:
        raise ValueError(
            "terrain_render: density_filter is only supported in smooth mode"
        )
    arguments = (
        materials_u8_device,
        palette,
        position.tolist(),
        forward.tolist(),
        right.tolist(),
        up.tolist(),
        half_width,
        half_height,
        width,
        height,
        light_direction.tolist(),
        max_steps,
        surface_mode,
        density_filter,
        detail,
    )
    # None/empty objects still select the established no-object C++ overload.
    if not objects:
        objects = None
    return _C.terrain_render(
        *arguments,
        objects,
        grounding,
        z_horizon,
        fog_start,
        fog_full,
    )


def argmax_last_axis(a):
    """Device-side argmax over the LAST axis only (returns int64, axis
    removed). Decode-loop fast path (KERNEL_OPT_IMPLEMENTATION_PLAN.md Phase
    1.1): block-per-row grid-stride + warp-shuffle reduction, so a single
    vocab-sized logits row still parallelizes across a full block instead of
    the generic one-thread-per-row `Tensor.argmax`. Tie-break matches
    numpy.argmax (lowest index wins). Detached (no autograd)."""
    return _C.argmax_last_axis(a)


def rms_norm(x, w, eps=1e-6):
    """Fused RMSNorm over the last dim (single kernel, fp32 accumulate,
    output in x's dtype). Inference-only: backward raises — training code
    must use an unfused op chain. w must be fp32."""
    return _C.rms_norm(x, w, eps)


def rope_apply(x, cos, sin, pos0=0, inverse=False, pair_swap=False):
    """Fused RoPE: out = x*cos[pos0+l] + rotate_half(x)*sin[pos0+l] in
    ONE launch (the composed chain is ~8). x (..., L, D); tables (T, D)
    in x's dtype. Inference-only: backward raises."""
    return _C.rope_apply(x, cos, sin, pos0, inverse, pair_swap)


def write_rows(buf, src, start=0):
    """IN-PLACE ring write: src rows land at (start+l) %% CAP along
    buf's dim -2. The decode-cache primitive (zero-copy appends).
    Inference-only (raises under grad). MUTATES buf — callers own the
    sharing contract: never alias a written buffer from a held cache."""
    _C.write_rows(buf, src, start)


def export_rows(cache, dim, start, length):
    return _C.export_rows(cache, dim, start, length)


def export_rope_rows(cache, cos, sin, dim, start, length, pos0,
                     inverse=False, pair_swap=False):
    return _C.export_rope_rows(cache, cos, sin, dim, start, length, pos0,
                               inverse, pair_swap)


def export_row_pair(raw_cache, rope_cache, cos, sin, raw_dim, rope_dim,
                    raw_start, rope_start, length, pos0, inverse=False,
                    pair_swap=False):
    return _C.export_row_pair(raw_cache, rope_cache, cos, sin, raw_dim,
                              rope_dim, raw_start, rope_start, length, pos0,
                              inverse, pair_swap)


def export_row_pairs(raw_caches, rope_caches, cos, sin, raw_dim, rope_dim,
                     raw_starts, rope_starts, length, pos0, inverse=False,
                     pair_swap=False):
    return _C.export_row_pairs(list(raw_caches), list(rope_caches), cos, sin,
                               raw_dim, rope_dim, list(raw_starts),
                               list(rope_starts), length, pos0, inverse,
                               pair_swap)


def swap_row_pairs_with_rope(raw_caches, rope_caches, raw_inserts,
                             rope_inserts, cos, sin, raw_dim, rope_dim,
                             head_tokens, tail_start, pos0, pair_swap=False):
    return _C.swap_row_pairs_with_rope(
        list(raw_caches), list(rope_caches), list(raw_inserts),
        list(rope_inserts), cos, sin, raw_dim, rope_dim, head_tokens,
        tail_start, pos0, pair_swap)


def evict_row_pairs(raw_caches, rope_caches, raw_dim, rope_dim, head_tokens,
                    drop_tokens):
    return _C.evict_row_pairs(list(raw_caches), list(rope_caches), raw_dim,
                              rope_dim, head_tokens, drop_tokens)


def arena_row_pair_transaction(raw_caches, rope_caches, raw_inserts,
                               rope_inserts, cos, sin, raw_dim, rope_dim,
                               sink_tokens, current_mount_tokens,
                               arena_width, pair_swap=False):
    return _C.arena_row_pair_transaction(
        list(raw_caches), list(rope_caches), list(raw_inserts),
        list(rope_inserts), cos, sin, raw_dim, rope_dim, sink_tokens,
        current_mount_tokens, arena_width, pair_swap)


def int4_linear(x, packed, scales, zeros, group_size=128):
    """INT4 group-quantized linear: y = x @ dequant(W)^T. Inference only.
    Two-stage: dequant W to a full (K,N) fp16 buffer then cuBLAS matmul."""
    return _C.int4_linear(x, packed, scales, zeros, group_size)


def int4_linear_fused(x, packed, scales, zeros, group_size=128):
    """INT4 linear via a fused dequant-GEMM — same result as int4_linear but
    dequantizes the weight in shared-memory tiles inside the GEMM, avoiding the
    full (K,N) fp16 weight transient. Opt-in: a hand GEMM can lose to cuBLAS at
    large N, so benchmark before defaulting to it."""
    return _C.int4_linear_fused(x, packed, scales, zeros, group_size)


def w8a16_matmul(x, codes, scales, launch_config="m64n16"):
    """FP16 activations times Trinity-compatible group-32 symmetric-INT8 weights.

    Args:
        x: Contiguous FP16 activation with shape ``(..., K)`` and rank >= 2.
        codes: UINT8 codes with shape ``(O, K)``; signed q is ``code - 128``.
        scales: FP16 scales with shape ``(O, K // 32)``.
        launch_config: ``"m64n16"`` (default) or ``"m16n64"``.

    Returns FP16 ``(..., O)`` with FP32 accumulation. K must be positive and
    divisible by 32. The fixed ``(O,K)`` weight broadcasts across every leading
    activation dimension, covering batched linears and im2col conv matrices.
    Weight tiles are dequantized to FP16 shared memory; no full FP16 weight is
    materialized. Frozen-weight inference only; backward is unsupported.
    """
    configs = {"m64n16": 0, "m16n64": 1}
    try:
        config_id = configs[launch_config]
    except (KeyError, TypeError):
        raise ValueError(
            f"launch_config must be one of {tuple(configs)}, got {launch_config!r}"
        ) from None
    return _C.w8a16_matmul(x, codes, scales, config_id)


def int4_dequant(packed, scales, zeros, group_size=128, out_dtype="float16"):
    """Dequantize packed INT4 weight to a (K, N) transposed fp16/fp32 matrix."""
    return _C.int4_dequant(packed, scales, zeros, group_size, out_dtype)


def intn_linear(x, packed, scales, zeros, bits, in_features, group_size=128):
    """INT2/INT3 group-quantized linear: y = x @ dequant(W)^T."""
    return _C.intn_linear(x, packed, scales, zeros, bits, in_features, group_size)


def intn_linear_fused(x, packed, scales, zeros, bits, in_features, group_size=128):
    """INT2/INT3 fused dequant-GEMM/GEMV path without a full weight transient."""
    return _C.intn_linear_fused(
        x, packed, scales, zeros, bits, in_features, group_size
    )


def intn_dequant(
    packed, scales, zeros, bits, in_features, group_size=128, out_dtype="float16"
):
    """Dequantize packed INT2/INT3 weight to a (K, N) transposed matrix."""
    return _C.intn_dequant(
        packed, scales, zeros, bits, in_features, group_size, out_dtype
    )


def mxfp4_linear(x, blocks, scales):
    """GPT-OSS MXFP4 expert linear: y = x @ dequant(blocks, scales).

    `blocks` is shaped `(out_features, groups, 16)` uint8 and `scales` is
    `(out_features, groups)` uint8. Each 16-byte group expands to 32 FP4
    weights with E8M0 exponent scales. Inference-only frozen-weight path.
    """
    return _C.mxfp4_linear(x, blocks, scales)


def mxfp4_linear_expert(x, blocks, scales, expert_idx):
    """GPT-OSS resident MXFP4 expert linear from `[experts, N, G, 16]`.

    `expert_idx` selects the packed expert inside the CUDA op, avoiding uint8
    slicing in Python.
    """
    return _C.mxfp4_linear_expert(x, blocks, scales, expert_idx)


def kv_int4_pack(x, group=32):
    """Quantize+pack a KV tensor (B,KV,S,D) to D-grouped 4-bit. Returns
    (packed_uint8 (B,KV,S,D/2), scales (B,KV,S,D/group) in x.dtype).
    Symmetric-8, group-32. Distinct from int4_dequant (weight, K-grouped)."""
    return _C.kv_int4_pack(x, group)


def kv_int4_unpack(packed, scales, group=32, lo=0, n=0, out_dtype="bfloat16"):
    """Dequantize rows [lo:lo+n) of a packed KV buffer -> (B,KV,n,D)."""
    return _C.kv_int4_unpack(packed, scales, group, lo, n, out_dtype)


def gated_delta_step(q, k, v, a, b, A_neg, dt_bias, state):
    """Fused Gated DeltaNet decode step (one token, one layer, ONE launch):
    l2norm(q,k) + gate math (sigmoid/softplus/exp) + decay-first delta-rule
    state update + readout. All fp32. FUNCTIONAL: returns (out, new_state)
    and leaves the input state untouched, so callers may branch or hold
    references freely (the GRM restore-once-decode-many contract).
    q,k (B,Hk,Dk) raw heads; v (B,H,Dv); a,b (B,H); A_neg = -exp(A_log),
    dt_bias: H elements; state (B,H,Dk,Dv). Inference only (no autograd)."""
    return _C.gated_delta_step(q, k, v, a, b, A_neg, dt_bias, state)


def apa_selective_attention(q, k, kq, v, scale, zthr, is_causal=False):
    """Fused sparse selective APA attention: full-precision dot only on the keys
    the bulk/quantized pass selects (|bulk| >= mean+zthr*std), rest stay quantized.
    q,k,kq,v: (B,H,L,D)/(B,H,S,D). Inference only (no autograd)."""
    return _C.apa_selective_attention(q, k, kq, v, scale, zthr, is_causal)


def apa_selective_attention_sink(q, k, kq, v, sinks, scale, zthr, is_causal=False):
    """Sink-aware fused sparse selective APA attention for GPT-OSS.

    `sinks` is `(H,)`. Each head's sink logit participates in the online softmax
    denominator and contributes no value vector, so the returned output remains
    `(B,H,L,VD)`.
    """
    return _C.apa_selective_attention_sink(q, k, kq, v, sinks, scale, zthr, is_causal)


def apa_blend_softmax(bulk, rank, zthr, Lq=0, row0=0, window=0):
    """Fused APA blend+softmax over precomputed bulk/rank score matrices (..., S):
    per row thr = mean(|bulk|)+zthr*std(|bulk|); score = |bulk|>=thr ? rank : bulk;
    returns softmax(score). Pairs with cuBLAS bulk/rank matmuls.

    Two bounds conventions (Phase 3.1, board item 4a):
      Lq=0 (default): legacy sentinel path. Causal/window masking must already
        be baked into bulk/rank as large-negative scores by the caller.
      Lq>0: index-arithmetic path — no mask tensor needed, masked keys are
        never read. `row0` is the absolute query-chunk start (tiled callers
        slice queries into blocks of `blk` rows); `Lq` is the FULL query
        length `L` (not the chunk length); `window` is the sliding-window
        width (0 = full causal from key 0). Bottom-right causal convention:
        row i sees keys `0..(S-Lq)+row0+i` inclusive, matching
        `functional._causal_mask`. Sliding window: keys
        `(q_abs-window, q_abs]`, matching `gpt_oss20b_tc._gpt_oss_attention_mask`.
    """
    return _C.apa_blend_softmax(bulk, rank, zthr, Lq, row0, window)


def apa_blend_softmax_sink(bulk, rank, sinks, zthr, Lq=0, row0=0, window=0):
    """Sink-aware APA blend weights for GPT-OSS attention.

    `bulk` and `rank` are `(B,H,L,S)` score tensors. `sinks` is `(H,)`.
    Selection stats are computed over valid key scores only; the sink logit
    participates in the softmax denominator but no sink column is returned,
    so the result remains `(B,H,L,S)` for `weights @ V`.

    Bounds conventions identical to `apa_blend_softmax` above: `Lq=0` is the
    legacy sentinel path (masks baked into bulk/rank as large-negative bias);
    `Lq>0` is index-arithmetic (no mask tensor needed; `row0`/`Lq`/`window` as
    above).
    """
    return _C.apa_blend_softmax_sink(bulk, rank, sinks, zthr, Lq, row0, window)


def mse_loss(pred, target):
    return _C.mse_loss(pred, target)


def where(cond, x, y):
    return _C.where(cond, x, y)


def einsum(equation, *operands):
    from . import functional
    return functional.einsum(equation, *operands)


def embedding(weight, idx):
    if not isinstance(idx, Tensor):
        idx = _C.tensor(np.ascontiguousarray(np.asarray(idx, dtype=np.int64)), "cuda", False)
    return _C.embedding(weight, idx)


def cat(tensors, dim=0):
    return _C.cat(list(tensors), dim)


def splice_rows(old_cache, insert, dim, head_tokens, tail_start):
    return _C.splice_rows(old_cache, insert, dim, head_tokens, tail_start)


def evict_rows(old_cache, dim, head_tokens, drop_tokens):
    return _C.evict_rows(old_cache, dim, head_tokens, drop_tokens)


def stack(tensors, dim=0):
    return _C.stack(list(tensors), dim)


def cross_entropy(logits, target, *, device="cuda"):
    """Cross-entropy from logits. `target` may be int class labels (1D) or a
    float one-hot Tensor matching `logits`."""
    if isinstance(target, Tensor):
        return _C.cross_entropy(logits, target)
    labels = np.asarray(target).astype(np.int64).ravel()
    num_classes = logits.shape[-1]
    onehot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    onehot[np.arange(labels.shape[0]), labels] = 1.0
    onehot = onehot.reshape(logits.shape)
    return _C.cross_entropy(logits, _C.tensor(onehot, device, False))


def save_checkpoint(path, model, **extra):
    """Save model parameters (+ optional extra scalars) to a .npz file."""
    sd = model.state_dict()
    payload = {f"model.{k}": v for k, v in sd.items()}
    for k, v in extra.items():
        payload[f"extra.{k}"] = np.array(v)
    np.savez(path, **payload)


def load_checkpoint(path, model):
    """Load parameters saved by save_checkpoint into `model`. Returns extras."""
    data = np.load(path, allow_pickle=True)
    sd = {k[len("model."):]: data[k] for k in data.files if k.startswith("model.")}
    model.load_state_dict(sd)
    return {k[len("extra."):]: data[k] for k in data.files if k.startswith("extra.")}


def weight_tie(src_module, src_attr, dst_module, dst_attr):
    """Tie two parameters to share one Tensor (e.g. embedding <-> LM head).

    Both modules then reference the same parameter object; gradients accumulate
    once and optimizers (which dedup by identity) update it once.
    """
    shared = getattr(src_module, src_attr)
    setattr(dst_module, dst_attr, shared)
    return shared


def checkpoint(fn, *inputs):
    """Gradient checkpointing.

    Runs `fn(*inputs)` under no_grad during the forward pass and replays it
    during backward, saving only the checkpoint inputs instead of the full
    interior activation graph. The function must return one Tensor.
    """
    return _C.checkpoint(fn, list(inputs))


def synchronize():
    _C.synchronize()


def empty_cache():
    """Release device blocks held idle by the caching allocator back to the
    driver. Live tensors are unaffected."""
    _C.empty_cache()


def set_alloc_pooling(enabled):
    """Enable the stream-ordered transients pool. Call AFTER model/weight
    loading: allocations made while disabled use raw cudaMalloc (persistents
    must stay raw — live pooled blocks pin pool chunks and cost context
    ceiling at OOM walls). Forward-pass transients allocated while enabled
    are pooled, removing the cudaMalloc/cudaFree serialization tax."""
    _C.set_alloc_pooling(bool(enabled))


def is_grad_enabled():
    return _C.is_grad_enabled()


@contextlib.contextmanager
def no_grad():
    prev = _C.is_grad_enabled()
    _C.set_grad_enabled(False)
    try:
        yield
    finally:
        _C.set_grad_enabled(prev)


from . import functional  # noqa: E402
from . import nn  # noqa: E402  (after _C and helpers are defined)
from . import optim  # noqa: E402
from . import quant  # noqa: E402
from . import quantization  # noqa: E402

apa_quant_attention = quant.apa_quant_attention

__all__ = [
    "Tensor", "tensor", "from_numpy", "zeros", "ones", "randn", "rand",
    "matmul", "dda_raycast", "raster_winner_scatter_min",
    "raster_triangle_winners", "raster_winner_resolve", "rasterize_clip",
    "bake_back_project", "bake_cosine_blend", "build_inpaint_island_csr",
    "inpaint_island_passes",
    "terrain_render", "rms_norm", "rope_apply", "write_rows", "export_rows",
    "export_rope_rows", "export_row_pair", "export_row_pairs",
    "swap_row_pairs_with_rope", "evict_row_pairs",
    "arena_row_pair_transaction", "causal_softmax", "fused_sdpa_noncausal", "apa_int4_sdpa_noncausal", "apa_refine_stats", "mse_loss", "cross_entropy", "where", "cat", "stack", "embedding",
    "synchronize", "empty_cache", "set_alloc_pooling", "no_grad", "is_grad_enabled", "nn", "optim", "functional",
    "quant", "quantization", "apa_quant_attention", "save_checkpoint", "load_checkpoint",
    "weight_tie", "checkpoint", "einsum", "int4_linear", "int4_linear_fused",
    "w8a16_matmul",
    "intn_linear", "intn_linear_fused",
    "mxfp4_linear", "mxfp4_linear_expert",
    "gated_delta_step",
    "int4_dequant", "intn_dequant", "apa_selective_attention",
    "apa_selective_attention_sink",
    "kv_int4_pack", "kv_int4_unpack",
    "apa_blend_softmax_sink", "argmax_last_axis",
    "apa_selective_fwd_train", "apa_selective_bwd", "apa_selective_train",
]


def apa_selective_fwd_train(q, k, kq, v, scale, zthr, is_causal=False):
    """O(L)-memory selective-attention training forward. Returns
    (out, lse, thr) — lse/thr are the saved per-row state the backward needs."""
    return _C.apa_selective_fwd_train(q, k, kq, v, scale, zthr, is_causal)


def apa_selective_bwd(q, k, kq, v, dO, lse, thr, scale, is_causal=False):
    """Selective-attention backward. Returns (dq, dk, dv)."""
    return _C.apa_selective_bwd(q, k, kq, v, dO, lse, thr, scale, is_causal)


def apa_selective_train(q, k, kq, v, scale, zthr, is_causal=False):
    """Differentiable O(L)-memory selective attention (graft-native training).
    Selection is a stop-gradient (kq detached); q,k,v receive gradients.
    Returns a single (B,H,L,D) tensor with autograd wired."""
    return _C.apa_selective_train(q, k, kq, v, scale, zthr, is_causal)
__version__ = "0.1.0-phase1"
