# splinemaths.py
import numpy as np

# Curve type tags
BEZIER, POLY = 0, 1


# ====================================================================================================
# Helpers
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Broadcast t
# ----------------------------------------------------------------------------------------------------

def _broadcast_t_for_batch(t, B, *, dtype=np.float32):
    """
    Broadcast parameter t to shape (B, T) and clamp into [0, 1) robustly.

    This function ensures that the upper bound 1.0 is mapped to the largest
    representable float strictly less than 1.0 (via np.nextafter), so that
    downstream computations like floor(t*(N-1)) never produce i == N-1.

    Parameters
    ----------
    t : scalar or array-like
        Accepted shapes:
          - scalar
          - (T,)
          - (1, T)
          - (B, T)
    B : int
        Batch size.
    dtype : np.dtype, optional
        Floating dtype to use (default: np.float32).

    Returns
    -------
    t_b : ndarray, shape (B, T), dtype dtype
        Broadcasted and clamped parameters in [0, 1).
    out_scalar : bool
        True if input t was scalar (useful to reshape outputs later).
    """
    t = np.asarray(t)

    # Keep f64 if already f64; otherwise cast to requested dtype without unnecessary copy
    if t.dtype == np.float64:
        t = t.astype(np.float64, copy=False)
        one = np.float64(1.0)
        zero = np.float64(0.0)
    else:
        t = t.astype(dtype, copy=False)
        one = dtype(1.0)
        zero = dtype(0.0)

    out_scalar = (t.shape == ())

    # Broadcast to (B, T)
    if t.shape == ():
        t_b = np.broadcast_to(t, (B, 1))
    elif t.ndim == 1:
        t_b = np.broadcast_to(t[None, :], (B, t.shape[0]))
    elif t.ndim == 2:
        if t.shape[0] == 1:
            t_b = np.broadcast_to(t, (B, t.shape[1]))
        elif t.shape[0] == B:
            t_b = t
        else:
            raise ValueError(f"_broadcast_t_for_batch> t shape {t.shape} incompatible with batch size {B}")
    else:
        raise ValueError(f"_broadcast_t_for_batch> unsupported t shape {t.shape}")

    # Robust clamp into [0, 1) using nextafter for the upper bound
    upper = np.nextafter(one, zero)  # largest float strictly < 1.0
    t_b = np.clip(t_b, zero, upper)  # avoid in-place to not mutate input

    return t_b, out_scalar


# ----------------------------------------------------------------------------------------------------
# Adjust attributes (utility used by attribute sampling)
# ----------------------------------------------------------------------------------------------------

def adjust_attributes(attr, target_N, cyclic):
    """
    Ensure an attribute block has the expected segment grid length.

    Parameters
    ----------
    attr : (N,) | (N,D) float32-compatible
        Attribute samples at control points (OPEN representation: no duplicate last row).
    target_N : int
        Expected length after adjustment (for cyclic curves this is often control_count+1).
    cyclic : bool
        Whether the curve is cyclic (closed).

    Returns
    -------
    a : (target_N,) | (target_N, D) float32
        Adjusted attribute array. For cyclic:
          - if len(attr) == target_N - 1: append attr[0] to close.
          - if len(attr) == target_N: keep as-is (assumed already closed).
        For non-cyclic:
          - require len(attr) == target_N.

    Raises
    ------
    ValueError
        If the attribute array length does not match the expected size,
        or if it is empty (N == 0).
    """
    a = np.asarray(attr, dtype=np.float32)
    N = a.shape[0]
    if N == 0:
        raise ValueError("adjust_attributes: empty attribute array (N==0) is not supported.")

    if cyclic:
        if N == target_N - 1:
            # Append first row to close the loop (temporary, not stored)
            a = np.concatenate([a, a[0:1]], axis=0)
        elif N != target_N:
            raise ValueError(
                f"Attribute length {N} incompatible with target_N {target_N} (cyclic)."
            )
        # else: already closed, keep as-is
    else:
        if N != target_N:
            raise ValueError(
                f"Attribute length {N} must match target_N {target_N} (non-cyclic)."
            )

    return a


# ----------------------------------------------------------------------------------------------------
# Gather OPEN batches (no persistent duplication)
# ----------------------------------------------------------------------------------------------------

def _gather_open_batch_for_group(curve, indices, need_handles=False):
    """
    Build OPEN batches (no duplicated first point) for a group of splines that share (N, cyclic).

    Parameters
    ----------
    curve : Curve
    indices : list[int]
        Spline indices in the group. Must not be empty and must share the same (N, cyclic).
    need_handles : bool
        If True, also return left/right handle batches.

    Returns
    -------
    P : (B, N, 3) float32
    (L, R) : (B, N, 3) float32 each, only if need_handles=True
    N : int
        Common control point count of the group (OPEN).
    cyclic_flag : bool
        Common cyclic flag of the group.
    """
    if not indices:
        raise ValueError("_gather_open_batch_for_group: 'indices' cannot be empty.")

    ls  = curve.splines.loop_start
    lt  = curve.splines.loop_total
    cyc = curve.splines.cyclic  # np.bool_

    N0   = int(lt[indices[0]])
    cyc0 = bool(cyc[indices[0]])
    for i in indices[1:]:
        if int(lt[i]) != N0 or bool(cyc[i]) != cyc0:
            raise ValueError("Open-batch group must share the same (N, cyclic).")

    B = len(indices)
    P = np.empty((B, N0, 3), dtype=np.float32)
    if need_handles:
        HL = np.empty((B, N0, 3), dtype=np.float32)
        HR = np.empty((B, N0, 3), dtype=np.float32)

    pos = curve.points.position
    if need_handles:
        hL = curve.points.handle_left
        hR = curve.points.handle_right

    for b, i in enumerate(indices):
        start, count = int(ls[i]), int(lt[i])
        P[b] = pos[start:start+count].astype(np.float32, copy=False)
        if need_handles:
            HL[b] = hL[start:start+count].astype(np.float32, copy=False)
            HR[b] = hR[start:start+count].astype(np.float32, copy=False)

    if need_handles:
        return P, HL, HR, N0, cyc0
    return P, N0, cyc0


# ----------------------------------------------------------------------------------------------------
# Build Bézier control blocks per segment from OPEN representation
# ----------------------------------------------------------------------------------------------------

def _build_bezier_ctrl_open(P, L, R, cyclic):
    """
    From OPEN control points (B, N, 3), build per-segment control blocks (B, S, 4, 3).
    S = N-1 if non-cyclic, else N with wrap on the next index.

    Returns
    -------
    ctrl : (B, S, 4, 3) float32
        Stacked [P0, R0, L1, P1] for each segment.
    """
    B, N, _ = P.shape
    if N < 2:
        return np.empty((B, 0, 4, 3), dtype=np.float32)

    if cyclic:
        P_next = np.concatenate([P[:, 1:, :], P[:, :1, :]], axis=1)  # (B, N, 3)
        L_next = np.concatenate([L[:, 1:, :], L[:, :1, :]], axis=1)
        C0, C1, C2, C3 = P, R, L_next, P_next                       # (B, N, 3)
    else:
        C0, C1, C2, C3 = P[:, :-1, :], R[:, :-1, :], L[:, 1:, :], P[:, 1:, :]  # (B, N-1, 3)

    return np.stack([C0, C1, C2, C3], axis=2)  # (B, S, 4, 3)


# ====================================================================================================
# Poly
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Poly - Evaluation (OPEN)
# ----------------------------------------------------------------------------------------------------

def poly_evaluate_batched_open(P, t, cyclic):
    """
    Evaluate batched polyline splines from OPEN representation.

    Parameters
    ----------
    P : (B, N, 3) float32
        Control points per spline (OPEN, no duplicated last row).
    t : scalar | (T,) | (B, T)
        Parameters in [0, 1). Broadcasted to (B, T).
    cyclic : bool
        If True, wrap to include the closing segment (S = N); else S = N-1.

    Returns
    -------
    pts : (B, 3) if scalar t, else (B, T, 3)
    """
    P = np.asarray(P, dtype=np.float32)
    B, N, _ = P.shape
    t_b, out_scalar = _broadcast_t_for_batch(t, B)

    if N < 2:
        empty = np.zeros((B, t_b.shape[1], 3), dtype=np.float32)
        return empty[:, 0] if out_scalar else empty

    S = N if cyclic else (N - 1)
    s = t_b * S
    i = np.floor(s).astype(np.int32)  # (B, T)
    rows = np.arange(B)[:, None]

    if cyclic:
        i1 = (i + 1) % N
    else:
        i1 = i + 1

    P0 = P[rows, i]
    P1 = P[rows, i1]
    u = (s - i).astype(t_b.dtype)[..., None]
    res = (1.0 - u) * P0 + u * P1
    return res[:, 0] if out_scalar else res


# ----------------------------------------------------------------------------------------------------
# Poly - Tangent (OPEN)
# ----------------------------------------------------------------------------------------------------

def poly_tangent_batched_open(P, t, cyclic, normalize=True, eps=1e-8):
    """
    Batched polyline tangents from OPEN representation.

    Parameters
    ----------
    P : (B, N, 3) float32
    t : scalar | (T,) | (B, T)
    cyclic : bool
    normalize : bool
    eps : float

    Returns
    -------
    T : (B, 3) if scalar t, else (B, T, 3)
    """
    P = np.asarray(P, dtype=np.float32)
    B, N, _ = P.shape
    t_b, out_scalar = _broadcast_t_for_batch(t, B)

    if N < 2:
        zeros = np.zeros((B, t_b.shape[1], 3), dtype=np.float32)
        return zeros[:, 0] if out_scalar else zeros

    S = N if cyclic else (N - 1)
    s = t_b * S
    i = np.floor(s).astype(np.int32)
    rows = np.arange(B)[:, None]
    i1 = (i + 1) % N if cyclic else (i + 1)

    seg = P[rows, i1] - P[rows, i]
    if normalize:
        n = np.linalg.norm(seg, axis=-1, keepdims=True)
        n[n < eps] = 1.0
        seg = seg / n

    return seg[:, 0] if out_scalar else seg


# ----------------------------------------------------------------------------------------------------
# Poly - Length (OPEN)
# ----------------------------------------------------------------------------------------------------

def poly_length_batched_open(P, cyclic):
    """
    Exact arc length for batched polylines from OPEN representation.

    Parameters
    ----------
    P : (B, N, 3) float32
        Control points (OPEN).
    cyclic : bool
        If True, include the closing edge from last to first.

    Returns
    -------
    lengths : (B,) float32
    """
    P = np.asarray(P, dtype=np.float32)
    B, N, _ = P.shape
    if N < 2:
        return np.zeros((B,), dtype=np.float32)

    if cyclic:
        P_next = np.concatenate([P[:, 1:, :], P[:, :1, :]], axis=1)  # (B, N, 3)
        seg = P_next - P                                             # (B, N, 3)
    else:
        seg = P[:, 1:, :] - P[:, :-1, :]                             # (B, N-1, 3)

    d = np.linalg.norm(seg, axis=-1)
    return d.sum(axis=1, dtype=np.float32)


# ----------------------------------------------------------------------------------------------------
# Poly - Sample attribute (kept as "closed" utility via adjust_attributes)
# ----------------------------------------------------------------------------------------------------

def sample_attribute_linear_batched(A, t):
    """
    Linear attribute sampling along a polyline, using a temporary CLOSED grid if needed.

    Parameters
    ----------
    A : (B, N) or (B, N, D) float32
        Attribute values at control points (N samples).
        If a CLOSED grid is passed (for cyclic), N is expected to be effective (N+1 open).
    t : scalar | (T,) | (B, T)

    Returns
    -------
    values : (B,) | (B, T) | (B, D) | (B, T, D)
    """
    A = np.asarray(A, dtype=np.float32)
    if A.ndim == 2:
        A = A[..., None]  # (B, N, 1)
        squeeze_D = True
    else:
        squeeze_D = False

    B, N, D = A.shape
    if N < 2:
        # broadcast the single value
        t_b, out_scalar = _broadcast_t_for_batch(t, B)
        base = A[:, 0:1, :]
        out = np.broadcast_to(base, (B, t_b.shape[1] if not out_scalar else 1, D))
        if out_scalar:
            out = out[:, 0]
        return out[..., 0] if squeeze_D else out

    t_b, out_scalar = _broadcast_t_for_batch(t, B)  # (B, T)
    s = t_b * (N - 1)

    # strict upper bound:
    dt = np.asarray(0.0, dtype=t_b.dtype)
    high = np.nextafter(dt + (N - 1), -np.inf)
    s = np.clip(s, 0.0, high, out=s)

    i = np.floor(s).astype(np.int32)                # (B, T)
    u = (s - i).astype(np.float32)                  # (B, T)
    rows = np.arange(B)[:, None]

    a0 = A[rows, i]          # (B, T, D)
    a1 = A[rows, i + 1]      # (B, T, D)
    out = (1.0 - u)[..., None] * a0 + u[..., None] * a1

    if out_scalar:
        out = out[:, 0]      # (B, D)
    return out[..., 0] if squeeze_D else out


# ====================================================================================================
# Bézier
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Bézier - Evaluation / Tangent (OPEN)
# ----------------------------------------------------------------------------------------------------

def bezier_evaluate_batched_open(P, L, R, t, cyclic):
    """
    Batched cubic Bézier evaluation from OPEN representation.

    Parameters
    ----------
    P : (B, N, 3) float32
    L : (B, N, 3) float32
    R : (B, N, 3) float32
    t : scalar | (T,) | (B, T)
    cyclic : bool

    Returns
    -------
    out : (B, 3) if scalar t, else (B, T, 3)
    """
    P = np.asarray(P, dtype=np.float32)
    L = np.asarray(L, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    B, N, _ = P.shape

    t_b, out_scalar = _broadcast_t_for_batch(t, B)  # (B, T)
    if N < 2:
        empty = np.zeros((B, t_b.shape[1], 3), dtype=np.float32)
        return empty[:, 0] if out_scalar else empty

    S = N if cyclic else (N - 1)
    s = t_b * S
    i = np.floor(s).astype(np.int32)         # (B, T)
    rows = np.arange(B)[:, None]
    i1 = (i + 1) % N if cyclic else (i + 1)

    u = (s - i).astype(t_b.dtype)
    u2 = u * u
    u3 = u2 * u
    omu = 1.0 - u
    omu2 = omu * omu
    omu3 = omu2 * omu

    P0 = P[rows, i]
    R0 = R[rows, i]
    L1 = L[rows, i1]
    P1 = P[rows, i1]

    res = (omu3[..., None] * P0 +
           3.0 * omu2[..., None] * u[..., None] * R0 +
           3.0 * omu[..., None]  * u2[..., None] * L1 +
           u3[..., None] * P1)

    return res[:, 0] if out_scalar else res


def bezier_tangent_batched_open(P, L, R, t, cyclic, normalize=True, eps=1e-8):
    """
    Batched cubic Bézier first derivative (optionally normalized) from OPEN representation.
    """
    P = np.asarray(P, dtype=np.float32)
    L = np.asarray(L, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    B, N, _ = P.shape

    t_b, out_scalar = _broadcast_t_for_batch(t, B)  # (B, T)
    if N < 2:
        zeros = np.zeros((B, t_b.shape[1], 3), dtype=np.float32)
        return zeros[:, 0] if out_scalar else zeros

    S = N if cyclic else (N - 1)
    s = t_b * S
    i = np.floor(s).astype(np.int32)
    rows = np.arange(B)[:, None]
    i1 = (i + 1) % N if cyclic else (i + 1)

    u   = (s - i).astype(t_b.dtype)
    u2  = u * u
    omu = 1.0 - u
    omu2= omu * omu

    P0 = P[rows, i]
    R0 = R[rows, i]
    L1 = L[rows, i1]
    P1 = P[rows, i1]

    dP = (3.0 * omu2[..., None] * (R0 - P0) +
          6.0 * omu[..., None]  * u[..., None] * (L1 - R0) +
          3.0 * u2[..., None]   * (P1 - L1))

    if normalize:
        n = np.linalg.norm(dP, axis=-1, keepdims=True)
        n[n < eps] = 1.0
        dP = dP / n

    return dP[:, 0] if out_scalar else dP


# ----------------------------------------------------------------------------------------------------
# Bézier - Length (OPEN, per-segment uniform sampling)
# ----------------------------------------------------------------------------------------------------

def bezier_length_batched(P, L, R, segment_resolution, cyclic):
    """
    Approximate arc length for batched cubic Bézier splines by uniform sampling per segment,
    from OPEN representation (no duplicated last row). Closure is handled on the fly.

    Parameters
    ----------
    P, L, R : (B, N, 3) float32
    segment_resolution : int
        Number of uniform subdivisions per segment (R). Each segment is approximated with
        `segment_resolution` straight-line steps.
    cyclic : bool

    Returns
    -------
    lengths : (B,) float32
    """
    B, N, _ = P.shape
    if N < 2:
        return np.zeros((B,), dtype=np.float32)

    ctrl = _build_bezier_ctrl_open(P, L, R, cyclic)  # (B, S, 4, 3)
    S = ctrl.shape[1]
    Rseg = max(1, int(segment_resolution))

    # Bernstein basis at Rseg+1 samples
    t = np.linspace(0.0, 1.0, Rseg + 1, dtype=np.float32)  # (R+1,)
    B0 = (1 - t) ** 3
    B1 = 3 * t * (1 - t) ** 2
    B2 = 3 * t**2 * (1 - t)
    B3 = t**3
    bern = np.stack([B0, B1, B2, B3], axis=0)              # (4, R+1)

    # Evaluate all segments in one shot: (B, S, R+1, 3)
    samples = np.einsum("bsqc,qr->bsrc", ctrl, bern)

    # Segment-wise polyline lengths: (B, S, R)
    diffs = samples[:, :, 1:, :] - samples[:, :, :-1, :]
    steps = np.linalg.norm(diffs, axis=-1)

    return steps.sum(axis=(1, 2), dtype=np.float32)  # (B,)


# ----------------------------------------------------------------------------------------------------
# Bézier - Sample attribute (cubic-preserving)
# ----------------------------------------------------------------------------------------------------

def sample_attribute_cubic_batched(A, t):
    """
    Cubic attribute sampling that preserves linear interpolation.

    The handles are chosen so that the cubic exactly reproduces linear interpolation:
      h0 = a0 + (a1 - a0)/3
      h1 = a0 + 2*(a1 - a0)/3

    Same signature/returns as the linear version.
    """
    A = np.asarray(A, dtype=np.float32)
    if A.ndim == 2:
        A = A[..., None]
        squeeze_D = True
    else:
        squeeze_D = False

    B, N, D = A.shape
    if N < 2:
        t_b, out_scalar = _broadcast_t_for_batch(t, B)
        base = A[:, 0:1, :]
        out = np.broadcast_to(base, (B, t_b.shape[1] if not out_scalar else 1, D))
        if out_scalar:
            out = out[:, 0]
        return out[..., 0] if squeeze_D else out

    t_b, out_scalar = _broadcast_t_for_batch(t, B)
    s = t_b * (N - 1)
    dt = np.asarray(0.0, dtype=t_b.dtype)
    high = np.nextafter(dt + (N - 1), -np.inf)
    s = np.clip(s, 0.0, high, out=s)

    i = np.floor(s).astype(np.int32)
    u = (s - i).astype(np.float32)

    rows = np.arange(B)[:, None]
    a0 = A[rows, i]          # (B, T, D)
    a1 = A[rows, i + 1]      # (B, T, D)

    # Linear-preserving cubic handles
    delta = a1 - a0
    h0 = a0 + (1.0 / 3.0) * delta
    h1 = a0 + (2.0 / 3.0) * delta

    u2 = u * u
    u3 = u2 * u
    omu = 1.0 - u
    omu2 = omu * omu
    omu3 = omu2 * omu

    out = (
        omu3[..., None] * a0 +
        3.0 * omu2[..., None] * u[..., None] * h0 +
        3.0 * omu[..., None] * u2[..., None] * h1 +
        u3[..., None] * a1
    )
    if out_scalar:
        out = out[:, 0]
    return out[..., 0] if squeeze_D else out


# ====================================================================================================
# Curve-level batched APIs
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Curve - Evaluation
# ----------------------------------------------------------------------------------------------------

def evaluate_curve(curve, t):
    """
    Evaluate all splines of a Curve at parameters t using batched kernels.

    Assumes canonical OPEN storage (no duplicated last row). Cyclic closure is applied on the fly.

    Parameters
    ----------
    curve : Curve
        Requires:
          - splines.loop_start (S,)
          - splines.loop_total (S,)
          - splines.cyclic     (S,) bool/0-1
          - splines.curve_type (S,) with values POLY or BEZIER
          - points.position (P, 3)
          - for BEZIER: points.handle_left/right (P, 3)
    t : scalar | (T,) | (S, T)
        Parameters in [0, 1). If shape is (S, T), each spline gets its own row of T params.

    Returns
    -------
    out : (S, 3) if scalar t, else (S, T, 3)
    """
    S = len(curve.splines)
    if S == 0:
        # Shape of empty result depends on t
        t_arr = np.asarray(t)
        if t_arr.shape == ():
            return np.empty((0, 3), dtype=np.float32)
        elif t_arr.ndim == 1:
            return np.empty((0, t_arr.shape[0], 3), dtype=np.float32)
        elif t_arr.ndim == 2:
            return np.empty((0, t_arr.shape[1], 3), dtype=np.float32)
        else:
            raise ValueError("evaluate_curve> unsupported t shape when S == 0")

    lt   = curve.splines.loop_total
    cyc  = curve.splines.cyclic
    ctyp = curve.splines.curve_type

    out = [None] * S
    t_arr = np.asarray(t)
    t_is_ST = (t_arr.ndim == 2 and t_arr.shape[0] == S)

    for ctype in (POLY, BEZIER):
        sel = np.nonzero(ctyp == ctype)[0]
        if sel.size == 0:
            continue

        # Bucket by (N, cyclic)
        buckets = {}
        for sidx in sel:
            key = (int(lt[sidx]), bool(cyc[sidx]))
            buckets.setdefault(key, []).append(sidx)

        for (Nkey, cyc_flag), idxs in buckets.items():
            if ctype == POLY:
                P, N0, cyc0 = _gather_open_batch_for_group(curve, idxs, need_handles=False)
                assert N0 == Nkey and cyc0 == cyc_flag
                t_g = t_arr[idxs] if t_is_ST else t
                vals = poly_evaluate_batched_open(P, t_g, cyclic=cyc_flag)
            else:
                P, L, R, N0, cyc0 = _gather_open_batch_for_group(curve, idxs, need_handles=True)
                assert N0 == Nkey and cyc0 == cyc_flag
                t_g = t_arr[idxs] if t_is_ST else t
                vals = bezier_evaluate_batched_open(P, L, R, t_g, cyclic=cyc_flag)

            # Scatter back
            for k, sidx in enumerate(idxs):
                out[sidx] = vals[k]

    return np.stack(out, axis=0)


# ----------------------------------------------------------------------------------------------------
# Curve - Tangent
# ----------------------------------------------------------------------------------------------------

def tangent_curve(curve, t, normalize=True, eps=1e-8):
    """
    Evaluate tangents for all splines of a Curve using batched kernels (OPEN representation).
    """
    S = len(curve.splines)
    if S == 0:
        t_arr = np.asarray(t)
        if t_arr.shape == ():
            return np.empty((0, 3), dtype=np.float32)
        elif t_arr.ndim == 1:
            return np.empty((0, t_arr.shape[0], 3), dtype=np.float32)
        elif t_arr.ndim == 2:
            return np.empty((0, t_arr.shape[1], 3), dtype=np.float32)
        else:
            raise ValueError("tangent_curve> unsupported t shape when S == 0")

    lt   = curve.splines.loop_total
    cyc  = curve.splines.cyclic
    ctyp = curve.splines.curve_type

    out = [None] * S
    t_arr = np.asarray(t)
    t_is_ST = (t_arr.ndim == 2 and t_arr.shape[0] == S)

    for ctype in (POLY, BEZIER):
        sel = np.nonzero(ctyp == ctype)[0]
        if sel.size == 0:
            continue

        # Bucket by (N, cyclic)
        buckets = {}
        for sidx in sel:
            key = (int(lt[sidx]), bool(cyc[sidx]))
            buckets.setdefault(key, []).append(sidx)

        for (Nkey, cyc_flag), idxs in buckets.items():
            if ctype == POLY:
                P, N0, cyc0 = _gather_open_batch_for_group(curve, idxs, need_handles=False)
                assert N0 == Nkey and cyc0 == cyc_flag
                t_g = t_arr[idxs] if t_is_ST else t
                Tvals = poly_tangent_batched_open(P, t_g, cyclic=cyc_flag, normalize=normalize, eps=eps)
            else:
                P, L, R, N0, cyc0 = _gather_open_batch_for_group(curve, idxs, need_handles=True)
                assert N0 == Nkey and cyc0 == cyc_flag
                t_g = t_arr[idxs] if t_is_ST else t
                Tvals = bezier_tangent_batched_open(P, L, R, t_g, cyclic=cyc_flag, normalize=normalize, eps=eps)

            for k, sidx in enumerate(idxs):
                out[sidx] = Tvals[k]

    return np.stack(out, axis=0)


# ----------------------------------------------------------------------------------------------------
# Curve - Length
# ----------------------------------------------------------------------------------------------------

def length_curve(curve):
    """
    Compute arc length of all splines in a Curve using batched kernels (OPEN representation).
    """
    S = len(curve.splines)
    if S == 0:
        return np.empty((0,), dtype=np.float32)

    lt   = curve.splines.loop_total
    cyc  = curve.splines.cyclic
    ctyp = curve.splines.curve_type
    rsl  = getattr(curve.splines, "resolution", None)  # int32 for Bézier

    out = np.zeros((S,), dtype=np.float32)

    for ctype in (POLY, BEZIER):
        sel = np.nonzero(ctyp == ctype)[0]
        if sel.size == 0:
            continue

        if ctype == POLY:
            # Bucket by (N, cyclic)
            buckets = {}
            for sidx in sel:
                if int(lt[sidx]) < 2:
                    out[sidx] = 0.0
                    continue
                key = (int(lt[sidx]), bool(cyc[sidx]))
                buckets.setdefault(key, []).append(sidx)

            for (Nkey, cyc_flag), idxs in buckets.items():
                P, N0, cyc0 = _gather_open_batch_for_group(curve, idxs, need_handles=False)
                assert N0 == Nkey and cyc0 == cyc_flag
                out[idxs] = poly_length_batched_open(P, cyclic=cyc_flag)

        else:  # BEZIER
            if rsl is None:
                raise ValueError("Bezier splines require 'splines.resolution'.")

            # Bucket by (N, cyclic, resolution)
            buckets = {}
            for sidx in sel:
                if int(lt[sidx]) < 2:
                    out[sidx] = 0.0
                    continue
                key = (int(lt[sidx]), bool(cyc[sidx]), int(rsl[sidx]))
                buckets.setdefault(key, []).append(sidx)

            for (Nkey, cyc_flag, resol), idxs in buckets.items():
                P, L, R, N0, cyc0 = _gather_open_batch_for_group(curve, idxs, need_handles=True)
                assert N0 == Nkey and cyc0 == cyc_flag
                out[idxs] = bezier_length_batched(P, L, R, segment_resolution=resol, cyclic=cyc_flag)

    return out


# ----------------------------------------------------------------------------------------------------
# Curve - Sample attribute (OPEN -> temp CLOSED via adjust_attributes)
# ----------------------------------------------------------------------------------------------------

def sample_attributes_curve(curve, t, names=None, cubic=False):
    """
    Sample per-point attributes along each spline at parameters t using batched kernels.

    Notes
    -----
    - Geometry arrays (position/handles) are not returned; only custom point-domain attributes.
    - This uses a temporary CLOSED grid (via `adjust_attributes`) for interpolation (linear/cubic).
      The underlying storage remains OPEN and is not modified.

    Parameters
    ----------
    curve : Curve
        Requires:
          - splines.loop_start (S,)
          - splines.loop_total (S,)
          - splines.cyclic     (S,)
          - points.<attr> arrays (P, ...) including at least:
            - position, handle_left, handle_right (these are auto-excluded)
    t : scalar | (T,) | (S, T)
        Parameter(s) in [0, 1).
    names : list[str] or None
        Attribute names from the points domain to sample. If None, uses all except geometry fields.
    cubic : bool
        If True, use cubic-style blending with linear-preserving fake handles; otherwise linear.

    Returns
    -------
    samples : dict[str, ndarray]
        For each name:
          - scalar t  -> (S,) or (S, D)
          - (T,)      -> (S, T) or (S, T, D)
          - (S, T)    -> (S, T) or (S, T, D)
    """
    S = len(curve.splines)
    geom_exclude = {"position", "handle_left", "handle_right"}

    if names is None:
        names = [nm for nm in curve.points.all_names if nm not in geom_exclude]
    else:
        for nm in names:
            if nm not in curve.points.all_names:
                raise AttributeError(f"Unknown point attribute '{nm}'")

    # Early exits
    if S == 0 or not names:
        t_arr = np.asarray(t)
        if t_arr.shape == ():
            return {nm: np.empty((0,), dtype=np.float32) for nm in names}
        else:
            T = t_arr.shape[-1]
            out_shape = (0, T) if t_arr.ndim >= 1 else (0,)
            empty = {}
            for nm in names:
                empty[nm] = np.empty(out_shape, dtype=np.float32)
            return empty

    ls   = curve.splines.loop_start
    lt   = curve.splines.loop_total
    cyc  = curve.splines.cyclic

    t_arr = np.asarray(t)
    t_is_ST = (t_arr.ndim == 2 and t_arr.shape[0] == S)

    # Bucket by (N, cyclic) to build consistent CLOSED batches per group
    buckets = {}
    for sidx in range(S):
        key = (int(lt[sidx]), bool(cyc[sidx]))
        buckets.setdefault(key, []).append(sidx)

    results = {nm: [None] * S for nm in names}

    for (Nkey, cyc_flag), idxs in buckets.items():
        B = len(idxs)

        # Select t for this bucket
        t_g = t_arr[idxs] if t_is_ST else t  # (B, T) or same scalar/(T,)

        for nm in names:
            arr_full = getattr(curve.points, nm)

            # Build a temporary CLOSED batch (B, n_eff[, D]) so that linear/cubic samplers can work
            n_eff = Nkey + (1 if cyc_flag else 0)
            blocks = []
            for sidx in idxs:
                start = int(ls[sidx])
                count = int(lt[sidx])
                a = arr_full[start:start+count]  # (N,) or (N, D)
                a_closed = adjust_attributes(a, target_N=n_eff, cyclic=bool(cyc[sidx]))
                blocks.append(a_closed.astype(np.float32, copy=False))

            A_batch = np.stack(blocks, axis=0)  # (B, n_eff[, D])

            # Sample
            V = sample_attribute_cubic_batched(A_batch, t_g) if cubic else sample_attribute_linear_batched(A_batch, t_g)

            # Scatter
            for k, sidx in enumerate(idxs):
                results[nm][sidx] = V[k]

    return {nm: np.stack(rows, axis=0) for nm, rows in results.items()}
