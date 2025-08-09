# splinemaths.py
import numpy as np

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

    # Force to float dtype (keep f64 if already f64; otherwise cast to requested dtype)
    if t.dtype == np.float64:
        t = t.astype(np.float64, copy=False)
        one = np.float64(1.0)
        zero = np.float64(0.0)
    else:
        t = t.astype(dtype, copy=False)
        one = dtype(1.0)   # ← change
        zero = dtype(0.0)  # ← change

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
    t_b = np.clip(t_b, zero, upper)  # no in-place to avoid mutating input

    return t_b, out_scalar

# ----------------------------------------------------------------------------------------------------
# Adjust attributes
# ----------------------------------------------------------------------------------------------------

def adjust_attributes(attr, target_N, cyclic):
    """Ensure an attribute block has the expected segment grid length.

    Parameters
    ----------
    attr : (N,) | (N,D) float32-compatible
        Attribute samples at control points (without closure unless already provided).
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
    # Force to float32 without unnecessary copy
    a = np.asarray(attr, dtype=np.float32)

    N = a.shape[0]
    if N == 0:
        # Explicitly forbid empty attribute arrays
        raise ValueError("adjust_attributes: empty attribute array (N==0) is not supported.")

    if cyclic:
        if N == target_N - 1:
            # Append first row to close the loop
            a = np.concatenate([a, a[0:1]], axis=0)
        elif N != target_N:
            raise ValueError(
                f"Attribute length {N} incompatible with target_N {target_N} (cyclic)."
            )
        # else: N == target_N → already closed, keep as-is
    else:
        if N != target_N:
            raise ValueError(
                f"Attribute length {N} must match target_N {target_N} (non-cyclic)."
            )

    return a

# ----------------------------------------------------------------------------------------------------
# Close batch
# ----------------------------------------------------------------------------------------------------

def _build_closed_batch_for_group(curve, indices, need_handles=False):
    """Build a closed, fixed-length batched tensor for a group of splines that share the same n_eff.

    Parameters
    ----------
    curve : Curve
    indices : list[int]
        Spline indices in the group. Must not be empty and must share the same effective length.
    need_handles : bool
        If True, also return closed L/R handle batches.

    Returns
    -------
    P : (B, n_eff, 3) float32
    (L, R) : (B, n_eff, 3) float32 each, only if need_handles=True
    """
    if not indices:
        raise ValueError("_build_closed_batch_for_group: 'indices' cannot be empty.")

    ls  = curve.splines.loop_start
    lt  = curve.splines.loop_total
    cyc = curve.splines.cyclic.astype(bool)

    # Effective length for the first spline; all others must match
    i0 = indices[0]
    n_eff = int(lt[i0] + (1 if cyc[i0] else 0))
    B = len(indices)

    P = np.empty((B, n_eff, 3), dtype=np.float32)
    if need_handles:
        HL = np.empty((B, n_eff, 3), dtype=np.float32)
        HR = np.empty((B, n_eff, 3), dtype=np.float32)

    pos = curve.points.position
    if need_handles:
        hL = curve.points.handle_left
        hR = curve.points.handle_right

    for b, i in enumerate(indices):
        # Sanity-check: consistent effective length across the group
        n_eff_i = int(lt[i] + (1 if cyc[i] else 0))
        if n_eff_i != n_eff:
            raise ValueError(
                f"_build_closed_batch_for_group: inconsistent n_eff "
                f"(got {n_eff_i} for spline {i}, expected {n_eff})."
            )

        start = ls[i]
        count = lt[i]
        is_cyc = bool(cyc[i])

        Pb = _close_block(pos[start:start+count], is_cyc)  # (n_eff, 3), float32
        P[b, :, :] = Pb

        if need_handles:
            HLb = _close_block(hL[start:start+count], is_cyc)
            HRb = _close_block(hR[start:start+count], is_cyc)
            HL[b, :, :] = HLb
            HR[b, :, :] = HRb

    if need_handles:
        return P, HL, HR
    return P

# ----------------------------------------------------------------------------------------------------
# Close a single block
# ----------------------------------------------------------------------------------------------------

def _close_block(P, is_cyclic):
    """Return a closed block.
    If cyclic is True, append the first point at the end.
    Shape in:  (N, 3)
    Shape out: (N, 3) if not cyclic, else (N+1, 3)
    """
    P = np.asarray(P, dtype=np.float32)
    if is_cyclic:
        out = np.empty((P.shape[0] + 1, 3), dtype=np.float32)
        out[:-1] = P
        out[-1] = P[0]
        return out
    return P


# ====================================================================================================
# Poly
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Poly - Evaluation
# ----------------------------------------------------------------------------------------------------

def poly_evaluate_batched(P, t):
    """Evaluate batched polyline splines.

    Parameters
    ----------
    P : (B, N, 3) float32
        Control points per spline. If cyclic, last row must duplicate first row.
    t : scalar | (T,) | (B, T)
        Parameters in [0, 1]. Broadcasted to (B, T).

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

    s = t_b * (N - 1)
    i = np.floor(s).astype(np.int32)  # (B, T)
    u = (s - i).astype(t_b.dtype)     # (B, T)

    rows = np.arange(B)[:, None]
    P0 = P[rows, i]
    P1 = P[rows, i + 1]

    res = (1.0 - u)[..., None] * P0 + u[..., None] * P1
    return res[:, 0] if out_scalar else res


# ----------------------------------------------------------------------------------------------------
# Poly - Tangent
# ----------------------------------------------------------------------------------------------------

def poly_tangent_batched(P, t, normalize=True, eps=1e-8):
    """Batched polyline tangents.

    Parameters
    ----------
    P : (B, N, 3) float32
        Poly control points; closed if cyclic.
    t : scalar | (T,) | (B, T)
        Parameters broadcasted to (B, T).
    normalize : bool
        If True, return unit direction. If False, return raw (P1 - P0).
    eps : float
        Epsilon for normalization in float32.

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

    s = t_b * (N - 1)
    i = np.floor(s).astype(np.int32)
    rows = np.arange(B)[:, None]
    seg = P[rows, i + 1] - P[rows, i]

    if normalize:
        n = np.linalg.norm(seg, axis=-1, keepdims=True)
        n[n < eps] = 1.0
        seg = seg / n

    return seg[:, 0] if out_scalar else seg


# ----------------------------------------------------------------------------------------------------
# Poly - Length
# ----------------------------------------------------------------------------------------------------

def poly_length_batched(P):
    """Exact arc length for batched polyline splines.

    Parameters
    ----------
    P : (B, N, 3) float32
        Control points; blocks must already be CLOSED if cyclic.

    Returns
    -------
    lengths : (B,) float32
    """
    P = np.asarray(P, dtype=np.float32)
    if P.shape[1] < 2:
        return np.zeros((P.shape[0],), dtype=np.float32)
    seg = P[:, 1:, :] - P[:, :-1, :]
    d = np.linalg.norm(seg, axis=-1)
    return d.sum(axis=1, dtype=np.float32)


# ----------------------------------------------------------------------------------------------------
# Poly - Sample attribute
# ----------------------------------------------------------------------------------------------------

def sample_attribute_linear_batched(A, t):
    """
    A : (B, N) ou (B, N, D) float32
    t : scalaire | (T,) | (B, T)
    return : (B,) | (B, T) | (B, D) | (B, T, D)
    """
    A = np.asarray(A, dtype=np.float32)
    if A.ndim == 2:
        A = A[..., None]  # (B, N, 1)
        squeeze_D = True
    else:
        squeeze_D = False

    B, N, D = A.shape
    if N < 2:
        # broadcast de la valeur unique
        t_b, out_scalar = _broadcast_t_for_batch(t, B)
        base = A[:, 0:1, :]                       # (B,1,D)
        out = np.broadcast_to(base, (B, t_b.shape[1] if not out_scalar else 1, D))
        if out_scalar:
            out = out[:, 0]
        return out[..., 0] if squeeze_D else out

    t_b, out_scalar = _broadcast_t_for_batch(t, B)  # (B,T)
    s = t_b * (N - 1)
    # borne haute stricte :
    dt  = np.asarray(0.0, dtype=(t_b.dtype))  # dtype de s/t_b
    high = np.nextafter(dt + (N - 1), -np.inf)
    s = np.clip(s, 0.0, high, out=s)

    i = np.floor(s).astype(np.int32)                # (B,T)
    u = (s - i).astype(np.float32)                  # (B,T)
    rows = np.arange(B)[:, None]

    a0 = A[rows, i]          # (B,T,D)
    a1 = A[rows, i + 1]      # (B,T,D)
    out = (1.0 - u)[..., None] * a0 + u[..., None] * a1

    if out_scalar:
        out = out[:, 0]      # (B,D)
    return out[..., 0] if squeeze_D else out


# ====================================================================================================
# Bezier
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Bezier - Evaluation (batched)
# ----------------------------------------------------------------------------------------------------

def bezier_evaluate_batched(P, L, R, t):
    """
    Batched cubic Bezier evaluation.

    Parameters
    ----------
    P : (B, N, 3) float32
        Anchor points; closed if cyclic (last row duplicates the first).
    L : (B, N, 3) float32
        Left handles; closed if cyclic.
    R : (B, N, 3) float32
        Right handles; closed if cyclic.
    t : scalar | (T,) | (B, T)
        Parameters in [0, 1], broadcast to (B, T).

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

    s = t_b * (N - 1)
    i = np.floor(s).astype(np.int32)         # (B, T)
    u = (s - i).astype(t_b.dtype)            # (B, T)

    rows = np.arange(B)[:, None]
    P0 = P[rows, i]                           # (B, T, 3)
    R0 = R[rows, i]
    L1 = L[rows, i + 1]
    P1 = P[rows, i + 1]

    u2  = u * u
    u3  = u2 * u
    omu = 1.0 - u
    omu2= omu * omu
    omu3= omu2 * omu

    res = (omu3[..., None] * P0 +
           3.0 * omu2[..., None] * u[..., None] * R0 +
           3.0 * omu[..., None]  * u2[..., None] * L1 +
           u3[..., None] * P1)

    return res[:, 0] if out_scalar else res


# ----------------------------------------------------------------------------------------------------
# Bezier - Tangent / First derivative (batched)
# ----------------------------------------------------------------------------------------------------

def bezier_tangent_batched(P, L, R, t, normalize=True, eps=1e-8):
    """
    Batched cubic Bezier first derivative (optionally normalized).

    Parameters
    ----------
    P, L, R : (B, N, 3) float32
    t : scalar | (T,) | (B, T)
    normalize : bool
        If True, returns unit vectors.
    eps : float
        Epsilon to avoid division by zero in normalization.

    Returns
    -------
    dP : (B, 3) if scalar t, else (B, T, 3)
    """
    P = np.asarray(P, dtype=np.float32)
    L = np.asarray(L, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    B, N, _ = P.shape

    t_b, out_scalar = _broadcast_t_for_batch(t, B)  # (B, T)

    if N < 2:
        zeros = np.zeros((B, t_b.shape[1], 3), dtype=np.float32)
        return zeros[:, 0] if out_scalar else zeros

    s = t_b * (N - 1)
    i = np.floor(s).astype(np.int32)
    u = (s - i).astype(t_b.dtype)

    rows = np.arange(B)[:, None]
    P0 = P[rows, i]
    R0 = R[rows, i]
    L1 = L[rows, i + 1]
    P1 = P[rows, i + 1]

    u2  = u * u
    omu = 1.0 - u
    omu2= omu * omu

    dP = (3.0 * omu2[..., None] * (R0 - P0) +
          6.0 * omu[..., None]  * u[..., None] * (L1 - R0) +
          3.0 * u2[..., None]   * (P1 - L1))

    if normalize:
        n = np.linalg.norm(dP, axis=-1, keepdims=True)
        n[n < eps] = 1.0
        dP = dP / n

    return dP[:, 0] if out_scalar else dP


# ----------------------------------------------------------------------------------------------------
# Bezier - Length (batched, approximated by uniform sampling)
# ----------------------------------------------------------------------------------------------------

def bezier_length_batched(P, L, R, segment_resolution):
    """
    Approximate arc length for batched cubic Bezier splines by uniform sampling per segment.

    P, L, R : (B, N, 3) float32 (déjà garanti)
    segment_resolution : np.int32 ou int
    """
    B, N, _ = P.shape
    if N < 2:
        return np.zeros((B,), dtype=np.float32)

    S = N - 1
    Rseg = max(1, int(segment_resolution))  # cast léger, pas de copie mémoire

    # (B, S, 4, 3)
    C0 = P[:, :-1, :]
    C1 = R[:, :-1, :]
    C2 = L[:, 1:, :]
    C3 = P[:, 1:, :]
    ctrl = np.stack([C0, C1, C2, C3], axis=2)

    t = np.linspace(0.0, 1.0, Rseg + 1, dtype=np.float32)
    B0 = (1 - t) ** 3
    B1 = 3 * t * (1 - t) ** 2
    B2 = 3 * t**2 * (1 - t)
    B3 = t**3
    bern = np.stack([B0, B1, B2, B3], axis=0)

    samples = np.einsum("bsqc,qr->bsrc", ctrl, bern)      # (B, S, R+1, 3)
    diffs   = samples[:, :, 1:, :] - samples[:, :, :-1, :]
    steps   = np.linalg.norm(diffs, axis=-1)              # (B, S, R)
    return steps.sum(axis=(1, 2), dtype=np.float32)       # (B,)


def bezier_length_batched_OLD(P, L, R, segment_resolution=10):
    """
    Approximate arc length for batched cubic Bezier splines by uniform sampling per segment.

    Parameters
    ----------
    P, L, R : (B, N, 3) float32
        Anchors (P), left handles (L), right handles (R).
        Curves must already be CLOSED if cyclic (last point == first point).
    segment_resolution : int
        Number of uniform samples PER SEGMENT (R). Each segment will be
        subdivided into `segment_resolution` straight-line steps.

    Returns
    -------
    lengths : (B,) float32
        Total length of each curve.
    """
    P = np.asarray(P, dtype=np.float32)
    L = np.asarray(L, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)

    B, N, _ = P.shape
    if N < 2:
        return np.zeros((B,), dtype=np.float32)

    S = N - 1  # same for all curves
    Rseg = max(1, int(segment_resolution))

    # Control points for each segment: (B, S, 4, 3)
    C0 = P[:, :-1, :]
    C1 = R[:, :-1, :]
    C2 = L[:, 1:, :]
    C3 = P[:, 1:, :]
    ctrl = np.stack([C0, C1, C2, C3], axis=2)  # (B, S, 4, 3)

    # Bernstein basis for cubic Bézier, evaluated at Rseg+1 points
    t = np.linspace(0.0, 1.0, Rseg + 1, dtype=np.float32)  # (R+1,)
    B0 = (1 - t) ** 3
    B1 = 3 * t * (1 - t) ** 2
    B2 = 3 * t**2 * (1 - t)
    B3 = t**3
    bern = np.stack([B0, B1, B2, B3], axis=0)              # (4, R+1)

    # Evaluate all segments in one shot: (B, S, R+1, 3)
    samples = np.einsum("bsqc,qr->bsrc", ctrl, bern)

    # Differences along each segment: (B, S, R, 3)
    diffs = samples[:, :, 1:, :] - samples[:, :, :-1, :]
    step_lengths = np.linalg.norm(diffs, axis=-1)  # (B, S, R)

    # Sum over R and S to get total length per curve
    return step_lengths.sum(axis=(1, 2), dtype=np.float32)  # (B,)


def bezier_length_batched_OLD(P, L, R, resolution=100):
    """
    Approximate arc length for batched cubic Bezier splines by uniform sampling.

    Parameters
    ----------
    P, L, R : (B, N, 3) float32
        Anchors and handles; blocks must already be CLOSED if cyclic.
    resolution : int
        Number of uniform samples in [0, 1].

    Returns
    -------
    lengths : (B,) float32
    """
    P = np.asarray(P, dtype=np.float32)
    L = np.asarray(L, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)

    B, N, _ = P.shape
    if N < 2:
        return np.zeros((B,), dtype=np.float32)

    T = max(2, int(resolution))
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)     # (T,)
    pts = bezier_evaluate_batched(P, L, R, t)          # (B, T, 3)
    seg = pts[:, 1:, :] - pts[:, :-1, :]               # (B, T-1, 3)
    d = np.linalg.norm(seg, axis=-1)                   # (B, T-1)
    return d.sum(axis=1, dtype=np.float32)             # (B,)


# ----------------------------------------------------------------------------------------------------
# Bezier - Sample attribute
# ----------------------------------------------------------------------------------------------------

def sample_attribute_cubic_batched(A, t):
    """
    Same signature/returns as the linear version.
    Handles chosen so that the cubic exactly reproduces linear interpolation:
      h0 = a0 + (a1 - a0)/3
      h1 = a0 + 2*(a1 - a0)/3
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
    a0 = A[rows, i]          # (B,T,D)
    a1 = A[rows, i + 1]      # (B,T,D)

    # Linear-preserving cubic handles
    delta = a1 - a0
    h0 = a0 + (1.0/3.0) * delta
    h1 = a0 + (2.0/3.0) * delta

    u2  = u * u
    u3  = u2 * u
    omu = 1.0 - u
    omu2= omu * omu
    omu3= omu2 * omu

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
# Curve
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Curve - Evaluation
# ----------------------------------------------------------------------------------------------------

def evaluate_curve(curve, t):
    """Evaluate all splines of a Curve at parameters t using batched kernels.

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
        Parameter(s) in [0, 1]. If shape is (S, T), each spline gets its own row of T params.

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

    # Prepare helpers
    ls   = curve.splines.loop_start
    lt   = curve.splines.loop_total
    cyc  = curve.splines.cyclic.astype(bool)
    ctyp = curve.splines.curve_type.astype(int)

    out = [None] * S
    t_arr = np.asarray(t)
    t_is_ST = (t_arr.ndim == 2 and t_arr.shape[0] == S)

    # Process by curve type
    for ctype in (POLY, BEZIER):
        sel = np.nonzero(ctyp == ctype)[0]
        if sel.size == 0:
            continue

        # Sub-bucket selected splines by effective length n_eff = N + cyclic
        buckets = {}
        for sidx in sel:
            n_eff = int(lt[sidx] + (1 if cyc[sidx] else 0))
            buckets.setdefault(n_eff, []).append(sidx)

        for n_eff, idxs in buckets.items():
            B = len(idxs)

            # Build closed, fixed-length batches
            if ctype == POLY:
                P = _build_closed_batch_for_group(curve, idxs, need_handles=False)      # (B, n_eff, 3)
            else:
                P, L, R = _build_closed_batch_for_group(curve, idxs, need_handles=True) # (B, n_eff, 3) each

            # Select t for this group: either per-spline (S,T) sliced, or the global t
            if t_is_ST:
                t_g = t_arr[idxs]   # (B, T)
            else:
                t_g = t             # scalar | (T,)

            # Evaluate with batched kernels
            if ctype == POLY:
                vals = poly_evaluate_batched(P, t_g)                 # (B,3) or (B,T,3)
            else:
                vals = bezier_evaluate_batched(P, L, R, t_g)         # (B,3) or (B,T,3)

            # Scatter back into output slots
            for k, sidx in enumerate(idxs):
                out[sidx] = vals[k]

    return np.stack(out, axis=0)

# ----------------------------------------------------------------------------------------------------
# Curve - Tangent
# ----------------------------------------------------------------------------------------------------

def tangent_curve(curve, t, normalize=True, eps=1e-8):
    """Evaluate tangents for all splines of a Curve using batched kernels.

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
        Parameter(s) in [0, 1]. If shape is (S, T), each spline gets its own row of T params.
    normalize : bool
        If True, return unit tangents; otherwise raw derivatives.
    eps : float
        Epsilon to avoid division by zero during normalization.

    Returns
    -------
    out : (S, 3) if scalar t, else (S, T, 3)
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

    ls   = curve.splines.loop_start
    lt   = curve.splines.loop_total
    cyc  = curve.splines.cyclic.astype(bool)
    ctyp = curve.splines.curve_type.astype(int)

    out = [None] * S
    t_arr = np.asarray(t)
    t_is_ST = (t_arr.ndim == 2 and t_arr.shape[0] == S)

    for ctype in (POLY, BEZIER):
        sel = np.nonzero(ctyp == ctype)[0]
        if sel.size == 0:
            continue

        # bucket par longueur effective n_eff = N + cyclic
        buckets = {}
        for sidx in sel:
            n_eff = int(lt[sidx] + (1 if cyc[sidx] else 0))
            buckets.setdefault(n_eff, []).append(sidx)

        for n_eff, idxs in buckets.items():
            B = len(idxs)

            if ctype == POLY:
                P = _build_closed_batch_for_group(curve, idxs, need_handles=False)       # (B, n_eff, 3)
            else:
                P, L, R = _build_closed_batch_for_group(curve, idxs, need_handles=True)  # (B, n_eff, 3) each

            # t pour ce groupe
            if t_is_ST:
                t_g = t_arr[idxs]   # (B, T)
            else:
                t_g = t             # scalar | (T,)

            # appel kernels batched
            if ctype == POLY:
                Tvals = poly_tangent_batched(P, t_g, normalize=normalize, eps=eps)
            else:
                Tvals = bezier_tangent_batched(P, L, R, t_g, normalize=normalize, eps=eps)

            # scatter
            for k, sidx in enumerate(idxs):
                out[sidx] = Tvals[k]

    return np.stack(out, axis=0)

# ----------------------------------------------------------------------------------------------------
# Curve - Length
# ----------------------------------------------------------------------------------------------------

def length_curve(curve):
    """Compute arc length of all splines in a Curve using batched kernels."""
    S = len(curve.splines)
    if S == 0:
        return np.empty((0,), dtype=np.float32)

    ls   = curve.splines.loop_start
    lt   = curve.splines.loop_total
    cyc  = curve.splines.cyclic          # np.bool_
    ctyp = curve.splines.curve_type      # int-like
    rsl  = getattr(curve.splines, "resolution", None)  # np.int32

    out = np.zeros((S,), dtype=np.float32)

    for ctype in (POLY, BEZIER):
        sel = np.nonzero(ctyp == ctype)[0]
        if sel.size == 0:
            continue

        if ctype == POLY:
            buckets = {}
            for sidx in sel:
                n_eff = int(lt[sidx] + (1 if cyc[sidx] else 0))
                if n_eff < 2:
                    out[sidx] = 0.0
                    continue
                buckets.setdefault(n_eff, []).append(sidx)

            for n_eff, idxs in buckets.items():
                P = _build_closed_batch_for_group(curve, idxs, need_handles=False)
                out[idxs] = poly_length_batched(P)

        else:  # BEZIER
            if rsl is None:
                raise ValueError("Bezier splines require 'splines.resolution'.")

            buckets = {}
            for sidx in sel:
                n_eff = int(lt[sidx] + (1 if cyc[sidx] else 0))
                if n_eff < 2:
                    out[sidx] = 0.0
                    continue
                key = (n_eff, int(rsl[sidx]))  # pas de cast de tableau, juste int() scalaire
                buckets.setdefault(key, []).append(sidx)

            for (n_eff, resol), idxs in buckets.items():
                P, L, R = _build_closed_batch_for_group(curve, idxs, need_handles=True)
                out[idxs] = bezier_length_batched(P, L, R, segment_resolution=resol)

    return out



# ----------------------------------------------------------------------------------------------------
# Curve - Sample attribute
# ----------------------------------------------------------------------------------------------------

def sample_attributes_curve(curve, t, names=None, cubic=False):
    """Sample per-point attributes along each spline at parameters t using batched kernels.

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
        Parameter(s) in [0, 1].
    names : list[str] or None
        Attribute names from the points domain to sample. If None, uses all except geometry fields.
    cubic : bool
        If True, use cubic-style blending with fake handles (h0=a0, h1=a1); otherwise linear.

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
    cyc  = curve.splines.cyclic.astype(bool)

    t_arr = np.asarray(t)
    t_is_ST = (t_arr.ndim == 2 and t_arr.shape[0] == S)

    # Bucket by effective length (n_eff = N + cyclic)
    buckets = {}
    for sidx in range(S):
        n_eff = int(lt[sidx] + (1 if cyc[sidx] else 0))
        buckets.setdefault(n_eff, []).append(sidx)

    results = {nm: [None] * S for nm in names}

    for n_eff, idxs in buckets.items():
        B = len(idxs)

        # Préparer t pour ce bucket
        if t_is_ST:
            t_g = t_arr[idxs]    # (B, T)
        else:
            t_g = t              # scalaire ou (T,)

        # Pour chaque attribut demandé, construire un batch (B, n_eff[, D]) fermé et sampler
        for nm in names:
            arr_full = getattr(curve.points, nm)

            # Construire A_batch en fermant chaque spline (via adjust_attributes)
            blocks = []
            for sidx in idxs:
                start = ls[sidx]
                count = lt[sidx]
                a = arr_full[start:start+count]  # (N,) ou (N, D)
                a_closed = adjust_attributes(a, target_N=n_eff, cyclic=bool(cyc[sidx]))
                blocks.append(a_closed.astype(np.float32, copy=False))

            # Stack en batch
            A_batch = np.stack(blocks, axis=0)  # (B, n_eff) ou (B, n_eff, D)

            # Sample via kernels batched
            if cubic:
                V = sample_attribute_cubic_batched(A_batch, t_g)   # (B, ...) suivant t_g
            else:
                V = sample_attribute_linear_batched(A_batch, t_g)

            # Scatter
            for k, sidx in enumerate(idxs):
                results[nm][sidx] = V[k]

    # Stack final par attribut
    return {nm: np.stack(rows, axis=0) for nm, rows in results.items()}







