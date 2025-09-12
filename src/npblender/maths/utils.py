# MIT License
#
# Copyright (c) 2025 Alain Bernard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the \"Software\"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Module Name: utils
Author: Alain Bernard
Version: 0.1.0
Created: 2025-09-02
Last updated: 2025-09-02

Summary:
    Some utilites.

"""

__all__ = ['get_axis', 'get_perp', 'get_angled', 'flat_top_gaussian', 'vonmises_angle_estimate']

import numpy as np

from .constants import *

# ====================================================================================================
# Normalize axis
# ====================================================================================================

def get_axis(v, null=[0, 0, 1]):
    """ Normalize a vector or an array of vectors.

    The vector can be specified as a string naming an axis : 'x', '-z', ...

    Arguments
    ---------
        - v (vector or array of vectors or str) : the vector to normalize
        - null (vector=(0, 0, 1)) : value to set to null vectors

    Returns
    -------
        - normalized vector(s), vector norm(s)
    """

    # ---------------------------------------------------------------------------
    # A string
    # ---------------------------------------------------------------------------

    if isinstance(v, str):

        upper = v.upper()

        if upper in ['X', '+X', 'POS_X', 'I', '+I']:
            return np.array((1., 0., 0.), dtype=bfloat), np.array(1., dtype=bfloat)
        elif upper in ['Y', '+Y', 'POS_Y', 'J', '+J']:
            return np.array((0., 1., 0.), dtype=bfloat), np.array(1., dtype=bfloat)
        elif upper in ['Z', '+Z', 'POS_Z', 'K', '+K']:
            return np.array((0., 0., 1.), dtype=bfloat), np.array(1., dtype=bfloat)

        elif upper in ['-X', 'NEG_X', '-I']:
            return np.array((-1., 0., 0.), dtype=bfloat), np.array(1., dtype=bfloat)
        elif upper in ['-Y', 'NEG_Y', '-J']:
            return np.array((0., -1., 0.), dtype=bfloat), np.array(1., dtype=bfloat)
        elif upper in ['-Z', 'NEG_Z', '-K']:
            return np.array((0., 0., -1.), dtype=bfloat), np.array(1., dtype=bfloat)
        else:
            raise RuntimeError(f"Unknwon axis spec: '{v}'")
        
    # ---------------------------------------------------------------------------
    # An array of vectors
    # ---------------------------------------------------------------------------
        
    vectors = np.asarray(v, dtype=bfloat)
    VNull = np.asarray(null)

    # Check input shapes
    if vectors.shape[-1] != 3 or VNull.shape != (3,):
        raise ValueError("vectors must have shape (..., 3) and null must have shape (3,)")

    # Compute the Euclidean norm along the last axis (vector dimension)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)

    # Identify zero-norm vectors to avoid division by zero
    is_zero = norms[..., 0] < ZERO

    # To avoid warning message with 0
    norms[is_zero] = 1

    # Normalize non-zero vectors, replace zero vectors with replace_nulls    
    normalized = np.where(
        is_zero[..., None],  # Expand dims for broadcasting
        VNull,               # Use fallback vector
        vectors / norms      # Normalize normally
    )

    return normalized, norms.reshape(norms.shape[:-1])

# ====================================================================================================
# Get vectors perpendicular to arguments
# ====================================================================================================

def get_perp(vectors, default=(0, 0, 1), normalize=False):
    """
    Compute a perpendicular vector to each input vector.
    
    Parameters
    ----------
    vectors : array_like
        Input vectors of shape (..., 3).
    default : tuple or array_like, optional
        Default perpendicular used when the vector is zero.
    
    Returns
    -------
    perp : ndarray
        Perpendicular vectors of shape (..., 3).
    """
    v = np.asarray(vectors, dtype=bfloat)
    d = np.asarray(default, dtype=bfloat)

    # Broadcast default to v's shape
    d = np.broadcast_to(d, v.shape)

    # Norms
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    mask_zero = norms.squeeze(-1) == 0

    # Find a test vector starting with (0, 0, 1) and then (0, 1, 0)
    test1 = np.array([0.0, 0.0, 1.0], dtype=bfloat)
    test2 = np.array([0.0, 1.0, 0.0], dtype=bfloat)
    test = np.where(np.abs(v[...,0]) + np.abs(v[...,1]) < 1e-6, test2, test1)

    # Cross product
    perp = np.cross(v, test)

    # Result

    if perp.ndim == 1:
        if mask_zero:
            perp = d
    else:
        perp[mask_zero] = d

    if normalize:
        n = np.linalg.norm(perp, axis=-1, keepdims=True)
        perp = perp / np.where(n == 0, 1, n)

    return perp

# ====================================================================================================
# Get vectors making a given angle with input vectors
# ====================================================================================================

def get_angled(vectors, angle, default=(0, 0, 1), keep_norm=False,
               twist=0.0, seed=None, eps=1e-6):
    """
    Return a vector making the given angle (radians) with each input vector.
    The direction around the cone is controlled by `twist` (angle in radians
    around the input direction). You can set twist='random' to sample a
    uniform angle in [0, 2*pi).

    Parameters
    ----------
    vectors : array_like, shape (..., 3)
        Input vectors.
    angle : float or array_like, broadcastable to vectors[..., 0]
        Cone half-angle (radians) with respect to each input vector.
    default : array_like, shape (3,)
        Fallback direction when input vector is zero.
    keep_norm : bool
        If True, scale the result by ||vectors|| (zeros stay unit).
    twist : float or array_like or 'random'
        Rotation (radians) around the input direction (choose the
        azimuth on the cone). If 'random', sample uniform in [0, 2*pi).
    seed : int or None
        RNG seed when twist='random'.
    eps : float
        Tolerance for near-colinearity with z-axis.

    Returns
    -------
    w : ndarray, shape (..., 3)
        Output vectors.
    """
    v = np.asarray(vectors, dtype=bfloat)
    if v.shape[-1] != 3:
        raise ValueError(f"Expected last dimension = 3, got {v.shape[-1]}")
    d = np.asarray(default, dtype=bfloat)
    if d.shape != (3,):
        raise ValueError("`default` must have shape (3,)")

    # Norms and zero-mask
    nv = np.linalg.norm(v, axis=-1, keepdims=True)             # (..., 1)
    mask_zero = (nv[..., 0] < 1e-12)

    # Replace null vectors with default direction, then normalize
    v_eff = np.where(mask_zero[..., None], d, v)               # (..., 3)
    n_eff = np.linalg.norm(v_eff, axis=-1, keepdims=True)
    v_hat = v_eff / np.where(n_eff == 0.0, 1.0, n_eff)         # (..., 3), unit

    # Build an orthonormal basis (p_hat, q_hat) spanning v_hat^⊥
    xy_sum = np.abs(v_hat[..., 0]) + np.abs(v_hat[..., 1])
    use_y  = (xy_sum <= eps)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=bfloat)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=bfloat)
    test   = np.where(use_y[..., None], y_axis, z_axis)        # (..., 3)

    p = np.cross(v_hat, test)
    np_norm_p = np.linalg.norm(p, axis=-1, keepdims=True)
    p_hat = p / np.where(np_norm_p == 0.0, 1.0, np_norm_p)     # (..., 3), unit

    q = np.cross(v_hat, p_hat)
    nq = np.linalg.norm(q, axis=-1, keepdims=True)
    q_hat = q / np.where(nq == 0.0, 1.0, nq)                   # (..., 3), unit

    # Determine twist angle(s)
    if isinstance(twist, str):
        if twist != 'random':
            raise ValueError("`twist` must be a float/array or 'random'.")
        rng = np.random.default_rng(seed)
        tw = rng.uniform(0.0, 2.0 * np.pi, size=v.shape[:-1]).astype(bfloat)
    else:
        tw = np.asarray(twist, dtype=bfloat)

    # Build the azimuth direction on the cone rim
    # u_hat = cos(twist)*p_hat + sin(twist)*q_hat
    ct = np.cos(tw)[..., None]
    st = np.sin(tw)[..., None]
    u_hat = ct * p_hat + st * q_hat                             # (..., 3), unit

    # Combine at the requested cone angle: w_hat = cos(a)*v_hat + sin(a)*u_hat
    ang = np.asarray(angle, dtype=bfloat)
    c = np.cos(ang)[..., None]
    s = np.sin(ang)[..., None]
    w_hat = c * v_hat + s * u_hat                               # (..., 3), unit

    # Optionally keep input norm; keep unit for zero inputs
    if keep_norm:
        w = w_hat * nv
        w[mask_zero] = w_hat[mask_zero]
        return w
    else:
        return w_hat



# ====================================================================================================
# flat top gaussian
# ====================================================================================================

def flat_top_gaussian(center, width, shape, seed=None):
    """
    Generate a flat-top Gaussian-like random distribution.

    The resulting distribution is centered around `center`, with values mostly
    uniformly distributed within [-width, +width], and Gaussian falloff outside.
    This creates a "flat-top" shape with smooth edges.

    Parameters
    ----------
    center : float or array-like
        The central value(s) around which the noise is distributed.
    width : float
        The width of the flat top region (values between -width and +width are uniformly distributed).
    shape : tuple of ints
        The shape of the output array.
    seed : int or None, optional
        Optional seed for deterministic output.

    Returns
    -------
    d : ndarray
        An array of shape `shape`, centered at `center`, with a flat-top Gaussian profile.
    """
    rng = np.random.default_rng(seed)

    d = rng.normal(0, 1, shape)

    bound = 2
    flat = abs(d) <= bound

    d[d < bound] -= width - bound
    d[d > bound] += width - bound

    d[flat] = rng.uniform(-width, width, np.shape(d[flat]))

    return center + d

# ====================================================================================================
# Von Mises angle estimate
# ====================================================================================================

def vonmises_angle_estimate(mu):
    """
    Estimate the characteristic angle (in radians) of a von Mises distribution
    based on its concentration parameter mu.

    For large mu (high concentration), the resulting angle is small.
    For small mu (low concentration), the angle approaches 2π (uniformity).

    Parameters
    ----------
    mu : float or array_like or None
        Concentration parameter of the von Mises distribution. If None,
        a default value of 1 radian is returned.

    Returns
    -------
    float or ndarray
        Estimated angular spread (in radians) corresponding to the given mu.
    """    
    if mu is None:
        return 1
    else:
        return np.exp(-np.array(mu)/6)*2*np.pi





