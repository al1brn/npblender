import numpy as np

bfloat = np.float32

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





