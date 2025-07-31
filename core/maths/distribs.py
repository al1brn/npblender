"""
This file is part of the geonodes distribution (https://github.com/al1brn/npblender).
Copyright (c) 2025 Alain Bernard.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------
numpy for Blender
-----------------------------------------------------


Distributions
-----------------------------------------------------

Distribution functions on curves, surfaces and volumes.

Distribution of locations and speeds



Created on Sat Nov 18 08:28:39 2023
Updated : 2025/07/18
"""

import numpy as np

print(__name__)

if __name__ == '__main__':
    from utils import get_axis, flat_top_gaussian, vonmises_angle_estimate
    from transformation import Rotation, Quaternion

else:
    from . utils import get_axis, flat_top_gaussian, vonmises_angle_estimate
    from . transformation import Rotation, Quaternion

PI  = np.pi
HPI = np.pi/2
TAU = 2*np.pi

# ====================================================================================================
# Regular distributions
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Line distribution
# ----------------------------------------------------------------------------------------------------

def regular_1d(point0=(0, 0, 0), point1=(0, 0, 1), count=10):
    return np.linspace(point0, point1, count)

# ----------------------------------------------------------------------------------------------------
# Grid distribution
# ----------------------------------------------------------------------------------------------------

def regular_2d(size=1, shape=(10, 10)):
    sx, sy = size, size if np.shape(size) == () else size
    x, y = np.meshgrid(np.linspace(-sx/2, sx/2, shape[0]), np.linspace(-sy/2, sy/2, shape[1]))
    return np.reshape(np.stack((x, y, np.zeros_like(x)), axis=-1), (np.prod(shape, dtype=int), 3))

# ----------------------------------------------------------------------------------------------------
# Cube distribution
# ----------------------------------------------------------------------------------------------------

def regular_3d(size=1, shape=(10, 10, 10)):
    sx, sy, sz = size, size, size if np.shape(size) == () else size
    return np.reshape(np.stack(np.meshgrid(
            np.linspace(-sx/2, sx/2, shape[0]),
            np.linspace(-sy/2, sy/2, shape[1]),
            np.linspace(-sz/2, sz/2, shape[2]),
            ), axis=-1), (np.prod(shape, dtype=int), 3))

# ----------------------------------------------------------------------------------------------------
# Circle distribution
# ----------------------------------------------------------------------------------------------------

def regular_circle(radius=1., angle=None, pie_angle=PI/4, count=16):
    if angle is None:
        ag = np.linspace(0, TAU, count, endpoint=False)
    else:
        ag = np.linspace(angle - pie_angle/2, angle + pie_angle/2, count, endpoint=True)

    return np.stack((radius*np.cos(ag), radius*np.sin(ag), np.zeros_like(ag)), axis=-1)

# ====================================================================================================
# Normal distribution
# ====================================================================================================

def normal_dist(shape, scale, dim=3, seed=None):
    """Generate normally distributed points in random directions.

    Points are distributed isotropically around the origin, with
    a normal distribution of distances (mean 0, std = scale).

    Arguments
    ---------
    - shape (int or tuple) : shape of the returned array
    - scale (int) : standard deviation
    - dim (int in (1, 2, 3)) : dimension
    - seed (int or Generator) : random seed

    Returns
    -------
    - array of values in dim dimensions
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # 1D
    # --------------------------------------------------

    if dim == 1:
        return rng.normal(0, scale, shape)

    # --------------------------------------------------
    # 2D
    # --------------------------------------------------

    elif dim == 2:
        theta = rng.uniform(0, 2*np.pi, shape)
        vectors = np.stack((np.cos(theta), np.sin(theta)), axis=-1)

        r = rng.normal(0, scale, shape)

        return vectors*r[..., None]

    # --------------------------------------------------
    # 3D
    # --------------------------------------------------

    z = rng.uniform(-1, 1, shape)
    theta = rng.uniform(0, 2*np.pi, shape)
    vectors = np.stack((np.sqrt(1 - z**2)*np.cos(theta),
                        np.sqrt(1 - z**2)*np.sin(theta),
                        z), axis=-1)

    r = rng.normal(0, scale, shape)

    return vectors*r[..., None]

# ====================================================================================================
# Shake points
# ====================================================================================================

def shake_points(points, scale=None, seed=None):
    """
    Slightly displace points using isotropic normal noise.

    Each point is moved randomly in all directions according to a
    normal (Gaussian) distribution centered at its original position.
    If `scale` is None, the points are returned unchanged.

    Parameters
    ----------
    points : array_like, shape (..., D)
        Input array of points (D is typically 2 or 3).
    scale : float or None, optional
        Standard deviation of the displacement. If None, no displacement is applied.
    seed : int or Generator, optional
        Random seed or NumPy random Generator for reproducibility.

    Returns
    -------
    displaced : ndarray, shape (..., D)
        Array of displaced points.
    """
    if points is None or scale is None:
        return points

    points = np.asarray(points)
    shape = points.shape[:-1]
    dim = points.shape[-1]

    noise = normal_dist(shape, scale, dim=dim, seed=seed)
    return points + noise

# ====================================================================================================
# Shake vectors
# ====================================================================================================

def shake_vectors(vectors, scale=None, length_only=False, lengths=None, seed=None):
    """
    Slightly perturb a set of vectors.

    The noise is proportional to the original vector norms. This can affect either
    only the magnitudes (`length_only=True`) or both directions and magnitudes.

    Parameters
    ----------
    vectors : array_like, shape (..., D)
        Input vectors to perturb.
    scale : float or None, optional
        Relative noise scale (as a fraction of the vector norm). If None, no perturbation is applied.
    length_only : bool, default False
        If True, only the magnitudes are modified (direction remains unchanged).
    lengths : array_like, optional
        Precomputed norms of the input vectors. Required only if `length_only=False`.
    seed : int or Generator, optional
        Seed or NumPy random Generator for reproducibility.

    Returns
    -------
    perturbed_vectors : ndarray, shape (..., D)
        The perturbed vectors.
    """
    if vectors is None or scale is None:
        return vectors

    vectors = np.asarray(vectors)
    shape = vectors.shape[:-1]

    if length_only:
        rng = np.random.default_rng(seed)
        # Multiplicative noise on magnitude only
        noise = rng.normal(loc=1.0, scale=scale, size=shape + (1,))
        return vectors * noise

    else:
        if lengths is None:
            lengths = np.linalg.norm(vectors, axis=-1)

        dim = vectors.shape[-1]
        perturb = normal_dist(shape, scale, dim=dim, seed=seed)
        return vectors + perturb * lengths[..., None]

# ====================================================================================================
# 1D distributions
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Points spread on a segment
# ----------------------------------------------------------------------------------------------------

def line_dist(point0=(-1, -1, -1), point1=(1, 1, 1), count=10, density=None, seed=None):
    """
    Distribute points randomly along a segment [point0, point1].

    Parameters
    ----------
    point0 : array_like, shape (D,)
        First endpoint of the segment.
    point1 : array_like, shape (D,)
        Second endpoint of the segment.
    count : int, default=10
        Number of points to generate (ignored if density is specified).
    density : float, optional
        If specified, determines point count via a Poisson distribution with mean = density * segment_length.
    seed : int or Generator, optional
        Random seed.

    Returns
    -------
    dict with:
        - 'points': ndarray of shape (count, D)
        - 'tangents': ndarray of shape (count, D)
    """

    point0 = np.asarray(point0, dtype=np.float32)
    point1 = np.asarray(point1, dtype=np.float32)

    rng = np.random.default_rng(seed)

    v = point1 - point0
    l = np.linalg.norm(v)

    if l == 0:
        return {'points': np.tile(point0, (count, 1)), 'tangents': np.zeros((count, len(point0)))}

    if density is not None:
        count = rng.poisson(l * density)

    if count == 0:
        return None

    t = rng.uniform(0, 1, count)[..., None]
    points = point0 + t * v
    tangents = np.tile(v / l, (count, 1))

    return {'points': points, 'tangents': tangents}

# ----------------------------------------------------------------------------------------------------
# Points on an arc of circle
# ----------------------------------------------------------------------------------------------------

def arc_dist(
    radius=1.0,
    scale=None,
    center=(0, 0, 0),
    arc_center=0.0,
    arc_angle=PI/2,
    use_vonmises=False,
    count=10,
    density=None,
    seed=None,
):
    """
    Distribute points along an arc of a circle.

    Parameters
    ----------
    radius : float, default=1.0
        Base radius of the arc.
    scale : float, optional
        Standard deviation for Gaussian noise around the radius.
    center : array_like, shape (D,), default=(0, 0, 0)
        Center of the arc (2D or 3D).
    arc_center : float, default=0.0
        Center angle (in radians) of the arc.
    arc_angle : float, default=TAU
        Total angle covered by the arc (in radians).
    use_vonmises : bool, default=False
        Whether to sample angles using a von Mises distribution.
    count : int, default=10
        Number of points to generate (ignored if density is given).
    density : float, optional
        If specified, the number of points is sampled from a Poisson distribution of expected arc length × density.
    seed : int or Generator, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'points': coordinates of sampled points
        - 'normals': normal unit vectors at each point
        - 'tangents': tangent unit vectors at each point
        - 'lengths': radius values used for each point
        - 'angles': angle values (in radians) used for each point
    """

    rng = np.random.default_rng(seed)
    center = np.asarray(center, dtype=float)
    dim = center.shape[-1]

    if dim < 2:
        raise ValueError("Center must be at least 2D")

    # --- Count estimation if density is given
    if density is not None:
        arc_len = vonmises_angle_estimate(arc_angle) * radius if use_vonmises else arc_angle * radius
        count = rng.poisson(arc_len * density)

    if count == 0:
        return None

    # --- Sample angles
    if use_vonmises:
        if arc_angle is None or arc_angle <= 0:
            ags = np.full(count, arc_center)
        else:
            mu = np.exp(arc_angle)  # higher arc_angle → tighter concentration
            ags = rng.vonmises(arc_center, mu, count)
    else:
        ags = arc_center + rng.uniform(-arc_angle / 2, arc_angle / 2, count)

    cags = np.cos(ags)
    sags = np.sin(ags)

    # --- Sample radius
    if scale is None:
        rs = np.full(count, radius)
    else:
        rs = rng.normal(radius, scale, count)

    # --- Compute points
    points = np.zeros((count, dim), dtype=float)
    points[:, 0] = rs * cags
    points[:, 1] = rs * sags

    # --- Compute tangents
    tangents = np.zeros((count, dim), dtype=float)
    tangents[:, 0] = -sags
    tangents[:, 1] = cags

    return {
        'points': points + center,
        'normals': np.stack((cags, sags, np.zeros_like(cags)), axis=-1),
        'tangents': tangents,
        'lengths': rs,
        'angles': ags,
    }


# ----------------------------------------------------------------------------------------------------
# Points on a circle
# ----------------------------------------------------------------------------------------------------

def circle_dist(
    radius=1.0,
    scale=None,
    center=(0, 0, 0),
    count=10,
    density=None,
    seed=None,
):
    """
    Distribute points along a full circle in the XY plane.

    Parameters
    ----------
    radius : float, default=1.0
        Base radius of the circle.
    scale : float, optional
        Standard deviation for Gaussian noise around the radius.
    center : array_like, shape (D,), default=(0, 0, 0)
        Center of the circle (2D or 3D).
    count : int, default=10
        Number of points to generate (ignored if density is given).
    density : float, optional
        If specified, the number of points is sampled from a Poisson distribution of expected arc length × density.
    seed : int or Generator, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'points': coordinates of sampled points
        - 'tangents': tangent unit vectors at each point
        - 'lengths': radius values used for each point
        - 'angles': angle values (in radians) used for each point
    """
    return arc_dist(
        radius=radius,
        scale=scale,
        center=center,
        arc_center=0.0,
        arc_angle=2 * np.pi,
        use_vonmises=False,
        count=count,
        density=density,
        seed=seed,
    )


# ----------------------------------------------------------------------------------------------------
# Spline distribution
# ----------------------------------------------------------------------------------------------------

def curve_dist_LATER(curve, t0=0., t1=1., count=10, density=None, seed=None):
    """ Distribute points on a curve.

    Arguments
    ---------
    - curve (Curve) : the curve
    - t0 (float = 0.) : start parameter
    - t1 (float = 1.) : end parameter
    - count (int = 10) : number of points to generate (overriden by density if not None)
    - density (float = None) : density of points (overrides count if not None)
    - seed (int = None) : random seed

    Returns
    -------
    - dict : 'points', 'tangents'
    """

    rng = np.random.default_rng(seed)

    funcs = curve.splines.functions

    locs = np.zeros((0, 3), float)
    tgs  = np.zeros((0, 3), float)

    for i, (func, length) in enumerate(zip(funcs, funcs.length)):

        if density is None:
            c = count
        else:
            c = rng.poisson(length*density)

        if c == 0:
            continue

        t = rng.uniform(t0, t1, c)

        pts  = func(t)
        locs = np.append(locs, pts, axis=0)

        tgs  = np.append(tgs, func.tangent(t), axis=0)

    if False and curve.dimensions == '2D':
        locs = np.array(locs[:, :2])
        tgs  = np.array(tgs[:, :2])

    return {'points': locs, 'tangents': tgs}


# ====================================================================================================
# 2D distributions
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Points spread on rectangle
# ----------------------------------------------------------------------------------------------------

def rect_dist(a=1, b=1, center=(0, 0, 0), count=10, density=None, seed=None):
    """
    Distribute points uniformly on a rectangle in the XY plane.

    Parameters
    ----------
    a : float, default=1
        Length of the rectangle along the X axis.
    b : float, default=1
        Length of the rectangle along the Y axis.
    center : array_like, shape (2,) or (3,), default=(0, 0, 0)
        Center of the rectangle.
    count : int, default=10
        Number of points to generate (overridden by density if specified).
    density : float, optional
        Point density per unit area. Overrides `count` if provided.
    seed : int or Generator, optional
        Random seed or generator.

    Returns
    -------
    dict
        Dictionary with:
        - 'points': (count, D) array of sampled points (D=2 or 3)
        - 'normals': (count, 3) array of normals (Z-up)
    """
    rng = np.random.default_rng(seed)

    center = np.asarray(center, dtype=float)
    dim = center.shape[-1]
    if dim not in (2, 3):
        raise ValueError("Center must have dimension 2 or 3")

    if density is not None:
        area = a * b
        count = rng.poisson(area * density)

    if count == 0:
        return None

    # Generate (x, y) in local rectangle coordinates
    xy = rng.uniform(low=[-a / 2, -b / 2], high=[a / 2, b / 2], size=(count, 2))

    # Assemble full points (2D or 3D)
    points = np.zeros((count, dim), dtype=float)
    points[:, 0:2] = xy
    points += center

    normals = np.tile((0., 0., 1.), (count, 1))  # always Z-up (3D)
    return {'points': points, 'normals': normals}


# ----------------------------------------------------------------------------------------------------
# Points on a disk pie
# ----------------------------------------------------------------------------------------------------

def pie_dist(
    radius=1,
    outer_radius=None,
    center=(0, 0, 0),
    pie_center=0.,
    pie_angle=PI/2,
    use_vonmises=False,
    count=10,
    density=None,
    seed=None
):
    """
    Distribute points uniformly inside a circular sector ("pie") in the XY plane.

    Parameters
    ----------
    radius : float, default=1
        Disk radius (or inner radius if `outer_radius` is specified).
    outer_radius : float, optional
        Outer radius. If provided, defines a ring sector from `radius` to `outer_radius`.
    center : array_like, shape (2,) or (3,), default=(0, 0, 0)
        Center of the pie.
    pie_center : float, default=0.
        Central angle of the pie (radians).
    pie_angle : float, default=TAU
        Total angular span of the pie (radians).
    use_vonmises : bool, default=False
        Whether to use a Von Mises distribution for angle sampling.
    count : int, default=10
        Number of points to generate (overridden by density if provided).
    density : float, optional
        Point density per unit area. Overrides `count` if given.
    seed : int or Generator, optional
        Random seed or NumPy Generator.

    Returns
    -------
    dict
        - 'points'   : (count, D) array of points (D=2 or 3)
        - 'tangents' : (count, 3) array of direction vectors (XY tangent)
        - 'normals'  : (count, 3) array of Z-up normals
        - 'lengths'  : (count,) array of radial distances
        - 'angles'   : (count,) array of polar angles (radians)
    """

    rng = np.random.default_rng(seed)
    center = np.asarray(center, dtype=float)
    dim = center.shape[-1]

    if dim not in (2, 3):
        raise ValueError("Center must be 2D or 3D")

    # --- Radii
    r0 = 0. if outer_radius is None else radius
    r1 = radius if outer_radius is None else outer_radius

    if r1 <= r0:
        raise ValueError("Outer radius must be greater than inner radius.")

    # --- Count from density
    if density is not None:
        effective_angle = vonmises_angle_estimate(pie_angle) if use_vonmises else pie_angle
        area = 0.5 * effective_angle * (r1**2 - r0**2)
        count = rng.poisson(area * density)

    if count == 0:
        return None

    # --- Angles
    if use_vonmises:
        if pie_angle is None:
            ags = np.full(count, pie_center)
        else:
            ags = rng.vonmises(pie_center, np.exp(pie_angle), count)
    else:
        ags = pie_center + rng.uniform(-pie_angle / 2, pie_angle / 2, count)

    # --- Radii (uniform in area)
    rs = r0 + np.sqrt(rng.uniform(0, 1, count)) * (r1 - r0)

    # --- Coordinates
    cos_a = np.cos(ags)
    sin_a = np.sin(ags)
    locs = np.zeros((count, dim), dtype=float)
    locs[..., 0] = rs * cos_a
    locs[..., 1] = rs * sin_a
    locs += center

    # --- Tangents (orthogonal to radius)
    tgs = np.zeros((count, 3), dtype=float)
    tgs[:, 0] = -sin_a
    tgs[:, 1] = cos_a

    # --- Normals
    normals = np.tile((0., 0., 1.), (count, 1))

    return {
        'points': locs,
        'tangents': tgs,
        'normals': normals,
        'lengths': rs,
        'angles': ags,
    }

# ----------------------------------------------------------------------------------------------------
# Points on a disk
# ----------------------------------------------------------------------------------------------------

def disk_dist(radius=1, outer_radius=None, center=(0, 0, 0), count=10, density=None, seed=None):
    """
    Distribute points uniformly on a 2D disk or annulus in the XY plane.

    Parameters
    ----------
    radius : float, default=1
        Radius of the disk (or inner radius if `outer_radius` is provided).
    outer_radius : float, optional
        Outer radius of the disk. If given, points are sampled in an annular ring [radius, outer_radius].
    center : array_like, shape (2,) or (3,), default=(0, 0, 0)
        Center of the disk.
    count : int, default=10
        Number of points to generate (overridden by `density` if given).
    density : float, optional
        Point density per unit area. Overrides `count` if provided.
    seed : int or Generator, optional
        Random seed or NumPy random generator.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'points'   : (count, D) point positions
        - 'tangents' : (count, 3) unit tangent vectors in-plane
        - 'normals'  : (count, 3) normal vectors (Z-up)
        - 'lengths'  : (count,) radial distances from center
        - 'angles'   : (count,) polar angles in radians
    """
    return pie_dist(
        radius=radius,
        outer_radius=outer_radius,
        center=center,
        pie_center=0.,
        pie_angle=TAU,
        use_vonmises=False,
        count=count,
        density=density,
        seed=seed
    )

# ----------------------------------------------------------------------------------------------------
# Points spread on a cylinder
# ----------------------------------------------------------------------------------------------------

def cylinder_dist(
    radius=1.0,
    scale=None,
    height=1.0,
    center=(0, 0, 0),
    arc_center=0.0,
    arc_angle=TAU,
    use_vonmises=False,
    count=10,
    density=None,
    seed=None,
):
    """
    Distribute points on the lateral surface of a cylinder.

    Points are placed on an arc of the circular base and uniformly along height.

    Parameters
    ----------
    radius : float, default=1.0
        Cylinder radius.
    scale : float, optional
        Radial noise (used inside `arc_dist`).
    height : float, default=1.0
        Height of the cylinder (Z direction).
    center : array_like, shape (3,), default=(0, 0, 0)
        Center of the cylinder.
    arc_center : float, default=0.0
        Center angle (in radians) for arc distribution.
    arc_angle : float, default=TAU
        Angular extent of the arc to sample.
    use_vonmises : bool, default=False
        Whether to use Von Mises distribution for angular sampling.
    count : int, default=10
        Number of points to sample (overridden by `density`).
    density : float, optional
        Surface density of points per unit area (overrides `count`).
    seed : int or Generator, optional
        Random seed or NumPy generator.

    Returns
    -------
    dict
        Dictionary with:
        - 'points'   : (count, 3) sampled positions
        - 'tangents' : (count, 3) tangent vectors along arc
        - 'normals'  : (count, 3) surface normals (radial)
        - 'lengths'  : (count,) radial distances
        - 'angles'   : (count,) angular positions (radians)
    """

    rng = np.random.default_rng(seed)

    # Estimate point count from density
    if density is not None:
        arc_len = vonmises_angle_estimate(arc_angle) if use_vonmises else arc_angle
        lateral_area = arc_len * radius * height
        count = rng.poisson(lateral_area * density)

    if count == 0:
        return None

    # Sample points on circular arc (XY plane)
    d = arc_dist(
        radius=radius,
        scale=scale,
        center=(0, 0, 0),
        arc_center=arc_center,
        arc_angle=arc_angle,
        use_vonmises=use_vonmises,
        count=count,
        seed=rng,
    )

    # Add height component (Z direction)
    d['points'][:, 2] = rng.uniform(-height / 2, height / 2, count)

    # Shift to final center
    d['points'] += np.asarray(center)

    return d

# ----------------------------------------------------------------------------------------------------
# Points spread on a sphere
# ----------------------------------------------------------------------------------------------------

def sphere_dist(
    radius=1.0,
    scale=None,
    center=(0, 0, 0),
    count=10,
    density=None,
    seed=None
):
    """
    Distribute points on the surface of a sphere.

    Parameters
    ----------
    radius : float, default=1.0
        Radius of the sphere.
    scale : float, optional
        Standard deviation of radial noise.
    center : array_like, shape (3,), default=(0, 0, 0)
        Center of the sphere.
    count : int, default=10
        Number of points to generate (overridden if `density` is provided).
    density : float, optional
        Surface density of points per unit area (overrides `count`).
    seed : int or Generator, optional
        Random seed or NumPy Generator.

    Returns
    -------
    dict :
        {
            'points'  : (count, 3) array of sampled positions,
            'normals' : (count, 3) unit vectors from center to points,
            'lengths' : (count,) sampled distances from center,
            'thetas'  : (count,) azimuthal angles ∈ [0, 2π),
            'phis'    : (count,) polar angles ∈ [–π/2, π/2]
        }
    """

    rng = np.random.default_rng(seed)

    # Estimate count from surface area
    if density is not None:
        area = 4 * np.pi * radius**2
        count = rng.poisson(area * density)

    if count <= 0:
        return None

    # Sample angles
    z = rng.uniform(-1.0, 1.0, count)     # cos(φ) uniformly distributed
    phi = np.arcsin(z)                   # polar angle ∈ [–π/2, π/2]
    theta = rng.uniform(0, 2 * np.pi, count)  # azimuthal angle ∈ [0, 2π)

    cos_phi = np.cos(phi)

    # Compute unit normals
    normals = np.stack([
        cos_phi * np.cos(theta),
        cos_phi * np.sin(theta),
        z
    ], axis=-1)

    # Compute radius
    if scale is None:
        rs = np.full(count, radius, dtype=float)
    else:
        rs = rng.normal(radius, scale, count)

    points = normals * rs[..., None]
    return {
        'points' : center + points,
        'normals': normals,
        'lengths': rs,
        'thetas' : theta,
        'phis'   : phi
    }

# ----------------------------------------------------------------------------------------------------
# Points spread on a spherical dome
# ----------------------------------------------------------------------------------------------------

def dome_dist(
    radius=1.0,
    scale=None,
    axis=(0, 0, 1),
    angle=np.pi / 2,
    use_vonmises=False,
    center=(0, 0, 0),
    count=10,
    density=None,
    seed=None
):
    """Distribute points on a spherical cap (dome).

    Parameters
    ----------
    radius : float, default=1.0
        Radius of the sphere.
    scale : float, optional
        Standard deviation of radial noise.
    axis : array_like, shape (3,), default=(0, 0, 1)
        Direction of the dome (default is +Z).
    angle : float, default=π/2
        Aperture angle of the dome (in radians), from center axis.
    use_vonmises : bool, default=False
        Whether to use Von Mises angular distribution.
    center : array_like, shape (3,), default=(0, 0, 0)
        Center of the sphere.
    count : int, default=10
        Number of points to generate (overridden by `density` if provided).
    density : float, optional
        Surface density of points per unit area.
    seed : int or Generator, optional
        Random seed or generator.

    Returns
    -------
    dict :
        {
            'points'  : (count, 3) array of sampled positions,
            'normals' : (count, 3) unit vectors from center to points,
            'lengths' : (count,) sampled distances from center
        }
    """

    rng = np.random.default_rng(seed)
    axis = np.asarray(axis, dtype=float)
    center = np.asarray(center, dtype=float)

    # ----- Estimate count from surface area
    if density is not None:
        if use_vonmises:
            area_angle = vonmises_angle_estimate(angle)
            area = 4 * area_angle * radius**2
        else:
            area = 2 * np.pi * radius**2 * (1 - np.cos(angle / 2))
        count = rng.poisson(area * density)

    if count <= 0:
        return None

    # ----- Sample polar angles φ ∈ [0, angle/2]
    if use_vonmises:
        phi = rng.vonmises(np.pi / 2, np.exp(angle), count)
        sphi = np.sin(phi)
        cphi = np.cos(phi)
    else:
        z = rng.uniform(np.cos(angle / 2), 1.0, count)
        sphi = np.sqrt(1 - z**2)
        cphi = z

    theta = rng.uniform(0, 2 * np.pi, count)

    # ----- Normals
    normals = np.stack([
        sphi * np.cos(theta),
        sphi * np.sin(theta),
        cphi
    ], axis=-1)

    # ----- Radii
    if scale is None:
        rs = np.full(count, radius)
    else:
        rs = rng.normal(radius, scale, count)

    points = normals * rs[:, None]

    # ----- Orientation along arbitrary axis
    rot = Rotation.look_at((0, 0, 1), axis)
    points_rot = rot @ points
    normals_rot = rot @ normals

    return {
        'points': center + points_rot,
        'normals': normals_rot,
        'lengths': rs
    }


# ----------------------------------------------------------------------------------------------------
# Point on a triangle
# ----------------------------------------------------------------------------------------------------

def triangle_dist(corners, count, rng=None):
    """Distribute points uniformly on a triangle.

    Parameters
    ----------
    corners : array_like, shape (3, 3)
        Coordinates of the three triangle corners.
    count : int
        Number of points to generate.
    rng : np.random.Generator, optional
        Numpy random generator. Created if not provided.

    Returns
    -------
    points : ndarray, shape (count, 3)
        Points uniformly distributed on the triangle.
    """
    if rng is None:
        rng = np.random.default_rng()

    if count == 0:
        return np.zeros((0, 3), dtype=float)

    corners = np.asarray(corners, dtype=float)
    O = corners[0]
    i = corners[1] - O
    j = corners[2] - O

    # Generate barycentric coordinates with uniform sampling
    p = rng.uniform(0, 1, (count, 2))
    mask = (p[:, 0] + p[:, 1]) > 1.0
    p[mask] = 1.0 - p[mask]

    return O + p[:, 0, None] * i + p[:, 1, None] * j

# ----------------------------------------------------------------------------------------------------
# Point on a surface
# ----------------------------------------------------------------------------------------------------

def surface_dist(surface, count=10, density=None, seed=None):
    """Distribute points on a triangulated surface.

    The surface is passed as a dictionary with the following keys:
        - 'triangles' : (n_faces, 3, 3) float array of triangle vertices
        - 'areas'     : (n_faces,) float array of face areas
        - 'normals'   : (n_faces, 3) float array of face normals

    Parameters
    ----------
    surface : dict
        Surface description with triangles, areas, and normals.
    count : int, default=10
        Number of points to generate (overridden by density).
    density : float, optional
        Point density per unit area (overrides count).
    seed : int, optional
        Random seed.

    Returns
    -------
    dict :
        {
            'points'  : (count, 3) sampled points on the surface,
            'normals' : (count, 3) normals at each sampled point
        }
    """
    rng = np.random.default_rng(seed)

    triangles = np.asarray(surface['triangles'])
    areas     = np.asarray(surface['areas'])
    normals   = np.asarray(surface['normals'])

    assert triangles.shape[1:] == (3, 3), "Triangles should have shape (n, 3, 3)"

    nfaces = len(triangles)
    total_area = np.sum(areas)

    # Compute count from density if provided
    if density is not None:
        count = rng.poisson(total_area * density)

    if count == 0:
        return None

    # Select face indices according to area distribution
    face_probs = areas / total_area
    face_indices = rng.choice(nfaces, size=count, p=face_probs)

    # Count how many points to generate per face
    per_face_counts = np.bincount(face_indices, minlength=nfaces)

    # Allocate outputs
    points = np.empty((count, 3), dtype=float)
    normal_array = np.empty((count, 3), dtype=float)

    i = 0
    for face_idx, n in enumerate(per_face_counts):
        if n == 0:
            continue
        tri = triangles[face_idx]
        points[i:i+n] = triangle_dist(tri, n, rng)
        normal_array[i:i+n] = normals[face_idx]
        i += n

    return {'points': points, 'normals': normal_array}

# ----------------------------------------------------------------------------------------------------
# Points on to a surface
# ----------------------------------------------------------------------------------------------------

def mesh_dist(mesh, selection=None, count=10, density=None, seed=None):
    """ > Distribute points on a mesh surface.

    Arguments
    ---------
    - mesh (Mesh) : mesh object
    - selection (array of bools = None) : face selection mask
    - count (int = 10) : number of points to generate (overridden by density if not None)
    - density (float = None) : density of points (overrides count if not None)
    - seed (int = None) : random seed

    Returns
    -------
    - dict : { 'points': (count, 3), 'normals': (count, 3) }
    """

    # Triangulate the selection
    tri_mesh = mesh.from_mesh(mesh, faces_sel=selection).triangulate()
    nfaces = len(tri_mesh.faces)

    normals = np.empty((nfaces, 3), dtype=np.float32)
    areas = np.empty(nfaces, dtype=np.float32)

    with tri_mesh.blender_data(readonly=True) as data:
        data.polygons.foreach_get('area', areas)
        data.polygons.foreach_get('normal', normals.ravel())

    corners = np.reshape(tri_mesh.corners, (nfaces, 3))

    surface = {
        'triangles': tri_mesh.vertices.position[corners],
        'normals': normals,
        'areas': areas,
    }

    return surface_dist(surface, count=count, density=density, seed=seed)

# ====================================================================================================
# 3D distributions
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Points spread into a cube
# ----------------------------------------------------------------------------------------------------

def cube_dist(size=1, center=(0, 0, 0), count=10, density=None, seed=None):
    """Distribute points uniformly inside a cube.

    Arguments
    ---------
    - size (float or array-like of 3 floats) : dimensions of the cube along each axis
    - center (array-like of 3 floats) : center of the cube
    - count (int) : number of points to generate (overridden by density if provided)
    - density (float, optional) : number of points per unit volume (overrides count)
    - seed (int or np.random.Generator, optional) : random seed

    Returns
    -------
    - dict : { 'points': (count, 3) array of sampled positions }
    """

    rng = np.random.default_rng(seed)

    size = np.full(3, size) if np.ndim(size) == 0 else np.asarray(size, dtype=float)
    center = np.asarray(center, dtype=float)

    if density is not None:
        volume = np.prod(size)
        count = rng.poisson(volume * density)

    if count <= 0:
        return None

    points = rng.uniform(center - size / 2, center + size / 2, (count, 3))
    return {'points': points}

# ----------------------------------------------------------------------------------------------------
# Points into a sphere
# ----------------------------------------------------------------------------------------------------

def ball_dist(
    radius=1.0,
    axis=(0, 0, 1),
    angle=np.pi,
    use_vonmises=False,
    center=(0, 0, 0),
    count=10,
    density=None,
    scale=None,
    seed=None,
    **kwargs
):
    """Distribute points inside a spherical volume (ball or cap).

    Parameters
    ----------
    radius : float
        Radius of the sphere.
    axis : array-like of shape (3,)
        Axis of the spherical cap (default is +Z).
    angle : float
        Angular aperture from the axis (0 to π).
    use_vonmises : bool
        Use Von Mises distribution for angular sampling.
    center : array-like of shape (3,)
        Center of the sphere.
    count : int
        Number of points to generate (overridden by density if provided).
    density : float, optional
        Target density (points per unit volume).
    seed : int or Generator, optional
        Random seed.
    
    Returns
    -------
    dict :
        {
            'points': (count, 3) array of positions,
            'normals': (count, 3) array of radial directions,
            'lengths': (count,) distances from center
        }
    """

    rng = np.random.default_rng(seed)
    center = np.asarray(center, float)
    axis = np.asarray(axis, float)

    # Estimate count from partial volume
    if density is not None:
        if use_vonmises:
            frac = vonmises_angle_estimate(angle)  # Approximate fraction of unit sphere
        else:
            # Fraction of volume of spherical cap (angle from axis)
            h = radius * (1 - np.cos(angle / 2))
            frac = (1 / 2) * (1 - np.cos(angle / 2))  # Approx fraction of full sphere volume
        volume = (4 / 3) * np.pi * radius**3 * frac
        count = rng.poisson(volume * density)

    if count <= 0:
        return {'points': np.zeros((0, 3)), 'normals': np.zeros((0, 3)), 'lengths': np.zeros((0,))}

    # Angular directions
    if use_vonmises:
        phi = rng.vonmises(np.pi / 2, np.exp(angle), count)
        sphi = np.sin(phi)
        cphi = np.cos(phi)
    else:
        z = rng.uniform(np.cos(angle / 2), 1.0, count)
        sphi = np.sqrt(1 - z**2)
        cphi = z

    theta = rng.uniform(0, 2 * np.pi, count)

    normals = np.stack([
        sphi * np.cos(theta),
        sphi * np.sin(theta),
        cphi
    ], axis=-1)

    # Radial distances
    rs = rng.uniform(0, radius, count)

    points = normals * rs[:, None]

    # Rotate to align with given axis
    rot = Rotation.look_at((0, 0, 1), axis)
    points_rot = rot @ points
    normals_rot = rot @ normals

    return {
        'points': center + points_rot,
        'normals': normals_rot,
        'lengths': rs
    }

# ====================================================================================================
# Speed distribution
# ====================================================================================================

def speed_dist(direction, speed, scale=None, mu=None, seed=None):
    """Generate velocity vectors distributed around a given direction.

    This uses a Von Mises-Fisher-like sampling in 2D or 3D,
    centered on the input direction, with angular dispersion controlled by `mu`.

    Parameters
    ----------
    direction : (N, D) array or (D,) array
        Target directions (2D or 3D), normalized or not.
    speed : float
        Base speed (magnitude of output vectors).
    scale : float, optional
        Std deviation for speed variation (in modulus).
    mu : float, optional
        Concentration parameter for angular dispersion (Von Mises). Higher = more concentrated.
    seed : int or Generator, optional
        Random seed.

    Returns
    -------
    speeds : (N, D) array
        Velocity vectors.
    """

    direction = np.asarray(direction, dtype=float)
    single = direction.ndim == 1
    if single:
        direction = direction[None, :]

    count, dim = direction.shape
    if count == 0:
        return np.zeros((0, dim))

    rng = np.random.default_rng(seed)

    # ===== Directional sampling =====

    if dim == 2:
        angles = np.arctan2(direction[:, 1], direction[:, 0])
        sampled = np.empty((count, 2))

        for i, theta in enumerate(angles):
            if mu is None:
                a = rng.uniform(0, 2 * np.pi)
            else:
                a = rng.vonmises(theta, mu)
            sampled[i] = [np.cos(a), np.sin(a)]

    elif dim == 3:
        sampled = np.empty((count, 3))
        for i, axis in enumerate(direction):
            sampled[i] = dome_dist(
                radius=1.0,
                axis=axis,
                angle=np.pi,
                use_vonmises=mu is not None,
                count=1,
                seed=rng if mu is not None else None,
            )['normals'][0]
    else:
        raise ValueError("Only 2D and 3D directions are supported.")

    # ===== Speed variation =====

    if scale is None:
        radii = np.full(count, speed)
    else:
        radii = rng.normal(speed, scale, count)

    result = sampled * radii[:, None]
    return result[0] if single else result

# ====================================================================================================
# Test with Blender
# ====================================================================================================

def test_distribs():

    print('-'*100)

    import numpy as np
    from npblender import Cloud, Mesh, Curve

    class X:
        DELTA = 5.
        VALUE = -5.

        @classmethod
        @property
        def offset(cls):
            cls.VALUE += cls.DELTA
            return (cls.VALUE, 0., 0.)

    def show_dist(d, name="Distrib", shake=None):

        pts = d['points'] + X.offset

        shake_points(pts, scale=shake)
        cloud = Cloud(pts)
        cloud.to_object(name)


    # ----- Normal dist

    pts = normal_dist(100, .2, dim=3)
    cloud = Cloud(pts)

    cloud.to_object("Normal dist")

    # ----- Line dist

    d = line_dist((-1, -1, -1), (2, 3, 4), density=10)
    show_dist(d, "Line dist")

    # ----- Arc dist

    d = arc_dist(radius=2., scale=.05, arc_center=np.pi/4, arc_angle=np.pi/2, use_vonmises=False, count=10, density=10)
    show_dist(d, "Arc dist")


    d = arc_dist(radius=2., scale=.05, arc_center=np.pi/4, arc_angle=5, use_vonmises=True, count=100, density=10)
    show_dist(d, "Arc dist (Von Mises)")

    # ----- Circle dist

    d = circle_dist(radius=1., scale=.05, count=10, density=10)
    show_dist(d, "Circle dist")

    # ----- curve dist

    curve = Curve.Spiral()
    d = curve_dist(curve, density=10)
    show_dist(d, "Curve dist", shake=.1)

    # ----- Rect dist

    d = rect_dist(3, 2, density=10)
    show_dist(d, "Rect dist", shake=.1)

    # ----- Pie dist

    d = pie_dist(radius0=1, radius1=2, pie_center=np.pi/2, pie_angle=np.pi, use_vonmises=False, density=10)
    show_dist(d, "Pie dist")

    d = pie_dist(radius0=1, radius1=2, pie_center=np.pi/2, pie_angle=4, use_vonmises=True, density=10)
    show_dist(d, "Pie dist (Von Mises)")

    # ----- Disk dist

    d = disk_dist(radius0=.5, radius1=2, density=10)
    show_dist(d, "Disk dist")

    # ----- Cylinder dist

    d = cylinder_dist(radius=2., scale=.02, height=2, arc_center=0., arc_angle=np.pi, use_vonmises=False, density=10)
    show_dist(d, "Cylinder dist")

    d = cylinder_dist(radius=2., scale=.02, height=2, arc_center=0., arc_angle=5, use_vonmises=True, density=10)
    show_dist(d, "Cylinder dist (Von Mises)")

    # ----- Sphere dist

    d = sphere_dist(radius=2., scale=.1, density=10)
    show_dist(d, "Sphere dist")

    # ----- Dome

    d = dome_dist(radius=2., scale=.1, axis=(1, 1, 1), angle=np.pi, use_vonmises=False, density=10)
    show_dist(d, "Dome dist")

    d = dome_dist(radius=2., scale=0, axis=(1, 1, 1), angle=5, use_vonmises=True, density=100)
    show_dist(d, "Dome dist (Von Mises)")

    # ----- Triangle

    pts = triangle_dist(np.array(((1, 1, 1), (1, 2, -1), (-1, 0, -2)), float), 100)
    show_dist({'points': pts}, "Triangle dist")

    # ----- Surface

    monkey = Mesh.Monkey()
    d = mesh_dist(monkey, selection=np.random.uniform(0, 1, len(monkey.faces))<.5, density=100)
    #d = distribs.mesh_dist(monkey, selection=None, density=100)
    show_dist(d, "Mesh dist")

    # ----- Cube

    d = cube_dist(size=(1, 2, 3), density=100)
    show_dist(d, "Cube dist")

    # ----- Ball

    d = ball_dist(radius=2., scale=.1, axis=(1, 1, 1), angle=np.pi, use_vonmises=False, density=10)
    show_dist(d, "Ball dist")

    # ----- Speeds

    d = rect_dist(4, 5, density=10)
    speeds = speed_dist(direction=d['normals'], speed=2, scale=.2, mu=None)
    mesh = Mesh(np.append(d['points'], d['points'] + speeds, axis=0))

    n = len(d['points'])
    mesh.edges.add(n, vertex0=np.arange(n), vertex1=np.arange(n) + n)

    mesh.points.position += X.offset

    mesh.to_object("Speed dist")
