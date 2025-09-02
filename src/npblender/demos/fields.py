import numpy as np

EPS = 1e-6

# ====================================================================================================
# Electric field
# ====================================================================================================

def electric_field(coords, charge_locations=[0, 0, 0], charges=1, charge_size=.1, speeds=None):
    """ Electric field produced by charges.

    If speeds are defined for the charges, the function returns the magnetic field.

    Arguments
    ---------
    - coords (array of vectors) : where to compute the field
    - charge_locations (array of vectors) : where the charges are located
    - charges (array of floats) : electric charges
    - charge_size (float) : field not defined close to the charges
    - speeds (array floats = None) : charge speed

    Returns
    -------
    - array ot vectors : field
    """

    # --------------------------------------------------
    # Coords
    # --------------------------------------------------

    if coords is None:
        return None
    
    coords = np.asarray(coords)
    if len(coords) == 0:
        return None
    
    single = coords.shape == (3,)
    coords = np.atleast_2d(coords) # (N, 3)

    # --------------------------------------------------
    # Charge locations
    # --------------------------------------------------

    charge_locations = np.asarray(charge_locations)
    if len(charge_locations) == 0:
        return np.zeros(3) if single else np.zeros_like(coords)
    
    charge_locations = np.atleast_2d(charge_locations) # (Q, 3)

    # --------------------------------------------------
    # Charges
    # --------------------------------------------------

    charges = np.atleast_1d(charges) # (Q,) or (1,)
    
    # --------------------------------------------------
    # Compute electric field
    # --------------------------------------------------

    E = coords[:, None] - charge_locations # (N, Q, 3)
    dist = np.linalg.norm(E, axis=2, keepdims=True)
    infinite = dist < charge_size
    dist[infinite] = np.nan
    E = E*charges[None, :, None] / dist**3

    if speeds is None:
        E = np.sum(E, axis=1)
        return E[0] if single else E

    # --------------------------------------------------
    # Magnetic field
    # --------------------------------------------------

    speeds = np.atleast_2d(speeds)
    B = np.cross(speeds[None], E)
    B = np.sum(B, axis=1)

    return B[0] if single else B


# ====================================================================================================
# Magnetic field
# ====================================================================================================

def magnetic_field(coords, location=[0, 0, 0], length=1, radius=.1, transfo=None):
    """ Approximation of the field produced by a cylindrical magnet.

    The field is approximated from a planar field produced by a single point
    oriented along x: circles centered on y axis and tangent in (0, 0):
    - x = r.sin(a)
    - y = r(1 - cos(a))
    When working with x > 0 and y > 0, r and a can be found with: f = x/y
    - s = f(1 - c) <=> s + fc = f <=> s² + 2fcs + f²c² = f² <=> s² + 2fcs + f²(c² - 1) = 0
    - s² + 2fcs - f²s² = 0 <=> (1 - f²)s = -2fc <=> s/c = 2f/(f² - 1)
    - a = atan(2f/(f² - 1))

    The circles are shifted along the y axis to approximate the magnet radius h:
    - x = r.sin(a)
    - y = dy + r(1 - cos(a))
    dy is such that:
    - dy(0) = 0
    - dy(h) = h
    - dy(inf) = 0
    Hence:
    - dy = h²y/(h² + y²)

    Then the circles are streched along x to approximate the magnet length:
    - x = x*(1 + length)
    """

    # --------------------------------------------------
    # Coords
    # --------------------------------------------------

    if coords is None:
        return None
    
    coords = np.asarray(coords)
    if len(coords) == 0:
        return None
    
    single = coords.shape == (3,)
    coords = np.atleast_2d(coords) # (N, 3)

    if transfo is not None:
        coords = transfo.inverse() @ coords

    # --------------------------------------------------
    # Canonical field
    # Coputed with x > 0 and y > 0 after stretching and shifting
    # --------------------------------------------------

    def _canonic(x, y):
        f = x/y   
        f2 = f**2 - 1
        a = np.arctan2(f2, 2*f)

        ca, sa = np.cos(a), np.sin(a)
        n = 1 + sa

        r = np.empty_like(x)
        mask = x > y
        r[mask] = x[mask]/sa[mask]
        mask = ~mask
        r[mask] = y[mask]/(1 + ca[mask])

        v = 3*n/r

        return v*sa, v*ca

    # --------------------------------------------------
    # Compute field
    # --------------------------------------------------

    # NOTE : Initialized at nan
    field = np.full(coords.shape, np.nan, dtype=np.float32)

    # Symmetry along x
    d = np.linalg.norm(coords[:, 1:], axis=1)
    # In formula, d is used as y (which is necessarily positive)
    x = np.abs(coords[:, 0])
    y = d

    # x stretching
    length = length/2
    x /= 1 + length

    # Particular cases
    y_not_null = y > EPS

    # Line of field along the x axis
    field[~y_not_null] = (1, 0, 0)

    # Valid points: x > 0 or y > radius
    valid = y_not_null

    y = y[valid]
    x = x[valid]

    # y shift
    dy = radius**2*y/(radius**2 + y*y)
    if False: # DEBUG
        y = y - dy + EPS

    # Canonic field
    fx, fy = _canonic(x, y)

    # Stretching back
    fx *= (1 + length)

    # To field
    fy /= d[valid]

    field[valid] = coords[valid]
    field[valid, 0] = fx
    field[valid, 1] *= fy
    field[valid, 2] *= fy

    # Symmetry
    field[coords[:, 0] < 0, 1:] *= -1

    # Back rotation
    if transfo is not None:
        field = transfo.rotation @ field

    return field[0] if single else field

# ====================================================================================================
# Induces field from a moving field
# ====================================================================================================

def induced_field(ifield_func, coords, speed=[0, 0, 0], factor=1., **kgwargs):

    field = ifield_func(coords, **kgwargs)
    speed = np.atleast_2d(speed)

    return factor * np.cross(speed, field)

# ====================================================================================================
# Lines of field
# ====================================================================================================

def line_of_fields(coords, field_func, count=100, ds=.1, **kwargs):

    coords = np.atleast_2d(coords)
    nsplines = len(coords)

    # ---------------------------------------------------------------------------
    # Move forwards or backwards
    # ---------------------------------------------------------------------------

    def _move(signed_ds, ends=None):

        lines = np.empty((nsplines, count, 3), dtype=np.float32)
        lines[:, 0] = coords

        weights = np.zeros((nsplines, count), dtype=np.float32)
        
        total = np.ones(nsplines, dtype=np.int32)

        active = np.ones(nsplines, dtype=bool)

        for i in range(count-1):

            locs = lines[active, i]
            field = field_func(locs, **kwargs)

            not_nan = ~np.isnan(field[:, 0])
            field = field[not_nan]

            nrm = np.linalg.norm(field, axis=1)
            not_null = nrm > 1e-6

            field = field[not_null] / nrm[not_null, None]

            active[active] = not_nan
            active[active] = not_null

            lines[active, i+1] = lines[active, i] + field*signed_ds
            weights[active, i+1] = nrm[not_null]
            if i == 0:
                weights[active, i] = nrm[not_null]

            total[active] += 1

            # ---- ends

            if i > 4:
                dist = np.linalg.norm(lines[active, i+1] - lines[active, 0], axis=-1)
                active[active] = dist > ds

                if ends is not None:
                    dist = np.linalg.norm(lines[active, i+1] - ends[active], axis=-1)
                    active[active] = dist > ds

        return lines, total, weights
    
    # ---------------------------------------------------------------------------
    # Move in the two directions
    # ---------------------------------------------------------------------------

    fwd_lines, fwd_total, fwd_weights = _move(ds)
    ends = fwd_lines[np.arange(nsplines), fwd_total-1]

    bwd_lines, bwd_total, bwd_weights = _move(-ds, ends=ends)

    # ---------------------------------------------------------------------------
    # Merge te resultings curves
    # ---------------------------------------------------------------------------

    npoints = np.sum(fwd_total) + np.sum(bwd_total) - nsplines
    points = np.empty((npoints, 3), np.float32)
    weights = np.empty(npoints, np.float32)

    offset = 0
    for i in range(nsplines):
        l = bwd_total[i]
        points[offset:offset + l] = np.flip(bwd_lines[i, :l], axis=0)
        weights[offset:offset + l] = np.flip(bwd_weights[i, :l], axis=0)
        # -1 : start point is at the end, it will be written twice
        offset += l - 1

        l = fwd_total[i]
        points[offset:offset + l] = fwd_lines[i, :l]
        weights[offset:offset + l] = fwd_weights[i, :l]
        offset += l

    return points, bwd_total + fwd_total - 1, weights

# ====================================================================================================
# Field visualisaton as field of vectors
# ====================================================================================================

def vis_field_of_vectors(field_func, coords, name="Field of vectors", materials=None, **kwargs):

    from npblender import Mesh

    field = field_func(coords, **kwargs)

    mesh = Mesh.vectors_field(
            coords, 
            field, 
            radius = .05,
            scale_length = 1.,
            angle = 24.,
            segments = 8,
            head = None,
            adjust_norm = None, 
            materials = materials)

    obj = mesh.to_object(name, shade_smooth=True)
    return obj

# ====================================================================================================
# Field visualisaton as field of vectors
# ====================================================================================================

def vis_lines_of_field(field_func, coords, name="Lines of field", as_mesh=False, count=100, ds=.1, **kwargs):

    from npblender import Curve

    field = field_func(coords, **kwargs)
    lines, total, weights = line_of_fields(coords, field_func, count=count, ds=ds, **kwargs)

    curve = Curve(points=lines, splines=total, radius=np.log(1 + weights))

    if as_mesh:
        curve = curve.to_mesh(profile=Curve.circle(resolution=6, radius=.01), use_radius=True)

    obj = curve.to_object("Lines of Electric Field")
    return obj

# ====================================================================================================
# Some demos
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Fields produced by moving electric charges
# ----------------------------------------------------------------------------------------------------

def demo_electric_charges(count=7, nsplines=100, magnetic=False, seed=0):

    from npblender import engine, Transformation, Rotation, Mesh

    rng = np.random.default_rng(seed)

    # Move the charges on circles
    radius = rng.uniform(1, 5, count)
    omega = (2*np.pi)*rng.uniform(1/10, 1, count)/10
    phi = rng.uniform(0, 2*np.pi, count)
    transfo = Transformation.from_components(
        rotation = Rotation.from_euler(rng.uniform(0, 2*np.pi, (count, 3))),
        translation = rng.uniform(-3, 3, (count, 3)),
        )
    q = rng.uniform(-1, 1, count)*100
    
    # Vis coords
    coords = rng.uniform(-10, 10, (nsplines, 3))

    def update():
        theta = omega*engine.time + phi
        x = radius*np.cos(theta)
        y = radius*np.sin(theta)
        z = np.zeros_like(x)
        loc = np.stack((x, y, z), axis=1)
        loc = transfo @ loc

        if magnetic:
            speeds = np.stack((-y, x, z), axis=1)*omega[:, None]
            speeds = transfo.rotation @ speeds
        else:
            speeds = None

        vis_lines_of_field(
            electric_field, 
            coords, 
            name="E from moving charges", 
            as_mesh=True, 
            count=100,
            ds=.1,
            charge_locations = loc,
            charges = q,
            charge_size = .1,
            speeds = speeds,
        )

        mesh = Mesh.icosphere(radius=.3)*count
        mesh.points.reshape(count, -1)
        mesh.points.position += loc[:, None]
        mesh.to_object("Charges")

    engine.go(update)

# ----------------------------------------------------------------------------------------------------
# Fields produced by a moving magnets
# ----------------------------------------------------------------------------------------------------

def demo_magnets(nsplines=100, magnetic=True, seed=0):

    from npblender import engine, Transformation, Rotation, Mesh

    rng = np.random.default_rng(seed)

    omega = 2*np.pi/3
    length = 3.
    mid = 2*length
    amp = mid - length/2

    def _field(coords, time):

        theta = omega*engine.time
        x0 = mid - amp*np.sin(theta)

        B0 = magnetic_field(coords, length=length, transfo=Transformation.from_components(translation=[x0, 0, 0]))
        B1 = magnetic_field(coords, length=length, transfo=Transformation.from_components(translation=[-x0, 0, 0]))

        if magnetic:
            return B0 + B1
        
        v0 = amp*omega*np.cos(theta)
        E0 = np.cross(np.array([v0, 0, 0]).reshape(1, 3), B0)
        E1 = np.cross(np.array([-v0, 0, 0]).reshape(1, 3), B1)

        print("SHAPES", B0.shape, E0.shape, E1.shape, (E0 + E1).shape)

        return E0 + E1

    # Vis coords
    coords = rng.uniform([-amp, -2, -.1], [amp, 2, .1], (nsplines, 3))
    assert(coords.shape == (nsplines, 3))

    def update():

        theta = omega*engine.time
        x0 = mid - amp*np.sin(theta)

        vis_lines_of_field(
            _field, 
            coords, 
            name="F from moving magnets" if magnetic else "E from moving magnets", 
            as_mesh=True, 
            count=100,
            ds=.1,
            time = engine.time,
        )

        mesh = Mesh.cube(size=(length, 1, .7))*2
        mesh.points.reshape(2, -1)
        v = np.array([x0, 0, 0]).reshape(1, 3)
        mesh.points[0].position -= v
        mesh.points[1].position += v
        mesh.to_object("Magnets")


    engine.go(update)

