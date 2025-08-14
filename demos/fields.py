
import numpy as np

# ====================================================================================================
# Electric field
# ====================================================================================================

def electric_field(coords, charge_locations=[0, 0, 0], charges=1, charge_size=.1):
    """ Electric field produced by charges
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
    # Compute field
    # --------------------------------------------------

    field = coords[:, None] - charge_locations # (N, Q, 3)
    dist = np.linalg.norm(field, axis=2, keepdims=True)
    infinite = dist < charge_size
    dist[infinite] = np.nan
    field = np.sum(field*charges[None, :, None] / dist**3, axis=1)

    return field[0] if single else field

# ====================================================================================================
# Magnetic field
# ====================================================================================================

def magnetic_field(coords, location=[0, 0, 0], length=1, radius=.1, transfo=None):

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
        coords = transfo.inverse(coords)

    # --------------------------------------------------
    # Compute field
    # --------------------------------------------------

    field = np.empty_like(coords)

    # Symmetry along x
    x_neg = coords[:, 0] < 0
    coords[x_neg, 0] *= -1

    # Symmetry around x axis
    r = np.linalg.norm(coords[:, 1:], axis=1)
    theta = np.atan2(coords[:, 2], coords[:, 1])

    




    # Outside the magnet

    out = np.logical_or(coords[:, 0] < length/2, coords[:, 0] > length/2)
    out = np.logical_or(out > radius)






    

    field = coords[:, None] - charge_locations # (N, Q, 3)
    dist = np.linalg.norm(field, axis=2, keepdims=True)
    infinite = dist < charge_size
    dist[infinite] = np.nan
    field = np.sum(field*charges[None, :, None] / dist**3, axis=1)

    
    



# ====================================================================================================
# Lines of field
# ====================================================================================================

def line_of_fields(coords, field_func, count=100, ds=.1, **kwargs):

    coords = np.atleast_2d(coords)
    nsplines = len(coords)

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
                    dist = np.linalg.norm(lines[active, i+1] - ends[active, 0], axis=-1)
                    active[active] = dist > ds



        return lines, total, weights

    fwd_lines, fwd_total, fwd_weights = _move(ds)
    ends = fwd_lines[np.arange(nsplines), fwd_total-1]

    bwd_lines, bwd_total, bwd_weights = _move(-ds)

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
# Field visu with arrows
# ====================================================================================================

def vis_field(field_func, seed=0, **kwargs):

    from npblender import Mesh

    rng = np.random.default_rng(seed)
    coords = rng.uniform(-10, 10, (1000, 3))
    field = field_func(coords, **kwargs)

    mesh = Mesh.vectors_field(coords, field, 
            radius=.05,
            scale_length=1.,
            angle=24.,
            segments=8,
            head=None,
            adjust_norm=None, 
            materials=None)

    mesh.to_object("Electric Field")

# ====================================================================================================
# Field visu with lines of field
# ====================================================================================================

def vis_lines_of_field(field_func, seed=0, as_mesh=False, **kwargs):

    from npblender import Curve

    rng = np.random.default_rng(seed)
    coords = rng.uniform(-10, 10, (1000, 3))
    field = field_func(coords, **kwargs)

    lines, total, weights = line_of_fields(coords, field_func, count=100, ds=.1, **kwargs)

    curve = Curve(points=lines, splines=total, radius=np.log(1 + weights))

    if as_mesh:
        curve = curve.to_mesh(profile=Curve.circle(resolution=6, radius=.01), use_radius=True)
    curve.      to_object("Lines of Electric Field")

def test(seed=0):
    rng = np.random.default_rng(seed)
    seed = rng.integers(1<<32, dtype=np.uint32)

    Q = 17
    charge_locations = rng.uniform(-10, 10, (Q, 3))
    charges = rng.uniform(-1, 1, Q)*100
    
    #vis_field(electric_field, charge_locations=charge_locations, charges=charges)
    vis_lines_of_field(electric_field, charge_locations=charge_locations, charges=charges)


class OLD:
    # ====================================================================================================
    # Field of vectors
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Field line
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def field_line(cls, field_func, start_point, max_len=10., prec=.01, sub_steps=10):

        pts = [start_point]
        rs  = [np.linalg.norm(field_func(start_point))]
        p   = np.array(start_point)
        l   = 0.
        for _ in range(10000):
            for _ in range(sub_steps):

                # ----- Vector at current location
                v0 = field_func(p)

                # ----- Precision along this vector
                norm  = np.sqrt(np.dot(v0, v0))
                factor = prec/norm
                v0 *= factor

                # ----- Average with target vector for more accurracy
                v1 = field_func(p + v0)*factor
                v = (v0 + v1)/2

                # ----- Next point
                p += v

            # ----- Segment length

            v = p - pts[-1]
            l += np.sqrt(np.dot(v, v))

            # ----- Add a new point

            pts.append(np.array(p))
            rs.append(norm)

            # ----- Done if loop or max_len is reached

            v = p - start_point
            cyclic = np.sqrt(np.dot(v, v)) < prec*(sub_steps-1)
            if cyclic or l >= max_len:
                pts.append(np.array(start_point))
                break

        if cyclic:
            pts.pop()

        return cls(pts, curve_type=POLY, cyclic=cyclic, radius=rs)
    
    # ----------------------------------------------------------------------------------------------------
    # Field lines
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def field_lines_OLD(cls, field_func, start_points,
        backwards=False, max_length=None, length_scale=None, end_points=None, zero=1e-6, max_points=1000,
        precision=.1, sub_steps=10, seed=0, **kwargs):

        """ Build splines showing lines of field

        Arguments :
        -----------
            - field_func (function of template (array of vectors, **kwargs) -> array of vectors) : the field function
            - start_points (array of vectors) : lines starting points
            - backwards (bool = False) : build lines backwards
            - max_length (float = None) : max line lengths
            - length_scale (float = None) : line length scale if random length scale around central value
            - end_points (array of vectors) : points where lines must end
            - zero (float = 1e-6) : value below which the field is null
            - max_points (int = 1000) : max number of points per spline
            - precision (float = 0.1) : step length
            - sub_steps (int = 10) : number of sub steps
        """

        splines = field.field_lines(field_func, start_points,
            backwards       = backwards,
            max_length      = max_length,
            length_scale    = length_scale,
            end_points      = end_points,
            zero            = zero,
            max_points      = max_points,
            precision       = precision,
            sub_steps       = sub_steps,
            seed            = seed,
            **kwargs)

        return cls(**splines)

        curves = cls()
        for avects, cyclic in lines:
            if len(avects) <= 1:
                continue
            curves.add(avects.co, curve_type='POLY', cyclic=cyclic, radius=avects.radius, tilt=avects.color)

        return curves
    
    # ----------------------------------------------------------------------------------------------------
    # Lines of electric field
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def electric_field_lines_OLD(cls, charge_locations, charges=1., field_color=True,
                           count=100, start_points=None, plane=None, plane_center=(0, 0, 0),
                           frag_length=None, frag_scale=None, max_points=1000,
                           precision=.1, sub_steps=10, seed=None):

        """ Create lines of field for a vector field generated by charges, typically an electric field.

        Arguments:
        ----------
            - charge_locations (array (n, 3) of vectors) : where the charges are located
            - charges (float or array (n) of floats = 1) : the charges
            - field_color (bool = True) : manage the field_color attribute
            - count (int = 100) : number of lines to create. Overriden by len(start_points) if not None
            - start_points (array (s, 3) of vectors = None) : the starting points to compute the lines from
            - plane (vector = None) = restrict start points to a plane defined by its perpendicular
            - plane_center (vector = (0, 0, 0)) : center of the plane
            - frag_length (float=None) : length of fragments, None for full lines
            - frag_scale (float=None) : length distribution scale
            - precision (float = .1) : step precision
            - sub_steps (int=10) : number of steps between two sucessive points of the lines
        """

        # ----------------------------------------------------------------------------------------------------
        # Field function

        poles = AttrVectors(charge_locations, charge=charges)
        field_func = lambda points: field.electric_field(points,
                            locations=poles.co, charges=poles.charge)

        # ----------------------------------------------------------------------------------------------------
        # Starting points

        rng = np.random.default_rng(seed=seed)
        n_charges = len(poles)

        if start_points is None:
            backwards = rng.choice([True, False], count)
            if frag_length is None:
                if plane is None:
                    start_points, _ = distribs.sphere_dist(radius=precision, count=count, seed=rng.integers(1<<63))
                else:
                    start_points, _ = distribs.circle_dist(radius=precision, count=count, seed=rng.integers(1<<63))
                    start_points = rotate_xy_into_plane(start_points, plane=plane, origin=plane_center)

                inds = rng.integers(0, n_charges, count)
                start_points += poles.co[inds]
                backwards[:] = poles.charge[inds] < 0

            else:
                center = np.average(poles.co, axis=0)
                bbox0, bbox1 = np.min(poles.co, axis=0), np.max(poles.co, axis=0)
                radius = 1.3*max(np.linalg.norm(bbox1 - center), np.linalg.norm(bbox0 - center))

                if plane is None:
                    start_points, _ = distribs.ball_dist(radius=radius, count=count, seed=rng.integers(1<<63))
                    start_points += center
                else:
                    start_points, _ = distribs.disk_dist(radius=radius, count=count, seed=rng.integers(1<<63))
                    start_points = rotate_xy_into_plane(start_points, plane=plane, origin=plane_center)

        else:
            if len(np.shape(start_points)) == 1:
                count = 1
            else:
                count = len(start_points)
            backwards = rng.choice([True, False], count)

        # ----------------------------------------------------------------------------------------------------
        # Full lines if frag_length is None

        full_lines = frag_length is None
        if full_lines:
            backwards[:] = False

        # ----------------------------------------------------------------------------------------------------
        # Field lines

        lines = field.field_lines(field_func,
            start_points    = start_points,
            backwards       = backwards,
            max_length      = frag_length,
            length_scale    = frag_scale,
            end_points      = charge_locations,
            max_points      = max_points,
            precision       = precision,
            sub_steps       = sub_steps,
            seed            = rng.integers(1 << 63),
            )

        # ----------------------------------------------------------------------------------------------------
        # Twice il full lines

        if full_lines:

            # ----- Exclude cyclic lines which are done

            open_lines = np.logical_not(lines['cyclic'] )

            # ----- Backwards lines

            backwards[:] = True
            back_lines = field.field_lines(field_func,
                start_points    = start_points[open_lines],
                backwards       = backwards[open_lines],
                max_length      = frag_length,
                length_scale    = frag_scale,
                end_points      = charge_locations,
                max_points      = max_points,
                precision       = precision,
                sub_steps       = sub_steps,
                seed            = rng.integers(1 << 63),
                )

            # ----- Merge the two dictionnaries

            all_lines = {'types':   list(lines['types']) + list(back_lines['types']),
                         'cyclic':  list(lines['cyclic']) + list(back_lines['cyclic']),
                         'splines': lines['splines'] + back_lines['splines'],
                        }
            lines = all_lines

        return cls(**lines)

    # ----------------------------------------------------------------------------------------------------
    # Lines of magnetic field
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def magnetic_field_lines_OLD(cls, magnet_locations, moments=(1, 0, 0), field_color=True,
                           count=100, start_points=None, min_width=.3, plane=None, plane_center=(0, 0, 0),
                           frag_length=None, frag_scale=None, max_points=1000,
                           precision=.1, sub_steps=10, seed=None):

        """ Create lines of field for a vector field generated by bipoles, typically an magnetic field.

        Arguments:
        ----------
            - magnet_locations (array (n, 3) of vectors) : where the bipoles are located
            - moments (vector or array (n) of vectors = (1, 0, 0)) : the moments of the magnets
            - field_color (bool = True) : manage the field_color attribute
            - count (int = 100) : number of lines to create. Overriden by len(start_points) if not None
            - start_points (array (s, 3) of vectors = None) : the starting points to compute the lines from
            - min_width (float = .3) : min width for volume generation when magnet locations are in a plane
            - plane (vector = None) = restrict start points to a plane defined by its perpendicular
            - plane_center (vector = (0, 0, 0)) : center of the plane
            - frag_length (float=None) : length of fragments, None for full lines
            - frag_scale (float=None) : length distribution scale
            - precision (float = .1) : step precision
            - sub_steps (int=10) : number of steps between two sucessive points of the lines
        """

        # ----------------------------------------------------------------------------------------------------
        # Field function

        magnets = AttrVectors(magnet_locations, moment=moments)
        field_func = lambda points: field.magnetic_field(points,
                            locations=magnets.co, moments=magnets.moment)

        # ----------------------------------------------------------------------------------------------------
        # Starting points

        rng = np.random.default_rng(seed=seed)
        n_magnets = len(magnets)

        backwards = rng.choice([True, False], count)
        if start_points is None:
            if frag_length is None:
                if plane is None:
                    start_points, _ = distribs.sphere_dist(radius=precision*10, count=count, seed=rng.integers(1<<63))
                else:
                    start_points, _ = distribs.circle_dist(radius=precision*10, count=count, seed=rng.integers(1<<63))
                    start_points = rotate_xy_into_plane(start_points, plane=plane, origin=plane_center)

                inds = rng.integers(0, n_magnets, count)
                mag_locs = magnets.co[inds]
                backwards[:] = np.einsum('...i, ...i', start_points, magnets.moment[inds]) < 0
                start_points += mag_locs

            else:
                center = np.average(magnets.co, axis=0)
                bbox0, bbox1 = np.min(magnets.co, axis=0), np.max(magnets.co, axis=0)
                radius = 1.3*max(1., max(np.linalg.norm(bbox1 - center), np.linalg.norm(bbox0 - center)))

                if plane is None:
                    dims = np.maximum(bbox1 - bbox0, (min_width, min_width, min_width))
                    center = (bbox0 + bbox1)/2
                    bbox0, bbox1 = center - 1.3*dims, center + 1.3*dims

                    start_points, _ = distribs.cube_dist(corner0=bbox0, corner1=bbox1, count=count, seed=rng.integers(1<<63))
                else:
                    start_points, _ = distribs.disk_dist(radius=radius, count=count, seed=rng.integers(1<<63))
                    start_points = rotate_xy_into_plane(start_points, plane=plane, origin=plane_center)

        else:
            if len(np.shape(start_points)) == 1:
                count = 1
            else:
                count = len(start_points)

        # ----------------------------------------------------------------------------------------------------
        # Field lines

        lines = field.field_lines(field_func,
            start_points    = start_points,
            backwards       = backwards,
            max_length      = frag_length,
            length_scale    = frag_scale,
            end_points      = magnet_locations,
            max_points      = max_points,
            precision       = precision,
            sub_steps       = sub_steps,
            seed            = rng.integers(1 << 63),
            )

        return cls(**lines)
    
    
