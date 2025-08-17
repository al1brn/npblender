#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 10:27:01 2023

@author: alain
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 07:45:22 2023

@author: alain
"""

import inspect

import numpy as np

from .sim_action import Action
from .sim_simul import Simulation
from .instances import Instances

class Kinematics(Simulation):
        
    def __init__(self, geometry=None, object=None):
        if geometry is None:
            self.own_geo = True
            geometry = Instances()
        else:
            self.own_geo = False
        
        if object is None:
            object = "Kinematics"

        super().__init__(geometry, object)

        # Create attributes dedicated to kinematics
        self.points.init_kinematics()

    # ====================================================================================================
    # Usefull actions
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Gravity
    # ----------------------------------------------------------------------------------------------------

    def gravity(self, g=(0, 0, -9.81)):
        self.points.accel += g

    # ----------------------------------------------------------------------------------------------------
    # Newton's Law
    # ----------------------------------------------------------------------------------------------------

    def newton_law(self, G=1, power=2, min_distance=.001):
        """ Newton's law between points
        
        The force between two points is given by:
        
        F = G.m1.m2 / dist**p
        """

        pts = self.points.ravel()
        if len(pts) < 2:
            return
        
        # Vectors between couples of points
        v = pts.position - pts.position[:, None]

        # Distance
        d = np.linalg.norm(v, axis=-1)
        close = d < min_distance

        d[close] = 1
        F = (G*v)*d[..., None]**(-power - 1)
        F *= pts.mass[:, None, None]
        F *= pts.mass[None, :, None]
        F[close] = 0

        pts.force += np.sum(F, axis=0)

    # ----------------------------------------------------------------------------------------------------
    # Central force
    # ----------------------------------------------------------------------------------------------------

    def central_force(self, location=(0, 0, 0), factor=1., power=-2, min_distance=.001):

        pts = self.points.ravel()

        v = pts.position - location
        d = np.linalg.norm(v, axis=-1)
        close = d < min_distance

        d[close] = 1
        F = (factor*v)*(d[:, None]**(power - 1))
        F[close] = 0

        pts.force += F

    # ----------------------------------------------------------------------------------------------------
    # Centrifugal and Coriolis acceleration
    # ----------------------------------------------------------------------------------------------------

    def centrifugal(self, location=(0, 0, 0), omega=1, axis=(0, 0, 1), coriolis_factor=1.):

        pts = self.points.ravel()

        axis = np.asarray(axis)

        # Position relatively to the axis of rotation
        v = pts.position - location

        # Decompose height / vector to axis then distance
        z  = np.einsum('...i, ...i', v, axis)
        xy = v - axis*z[..., None]

        d = np.linalg.norm(xy, axis=-1)

        # Centrifugal
        acc = xy*(omega*omega)

        # Coriolis
        # Factor articially controls the intensity of the Coriolis force

        acc += np.cross((-2*omega)*axis, pts.speed)*coriolis_factor

        pts.accel += acc

    # ----------------------------------------------------------------------------------------------------
    # Viscosity :  slow down speed according a power law
    # ----------------------------------------------------------------------------------------------------

    def viscosity(self, factor=1., power=2, max_force=None, fluid_speed=None):

        from collections.abc import Callable

        pts = self.points.ravel()
        speed = pts.speed

        # Speed relative to fluid speed
        if fluid_speed is None:
            pass

        elif callable(fluid_speed):
            spped -= fluid_speed(pts.position)

        else:
            speed -= fluid_speed

        # Direction and norm
        nrm = np.linalg.norm(speed, axis=-1)
        nrm[nrm < .001] = 1
        u = speed/nrm[:, None]

        # Raw force
        if 'viscosity' in pts.actual_names:
            factor = factor*pts.viscosity
        
        F = factor*(nrm**power)
        
        # Make sure the force doesn't invese de speed
        max_F = pts.mass*nrm/self.delta_time
        F = np.minimum(F, max_F)

        pts.force -= u*F[:, None]

    # ====================================================================================================
    # Simulation
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------------------------------------

    def reset(self):
        if self.own_geometry:
            self.geometry.clear()

        super().reset()

    # ----------------------------------------------------------------------------------------------------
    # Compute
    # ----------------------------------------------------------------------------------------------------

    def compute(self):

        pts = self.points.ravel()

        # Reset the acceleration and force
        pts.accel = 0
        pts.force = 0

        # Loop on the actions
        super().compute()

        # Take forces into account
        F = pts.force
        mask = pts.mass != 0
        F[mask] /= pts.mass[mask, None]
        F[~mask] = 0

        # Full acceleration
        accel = pts.accel + F

        # Move the particles
        new_speed = pts.speed + accel*self.delta_time
        pts.position += (pts.speed + new_speed)*(self.delta_time/2)
        pts.speed = new_speed

        # Update the age
        if 'age' in pts.actual_names:
            self.points.age += self.delta_time



# =============================================================================================================================
# Kinematics functions

# ----------------------------------------------------------------------------------------------------
# Wind with viscosity and noise

def wind(self, force=5, axis='X', viscosity=1., power=2, noise=None, noise_scale=1., noise_detail=1., time_scale=1.):

    wind_speed = force*axis

    if noise is not None:

        noise_scale  = self.keyed(noise_scale)
        noise_detail = self.keyed(noise_detail)
        time_scale   = self.keyed(time_scale)

        bnoise = BNoise(scale=noise_scale/10, detail=noise_detail, dimension=4, noise_basis='PERLIN_ORIGINAL', seed=self.seed)
        wind_speed += noise*bnoise.vector(self.locations, w=self.time*time_scale)

    return self.viscosity(wind_speed, viscosity, power)

# ----------------------------------------------------------------------------------------------------
# Thrust

def thrust(self, thrust=1., axis=None, noise=None, noise_scale=1., noise_detail=1., time_scale=1.):

    if len(self) == 0:
        return []

    thrust = self.keyed(thrust, t=self.get_attribute('ages', None))
    axis   = self.keyed(axis)
    noise  = self.keyed(noise)

    #print(f"{self.time:.1f}: {thrust[:3]}")

    # ----- Axis is not defined, we use speed direction

    if axis is None:
        axis = Vectors.Axis(np.array(self.speeds)).a

    # ----- Axis is defined, we take the rotation of the point into account

    else:
        axis = self.matrices @ axis

    # ----- We can compute the force

    if np.shape(thrust) == ():
        force = thrust*axis
    else:
        force = np.array(thrust)[None, :]*axis

    # ----- Less borrowing

    if noise is not None:

        noise_scale  = self.keyed(noise_scale)
        noise_detail = self.keyed(noise_detail)
        time_scale   = self.keyed(time_scale)

        bnoise = BNoise(scale=noise_scale/10, detail=noise_detail, dimension=4, noise_basis='PERLIN_ORIGINAL', seed=self.seed)
        force += noise*bnoise.vector(self.locations, w=self.time*time_scale)

    return force

def add_thrust(self, *args, **kwargs):
    return self.add_force(Points.thrust, *args, **kwargs)


# ----------------------------------------------------------------------------------------------------
# Vortex

def add_vortex(self,
        location     = (0, 0, 0),
        axis         = 'Z',
        omega        = 1.,
        vert_factor  = 1,
        vert_power   = .5,
        hrz_factor   = 1,
        hrz_power    = -2,
        angle        = np.pi/6,
        viscosity    = 1.,
        power        = 2,
        noise        = 0,
        noise_scale  = 1.,
        noise_detail = 1.,
        time_scale   = 1.,
        ):

    bnoise = BNoise(scale=noise_scale, detail=noise_detail, dimension=4, noise_basis='PERLIN_ORIGINAL', seed=self.seed)

    axis = axis_vector(axis)
    def vortex(points, dt):

        # Locations relative to the bottom point

        rel_locs = points.locations - location

        # Location on the axis and vector perpendicular

        loc_z  = np.maximum(.001, np.abs(np.einsum('...i, ...i', rel_locs, axis)))
        loc_xy = rel_locs - axis*loc_z[:, None]

        rs = np.linalg.norm(loc_xy, axis=-1)

        # ----- Omega decreases with altitude

        omg = omega*vert_factor*np.power(loc_z, vert_power)

        # ----- Omega decreases with distance to the cone

        r_max = loc_z*np.tan(angle)

        omg *=hrz_factor*np.power(np.maximum(0.001, np.abs(rs - r_max)), hrz_power)

        # ----- We can compute  the speed

        # Unary vectors

        vortex_speed = np.cross(axis, loc_xy/(rs[:, None]))

        # We add some noise

        if noise != 0:
            vortex_speed += noise*bnoise.vector(points.locations, w=points.t*time_scale)

        # We scale with the rotation speed

        vortex_speed *= np.minimum(10, omg)[:, None]

        return Points.vicosity_acc(points, fluid_speeds=vortex_speed, factor=viscosity, power=power, dt=dt)

    self.add_force(vortex)

# =============================================================================================================================
# Kinematics simulation

class Kinematics_OLD(Simulation):

    def __init__(self, geometry=None, count_down=False, max_force=None, max_acc=None, min_speed=None, max_speed=None, seed=0):
        """ > Kinematics simulation

        Animate points.

        Points are initialized with "auto" attributes:
        - speed (vector) : speed
        - acceleration (vector) : acceleration
        - mass (float) : mass
        - age (float) : age
        - locked (bool) : point doesn't move
        - last_pos (vector) : last position
        - viscosity (float) : viscosity
        - scales (vector) : scale x, y, z (not to be confused with scale attribute)
        - moment (float) : moment
        - eulers (vector) : rotation x, y, z (not to be confused with euler attribute)
        - omega (vector) : rotation speed

        Properties
        ----------
        - geometry (Geometry) : geometry with points
        - points (points) : points of geometry
        - count_down (bool = False) : decreases age rather than increase
        - max_force (float = None) : maximum force
        - max_acc (float = None) : maxium acceleration
        - min_speed (float = None) : minimum speed
        - max_speed (float = None) : maximum speed
        - seed (int = 0) : random seed

        Arguments
        ---------
        - geometry (Geometry) : geometry with points
        - count_down (bool = False) : decreases age rather than increase
        - max_force (float = None) : maximum force
        - max_acc (float = None) : maxium acceleration
        - min_speed (float = None) : minimum speed
        - max_speed (float = None) : maximum speed
        - seed (int = 0) : random seed
        """

        super().__init__()

        if geometry is None:
            self.geometry = Cloud()
            self.own_geometry = True
        else:
            self.geometry     = geometry
            self.own_geometry = False
        self.points = self.geometry.points

        self.count_down  = count_down
        self.max_force   = max_force
        self.max_acc     = max_acc
        self.min_speed   = min_speed
        self.max_speed   = max_speed
        self._ignore_acc = None




    # ====================================================================================================
    # Actions and event

    def set_count(self, count):
        diff = count - len(self.points)
        if diff < 0:
            self.points.delete(np.arange(len(self)+diff, len(self)))
        elif diff > 0:
            self.add_points(np.zeros((diff, 3), float))

    def kill_points(self, selection):
        self.points.delete(selection)

    # ====================================================================================================
    # Forces / Accelerations

    def add_force(self, func, *args, start=0, duration=None, **kwargs):
        """ Add a force

        Acceleration if computed by dividing the force by the mass. To add an
       acceleration such as gravity use <#add_acceleration>.

        Arguments
        ---------
        - func (function) : function of template f(kinematics, *args, *kwargs)
        - top (float = 0) : start time
        - duration (float = None) : force duration

        Returns
        -------
        - Force : the force added to the kinematics
        """
        force = Force(self.str_to_func(func), *args, start=start, duration=duration, is_force=True, **kwargs)
        self.append(force)

        return force

    def add_acceleration(self, func, *args, start=0, duration=None, **kwargs):
        """ Add an acceleration

        The acceleration is directly applied to the object. To take the mass into account
        see <#add_force>.

        Arguments
        ---------
        - func (function) : function of template f(kinematics, *args, *kwargs)
        - top (float = 0) : start time
        - duration (float = None) : force duration

        Returns
        -------
        - Force : the force added to the kinematics
        """
        force = Force(self.str_to_func(func), *args, start=start, duration=duration, is_force=False, **kwargs)
        self.append(force)

        return force

    # ====================================================================================================
    # Actions to create / kill particles

    # ----------------------------------------------------------------------------------------------------
    # Create

    @staticmethod
    def create_points(kinematics, position, **attributes):
        """ Create points action function

        Create points at given position with given attributes

        ``` python
        kinematics.add_event("create_points", (0, 0, 0), speed=(1, 0, 0))
        ```

        Arguments
        ---------
        - kinematics (Kinematics) : Kinematics instance
        - position (array (n, 3)) : positions of the points to create
        - attributes : attributes of the created points
        """
        n = 1 if np.shape(position) == (3,) else len(position)
        return kinematics.points.add(n, position=position, **attributes)

    def emit(self, dist_func, *args, count=10, density=None, attributes={}, **kwargs):
        """ Create points action function

        Create points using a distribution function

        Arguments
        ---------
        - kinematics (Kinematics) : Kinematics instance
        - dist_func (function) : distribution function
        - args : args to pass to the distribution function
        - count (int=10) : number of particles to create
        - density (float : None) : density
        - attributes (dict = {}) : created points atttibutes
        - kwargs : kwargs to pass to the distribution function
        """

        if density is not None:
            density *= self.delta_time
        count = int(count*self.delta_time)

        position, speed = dist_func(*args, count=count, density=density, seed=self.rng, **kwargs)

        return self.points.add(len(position), position=position, speed=speed, **attributes)

    # ----------------------------------------------------------------------------------------------------
    # Kill

    @staticmethod
    def kill(kinematics, selection):
        kinematics.kill_points(selection)

    @staticmethod
    def kill_old(kinematics, age, scale=None):
        """ Kill particles action function

        Kill particles older than the given age

        Arguments
        ---------
        - kinematics (Kinematics) : Kinematics instance
        - age (float) : limit age
        - scale (float = None) : age is randomly shaken around the given value is scale is not None
        """
        ages = age if scale is None else kinematics.rng.normal(age, scale, len(kinematics.points))
        if kinematics.count_down:
            selection = kinematics.points.age < ages
        else:
            selection = kinematics.points.age > ages

        if np.sum(selection) > 0:
            kinematics.points.delete(selection)

    @staticmethod
    def kill_z0(kinematics, z=0, scale=None):
        """ Kill particles action function

        Kill particles below the given z value

        Arguments
        ---------
        - kinematics (Kinematics) : Kinematics instance
        - z (float) : limit z
        - scale (float = None) : z is randomly shaken around the given value is scale is not None
        """

        zs = z if scale is None else kinematics.rng.normal(z, scale, len(kinematics.points))
        selection = kinematics.points.position[:, 2] <= zs

        if np.sum(selection) > 0:
            kinematics.points.delete(selection)


    # =============================================================================================================================
    # Misceleanous

    @staticmethod
    def orient_to_speeds(kinematics, track_axis='Z', up_axis='Y'):
        kinematics.eulers = tracker(kinematics.speed, track_axis=track_axis, up_axis=up_axis).eulers

    # =============================================================================================================================
    # Bounce on a infinite plane or disk

    @staticmethod
    def bounce_on_plane_OLD(kinematics, origin=(0, 0, 0), direction='Z', radius=None, epsilon=.1, energy_factor=.95, noise=None):

        z_axis = axis_vector(direction)
        rel_loc = kinematics.points.position - origin

        # ----- Position of the particles along direction

        z = np.einsum('...i, ...i', z_axis, rel_loc)

        # ----- Incident speed

        in_speed = np.einsum('...i, ...i', kinematics.points.speed, z_axis)

        # ----- Particles which will pass through the plane

        sel = np.logical_and(in_speed < 0, np.logical_and(z > -epsilon, z + in_speed*kinematics.delta_time < epsilon))
        if not np.sum(sel):
            return

        # ----- Particles location in the plane

        xy = rel_loc - z_axis*z[:, None]

        # Possible constraint on a radus around  the origin

        if radius is not None:
            u, r = normalize(xy)
            np.logical_and(sel, r < radius, out=sel)
            if not np.sum(sel):
                return

        # ----- new speed

        new_speed = kinematics.points.speed[sel] - z_axis*(2*in_speed)[sel, None]

        if noise is not None:
            new_speed += kinematics.rng.uniform(-noise, noise, (len(new_speed), 3))

        E0 = np.sum(kinematics.points.speed[sel]**2, axis=-1)
        E1 = np.sum(new_speed**2, axis=-1)
        new_speed *= (energy_factor*E1/E0)[:, None]

        kinematics.points.speed[sel] = new_speed

    @staticmethod
    def bounce_on_plane(kinematics, origin=(0, 0, 0), direction='Z', radius=0, epsilon=.1, energy_factor=.95, noise=None):

        z_axis = axis_vector(direction)

        positions = kinematics.points.position
        speeds    = kinematics.points.speed

        # ----- Position of the particles along direction

        rel_loc = positions - origin
        z = np.einsum('...i, ...i', z_axis, rel_loc)
        below = z < radius

        # ----- Incident speed

        in_speed = np.einsum('...i, ...i', speeds, z_axis)

        # ----- Bounce when particles are below the plane and speed is towards negative z

        bounce = np.logical_and(z < radius, in_speed < 0)

        if np.sum(bounce) > 0:
            kinematics.ignore_acc[bounce] = True
            kinematics.points.position[bounce] -= z_axis*(z[bounce] + radius)
            kinematics.points.speed[bounce] = reflect(speeds[bounce], z_axis, factor=energy_factor)

    # =============================================================================================================================
    # Bounce within a box

    @staticmethod
    def bounce_in_box(kinematics, x0=-1, x1=1, y0=-1, y1=1, z0=-1, z1=1, epsilon=.1, energy_factor=1, noise=None):
        Kinematics.bounce_on_plane(kinematics, origin=(x0, 0, 0), direction=( 1, 0, 0), epsilon=epsilon, energy_factor=energy_factor, noise=noise)
        Kinematics.bounce_on_plane(kinematics, origin=(x1, 0, 0), direction=(-1, 0, 0), epsilon=epsilon, energy_factor=energy_factor, noise=noise)
        Kinematics.bounce_on_plane(kinematics, origin=(0, y0, 0), direction=( 0, 1, 0), epsilon=epsilon, energy_factor=energy_factor, noise=noise)
        Kinematics.bounce_on_plane(kinematics, origin=(0, y1, 0), direction=( 0,-1, 0), epsilon=epsilon, energy_factor=energy_factor, noise=noise)
        Kinematics.bounce_on_plane(kinematics, origin=(0, 0, z0), direction=( 0, 0, 1), epsilon=epsilon, energy_factor=energy_factor, noise=noise)
        Kinematics.bounce_on_plane(kinematics, origin=(0, 0, z1), direction=( 0, 0,-1), epsilon=epsilon, energy_factor=energy_factor, noise=noise)

    @staticmethod
    def bounce_on_surface(kinematics, mesh_dict, radius=.1, precision=None, ignore_axis=None, energy_factor=.95, flip=False, noise=None):
        """ > Bounce on the faces of a mesh

        The mesh is passed in a dict keyd by 'mesh': `mesh_dict = {'mesh': my_mesh}`.
        The dict is enriched at the first call.

        Bounces are computed by distributing random points on the surface.
        If `ignore_axis` argument is not None, point normals oriented along this axis are ignored.

        Particles close to a point on the surface and below the surface, are move to the surface and
        their speed is reflected. The flag <#ignore_acc> is set to True.

        Arguments
        ---------
        - kinematics (Kinematics) : kinematics object
        - mesh_dict (dict) : dict containing an entry 'mesh': Mesh
        - radius (float=.1) : particles radius
        - precision (float = None) : precision (the lower the more precise)
        - ignore_axis (vector = None) : ignore faces with normal along this vector
        - energy_factor (float = .95) : kinetical energey factor
        - flip (bool = False) : flip faces
        - noise (bool = None) : add randomness to the reflected speed
        """

        if precision is None:
            precision = radius*2

        # ----------------------------------------------------------------------------------------------------
        # Bounce is computed on random points distributed on the surface

        kdtree = mesh_dict.get("kdtree")
        if kdtree is None:

            mesh = mesh_dict["mesh"]
            cloud = mesh.distribute_poisson(distance_min=precision, density_max=100., density_factor=100., seed=mesh_dict.get('seed', 0))
            normal = np.array(cloud.points.Normal)
            if flip:
                normal *= -1

            if ignore_axis is None:
                position = np.array(cloud.points.position)

            else:
                sel = np.einsum('...i, ...i', normal, ignore_axis) <= 0
                position = np.array(cloud.points.position[sel])
                normal   = np.array(normal[sel])


            kdtree = KDTree(position)

            mesh_dict["position"] = position
            mesh_dict["normal"]   = normal
            mesh_dict['kdtree']   = kdtree

        # ----------------------------------------------------------------------------------------------------
        # Query the KDTree
        # - dists : distances for each kinematics points to one point on the surface
        # - indices : indices of close points on the surface

        dists, indices = kdtree.query(kinematics.points.position, k=1, eps=0, p=2)

        # ----------------------------------------------------------------------------------------------------
        # Bounce computation

        position = mesh_dict["position"][indices]
        normal   = mesh_dict["normal"][indices]

        rel_pos = kinematics.points.position - position
        z = np.einsum('...i, ...i', rel_pos, normal)

        bounce = np.logical_and(dists < 2*precision, z < radius)

        # No bounce

        if np.sum(bounce) == 0:
            return

        # Let's reflect the bouncing points

        normal = np.array(normal[bounce])

        new_speed = reflect(kinematics.points.speed[bounce], normal, factor=energy_factor)
        if noise is not None:
            new_speed = distribs.vector_noise(new_speed, scale=noise, angle_scale=noise, seed=kinematics.seed)

        kinematics.points.position[bounce] -= normal*(z[bounce, None] - radius)
        kinematics.points.speed[bounce] = new_speed

    # =============================================================================================================================
    # Collisions between particles
    #
    # Spheres with same radius and particles share the same mass

    @staticmethod
    def collisions(kinematics, r=.1, energy_factor=1):
        """ Collisions between particles

        Particles have the same radius r

        1) Candidates

        A particle can collide with particles closer than v.dt + r

        2) Collision occurs

        Particle is considered at rest at the origin of the frame by subtracting its
        speed to the candidate speed

        Collision occurs if:
        - other is closer than r
        - or distance decreases with time

        Collision time is such as distance between the two particles is 2r

        3) Speeds

        The particle speed is decomposed in two parts:
        - the part along the line between the two centers
        - the part perpendicular to this line

        The part along is fully transferred to the particle at rest
        The part perp is unchanged

        4) Energy

        Back to the world frame, we have the speed of each particles.
        We adjust to make sure kinetical energy is preserved : a factor is applied
        to be sure that no kinetical energy is created
        """

        print("Collision")

        points = kinematics.points

        kdtree = KDTree(points.position)

        r2  = 2*r
        r22 = r2*r2

        epsilon = r/20
        epsilon2 = epsilon*epsilon

        pairs = kdtree.query_pairs(2*r, eps=0., output_type='set')

        for pair in pairs:

            i, j = pair

            P = points[i].position
            p_speed = points[i].speed

            Q = points[j].position
            q_speed = points[j].speed

            # ----- Frame centered on P at rest

            Q = points.position[j]
            L = Q - P

            v = q_speed - p_speed
            v_norm = np.sqrt(sum(v*v))

            # ----- Distance between the centers

            d2 = sum(L*L)
            d  = np.sqrt(d2)
            if d < epsilon:
                d = r/10
                u = np.array((1., 0., 0.))
            else:
                u = L/d

            # ----------------------------------------------------------------------------------------------------
            # Low speed : crossing r takes time

            small_speed = r/kinematics.delta_time/100

            if v_norm < small_speed:

                i_speed = -u*small_speed*1.1
                j_speed = -i_speed

                #accs[i] += i_speed/kinematics.delta_time
                #accs[j] += j_speed/kinematics.delta_time

                kinematics.points.position[i] = P - u*(r - d/2)
                kinematics.points.position[j] = Q + u*(r - d/2)

                kinematics.points.speed[i] = i_speed
                kinematics.points.speed[j] = j_speed

                kinematics.ignore_acc[i] = True
                kinematics.ignore_acc[j] = True

                #kinematics.constraints.append((i, P - u*(r - d/2), i_speed, u))
                #kinematics.constraints.append((j, Q + u*(r - d/2), j_speed, -u))

                continue

            # ----------------------------------------------------------------------------------------------------
            # Normal speed

            v_along = np.dot(u, v)
            if v_along > 0:
                continue

            v_along = v_along*u
            v_perp  = v - v_along

            # ----- Back to world frame

            i_speed = p_speed + v_along
            j_speed = p_speed + v_perp

            i_speed2 = sum(i_speed*i_speed)
            j_speed2 = sum(j_speed*j_speed)

            # ----- Kinetical Energy preservation

            E0 = sum(p_speed*p_speed) + sum(q_speed*q_speed)
            E1 = i_speed2 + j_speed2

            f = E0/E1*energy_factor

            i_speed *= f
            j_speed *= f

            # ----- Constraints

            u = i_speed/np.sqrt(i_speed2)
            v = j_speed/np.sqrt(j_speed2)

            kinematics.points.position[i] = P + u*(r - d/2)
            kinematics.points.position[j] = Q + v*(r - d/2)

            kinematics.points.speed[i] = i_speed
            kinematics.points.speed[j] = j_speed

            kinematics.ignore_acc[i] = True
            kinematics.ignore_acc[j] = True

            #kinematics.constraints.append((i, P + u*(r - d/2), i_speed, u))
            #kinematics.constraints.append((j, Q + v*(r - d/2), j_speed, v))



    # ====================================================================================================
    # Emitters

    def add_emitter(self, dist_func, *args, top=0, duration=None, count=10, density=None, **kwargs):
        self.add_event(top, Simulation.emit, dist_func, *args, duration=duration, count=count, density=density, **kwargs)


    def emit_track_particles(self, tracked_points, distribution='SEGMENT',
            radius=.1, count=0, density=1.,
            distance=None, speed=0., speed_scale=None, speed_angle_scale=None,
            stop_when_locked=False, attributes_transfer=None, **attributes):

        # Keyable arguments

        radius              = self.keyed(radius)
        count               = self.keyed(count)
        density             = self.keyed(density)
        distance            = self.keyed(distance)
        speed               = self.keyed(speed)
        speed_scale         = self.keyed(speed_scale)
        speed_angle_scale   = self.keyed(speed_angle_scale)

        # Short cut !

        position = tracked_points.position

        # ----- If distribution on a trajectory segment, we need the last locations

        if distribution == 'SEGMENT':
            last_pos = np.array(tracked_points.last_pos)
            tracked_points.last_pos = position
        else:
            last_pos = position

        # ----- The speeds of the points give the ejection direction

        tracked_speed_dir, tracked_speed_norm = distribs.normalize_vectors(tracked_points.speed, keep_zeros=True)
        tracked_speed_dir *= -1

        # ----- Particles can be emitted at a given distance behind the tracked points

        if distance is not None:
            position  = position + tracked_speed_dir*distance
            last_pos  = last_pos + tracked_speed_dir*distance

        # ----- Density and count

        if density is not None:
            density = density * self.delta_time

        count = self.rng.poisson(self.delta_time*count)

        # ----- Loop on the tracked points

        for i_tracked, (p_pos, p_last) in enumerate(zip(position, last_pos)):

            if stop_when_locked and tracked_points.points.locked[i_tracked]:
                continue

            # ----- Segment distribution : we emit on a cylinder between last and current locations

            if distribution == 'SEGMENT':

                locs, speeds = distribs.cylinder_dist(p_last, p_pos, radius=radius, count=count, density=density,
                        scale=radius/3, seed=self.rng, speed=0, speed_dir='NORMAL')

            # ----- Sphere distribution : we emit on a sphere

            elif distribution == 'SPHERE':
                locs, speeds = dstribs.sphere_dist(radius, center=p_pos, scale=radius/3, count=count, density=density,
                                        speed=speed, speed_scale=speed_scale, speed_angle_scale=speed_angle_scale,
                                        seed=self.rng)

            else:
                raise Exception(f"emit_track_particles error: invalid distribution code: '{distribution}")

            # Add the points

            inds = self.points.add(len(locs), position=locs, speed=speed, **attributes)

            # Transfer attributes

            if attributes_transfer is not None:
                if isinstance(attributes_transfer, str):
                    attributes_transfer = [attributes_transfer]
                for name in attributes_transfer:
                    self.points.attributes[name][inds] = tracked_points.attributes[name][i_tracked]


    # ====================================================================================================
    # Baking

    # ----------------------------------------------------------------------------------------------------
    # Positions

    def sim_curves(self, duration=10., count=100, sub_steps=1):

        self.reset()

        curves = np.empty(self.shape + (count, 3))

        dt = duration/(count-1)
        for index in range(count):
            curves[..., index, :] = self.locations
            self.step(dt, sub_steps=sub_steps)

        return curves

    # ----------------------------------------------------------------------------------------------------
    # Bake particles
    # Location is saved only for live points

    def bake_particles(self, duration=10., count=100, sub_steps=1):

        self.reset()

        points_list = []

        dt = duration/(count-1)
        for index in range(count):
            points_list.append(self.clone())
            self.step(dt, sub_steps=sub_steps)

        return points_list

    # ----------------------------------------------------------------------------------------------------
    # Bake with ID

    def bake(self, duration=10., count=100, sub_steps=1):

        self.reset()
        self.create_ID()

        chaos = self.bake_particles(duration=duration, count=count, sub_steps=sub_steps)

        # ----- Let's rebuild with IDs
        # The ID of attributes gives the number of create IDs

        total = chaos[-1].attributes.ID

        points_list = []
        for points in chaos:

            all_points = Points(shape=total)
            points_list.append(all_points)

            all_points.attributes.copy(self.attributes)
            all_points.new_bool('alive', False)
            points.alive = False

            points.reshape(points.size)
            if len(points) == 0:
                continue

            points.new_bool('alive', True)
            points.alive = True

            IDs = points.ID
            all_points.a[IDs] = points.a
            all_points.attributes.set_selection(IDs, points.attributes)


        return points_list

    # ----------------------------------------------------------------------------------------------------
    # Key frames locations

    def set_key_frames(self, objects, frame, properties=['locations', 'scales', 'eulers']):

        from npblender.core import blender

        if len(objects) != self.size:
            raise Exception(f"bake_locations error: {len(objects)} are provided for {self.size} points!")

        locs   = np.reshape(self.locations, (len(objects), 3))
        scales = np.reshape(self.scales,    (len(objects), 3))
        eulers = np.reshape(self.eulers.a,  (len(objects), 3))

        for obj, loc, scale, euler in zip(objects, locs, scales, eulers):
            if 'locations' in properties:
                blender.kf_set(obj, "location.x", frame, loc[0])
                blender.kf_set(obj, "location.y", frame, loc[1])
                blender.kf_set(obj, "location.z", frame, loc[2])

            if 'scales' in properties:
                blender.kf_set(obj, "scale.x", frame, scale[0])
                blender.kf_set(obj, "scale.y", frame, scale[1])
                blender.kf_set(obj, "scale.z", frame, scale[2])

            if 'eulers' in properties:
                blender.kf_set(obj, "rotation_euler.x", frame, euler[0])
                blender.kf_set(obj, "rotation_euler.y", frame, euler[1])
                blender.kf_set(obj, "rotation_euler.z", frame, euler[2])

    # ----------------------------------------------------------------------------------------------------
    # Compute the trajectories

    def get_trajectories(self, duration=10., count=100, sub_steps=1, dt=None):

        self.reset()
        self.create_ID()

        locs = []
        tops = []

        if dt is None:
            dt = duration/(count-1)
        else:
            duration = dt*(count-1)

        for index in range(count):

            # ----- Capture the locations if any

            IDs = self.ID
            if len(IDs) > 0:
                nb_ID = np.max(IDs) + 1

                n =  nb_ID - len(locs)
                if n > 0:
                    tops.extend([(index, index*dt)]*n)
                    locs.extend([[] for _ in range(n)])

                for ID, loc in zip(IDs, self.locations):
                    locs[ID].append(list(loc))

            self.step(dt, sub_steps=sub_steps)

        return {
            'frames':     np.array([top[0] for top in tops]),
            'times':      np.array([top[1] for top in tops]),
            'locations': [np.array(loc) for loc in locs],
            }

    # =============================================================================================================================
    # To mesh object

    def to_mesh_object(self, spec, model=None, update=False, attributes=[], shade_smooth=True):

        import bpy
        #from npblender.core.meshbuilder import MeshBuilder

        #if model is not None and isinstance(model, (str, bpy.types.Object)):
        #    model = MeshBuilder.FromObject(model)

        # ----------------------------------------------------------------------------------------------------
        # Update an existing mesh with the proper number of vertices

        if update:
            mesh = blender.get_object(spec).data

            # ----- Update the vertices

            # No model : we have a cloud of vertices
            if model is None:
                n = 1
                mesh.vertices.foreach_set('co', np.array(self.locations.flatten()))

            # Model size duplicated by self.shape
            else:
                n = model.verts_len
                if n == 1:
                    verts = self @ model.verts
                else:
                    verts = self[..., None] @ model.verts

                mesh.vertices.foreach_set('co', np.reshape(verts, verts.size))

            # ----- Update the attributes

            for name in ['scales', 'eulers']:
                if name in attributes:
                    blender.set_attribute(mesh, name, np.array(self.get_attribute(name)))

            self.attributes.to_mesh(mesh, attributes=attributes, update=True)

            mesh.update()

        # ----------------------------------------------------------------------------------------------------
        # Let's create the object

        else:

            # ----- Create the mesh

            # A cloud of vertices
            if model is None:
                obj = blender.create_mesh_object(spec)
                mesh = obj.data

                mesh.clear_geometry()
                if not self.size:
                    return

                mesh.vertices.add(self.size)
                mesh.vertices.foreach_set('co', self.locations.flatten())

                mesh.update()

            # Model duplicated along the shape
            else:
                mb = model*self.size
                mb.transform(self)

                obj = mb.to_object(spec, shade_smooth=shade_smooth)

            # ----- Create the attributes

            mesh = obj.data
            for name in ['scales', 'eulers']:
                blender.create_attribute(mesh, name, 'VECTOR', domain='POINT', value=np.array(self.get_attribute(name)))

            self.attributes.to_mesh(mesh, attributes=attributes, update=False)

    def to_cloud(self, spec, attributes=[]):

        from npblender.core import blender

        obj = blender.create_mesh_object(spec)
        mesh = obj.data
        mesh.clear_geometry()
        mesh.from_pydata(np.array(self.locations), (), ())

        self.attributes.to_mesh(mesh, attributes=attributes, update=False)

        return obj

    # =============================================================================================================================
    # Particles animation

    @staticmethod
    def anime_particles(points_list, model=None, name="Particles", dead_location=None, frame0=1, frame1=None, interpolation='LINEAR'):

        from npblender.core.meshbuilder import MeshBuilder
        from npblender.core.shapekeys import MeshShapeKeys

        frames = len(points_list)
        if frames == 0:
            return

        # ----- Number of particles

        count     = 0
        use_alive = False
        for points in points_list:
            count = max(count, points.size)
            use_alive = points.attributes.exists('alive') and points.attributes.exists('ID')

        # ----- Create the mesh

        if model is None:
            mb = MeshBuilder()
            mb.add_verts((0, 0, 0))
        else:
            mb = MeshBuilder(model)

        block_size = mb.verts_len
        mb = mb*count

        # ----- Frames shape keys

        sks = MeshShapeKeys(mb, frames)

        sk_shape = np.shape(sks[0])
        base_points = np.reshape(sks[0], (count, block_size, 3))

        # ----- Loop with 0 at last key to keep origin

        if not use_alive:
            work = TMatrices(shape=count)

        for ikey in reversed(range(frames)):

            points = points_list[ikey]

            if use_alive:
                points.scales[np.logical_not(points.alive)] = (0, 0, 0)
                if dead_location is not None:
                    points.locations[np.logical_not(points.alive)] = dead_location

                sks[ikey] = np.reshape(points[:, None] @ base_points, sk_shape)

            else:
                work.scales = 0
                if dead_location is not None:
                    work.locations = dead_location

                work[:points.size] = points

                sks[ikey] = np.reshape(work[:, None] @ base_points, sk_shape)

        # ----- To object

        obj = sks.to_object(name)

        # ----- Animation

        if frame0 is not None:
            MeshShapeKeys.set_keyframes(obj, frame0=frame0, frame1=frame1, interpolation=interpolation)

        return obj


    # ====================================================================================================
    # Visualization

    def visu(self, size=10, resolution=100, z=0, speeds=0, scale=1., name="Field"):

        from npblender.core.meshbuilder import MeshBuilder

        x, y = np.meshgrid(np.linspace(-size/2, size/2, resolution), np.linspace(-size/2, size/2, resolution))
        locs = np.stack((x, y, np.ones_like(x)*z), axis=-1)

        self.resize(locs.shape[:-1])
        self.locations = locs

        self.mass = 1
        self.viscosity = 1

        acc = self.get_acceleration(dt=.1)
        nacc = np.linalg.norm(acc, axis=-1)

        # ----- To object

        count = self.size

        mb = MeshBuilder.Circle(segments=3)*count
        mb.scale_x(.2)
        mb.scale(.05)

        mb.scale_y(nacc*scale)
        mb.toward(acc, track_axis='Y', up_axis='Z')
        mb.locate(locs)

        mb.to_object(name)

    # ====================================================================================================
    # Demos

    # ----------------------------------------------------------------------------------------------------
    # Emitters

    @staticmethod
    def demo_emit(count=40, density=10, scale=.1, speed=5, speed_scale=.1, speed_angle_scale=.1, emitters=None):

        engine.init()

        simul = Simulation()
        simul.add_kill_old(3, .3)

        # ----- Emitters locations

        emit_count = 0
        center = np.zeros(3, float)
        delta  = 20

        def next_center(emit_count):
            emit_count += 1
            if emit_count % 4 == 0:
                center[0] = 0
                center[1] += delta
            else:
                center[0] += delta
            return emit_count

        # ----- Line emitter

        if emitters is None or 'line' in emitters:
            simul.add_emitter(distribs.line_dist, top=0, duration=10, count=count, density=density,
                    point0=(0, 0, 0), point1=(10, 10, 10), scale=scale,
                    speed=speed, speed_scale=speed_scale, speed_angle_scale=speed_angle_scale, speed_pie_angle=TAU)
            emit_count = next_center(emit_count)


        if emitters is None or 'line' in emitters:
            simul.add_emitter(distribs.line_dist, top=1, duration=10, count=count, density=density,
                              point0=center+(0, 0, 0), point1=center+(10, 10, 10), scale=scale,
                               speed=speed, speed_scale=speed_scale, speed_angle_scale=speed_angle_scale, speed_pie_angle=.05)
            emit_count = next_center(emit_count)

        if emitters is None or 'circle' in emitters:
            simul.add_emitter(distribs.circle_dist, top=2, duration=10, count=count, density=density,
                              radius=2, center=np.array(center),
                              speed=speed, speed_scale=speed_scale, speed_angle_scale=speed_angle_scale, speed_pie_angle=.05)
            emit_count = next_center(emit_count)

        if emitters is None or 'curve' in emitters:
            curve = Curve.Spiral(start_radius=2, end_radius=4, height=4)
            curve.points.translate(center)

            simul.add_emitter(distribs.curve_dist, curve, top=3, duration=10, count=count, density=density,
                              t0=0., t1=1., scale=scale,
                              speed=speed, speed_scale=speed_scale, speed_angle_scale=speed_angle_scale, speed_pie_angle=.05)
            emit_count = next_center(emit_count)

        if emitters is None or 'rect' in emitters:
            simul.add_emitter(distribs.rect_dist, top=4, duration=10, count=count, density=density,
                              a=6, b=4, center=np.array(center), scale=scale, z_scale=None,
                              speed=speed, speed_scale=speed_scale, speed_angle_scale=speed_angle_scale)
            emit_count = next_center(emit_count)

        if emitters is None or 'disk' in emitters:
            simul.add_emitter(distribs.disk_dist, 4, top=5, duration=10, count=count, density=density,
                              center=np.array(center), scale=scale, z_scale=None,
                              speed=speed, speed_scale=speed_scale, speed_angle_scale=speed_angle_scale)
            emit_count = next_center(emit_count)

        if emitters is None or 'cylinder' in emitters:
            simul.add_emitter(distribs.cylinder_dist, top=6, duration=10, count=count, density=density,
                              point0=center + (0, 0, 0), point1=center+(0, 0, 10), radius=3, scale=scale,
                              speed=speed, speed_scale=speed_scale, speed_angle_scale=speed_angle_scale, )
            emit_count = next_center(emit_count)

        if emitters is None or 'sphere' in emitters:
            simul.add_emitter(distribs.sphere_dist, 4, top=7, duration=10, count=count, density=density,
                              center=np.array(center), scale=scale,
                              speed=speed, speed_scale=speed_scale, speed_angle_scale=speed_angle_scale, )
            emit_count = next_center(emit_count)

        if emitters is None or 'surface' in emitters:
            monkey = Mesh.Monkey()
            monkey.points.scale(3)
            monkey.points.translate(center)

            simul.add_emitter(distribs.surface_dist, monkey.faces.get_surface(), top=8, duration=10, count=count, density=density,
                              scale=scale,
                              speed=speed, speed_scale=speed_scale, speed_angle_scale=speed_angle_scale, )
            emit_count = next_center(emit_count)

        if emitters is None or 'cube' in emitters:
            simul.add_emitter(distribs.cube_dist, top=9, duration=10, count=count, density=density,
                              corner0=center + (-2, -2, -2), corner1=center+(2, 2, 2), scale=scale,
                              speed=speed, speed_scale=speed_scale, speed_angle_scale=speed_angle_scale, )
            emit_count = next_center(emit_count)

        if emitters is None or 'ball' in emitters:
            simul.add_emitter(distribs.ball_dist, 2, top=10, duration=10, count=count, density=density, scale=scale,
                              center=np.array(center),
                              speed=speed, speed_scale=speed_scale, speed_angle_scale=speed_angle_scale, )
            emit_count = next_center(emit_count)

        # ----- Launch

        simul.to_engine(to_object="Demo Emit")

    # ----------------------------------------------------------------------------------------------------
    # Tracks

    def demo_track():

        # ----- Balls

        count = 20
        balls = Points()
        balls.init_simulation()
        balls.add_gravity()
        balls.add_action(lambda points: points.kill(points.z <= 0))

        locs, speeds = distribs.rect_dist(40, 5, count=count,
                            speed=50, speed_scale=3, speed_angle_scale=.2, seed=balls.seed)

        balls.create(tops = balls.rng.uniform(0, 5, count),
            locations         = locs,
            speeds            = speeds,
            )

        balls.set_reset_state()

        # ----- Particles

        particles = Points()
        particles.init_simulation()
        particles.add_kill_old(3, .5)
        particles.add_viscosity()
        particles.add_emit_track(balls, distribution='SEGMENT', radius=.5, count=0, density=100., distance=None,
            speed=10., speed_scale=3, speed_angle_scale=.5, stop_when_locked=False, seed=0)

        #balls.add(locations=balls.rng.uniform(-5, 5, (1000, 3)))

        particles.set_reset_state()

        def setup():
            print("SETUP INIT")
            balls.reset()
            particles.reset()
            print("SETUP DONE")

        def update(eng):
            balls.to_mesh_object("Balls")
            balls.step(1/24)

            particles.to_mesh_object("Particles")
            particles.step(1/24)

        engine.go(update, setup)





# ====================================================================================================
# Demos

def demo_surface():
    # ----- Let's create Suzan

    monkey = MeshBuilder.Monkey()

    # ----- We select the faces facingupwards

    faces = monkey.normals()[:, 2] > 0.1
    surf = monkey.get_surface(faces)

    # ----- Let's locate points on the faces with a certain density

    locs, normals = Points.surf_locs(surf, density=100.)

    # ----- To object

    mb = MeshBuilder.Cube(size=.05)*len(locs)

    # Orient according the normals
    mb.toward(normals)

    # Locate at the points positions
    mb.locate(locs)

    monkey.append(mb)

    monkey.to_object("Points on Suzan", shade_smooth=False)

def demo_wind():
    import numpy as np

    from npblender.core.points import Points
    from npblender.core.meshbuilder import MeshBuilder

    from npblender.core import engine

    engine.init()

    rng = np.random.default_rng(0)

    count = 200
    max_count = 10000
    points = Points(rng=rng)
    points.init_simulation()

    points.add_gravity()
    points.add_wind(
        force        = 20,
        axis         = 'Y',
        viscosity    = 1.,
        power        = 2,
        noise        = 3,
        noise_scale  = .5,
        noise_detail = 3,
        time_scale   = 10,
        )

    def update(scene):

        # Add some points
        n = min(count, max_count - len(points))
        if n > 0:
            points.add(
                position  = Points.normal_sphere_locs(count=n, scale=10, seed=points.seed) + (0, 0, 10),
                speed     = Points.normal_sphere_locs(count=n, scale=5, seed=points.seed),
                mass      = rng.normal(1, .3, n),
                viscosity = rng.normal(3, .5, n),
                )

        print("Count", len(points))
        print("Max speed", np.max(np.linalg.norm(points.speeds, axis=-1)))

        # Simulation

        points.step(1/25, sub_steps=1)

        #  Delete particle on the ground

        points.kill(points.z < 0)

        # ----- To object

        mb = MeshBuilder.Cube(.5)*len(points)
        mb.transform(points)
        mb.to_object("Wind")

    engine.go(update)

def demo_centrifugal():

    # ----- Free fall in a rotating frame
    # This demo illustrates free falls of 10 balls into a centrifugal frame
    # The trajectories of the balls are illustrated with curves

    import numpy as np

    from npblender.core.points import Points
    from npblender.core.meshbuilder import MeshBuilder
    from npblender.core.curvebuilder import PolyBuilder

    from npblender.core import engine

    engine.init()

    count = 10
    points = Points(locations=Points.line_locs((-10, -10, 0), (10, -10, 0), count))
    points.init_simulation(speeds=(0, 0, 0))

    points.add_centrifugal(omega=1)

    curves = points.sim_curves(duration=10, count=100, sub_steps=1)

    cb = PolyBuilder()
    cb.add_splines(curves)

    # ----- To object

    balls = MeshBuilder.UVSphere()*count
    balls.to_object("Balls")

    obj = cb.to_object("Centrifugal")
    obj.data.bevel_depth = .05
    obj.data.bevel_factor_end = 0

    def update(scene):
        obj.data.bevel_factor_end  = scene.time/10
        balls.locate(cb(scene.time/10))
        balls.update_object("Balls")

    engine.go(update)
