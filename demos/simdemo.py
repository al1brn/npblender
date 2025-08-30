# Simulation demo

import numpy as np

from npblender import Simulation, Mesh, Curve, noise, maprange, Instances, Cloud, Meshes
from npblender.maths import distribs


def demo(name='explosion', **kwargs):

    classes = {
        'bouncing balls': BouncingBallsRain,
        'curves field' : CurvesField,
        'explosion' : Explosion,
    }

    class_ = classes.get(name.lower(), None)
    if class_ is None:
        raise AttributeError(f"Invalid demo name, valids are {list(classes.keys())}.")
    
    simul = class_(**kwargs)
    simul.go()

# ====================================================================================================
# Rain of bouncing balls
# ====================================================================================================

class BouncingBallsRain(Simulation):
    def __init__(self, count=100, instances=False):

        rng = np.random.default_rng(0)

        radius = .1

        if instances:
            self.geometry = Instances(models=[Mesh.uvsphere(radius=radius)])
        else:
            self.geometry = Cloud()

        self.points.disk_dist(radius=15*radius, count=count, seed=rng)
        self.points.z = 5
        self.init_pos = self.points.position

        self.points.new_float("t0")
        self.points.new_float("t1")
        self.points.new_vector("speed")

        self.points.t0 = rng.uniform(0, 5, count)
        self.points.t1 = self.points.t0 + rng.normal(4, .2, count)

        speeds = self.points.disk_speed(.1, 1.3, seed=rng)
        # Animation with variable pos
        pos = lambda t: np.stack((
                self.init_pos[:, 0] + maprange(t, t0=self.points.t0, t1=self.points.t1, v0=0, v1=speeds[:, 0], mode="LINEAR"), 
                self.init_pos[:, 1] + maprange(t, t0=self.points.t0, t1=self.points.t1, v0=0, v1=speeds[:, 1], mode="LINEAR"),
                maprange(t, t0=self.points.t0, t1=self.points.t1, v0=5, v1=radius, mode="BOUNCE.OUT"),
                ), axis=-1)

        self.add_action(
            "change_attribute",
            "position",
            pos,
            )

    def reset(self):
        self.points.position = self.init_pos
        self.points.speed = 0

# ====================================================================================================
# Field of curves
# ====================================================================================================

class CurvesField(Simulation):
    """ A field of vertical lines
    """
    def __init__(self, count=1000, resolution=5):
        rng = np.random.default_rng()
        self.perlin_seed = rng.integers(1<<32)

        start_pos = rng.uniform((-10, -10, 0), (10, 10, 0), (count, 3))

        self.geometry = Curve.line(start_pos, start_pos + [0, 0, 3], resolution=resolution)
        self.ref_pos = np.array(self.points.position)

    def reset(self):
        self.points.position = self.ref_pos

    # ---------------------------------------------------------------------------
    # Move the points with noise
    # ---------------------------------------------------------------------------

    def compute(self):
        N, S = len(self.geometry.splines), self.geometry.splines.loop_total[0]
        t = self.time

        ns = noise(self.ref_pos, t=t, scale=1.0, octaves=5, lacunarity=2.0, gain=0.5,
          normalize=True, period=None, algo='fBM', perlin=self.perlin_seed, out_dim=2)
        ns = np.append(ns, np.zeros_like(ns[:, 0]))
        
        ns = ns.reshape((N, S, 3))
        ns *= maprange(np.linspace(0, 1, S), mode='SMOOTH')[None, :, None]
        
        self.points.position = self.ref_pos + ns.reshape(-1, 3)

# ====================================================================================================
# Explosion
# ====================================================================================================

class Explosion(Simulation):
    
    def __init__(self, subdivs=3, force=100):
        
        super().__init__()

        rng = np.random.default_rng(0)
        
        # Let's explode an ico sphere
        ico = Mesh.icosphere(subdivisions=subdivs)
        
        # ---------------------------------------------------------------------------
        # We randomly group neighbor faces to have pices of different sizes
        # ---------------------------------------------------------------------------

        nfaces = len(ico.faces)
        groups = np.arange(nfaces)
        neighbors = np.asarray(ico.faces_neighbors())

        for _ in range(8*subdivs):
            sel = rng.uniform(0, 1, nfaces) < .1
            n = np.sum(sel)
            neigh_sel = neighbors[sel][np.arange(n), rng.integers(0, 3, n)]
            groups[sel] = groups[neigh_sel]
            
        # ---------------------------------------------------------------------------
        # Pieces are made as mesh islands
        # ---------------------------------------------------------------------------
        
        ico.faces.new_int("group")
        ico.faces.group = groups
        ico = ico.separate_faces(groups=groups)
        ico=ico.solidify()
        
        # ---------------------------------------------------------------------------
        # Let's treat this single mesh ad individual meshes
        # ---------------------------------------------------------------------------
        
        meshes = Meshes.from_mesh_islands(ico)
        self.geometry = meshes

        npieces = len(meshes)
        self.init_position = np.array(self.points.position)
        
        mass = np.empty(npieces, dtype=int)
        for bucket, offset in meshes:
            mass[offset:offset + len(bucket)] = bucket.shape[1]
            
        self.points.init_kinematics()
        self.points.mass = mass/np.max(mass)
        
        # ---------------------------------------------------------------------------
        # Let'as add the actions
        # ---------------------------------------------------------------------------

        self.add_action("gravity", start=1)
        uforce = self.init_position/np.linalg.norm(self.init_position, keepdims=True)
        uforce = distribs.shake_vectors(uforce, scale=.2, length_only=False, lengths=None, seed=rng)
        self.add_action("force", uforce*rng.normal(force, force*.15, (npieces, 1)), start=1, duration=.2)
        self.add_action("change_attribute", "omega",
            distribs.sphere_dist(radius=10, scale=1, count=len(self.points), seed=0)['points'], 
            start=1.1,
            duration=0)
        self.add_action("change_attribute", "omega", .98, incr='*', start=1.1)
        
    # ----------------------------------------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------------------------------------
        
    def reset(self):
        super().reset()
        self.points.position = self.init_position
        self.points.quat[:] = (0, 0, 0, 1)
        self.points.speed[:] = 0
        self.points.omega[:] = 0
        
    # ----------------------------------------------------------------------------------------------------
    # Compute
    # ----------------------------------------------------------------------------------------------------

    def compute(self):
        self.compute_motion()
    
    # ----------------------------------------------------------------------------------------------------
    # View
    # ----------------------------------------------------------------------------------------------------
    
    def view(self):
        self.geometry.realize().to_object("Explosion")

    # ----------------------------------------------------------------------------------------------------
    # Backing
    # ----------------------------------------------------------------------------------------------------
    
    def get_frame_data(self):
        return self.meshes.points.to_dict()
    
    def set_frame_data(self, data):
        self.meshes.points = self.points.from_dict(data)

# ====================================================================================================
# Fireworks
# ====================================================================================================

class Rocket(Simulation):

    def __init__(self, count=30, duration=10, seed=0):

        rng = np.random.default_rng(seed)

        self.insts = Instances(model=Mesh.cube(radius=.05))
        self.insts.points.line_dist((-5, 0, 0), (5, 0, 0), count=count)

        self.start_pos = np.array(self.points.position)

        self.points.new_vector("launch_force")
        self.points.new_float("launch_time")
        self.points.new_bool("launched")

        self.points.launch_force = self.insts.points.speed_along(
            speed=100, 
            direction=(0, 0, 1),
            scale = 10,
            angle = np.radians(20),
            use_vonmises=True,
            )
        self.points.launch_time = rng.uniform(0, 5, count)
        self.points.launched = False

        # ---------------------------------------------------------------------------
        # Actions
        # ---------------------------------------------------------------------------

        self.add_action("gravity")
        self.add_action("launch")

    @property
    def points(self):
        return self.insts.points

    def reset(self):
        self.points.launched = False
        self.points.position = self.start_pos
        self.points.speed = 0

    def launch(self):
        pts = self.points
        mask = np.logical_and(~pts.launched, pts.launch_time[i] >= self.time)
        pts.force += pts.launch_foce[mask]






