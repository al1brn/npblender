# Simulation demo

import numpy as np

from npblender import Simulation, Kinematics, Mesh, Curve, noise, maprange

# ====================================================================================================
# Simulation simple demo
# ====================================================================================================

def simu_demo():
    
    # ---------------------------------------------------------------------------
    # Initialize a field of vertical lines
    # ---------------------------------------------------------------------------

    N, S = 1000, 10

    rng = np.random.default_rng()
    pseed = rng.integers(1<<32)

    start = rng.uniform([-10, -10, 0], [10, 10, 0], (N, 3))
    end = start + [0, 0, 3]

    curve = Curve.line(start, end, resolution=S)
    ref = curve.points.position

    # ---------------------------------------------------------------------------
    # Move the points with noise
    # ---------------------------------------------------------------------------

    def move(simulation, t=None):
        pts = simulation.geometry.points.position

        ns = noise(ref, t=t, scale=1.0, octaves=5, lacunarity=2.0, gain=0.5,
          normalize=True, period=None, algo='fBM', perlin=pseed, out_dim=2)
        ns = np.append(ns, np.zeros_like(ns[:, 0]))
        
        ns = ns.reshape((N, S, 3))
        ns *= maprange(np.linspace(0, 1, S), mode='SMOOTH')[None, :, None]
        
        simulation.geometry.points.position = ref + ns.reshape(-1, 3)

    # ---------------------------------------------------------------------------
    # Build the simulation
    # ---------------------------------------------------------------------------

    simul = Simulation(curve, "Simulation Demo")
    action = simul.add_action(move, None)

    simul.go()

# ====================================================================================================
# Explosion
# ====================================================================================================

class ExplosionSim(Kinematics):
    
    def __init__(self, subdivs=3, force=100):
        
        from npblender.core.maths import distribs
        from npblender import Meshes
        
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
        
        self.meshes = Meshes.from_mesh_islands(ico)
        npieces = len(self.meshes)
        self.init_position = self.meshes.points.position
        
        mass = np.empty(npieces, dtype=int)
        for bucket, offset in self.meshes:
            mass[offset:offset + len(bucket)] = bucket.shape[1]
            
        self.meshes.points.init_kinematics()
        self.meshes.points.mass = mass/np.max(mass)
        
        # ---------------------------------------------------------------------------
        # Let'as add the actions
        # ---------------------------------------------------------------------------
        self.add_action("gravity", start=1)
        uforce = self.init_position/np.linalg.norm(self.init_position, keepdims=True)
        uforce = distribs.shake_vectors(uforce, scale=.2, length_only=False, lengths=None, seed=rng)
        self.add_action("force", uforce*rng.normal(force, force*.15, (npieces, 1)), start=1, duration=.2)
        self.add_setter("omega",
            value=distribs.sphere_dist(radius=10, scale=1, count=len(self.points), seed=0)['points'], 
            start=1.1,
            duration=0)
        self.add_setter("omega", value=.98, incr='*', start=1.1)
        
        
    # ----------------------------------------------------------------------------------------------------
    # Points
    # ----------------------------------------------------------------------------------------------------
    
    @property
    def points(self):
        return self.meshes.points
        
    # ----------------------------------------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------------------------------------
        
    def reset(self):
        super().reset()
        self.meshes.points.position = self.init_position
        self.meshes.points.quat[:] = (0, 0, 0, 1)
        self.meshes.points.speed[:] = 0
        self.meshes.points.omega[:] = 0
        
    # ----------------------------------------------------------------------------------------------------
    # View
    # ----------------------------------------------------------------------------------------------------
    
    def view(self):
        self.meshes.realize().to_object("Explosion")

    # ----------------------------------------------------------------------------------------------------
    # Backing
    # ----------------------------------------------------------------------------------------------------
    
    def get_frame_data(self):
        return self.meshes.points.to_dict()
    
    def set_frame_data(self, data):
        self.meshes.points = self.points.from_dict(data)
        


def explosion_demo(subdivs=3, force=100):
    ExplosionSim(subdivs=subdivs, force=force).go()



