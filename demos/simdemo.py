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





