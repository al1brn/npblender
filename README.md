# NPBlender

**NumPy‑based geometry for Blender** — build and manipulate meshes, curves, splines, and point clouds with fast, vectorized Python.

> TL;DR: Work with Blender geometry using **NumPy**. Read/write Blender data, run batch computations, and keep everything scriptable.

---

## Why NPBlender?

- **Numerical workflows**: when you need custom math (simulation, sampling, meshing, attribute baking) beyond regular operators.  
- **External data**: bring in geospatial data, scans, CSV/NPZ datasets, or anything you can load into NumPy, then turn it into Blender geometry.  
- **Interoperable**: everything stays as plain arrays; easy to integrate with your pipeline or other Python tools.

### Works alongside Geometry Nodes

NPBlender does not replace GN — it complements it. Generate or edit geometry with NumPy (including attributes), send it back to Blender as objects or mesh/curve data, then continue with **Geometry Nodes** for non‑destructive setups.

---

## Installation

> You don’t need to “install” an add‑on. Just make the package importable by Blender’s Python.

1. **Clone (or symlink) the repository** into Blender’s *scripts/modules* directory so that the path ends with `.../scripts/modules/npblender`:

- **Linux**: `~/.config/blender/<version>/scripts/modules/`
- **macOS**: `~/Library/Application Support/Blender/<version>/scripts/modules/`
- **Windows**: `%APPDATA%\Blender Foundation\Blender\<version>\scripts\modules\`

2. Restart Blender, then in the *Scripting* workspace:

```python
import npblender as npb
```

For details on Blender’s script directories, see the Blender Manual:  
<https://docs.blender.org/manual/en/latest/advanced/blender_directory_layout.html>

> Tip: You can also keep your working copy elsewhere and create a **symlink** named `npblender` inside `scripts/modules/` pointing to it.

---

## Quick Start

```python
# Run inside Blender's Python console or Text Editor
import numpy as np
from npblender import Mesh

# 1) Create a parametric grid
plate = Mesh.grid(size_x=2, size_y=2, vertices_x=50, vertices_y=50)

# 3) NumPy-driven deformation
plate.points.position[:, 2] += 0.1 * np.sin(plate.points.position[:, 0] * 4.0)

# 2) Apply a thickness and send it to Blender as an object
plate=plate.solidify(thickness=.1)
obj = plate.to_object("NPB Plate", shade_smooth=True)

```

Another example with curves:

```python
import numpy as np
from npblender import Curve

# Twisted pole

# Start by a curve line of 100 points
n = 100
pole = Curve.line(start=(0, 0, 0), end=(0, 0, 1), resolution=n)

# Twist and scale each point
pole.points.tilt = np.linspace(0, 3*np.pi, n)
pole.points.radius = np.linspace(1, 0, n)

# To mesh with a 5 segments circle
profile = Curve.circle(radius=.2, resolution=5)

mesh = pole.to_mesh(profile=profile, caps=True, use_radius=True)
mesh.to_object("NPB Twisted Pole", shade_smooth=False)
```

Animation example (play animation in Blender once the program executed):

``` python
import numpy as np
from npblender import Mesh, engine
from npblender.maths import maprange

amplitude = 4
radius = 1

# Bouncing ball
ball = Mesh.uvsphere(radius=radius, rings=64)

# Copy initial position
pos = np.array(ball.points.position)

# Called at each frame change
def update():
    # Get current time from engine
    t = engine.time
    
    # Simulate infinite bounce
    z = amplitude*np.abs(np.sin(np.pi*t/2))
    
    # Refresh to initial shape
    ball.points.position = pos
    
    # Vertical location
    ball.points.z += z
    
    # Get the resulting z of each vertex
    bz = ball.points.z 
    
    # Widen points below zero
    ball.points.position[:, :2] *= maprange(bz, -1, -.5, 1.6, 1, mode='SMOOTH')[:, None]
    
    # Shift upwards points below zero
    ball.points.z += maprange(bz, -1, -.5, .3, 0, mode='QUAD.OUT')
    
    # Global upward to stick to z=0
    ball.points.z += .73
    
    # To object
    ball.to_object("NPB Bouncing Ball")


# Declare update method and launch engine
engine.go(update)
```

---

## Documentation

Full documentation is available [here](https://al1brn.github.io/npblender)
API reference is [here](https://al1brn.github.io/npblender/api)

---

## License

MIT
