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
from npblender.mesh import Mesh

# 1) Create a parametric grid
mesh = Mesh.grid(x=2, y=2, nx=50, ny=50)

# 3) NumPy-driven deformation
plate.points.position[:, 2] += 0.1 * np.sin(plate.points.position[:, 0] * 4.0)

# 2) Apply a thickness and send it to Blender as an object
plate = mesh.solidify(thickness=0.2)
obj = plate.to_object("NPB_Plate", shade_smooth=True)
```

Another example with curves:

```python
from npblender.geometry.curve import Curve

# Make a Bezier circle, convert to mesh with a profile sweep
circle = Curve.bezier_circle()
profile = Curve.line(start=(0, 0, -0.05), end=(0, 0, 0.05), resolution=8)
tube = circle.to_mesh(profile=profile, caps=True, use_radius=False)
tube.to_object("NPB_Tube", shade_smooth=True)
```

---

## Documentation

Full documentation is available [here](docs/)

---

## License

MIT
