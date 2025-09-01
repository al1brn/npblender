# Mesh

`Mesh` is the mesh class of **npblender**. It inherits from `Geometry` and represents a Blender mesh (vertices, edges, faces, loops) while remaining fully NumPy-compatible. The four main domains are: Points, Corners, Faces, and Edges.

## Inheritance and Domains

- Inherits from: `Geometry`
- Domains:
  - `points` - vertex positions and per-vertex attributes
  - `corners` - loops (vertex indices per corner)
  - `faces` - face sizes and offsets
  - `edges` - edges as pairs of vertex indices

Each domain can hold additional user attributes.

## Correspondence with Blender Mesh

The class can import from and export to Blender mesh data:

- `from_mesh_data(data)`: build a `Mesh` from a `bpy.types.Mesh` (reads vertices, loops, polygons, edges, materials, attributes).
- `to_mesh_data(data)`: write the mesh back into a `bpy.types.Mesh`.

Object-level transfer:

- `from_object(obj, evaluated=False)`: capture a mesh from a Blender object (`evaluated=True` applies modifiers).
- `to_object(obj, shade_smooth=None, shapekeys=None, collection=None)`: create or update a Blender object from this `Mesh`.

See: [`from_mesh_data`](np.blender.mesh.Mesh.from_mesh_data), [`to_mesh_data`](np.blender.mesh.Mesh.to_mesh_data), [`from_object`](np.blender.mesh.Mesh.from_object), [`to_object`](np.blender.mesh.Mesh.to_object)

## Editing with Blender

Two context managers are provided:

- `with mesh.bmesh(readonly=False) as bm:`  
  Opens a temporary BMesh, allows calling `bmesh.ops.*`, and re-captures data into the `Mesh` (unless `readonly=True`).

- `with mesh.blender_data(readonly=False) as data:`  
  Gives direct access to the underlying `bpy.types.Mesh` for reading/writing attributes, normals, materials, etc.

Typical usage includes triangulation, removing doubles, extrusion, inset, or applying modifiers like Solidify.

See: [`bmesh`](np.blender.mesh.Mesh.bmesh), [`blender_data`](np.blender.mesh.Mesh.blender_data), [`triangulate`](np.blender.mesh.Mesh.triangulate), [`remove_doubles`](np.blender.mesh.Mesh.remove_doubles), [`extrude_region`](np.blender.mesh.Mesh.extrude_region), [`inset_faces`](np.blender.mesh.Mesh.inset_faces), [`solidify`](np.blender.mesh.Mesh.solidify)

## Primitives

The class provides many primitive constructors:

- Blender-based (prefix `bl_`):  
  [`bl_grid`](np.blender.mesh.Mesh.bl_grid), [`bl_circle`](np.blender.mesh.Mesh.bl_circle), [`bl_cone`](np.blender.mesh.Mesh.bl_cone), etc. (internally call `bmesh.ops.create_*`).

- npblender-based:  
  [`points_cloud`](np.blender.mesh.Mesh.points_cloud), [`line`](np.blender.mesh.Mesh.line), [`grid`](np.blender.mesh.Mesh.grid), [`cube`](np.blender.mesh.Mesh.cube), [`circle`](np.blender.mesh.Mesh.circle), [`cylinder`](np.blender.mesh.Mesh.cylinder), [`cone`](np.blender.mesh.Mesh.cone), [`uvsphere`](np.blender.mesh.Mesh.uvsphere), [`icosphere`](np.blender.mesh.Mesh.icosphere), [`torus`](np.blender.mesh.Mesh.torus), [`pyramid`](np.blender.mesh.Mesh.pyramid), [`monkey`](np.blender.mesh.Mesh.monkey)

## Examples

### 1) Create, edit, and export to Blender
```python
mesh = Mesh.grid(2, 2, 50, 50)            # primitive
with mesh.bmesh() as bm:                  # edit in BMesh
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)

mesh = mesh.solidify(thickness=0.2)       # apply Solidify modifier
obj = mesh.to_object("Plate", shade_smooth=True)
```

### 2) Roundtrip with a Blender object
```python
# Import existing object with modifiers applied
m = Mesh.from_object("MyMesh", evaluated=True)

# Modify vertices in NumPy and write back
m.points.position[:, 2] += 0.1
m.to_object("MyMesh")
```

### 3) Primitives and boolean operations
```python
cube = Mesh.cube(size=2)
cyl  = Mesh.cylinder(radius=0.5, depth=3)
res  = cube.boolean(cyl, operation='DIFFERENCE')
res.to_object("CubeMinusCyl", shade_smooth=False)
```

### 4) Access Blender mesh data directly
```python
with mesh.blender_data() as data:
    face_normals = np.array([p.normal[:] for p in data.polygons])

new_data = bpy.data.meshes.new("TmpMesh")
mesh.to_mesh_data(new_data)
```

## Best Practices

- Use `bmesh()` for geometry operations (`subdivide`, `bridge`, `triangulate`, etc.).
- Use `blender_data()` for direct attribute access (normals, materials, custom layers).
- Methods preserve materials and attributes across transfers.

---

This provides a full overview of the `Mesh` class in npblender:  
- Domains consistent with Blender  
- Import/export with objects and mesh data  
- Contexts for editing  
- A wide set of primitive constructors  


## Mesh - Methods

::: npblender.mesh.Mesh
    options:
      inherited_members: true
      show_root_heading: false      # évite de générer un 2ᵉ H1
      heading_level: 2              # la racine serait H2 (si affichée) → méthodes en H3
      inherited_members: true       # inclut les méthodes héritées (si souhaité)
