# npblender API

## Table of Contents

## Geometry
- [Geometry][npblender.Geometry] : root class for actual geometries
- [Mesh][npblender.Mesh]
    - `points`: [Vertex][npblender.Vertex]
    - `corners`: [Corner][npblender.Corner]
    - `edges`: [Edge][npblender.Edge]
    - `faces`: [Face][npblender.Face]
- [Curve][npblender.Curve]
    - `points`: [ControlPoint][npblender.ControlPoint]
    - `splines`: [Spline][npblender.Spline]
- [Cloud][npblender.Cloud]
    - `points`: [Point][npblender.Point]
- [Instances][npblender.Instances]
    - `points`: [Point][npblender.Point]
- [Meshes][npblender.Meshes]
    - `points`: [Point][npblender.Point]
- [Text][npblender.Text]
- [Formula][npblender.Formula]

### Animation
- [Camera][npblender.Camera] : for camera culling
- [engine][npblender.Engine] : animation engine
- [Animation][npblender.Animation] : basic yet powerful Animation class
- [Simulation][npblender.Simulation] : advanced simulation

### Maths
- [Rotation][npblender.Rotation]
- [Quaternion][npblender.Quaternion]
- [Transformation][npblender.Transformation]
- [maths][npblender.maths] module
- [maths.distribs][npblender.maths.distribs] module

