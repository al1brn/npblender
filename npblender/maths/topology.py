#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blender Python Geometry module

Created on Wed Jun 29 17:03:43 2022

@author: alain.bernard
@email: alain@ligloo.net

-----

Topology functions for Meshes.

Topology build both meshes and uv mapping in a consistent way.
Grid (x, y) are implemented as x lines of y points.

"""

import numpy as np

# ====================================================================================================
# Grid topology
# ====================================================================================================

def grid_corners(nx, ny, row_first=False, close_x=False, close_y=False, clockwise=False):
    """ Build corners of 4-point loops on a grid

    The grid corners can be used:
    - to build the UV Map
    - to build the grid faces
    
    Grid is shaped using 'ij' indexing:
    - grid[i, j] = point at coordinate n(x=i, y=j)
    
    Hence, horizontal lines along x are grid columns grid[:, j], and vertical lines
    along y are lines grid[i, :].
    
    > [!CAUTION]
    > Flattening a grid gives a series of vertical columns, not a serie of horizontal lines.
    > Use flatten(order='F') to flatten as a series of x lines.
    
    The function grid_corners returns indices in range(nx*ny). The indices can computed:
    - row_first = True (default) : +1 along x and +nx along y
    - row_first = False : +ny along x and + 1 along y

    Arguments
    ---------
    - nx (int) : number of points in x direction
    - ny (int) : number of points in y direction
    - row_first : nx*ny are arranged row first (True) or col first (False)
    - close_x (bool = False) : close the grid in x direction
    - close_y (bool = False) : close the grid in y direction
    - clockwise (bool = False) : order of the corners in the loop

    Returns
    -------
    - array (nx_, ny_) of ints, nx_ and ny_ depend upon close_x and close_y
    """
    
    if row_first:
        indices = np.arange(nx*ny).reshape(ny, nx).T
    else:
        indices = np.arange(nx*ny).reshape(nx, ny)
        
    #print(np.flip(indices.T, axis=0))
        
    if close_x:
        irange = np.arange(nx + 1)
        irange[-1] = 0
    else:
        irange = np.arange(nx)
        
    if close_y:
        jrange = np.arange(ny + 1)
        jrange[-1] = 0
    else:
        jrange = np.arange(ny)

    i, j = np.meshgrid(irange, jrange, indexing='ij')
    
    c0 = indices[i[:-1, :-1], j[:-1, :-1]].ravel()
    c1 = indices[i[1:, 1:],   j[:-1, :-1]].ravel()
    c2 = indices[i[1:, 1:],   j[1:, 1:]].ravel()
    c3 = indices[i[:-1, :-1], j[1:, 1:]].ravel()

    if clockwise:
        return np.stack((c3, c2, c1, c0), axis=-1)
    else:
        return np.stack((c0, c1, c2, c3), axis=-1)
    
# ====================================================================================================
# Grid map
# ====================================================================================================
    
def grid_uv_map(nx, ny, close_x=False, close_y=False, clockwise=False, bbox=[0, 0, 1, 1]):
    """ Build a grid uv map

    Arguments
    ---------
    - nx (int) : number of points in x direction
    - ny (int) : number of points in y direction
    - close_x (bool = False) : close the grid in x direction
    - close_y (bool = False) : close the grid in y direction
    - clockwise (bool = True) : order of the corners in the loop
    - bbox (list of 4 floats) : bounding box [xmin, ymin, xmax, ymax]

    Returns
    -------
    - array (nx_, ny_, 4, 2) of floats, nx_ and ny_ depend upon close_x and close_y
    """

    if close_x:
        nx += 1
    if close_y:
        ny += 1

    topo = grid_corners(nx, ny, row_first=False, clockwise=clockwise)
    
    u, v = np.meshgrid(
        np.linspace(bbox[0], bbox[2], nx),
        np.linspace(bbox[1], bbox[3], ny), 
        indexing='ij')
    
    return np.stack((u.ravel(), v.ravel()), axis=-1)[topo].reshape(nx-1, ny-1, 4, 2)

# ====================================================================================================
# Fans topology
# ====================================================================================================

def fans_corners(count, close=True, clockwise=False):
    """ Build corners of 3-point loops on a disk

    The fans corners can be used:
    - to build the UV Map
    - to build the mesh faces

    Arguments
    ---------
    - count (int) : number of points on the circle
    - close (bool = True) : close the loop
    - clockwise (bool = False) : order of the corners in the loop

    Returns
    -------
    - array (count_, 3) of ints, count_ depends upon close
    """
    if close:
        brange = np.arange(count + 1)
        brange[-1] = 0
    else:
        brange = np.arange(count)

    c0 = brange[:-1]
    c1 = brange[1:]
    c2 = np.ones(len(brange)-1, dtype=int)*count

    # c2 (top index) in last scolumn
    if clockwise:
        return np.stack((c1, c0, c2), axis=-1)
    else:
        return np.stack((c0, c1, c2), axis=-1)
    
# ====================================================================================================
# Fans uv map
# ====================================================================================================
    
def disk_uv_map(count, mode='NGON', clockwise=False, bbox=[0, 0, 1, 1]):
    """ Build an uv map for a disk

    Arguments
    ---------
    - count (int) : number of points on the circle
    - mode (FillCap = 'NGON') : NGON or FANS
    - clockwise (bool = False) : order of the corners in the loop
    - bbox (list of 4 floats) : bounding box [xmin, ymin, xmax, ymax]

    Returns
    -------
    - array (count, 3, 2) of floats
    """

    center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]

    ag = np.linspace(0, 2*np.pi, count, endpoint=False)

    u = center[0] + np.cos(ag)*((bbox[2] - bbox[0])/2)
    v = center[1] + np.sin(ag)*((bbox[3] - bbox[1])/2)

    uv = np.stack((u, v), axis=-1)

    if mode == 'NONE':
        return None

    elif mode == 'NGON':
        return uv
    
    elif mode == 'FANS':
        uv = np.append(uv, [center], axis=0)
        return uv[fans_corners(count, close=True, clockwise=clockwise)] #.reshape(count, 3, 2)
    
    assert(False)

# ====================================================================================================
# Border edges
# ====================================================================================================

def border_edges(*loops, close=False):
    edges = []
    for loop in loops:
        n = len(loop)
        if n < 2:
            edges.append(np.zeros((0, 2), dtype=np.int32))
        else:
            a = [[i0, i1] if i0 < i1 else [i1, i0] for (i0, i1) in zip(loop[:-1], loop[1:])]
            if close and n > 2:
                if loop[0] < loop[-1]:
                    a.append([loop[0], loop[-1]])
                else:
                    a.append([loop[-1], loop[0]])

            edges.append(np.array(a))

    if len(loops) == 1:
        return edges[0]
    else:
        return edges

# ====================================================================================================
# Edges between
# ====================================================================================================

def edges_between(loop0, loop1):
    return np.array([[i0, i1] if i0 < i1 else [i1, i0] for (i0, i1) in zip(loop0, loop1)])

# ====================================================================================================
# Grid edges
# ====================================================================================================

def row_edges(corners, close=False):
    n = corners.shape[1]
    if n < 2:
        return np.zeros((0, 0, 2), dtype=np.int32)

    edges = np.stack((corners[:, :-1], corners[:, 1:]), axis=-1)
    if close and n > 2:
        last = np.stack((corners[:, 0], corners[:, -1]), axis=-1)
        edges = np.append(edges, last[:, None], axis=1)

    return np.sort(edges, axis=-1)

def col_edges(corners, close=False):

    n = corners.shape[0]
    if n < 2:
        return np.zeros((0, 0, 2), dtype=np.int32)

    edges = np.stack((corners[:-1], corners[1:]), axis=-1)
    if close and n > 2:
        last = np.stack((corners[0], corners[-1]), axis=-1)
        edges = np.append(edges, last[None], axis=0)

    return np.sort(edges, axis=-1)

      

# OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD 
# OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD 
# OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD 
# OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD 
# OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD 

            
        














# ====================================================================================================
# Bound Box

class Box:
    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], Box):
                self.x0, self.y0, self.x1, self.y1 = args[0].x0, args[0].y0, args[0].x1, args[0].y1
            elif isinstance(args[0], tuple):
                self.x0, self.y0, self.x1, self.y1 = args[0]
            else:
                raise Exception(f"Argument error: {args}")

        else:
            if len(args) == 0:
                args = (0., 0., 1., 1.)
            self.x0, self.y0, self.x1, self.y1 = args

        try:
            _ = 1/self.sx + 1/self.sy
        except:
            raise Exception(f"Strange: Box({self.x0:.3f}, {self.y0:.3f}, {self.x1:.3f}, {self.y1:.3f}), size ({self.sx:.3f}, {self.sy:.3f})")

    def __str__(self):
        return f"Box({self.x0:.3f}, {self.y0:.3f}, {self.x1:.3f}, {self.y1:.3f}), size ({self.sx:.3f}, {self.sy:.3f})"

    @classmethod
    def FromUvs(cls, uvs, box=(0., 0., 1., 1.)):
        if box is None:
            return cls(np.min(uvs[..., 0]), np.min(uvs[..., 1]), np.max(uvs[..., 0]), np.max(uvs[..., 1]))
        else:
            return Box(box)

    @property
    def sx(self):
        return self.x1 - self.x0

    @property
    def sy(self):
        return self.y1 - self.y0

    @property
    def P0(self):
        return (self.x0, self.y0)

    @property
    def size(self):
        return (self.x1 - self.x0, self.y1 - self.y0)

    def ratio(self, other):
        return (self.sx/other.sx, self.sy/other.sy)

    def resize(self, uvs, box=(0., 0., 1., 1.)):
        box = Box.FromUvs(uvs, box)
        return self.P0 + (uvs - box.P0)*self.ratio(box)

# ====================================================================================================
# ====================================================================================================
# Utility functions
# ====================================================================================================


# ====================================================================================================
# Corners
#
# A face is a loop of corners. Each corner has a vertex index and UVMap coordinates
# The agorithms to compute the faces as a series of corners are identical for vertex indices and uvs
# with a difference for close surfaces.
#
# A close surface is a surface where the last points are connected to the first ones.
# For vertices, the last index is connected to the first one
# It is not the case for uvs which need an additional points
#
# For instance for a cylinder of two rings of 4 points:
#
# Vertex indices are [0, 1, 2, 3] and [4, 5, 6, 7]
# The face closing the cylinder is [3, 0, 4, 7]
#
# uvs vectors are [(0, 0), (.25, 0), (.5, 0), (.75, 0), (1, 0)] and [(0, 1), (.25, 1), (.5, 1), (.75, 1), (1, 1)]
# The last uv face is [(.75, 0), (1, 0), (1, 1), (.75, 1)]
#
# The function grid_corners builds the faces and can be called either with a array of indices or with
# an array of 2_vectors. For closed surfaces:
# - Vertices : use the number of vertices with close argument = True
# - uvs : create one additional 2-vector with close argument = False

# ----------------------------------------------------------------------------------------------------
# Corners are computed either on arrays of 2-vectors or arrays of int indices

def get_item_shape(array):
    is_uvs = array.dtype.name[:5] == 'float'
    assert(not is_uvs or np.shape(array)[-1] == 2)
    return (2,) if is_uvs else ()

# ----------------------------------------------------------------------------------------------------
# Bridge two rings

def bridge_corners(ring0, ring1, clockwise, close=True):
    """ Return corners bridging two rings of indices.

    In the simpler use, ring0 and ring1 are array of ints representings the indices.

    In a more complex use, these arguments are arrays of such rings.
    This allows to build a whole grid by bridging eachring with its successor ring.

    Arguments
    ---------
        - ring0 (array[n] of ints or 2-vectors or array of) : start rings
        - ring1 (array[n] or ints or 2-vectors or array of) : end rings
        - close (bool=True) : the rings are closed
        - clockwise (bool=False) : order of the corners

    Returns
    -------
        - array of ints or 2-vectors : corners
    """

    assert(np.shape(ring0)==np.shape(ring1))

    # ----- uvs or vertex indices

    item_shape = get_item_shape(ring0)

    # ----- Simple --> let's make it complex :-)

    single = len(np.shape(ring0)) - len(item_shape) == 1
    if single:
        n = len(ring0)
        ring0 = np.reshape(ring0, (1, n) + item_shape)
        ring1 = np.reshape(ring1, (1, n) + item_shape)
    else:
        n = np.shape(ring0)[1]

    nrings = len(ring0)

    # ----- If close, we add the first ring at the end

    if close:
        ring0 = np.append(ring0, ring0[:, 0][:, None], axis=1)
        ring1 = np.append(ring1, ring1[:, 0][:, None], axis=1)
    else:
        n -= 1

    # ----- We can build the corners

    corners = np.empty((nrings, n, 4) + item_shape, ring0.dtype)

    if clockwise:
        corners[:, :, 0] = np.roll(ring0, -1, axis=1)[:, :n]
        corners[:, :, 1] = ring0[:, :n]
        corners[:, :, 2] = ring1[:, :n]
        corners[:, :, 3] = np.roll(ring1, -1, axis=1)[:, :n]

    else:
        corners[:, :, 0] = ring0[:, :n]
        corners[:, :, 1] = np.roll(ring0, -1, axis=1)[:, :n]
        corners[:, :, 2] = np.roll(ring1, -1, axis=1)[:, :n]
        corners[:, :, 3] = ring1[:, :n]

    if single:
        return corners[0]
    else:
        return corners

# ----------------------------------------------------------------------------------------------------
# Grid corners

def grid_corners_OLD(rings, clockwise, x_close=False, y_close=False):
    """ Corners of a grid.

    Arguments
    ---------
        - rings (array(ny, nx) of ints) : indices
        - x_close (bool=False) : close the grid along x (True for cylinders)
        - y_close (bool=False) : close the grid along y (True for torus)
        - clockwise (bool=False) : order of the corners

    Returns
    -------
        - array (ny, nx, 4) of ints or 2-vectors
    """

    nx = np.shape(rings)[1]
    ny = np.shape(rings)[0]

    # If y_close, add the initial ring to the end
    # Item shape is not needed because this can occur only with indices

    if y_close:
        rings = np.resize(rings, (ny+1, nx))

    return bridge_corners(rings[:-1], rings[1:], close=x_close, clockwise=clockwise)
# ----------------------------------------------------------------------------------------------------
# Fans corners

def fans_corners_OLD(ring, center, clockwise, close=False):
    """ Corners shaping fans around a center

    Arguments
    ---------
    - ring (array of ints or 2-vectors) : the surrounding ring
    - center (int or 2-vector or array of) : center
    - clockwise (bool=False) : face orientation
    - close (bool=True) : ring is closed

    Returns
    -------
    - array (n, 3) of ints or 2-vectors
    """

    # TBD

    # ----- uvs or vertex indices

    item_shape = get_item_shape(ring)

    # ----- close : we add the initial index to the end
    # Close if necessarily False for uvs

    if close:
        ring = np.resize(ring, len(ring) + 1)
    n = len(ring)

    corners = np.empty((n-1, 3) + item_shape, ring.dtype)
    corners[:, 0] = ring[:-1]
    if clockwise:
        corners[:, 1] = center
        corners[:, 2] = ring[1:]
    else:
        corners[:, 1] = ring[1:]
        corners[:, 2] = center

    return corners

# ====================================================================================================
# uvs
#
# uvs functions return components corners, sizes and UVMap
# It is up to the caller to create the vertices of the geometry

# ----------------------------------------------------------------------------------------------------
# Grid

def grid_uvs(x_count, y_count, clockwise=False, x_close=False, y_close=False, vert_indices=None, indexing='ij'):

    # ----- Uvs

    x_uv = x_count + (1 if x_close else 0)
    y_uv = y_count + (1 if y_close else 0)

    v2s = np.stack(np.meshgrid(np.linspace(0, 1, x_uv), np.linspace(0, 1, y_uv), indexing=indexing), axis=-1)
    uvs = grid_corners(v2s, clockwise)

    # ----- Vertices

    if vert_indices is None:
        vert_indices = np.reshape(range(x_count*y_count), (y_count, x_count))
    else:
        assert(np.shape(vert_indices) == (y_count, x_count))

    corners = grid_corners(vert_indices, clockwise, x_close=x_close, y_close=y_close)
    sizes = np.size(corners)//4

    return {'corners': corners, 'sizes': [4]*sizes, 'UVMap': np.reshape(uvs, (np.size(uvs)//2, 2))}
# ----------------------------------------------------------------------------------------------------
# Disk

def disk_uvs(x_count, y_count, clockwise=False, fans=False, vert_indices=None):


    # TBD

    # ----- Uvs

    x_uv = x_count + 1

    ag   = np.linspace(0, 2*np.pi, x_uv, endpoint=True)
    vs2  = np.stack((np.cos(ag), np.sin(ag)), axis=-1)

    # ----- Vertices

    nvertices = x_count*y_count + (1 if fans else 0)
    if vert_indices is None:
        vert_indices = np.arange(nvertices)
    else:
        assert(len(vert_indices) == nvertices)

    # ----- No inside circles

    if y_count == 1:
        uvs = np.empty((0, 2), float)
        vs2 = vs2/2 + (.5, .5)

        corners = np.empty(0, int)
        sizes = []
        last_inds = vert_indices[:x_count]

    # ----- Inside circles

    else:
        # uvs

        vs2 = np.resize(vs2, (y_count,) + np.shape(vs2))
        vs2 *= np.linspace(1, 0, y_count, endpoint=False)[:, None, None]
        vs2 = vs2/2 + (.5, .5)

        uvs = grid_corners(vs2, clockwise)
        uvs = np.reshape(uvs, (np.size(uvs)//2, 2))

        vs2 = vs2[-1]

        # vertex indices

        if fans:
            inds = np.reshape(vert_indices[:-1], (y_count, x_count))
        else:
            inds = np.reshape(vert_indices, (y_count, x_count))

        corners = grid_corners(inds, clockwise, x_close=True)
        corners = np.reshape(corners, np.size(corners))
        sizes   = [4]*(len(corners)//4)

        last_inds = inds[-1]

    # ----- Circle to center

    if fans:
        # uvs

        f_uvs = fans_corners(vs2, (.5, .5), clockwise)
        uvs = np.append(uvs, np.reshape(f_uvs, (np.size(f_uvs)//2, 2)), axis=0)

        # vertices

        f_cs = fans_corners(last_inds, vert_indices[-1], clockwise, close=True)
        corners = np.append(corners, np.reshape(f_cs, np.size(f_cs)), axis=0)
        sizes.extend([3]*len(f_cs))

    else:
        # uvs

        if clockwise:
            uvs = np.append(uvs, [v for v in reversed(vs2[:-1])], axis=0)
        else:
            uvs = np.append(uvs, vs2[:-1], axis=0)

        # vertices

        if clockwise:
            corners = np.append(corners, [i for i in reversed(last_inds)])
        else:
            corners = np.append(corners, last_inds)

        sizes.append(x_count)


    return {'corners': corners, 'sizes': np.array(sizes), 'UVMap': uvs}

# ----------------------------------------------------------------------------------------------------
# Triangle uvs (used for spheres)

def triangles_uvs(x_count, clockwise=False, up=True, x_close=True, vert_indices=None, top_index=None):

    # TBD

    # ----- Uvs

    x_uv = x_count + (1 if x_close else 0)
    vs = np.stack((np.linspace(0, 1, x_uv), np.zeros(x_uv, float)), axis=-1)
    if not up:
        vs[:, 1] = 1

    delta = 1/(x_uv - 1)
    tops  = np.stack((delta/2 + np.linspace(0, 1 - delta, x_uv - 1), np.ones(x_uv-1, float)), axis=-1)
    if not up:
        tops[:, 1] = 0

    uvs = fans_corners(vs, tops, clockwise)
    uvs = np.reshape(uvs, (np.size(uvs)//2, 2))

    # ----- Vertices

    if vert_indices is None:
        vert_indices = np.arange(x_count)
    if top_index is None:
        top_index = len(vert_indices)

    corners = fans_corners(vert_indices, top_index, clockwise, close=x_close)
    corners = np.reshape(corners, np.size(corners))

    return {'corners': corners, 'sizes': np.array([3]*(len(corners)//3)), 'UVMap': uvs}


# ====================================================================================================
# ====================================================================================================
# Geometry primitives
# ====================================================================================================

# ====================================================================================================
# Grid

def grid(x_size=2, y_size=2, x_count=2, y_count=2, clockwise=False, indexing='xy'):
    """ Grid

    Create a grid

    Arguments
    ---------
    - x_size (float=2) : size along x axis
    - y_size (float=2) : size along y axis
    - x_count (int=2) : number of vertices alongs x axis
    - y_count (int=2) : number of vertices along y axis
    - clockwise (bool = False) : faces orientation
    - indexing (str = 'ij') : indexing type in ('xy', 'ij') as for numpy.meshgrid

    Returns
    -------
    - dict : keys = 'verts', 'corners', 'sizes', 'UVMap'
    """

    x_count = max(x_count, 2)
    y_count = max(y_count, 2)

    # ----- Vertices

    verts = np.stack(np.meshgrid(np.linspace(-x_size/2, x_size/2, x_count, endpoint=True), np.linspace(-y_size/2, y_size/2, y_count, endpoint=True), indexing=indexing), axis=-1)
    shape = np.shape(verts)
    verts = np.append(verts, np.zeros(shape[:-1] + (1,), float), axis=-1)

    #verts = np.zeros((y_count, x_count, 3), float)
    #verts[..., 0] = np.linspace(-x_size/2, x_size/2, x_count)
    #verts[..., 1] = np.linspace(-y_size/2, y_size/2, y_count)[:, None]

    # ----- Corners and uvs

    comps = grid_uvs(shape[1], shape[0], clockwise=indexing=='ij', indexing=indexing)

    comps['verts'] = verts

    return comps

# ====================================================================================================
# Cap
# Inside an existing circle

def cap(circle_indices, offset_new=0, radius=1, x_count=None, y_count=1, fans=False, clockwise=False):

    # ----- For a disk, we create everything

    x_count = x_count if circle_indices is None else len(circle_indices)
    assert(x_count is not None)
    y_count = max(y_count, 1)

    # ----- Vertices

    ag    = np.linspace(0, 2*np.pi, x_count, endpoint=False)
    verts = np.stack((radius*np.cos(ag), radius*np.sin(ag), np.zeros_like(ag)), axis=-1)

    if y_count > 1:
        verts = np.resize(verts, (y_count, x_count, 3))
        verts *= np.linspace(1, 0, y_count, endpoint=False)[:, None, None]

    verts = np.reshape(verts, (np.size(verts)//3, 3))
    if fans:
        verts = np.append(verts, [(0, 0, 0)], axis=0)

    # ----- Corners and uvs

    if circle_indices is None:
        vert_indices = np.arange(len(verts))
    else:
        vert_indices = np.append(circle_indices, np.arange(len(verts) - len(circle_indices)) + offset_new)
    #if circle_indices is not None:
    #    vert_indices[:x_count] = circle_indices

    comps = disk_uvs(x_count, y_count, clockwise, fans=fans, vert_indices=vert_indices)

    if circle_indices is None:
        comps['verts'] = verts
    else:
        comps['verts'] = verts[x_count:]

    return comps


# ====================================================================================================
# Disk
# A cap without existing circle

def disk(radius=1, x_count=16, y_count=1, fans=False, clockwise=False):

    x_count = max(x_count, 2)
    y_count = max(y_count, 1)

    return cap(None, 0, radius=radius, x_count=x_count, y_count=y_count, fans=fans, clockwise=clockwise)

# ====================================================================================================
# Cone

def cone(radius=1, depth=2, x_count=16, y_count=1, z_count=2, caps='NGON'):

    x_count = max(x_count, 3)
    y_count = max(y_count, 1)
    z_count = max(z_count, 2)

    # ----- A cone is an "elevated disk"

    comps = disk(radius=radius, x_count=x_count, y_count=y_count, fans=True)
    comps['verts'][ -1, 2] =  depth/2

    if y_count == 1:
        comps['verts'][:-1, 2] = -depth/2
        nverts = len(comps['verts'])
    else:
        verts = np.reshape(comps['verts'][:-1], (y_count, x_count, 3))
        verts[..., 2] = np.linspace(-depth/2, depth/2, y_count, endpoint=False)[:, None]
        nverts = len(verts)

    # ----- Bottom cap

    if caps is not None and caps != 'NONE':
        bot_comp = cap(np.arange(x_count), nverts, radius=radius, y_count=y_count, fans=caps in ['FANS', 'TRIANGLES', 'TRIANGLE_FAN'], clockwise=True)
        bot_comp['verts'][..., 2] = -depth/2


        comps['verts']   = np.append(comps['verts'],   bot_comp['verts'],   axis=0)
        comps['corners'] = np.append(comps['corners'], bot_comp['corners'], axis=0)
        comps['sizes']   = np.append(comps['sizes'],   bot_comp['sizes'],   axis=0)

        comps['UVMap'] = Box(0, 0, .5, .5).resize(comps['UVMap'])
        comps['UVMap'] = np.append(comps['UVMap'], Box(.5, 0, 1, .5).resize(bot_comp['UVMap']), axis=0)

    return comps

# ====================================================================================================
# Cylinder

def cylinder(radius=1, radius1=None, depth=2, x_count=16, y_count=1, z_count=2, caps='NGON', verts=None):

    if radius1 == 0:
        return cone(radius=radius, depth=depth, x_count=x_count, y_count=y_count, z_count=z_count, caps=caps)

    x_count = max(x_count, 3)
    y_count = max(y_count, 1)
    z_count = max(z_count, 2)

    # ----- Body vertices and uvs

    if verts is None:
        ag    = np.linspace(0, 2*np.pi, x_count, endpoint=False)
        verts = np.resize(np.stack((np.cos(ag), np.sin(ag), np.zeros_like(ag)), axis=-1), (z_count, x_count, 3))
        verts[..., 2] = np.linspace(-depth/2, depth/2, z_count)[:, None]
        if radius1 is None:
            verts[..., :2] *= radius
        else:
            verts[..., :2] *= np.linspace(1, radius1/radius, z_count)[:, None, None]
    else:
        assert(np.shape(verts)==(z_count, x_count, 3))

    verts = np.reshape(verts, (np.size(verts)//3, 3))

    comps = grid_uvs(x_count, z_count, False, x_close=True)
    comps['corners'] = np.reshape(comps['corners'], np.size(comps['corners']))
    comps['UVMap']   = np.reshape(comps['UVMap'], (np.size(comps['UVMap'])//2, 2))

    # ----- Caps

    if isinstance(caps, tuple):
        bot_cap, top_cap = caps
    else:
        bot_cap, top_cap = caps, caps

    # Bot

    bot_comp, top_comp = None, None
    if bot_cap is not None and bot_cap != 'NONE':
        bot_comp = cap(np.arange(x_count), len(verts), radius=radius, y_count=y_count, fans=bot_cap in ['FANS', 'TRIANGLES', 'TRIANGLE_FAN'], clockwise=True)

        bot_comp['verts'][..., 2] = -depth/2
        verts = np.append(verts, bot_comp['verts'], axis=0)

        comps['corners'] = np.append(comps['corners'], bot_comp['corners'], axis=0)
        comps['sizes']   = np.append(comps['sizes'],   bot_comp['sizes'],   axis=0)

    # Top

    if top_cap is not None and top_cap != 'NONE':
        top_comp = cap(np.arange(x_count)+x_count*(z_count-1), len(verts), radius=radius, y_count=y_count, fans=top_cap in ['FANS', 'TRIANGLES', 'TRIANGLE_FAN'], clockwise=False)

        top_comp['verts'][..., 2] = depth/2
        verts = np.append(verts, top_comp['verts'], axis=0)

        comps['corners'] = np.append(comps['corners'], top_comp['corners'], axis=0)
        comps['sizes']   = np.append(comps['sizes'],   top_comp['sizes'],   axis=0)

    if bot_comp is not None or top_comp is not None:
        comps['UVMap'] = Box(0, .5, 1, 1).resize(comps['UVMap'])

        if bot_comp is not None:
            comps['UVMap'] = np.append(comps['UVMap'], Box(0, 0, .5, .5).resize(bot_comp['UVMap']), axis=0)
        if top_comp is not None:
            comps['UVMap'] = np.append(comps['UVMap'], Box(.5, 0, 1, .5).resize(top_comp['UVMap']), axis=0)

    comps['verts'] = verts

    return comps

# ====================================================================================================
# Torus

def torus(major_radius=1, minor_radius=.25, x_count=12, y_count=32, verts=None):

    # ----- Vertices

    if verts is None:
        ag = np.linspace(0, 2*np.pi, x_count, endpoint=False)
        prof = np.stack((minor_radius*np.cos(ag), np.zeros_like(ag), minor_radius*np.sin(ag)), axis=-1)
        prof[:, 0] += major_radius

        ag = np.linspace(0, 2*np.pi, y_count, endpoint=False)
        M = np.zeros((y_count, 3, 3), float)

        M[:, 0, 0] = np.cos(ag)
        M[:, 1, 1] = M[..., 0, 0]
        M[:, 0, 1] = np.sin(ag)
        M[:, 1, 0] = -M[:, 0, 1]
        M[:, 2, 2] = 1

        verts = np.einsum('...ij, ...j', M[:, None], prof)

    else:
        assert(np.shape(verts) == (y_count, x_count, 3))

    comps = grid_uvs(x_count, y_count, False, x_close=True, y_close=True)

    comps['verts'] = verts

    return comps

# ====================================================================================================
# UV sphere

def uv_sphere(radius=1, x_count=32, y_count=17):

    x_count = max(x_count, 3)
    y_count = max(y_count, 3)

    # ----- Body vertices and uvs

    ag    = np.linspace(0, 2*np.pi, x_count, endpoint=False)
    verts = np.resize(np.stack((np.cos(ag), np.sin(ag), np.zeros_like(ag)), axis=-1), (y_count, x_count, 3))


    ag = np.linspace(-np.pi/2, np.pi/2, y_count)
    z = radius*np.sin(ag)
    verts[..., :2] *= np.sqrt(radius*radius - z*z)[:, None, None]
    verts[..., 2] = z[:, None]

    # Remove the first and last circles and replace by two poles

    verts = np.append(np.reshape(verts[1:-1], (x_count*(y_count-2), 3)),  [(0, 0, -radius), (0, 0, radius)], axis=0)
    bot_index = x_count*(y_count-2)
    top_index = bot_index + 1

    # Corners

    if y_count == 3:
        comps = {'corners': np.empty(0, int), 'sizes': np.empty(0, int), 'UVMap': np.empty((0, 2), float)}

    else:
        comps = grid_uvs(x_count, y_count-2, False, x_close=True)

        comps['corners'] = np.reshape(comps['corners'], np.size(comps['corners']))
        comps['UVMap']   = np.reshape(comps['UVMap'], (np.size(comps['UVMap'])//2, 2))
        comps['UVMap']   = Box(0, 1/(y_count-1), 1, 1 - 1/(y_count-1)).resize(comps['UVMap'])

    # ----- Fans

    htris = 1/(y_count-1)

    bot_comp = triangles_uvs(x_count, clockwise=True, up=False, x_close=True, vert_indices=np.arange(x_count), top_index=bot_index)

    comps['corners'] = np.append(comps['corners'], bot_comp['corners'], axis=0)
    comps['sizes']   = np.append(comps['sizes'],   bot_comp['sizes'],   axis=0)
    comps['UVMap']   = np.append(comps['UVMap'], Box(0, 0, 1, htris).resize(bot_comp['UVMap']), axis=0)

    top_comp = triangles_uvs(x_count, clockwise=False, up=True, x_close=True, vert_indices=np.arange(x_count) + (y_count-3)*x_count, top_index=top_index)

    comps['corners'] = np.append(comps['corners'], top_comp['corners'], axis=0)
    comps['sizes']   = np.append(comps['sizes'],   top_comp['sizes'],   axis=0)
    comps['UVMap']   = np.append(comps['UVMap'], Box(0, 1-htris, 1, 1).resize(top_comp['UVMap']), axis=0)

    comps['verts'] = verts

    return comps



# ====================================================================================================
# Utilities

# ---------------------------------------------------------------------------
# Orientation

def orientation(uvs, sizes):

    if isinstance(sizes, int):
        sizes = np.ones(len(uvs)//sizes, int)*sizes

    ors = np.zeros(len(sizes), int)

    offset = 0
    for i, size in enumerate(sizes):
        #vs = a[1:] - a[:-1]
        vs = uvs[offset + 1:offset + size] - uvs[offset:offset + size - 1]
        f = np.sum(np.cross(vs[:-1], vs[1:]))
        ors[i] = 1 if f > 0 else -1

        offset += size

    return ors


# ====================================================================================================
# Debug / test

if __name__ == '__main__':

    # ---------------------------------------------------------------------------
    # Dump

    def dump(comps, title=None):
        if title is not None:
            print('-'*50)
            print(title)
            print()

            verts   = comps['verts']
            corners = comps['corners']
            sizes   = comps['sizes']
            uvs     = comps['UVMap']

            print(f"Verts:   {np.shape(verts)[:-1]}")
            print(f"Corners: {np.shape(corners)}, size: {np.size(corners)}, max: {np.max(corners)}, check: {np.max(corners)==np.size(verts)//3 - 1}")
            print(f"Sizes:   {len(sizes)}, sum: {np.sum(sizes)}, check: {np.sum(sizes) == np.size(corners)}")
            print(f"UVMap:   {np.shape(uvs)[:-1]}, size: {np.size(uvs)//2}, check: {np.size(uvs)//2 == np.size(corners)}")

            print()

    # ---------------------------------------------------------------------------
    # plot

    def plot(comps, title=""):

        title = f"{title} - {np.shape(comps['UVMap'])} {len(comps['sizes'])}"

        uvs     = comps['UVMap']
        uvs     = np.reshape(uvs, (np.size(uvs)//2, 2))
        sizes   = comps['sizes']
        corners = comps['corners']
        corners = np.reshape(corners, np.size(corners))

        halt = True
        if np.sum(sizes) != len(uvs):
            dump(comps, title)
            if halt:
                raise Exception(f"{title}\nIntegrity error: {np.sum(sizes)=} != {len(uvs)=}\n{sizes}")

        if len(corners) != len(uvs):
            dump(comps, title)
            if halt:
                raise Exception(f"{title}\nIntegrity error: {len(corners)=} != {len(uvs)=}\n{sizes}")

        import matplotlib.pyplot as plt

        print(f"Plot {title}")

        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        offsets = np.append([0], np.cumsum(sizes))
        ors = orientation(uvs, sizes)

        ax.plot((0, 1, 1, 0, 0), (0, 0, 1, 1, 0), 'b')

        for i in range(len(sizes)):

            face = uvs[offsets[i]: offsets[i] + sizes[i]]

            if ors[i] < 0:
                ax.fill(face[:, 0], face[:, 1], 'r')
            else:
                ax.fill(face[:, 0], face[:, 1], 'g')

            x = list(face[:, 0]) + [face[0, 0]]
            y = list(face[:, 1]) + [face[0, 1]]

            ax.plot(x, y, 'k')

            #break

        plt.title(title)

        plt.show()


    comp = grid(x_count=5, y_count=3, indexing='ij')
    plot(comp, "Grid ij")

    comp = grid(x_count=5, y_count=3, indexing='xy')
    plot(comp, "Grid xy")

    comp = disk(y_count=1, fans=False)
    plot(comp, "Disk 1 Plain")

    comp = disk(y_count=1, fans=True)
    plot(comp, "Disk 1 Fans")

    comp = disk(x_count=6, y_count=2, fans=False)
    plot(comp, "Disk 2 Plain")

    comp = disk(y_count=3, fans=True)
    plot(comp, "Disk 3 Fans")

    comp = cylinder(radius1=.8, y_count=2, z_count=4, caps=('FANS', 'NGON'))
    plot(comp, "Cylinder FANS NGON")

    comp = cylinder(radius1=.8, y_count=2, z_count=4, caps=None)
    plot(comp, "Cylinder NO CAP")

    comp = cone(z_count=7, caps=None)
    plot(comp, "Cone NO CAP")

    comp = cone(z_count=7, caps='FANS')
    plot(comp, "Cone FANS")

    comp = torus()
    plot(comp, "Torus")

    comp = uv_sphere(x_count=6, y_count=5)
    plot(comp, "UVSphere 5 x 6")

    comp = uv_sphere(x_count=6, y_count=3)
    plot(comp, "UVSphere 3 x 6")

    comp = uv_sphere(x_count=3, y_count=3)
    plot(comp, "UVSphere 3 x 3")


if False:
    nx = 7
    ny = 4


    g = grid(x_size=2, y_size=2, x_count=nx, y_count=ny, clockwise=False, indexing='ij')
    print(g['verts'].shape)
