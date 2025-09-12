__all__ = ["Formula"]

import numpy as np

from . import blender
from .constants import bfloat
from .mesh import Mesh
from .curve import Curve
from .textutils import parse_latex
from .maths import Rotation, maprange

# ====================================================================================================
# Formula item
# ====================================================================================================

class Item:
    def __init__(self, geometry, loc=None, center=None, scale=None, rot=None, margins=None, *attrs):

        self.geometry = geometry

        self._loc = loc
        self._center = center
        self._scale = scale
        self._margins = margins
        self.rot = rot

        self.update_bbox()

        self.attrs = dict(attrs)

    # ====================================================================================================
    # Properties
    # ====================================================================================================

    @property
    def transformed(self):

        geo = self.geometry.clone()
        margins = self.margins
        
        if self._scale is not None or self._rot is not None:
            if self._rot is None:
                rot = None
            else:
                rot = Rotation.from_euler([0, 0, self._rot])

            if self._scale is None:
                scale = None
            else:
                scale = self.scale
                margins = margins[0]*scale[0], margins[1]*scale[1], margins[2]*scale[0], margins[3]*scale[1]

            geo.transformation(self, rotation=rot, scale=scale, translation=self.loc, pivot=self.center)

        elif self._loc is not None:
            geo.points.translate(self.loc)

        # ----- Attributes

        for k, v in self.attrs.items():
            if 'faces' in geo.domain_names:
                geo.faces[k] = v
            elif 'splines' in geo.domaine_names:
                geo.faces[k] = v
            else:
                geo.points[k] = v

        # ----- Return as item for amrgin and true dims

        return Item(geo, margins=margins)
    
    # ====================================================================================================
    # Update bbox
    # ====================================================================================================

    def update_bbox(self):
        self.vmin, self.vmax = np.min(self.geometry.points.position, axis=0), np.max(self.geometry.points.position, axis=0)
        self.true_center = (self.vmin + self.vmax)/2
        self.true_box = self.vmax - self.vmax

    # ====================================================================================================
    # Properties
    # ====================================================================================================

    @staticmethod
    def _get_vect(attr, default):
        if attr is None:
            return default
        elif len(attr) == 2:
            return np.array([attr[0], attr[1], 0.])
        else:
            assert(len(attr) == 3)
            return np.array(attr)

    @property
    def loc(self):
        return self._get_vect(self._loc, [0., 0., 0.])
    
    @property
    def center(self):
        return self._get_vect(self._center, self.true_center)
    
    @property
    def scale(self):
        return self._get_vect(self._scale, [1., 1., 1.])
    
    @property
    def margins(self):
        return self._get_vect(self._margins, [0., 0., 0., 0.])

    @property
    def x0(self):
        return self.vmin[0] - self.margins[0]

    @property
    def y0(self):
        return self.vmin[1] - self.margins[1]

    @property
    def x1(self):
        return self.vmax[0] + self.margins[2]

    @property
    def y1(self):
        return self.vmax[1] + self.margins[3]
    
    @property
    def width(self):
        return self.vmax[0] - self.vmin[0]

    @property
    def height(self):
        return self.vmax[1] - self.vmin[1]
    
    # ====================================================================================================
    # Some useful operations
    # ====================================================================================================

    def x_align(self, x, align='left'):
        align = align.lower()
        delta = x - self.x0
        if align == 'center':
            delta -= self.width/2
        elif align == 'right':
            delta =- self.width
        
        self.geometry.points.x += delta
        self.vmin[0] += delta
        self.vmax[0] += delta


        return self

    def y_align(self, y, align='middle'):
        align = align.lower()
        delta = y - self.y0
        if align in ['center', 'middle']:
            delta -= self.height/2
        elif align == 'top':
            delta =- self.height
        
        self.geometry.points.y += delta
        self.vmin[1] += delta
        self.vmax[1] += delta

        return self

    def apply_scale(self, x_scale=1., y_scale=1.):
        scale = (x_scale, y_scale, 1.0)
        self.geometry.points.apply_scale(scale)
        self.update_bbox

        return self
    
    def set_height(self, height, mode='scale_y', down_scale=False):

        if height <= self.height and not down_scale:
            return self

        mode = mode.lower()

        if mode in ['scale', 'scale_y', 'scale_xy']:
            scale = height/self.height
            x_scale = scale if mode == 'scale_xy' else 1.
            return self.geometry.apply_scale(x_scale, height/self.height)
        
        points = self.geometry.points
        
        y = points.y
        y0, y1 = np.min(y), np.max(y)
        h = y1 - y0
        mrg = h*0.
        cy = (y0 + y1)/2
        
        dh = height - h
        dy = maprange(np.abs(y - cy), mrg, h/2 - mrg, 0, dh/2, mode='SMOOTH') * np.sign(y - cy)

        points.y += dy

        self.update_bbox()

        return self
    
    def set_width(self, width, mode='scale_x', down_scale=False):

        if width <= self.width and not down_scale:
            return self

        mode = mode.lower()

        scale = width/self.width
        y_scale = scale if mode == 'scale_xy' else 1.
        return self.geometry.apply_scale(scale, y_scale)




        



# ====================================================================================================
# Formula
# ====================================================================================================

class Formula:

    def __init__(self, *compos, block_type='CONTENT', name="Content", factor=1., **options):

        self.block_type = block_type
        self.name = name
        for k, v in options.items():
            self.set_prop(k, v)

        # Animation
        self.factor = 1.

        # Owner / content
        self.owner = None 
        self.content = []
        self.add_content(*compos)

    def __str__(self):
        return f"<Composition {self.name} ({self.block_type})>"
    
    

    


    
    # ====================================================================================================
    # Root initialisation
    # ====================================================================================================

    @classmethod
    def new(cls, font=None, italic=True, bold=False, color=(0, 0, 0), **options):

        root = cls(block_type='CONTENT', name="Root",
                font = Font() if font is None else font,
                italic=italic, bold=bold, color=color,
                **options)

        # ----- Font dimensions

        txt = Text("a", font=root.font)
        x0, y0, x1, y1 = root.bbox(txt.to_mesh())
        root._x_space = x1 - x0
        root._x_sepa = root._x_space*0.3
        root._y_a = y1

        txt = Text("Ap", font=root.font)
        x0, y0, x1, y1 = root.bbox(txt.to_mesh())
        root._y_height = y1
        root._y_depth = -y0

        # ----- Done

        return root
    
    # ----------------------------------------------------------------------------------------------------
    # Add a sub content
    # ----------------------------------------------------------------------------------------------------

    def add_child(self, child, to_content=False):
        if child is None:
            return None
        elif isinstance(child, str):
            child = Composition.new_string(child)
        elif isinstance(child, Mesh):
            child = Composition.new_mesh(child)
        
        child.owner = self
        if to_content:
            self.content.append(child)
        return child
    
    def add_content(self, *compos):
        for compo in compos:
            self.add_child(compo, to_content=True)

    # ----------------------------------------------------------------------------------------------------
    # Get / set a property
    # ----------------------------------------------------------------------------------------------------

    def get_prop(self, name, default=None):
        _name = f"_{name}"
        if _name in self.__dict__:
            return self.__dict__[_name]
        
        if self.owner is None:
            return default
        else:
            return self.owner.get_prop(name, default)

    def set_prop(self, name, value):
        setattr(self, f"_{name}", value)

    def __getattr__(self, name):
        val = self.get_prop(name, "NO WAY")
        if val == "NO WAY":
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        else:
            return val

    # ====================================================================================================
    # Initialization
    # ====================================================================================================

    def append(self, *compos):
        if self.block_type == 'CONTENT':
            self.add_content(*compos)
            return self
        else:
            return Composition(self, *compos)

    @classmethod
    def new_mesh(cls, mesh, name="Mesh", **options):
        compo = cls(block_type='MESH', name=name, **options)
        compo.mesh = mesh
        return compo

    @classmethod
    def new_string(cls, string, name="Text", **options):
        compo = cls(block_type='STRING', name=name, **options)
        compo.string = string
        return compo

    @classmethod
    def new_decorator(cls, content, decorator, name=None, **options):
        if name is None:
            name = decorator.title()

        compo = cls(content, block_type='DECORATOR', name=name, **options)
        compo.deco = decorator
        
        return compo
    
    @classmethod
    def new_sqrt(cls, *compos, name="Sqrt", **options):
        compo = cls(*compos, block_type='SQRT', name=name, **options)
        return compo
    
    @classmethod
    def new_script(cls, content, script, location='SUB', name=None, merge=True, **options):

        if merge and content.block_type == 'SCRIPT':
            if content.scripts.get(location) is None:
                content.scripts[location] = content.add_child(script)
                return content

        compo = cls(content, block_type='SCRIPT', name=name, **options)
        compo.scripts = {location.upper(): compo.add_child(script)}
        return compo
    
    @classmethod
    def new_ind_exp(cls, content, subscript, superscript, name="Ind exp", **options):
        compo = cls(content, block_type='SCRIPT', name=name, **options)
        compo.scripts = {}
        if subscript is not None:
            compo.scripts['SUB'] = compo.add_child(subscript)
        if superscript is not None:
            compo.scripts['SUP'] = compo.add_child(superscript)
        return compo
    
    @classmethod
    def new_integral(cls, content, subscript=None, superscript=None, sigma=False, name=None, **options):
        if name is None:
            name = "Sigma" if sigma else "Integral"
            
        compo = cls(content, block_type='INTEGRAL', name=name, **options)
        compo.is_sigma = sigma
        compo.scripts = {}

        if subscript is not None:
            subscript = compo.add_child(subscript)
            if sigma:
                compo.scripts['BOT'] = subscript
            else:
                compo.scripts['SUB'] = subscript

        if superscript is not None:
            superscript = compo.add_child(superscript)
            if sigma:
                compo.scripts['TOP'] = superscript
            else:
                compo.scripts['SUP'] = superscript

        return compo
    
    @classmethod
    def new_fraction(cls, numerator, denominator, name="Fraction", **options):
        compo = cls(numerator, block_type='FRACTION', name=name, **options)
        compo.denominator = compo.add_child(denominator)
        return compo
    
    # ====================================================================================================
    # Utility
    # ====================================================================================================

    @staticmethod
    def bbox(mesh):
        if mesh is None or len(mesh.points) == 0:
            return (0.0, 0.0, 0.0, 0.0)

        vmin, vmax = np.min(mesh.points.position, axis=0), np.max(mesh.points.position, axis=0)

        x0, y0 = vmin[:2]
        x1, y1 = vmax[:2]

        return (x0, y0, x1, y1)

    # ====================================================================================================
    # Decorators
    # ====================================================================================================

    @staticmethod
    def bar_mesh(x0, y0, x1, y1, width=.1):
        prof = Curve.line((width/2, 0, 0), (-width/2, 0, 0))
        return Curve.line((x0, y0, 0), (x1, y1, 0)).to_mesh(profile=prof)
    
    @staticmethod
    def arrow_mesh(x0, y0, x1, y1, width=.1, invert=False):
        dx0, dx1 = 4*width, 7*width
        dy = 2*width
        
        xs = x0 - width
        xm = x1 + 2*width

        dw = width/2

        points = [
            [xs, y0-dw, 0], [xm - dx0, y0-dw, 0], [xm - dx1, y0 - dy, 0],
            [xm, y0, 0],
            [xm - dx1, y0 + dy, 0], [xm - dx0, y0 + dw, 0], [xs, y0 + dw, 0],
            ]
        if invert:
            points = np.array(points)
            points[:, 0] = x0 + x1 - points[:, 0]
            points = np.flip(points, axis=0)

        return Mesh(points=points, corners=np.arange(len(points)))
    
    @staticmethod
    def point_mesh(x0, x1, y, radius=.1, count=1):

        mesh = Mesh.disk(radius=radius, segments=16)*count

        x = (x0 + x1)/2
        if count <= 1:
            mesh.points.position += (x, y, 0)
            
        else: 
            tr = np.zeros((count, 3), dtype=bfloat)
            dx = radius*(count-1)*1.2
            tr[:, 0] = np.linspace(x - dx, x + dx, count)
            tr[:, 1] = y

            mesh.points.translate(tr)

        return mesh
    
    @staticmethod
    def block_mesh(deco, y0, y1, margin, width=.1):

        y0 -= margin
        y1 += margin

        # ---------------------------------------------------------------------------

        if deco == '||':
            return Mesh(points=[
                [-width/2, y0, 0], [width/2, y0, 0], [width/2, y1, 0], [-width/2, y1, 0],
                ], corners=[0, 1, 2, 3])

        # ---------------------------------------------------------------------------

        elif deco == '‖‖':
            mesh = Mesh(points=[
                [-width/2, y0, 0], [width/2, y0, 0], [width/2, y1, 0], [-width/2, y1, 0],
                ], corners=[0, 1, 2, 3])*2
            mesh.points.translate([[0, 0, 0], [width*1.8, 0, 0]])
            return mesh

        # ---------------------------------------------------------------------------

        elif deco == '[]':
            w = width
            cr = w*3
            pts = [
                [0, y0, 0], [cr, y0, 0], [cr, y0+w, 0], [w, y0+w, 0],
                [w, y1-w, 0], [cr, y1-w, 0], [cr, y1, 0], [0, y1, 0],
                ]
            return Mesh(points=pts, corners=np.arange(len(pts)))

        # ---------------------------------------------------------------------------

        elif deco == '()':
            w = width*3.5
            y = (y0 + y1)/2

            h = (y1 - y0)/6
            tx, ty = w*.6, y*.2

            pts = np.array([[w, y0, 0], [0, y, 0], [w, y1, 0]])

            lhs = np.array(pts)
            lhs[0] += (tx, -ty, 0)
            lhs[1] += (0, -h, 0)
            lhs[2] += (-tx, -ty, 0)

            rhs = np.array(pts)
            rhs[0] += (-tx, ty, 0)
            rhs[1] += (0, h, 0)
            rhs[2] += (tx, ty, 0)

            curve = Curve(
                points=pts, 
                curve_type='BEZIER', 
                radius=[1, 2.5, 1],
                handle_left = lhs,
                handle_right = rhs,
                )
            prof = Curve.line((width/4, 0, 0), (-width/4, 0, 0))
            return curve.to_mesh(profile=prof)

        # ---------------------------------------------------------------------------

        elif deco == '{}':
            # Used for thickness
            dx, dy = width*1., width*1.3

            # y mid
            y = (y0 + y1)/2

            # Height
            h = (y1 - y0)/2

            # Shape thickness
            w = width*3

            # Handles
            tx = w*.5
            mx = tx*2

            # Bottom half
            curve = Curve(
                points = [[  w, -h, 0], [ 0, 0, 0]], 
                curve_type='BEZIER', 
                radius=[1, 2.5],
                handle_left = [[ tx, -h, 0], [mx, 0, 0]],
                handle_right = [[-tx, -h, 0], [mx, 0, 0]],
                )
            
            # Make thick
            curve = curve.to_poly(resolution=20)
            pts = curve.points.position
            xpts = pts + (dx, 0, 0)
            ypts = pts + (0, dy, 0)
            n = len(pts)
            p = maprange(np.arange(n), 0, n-1, 0, 1, mode='LINEAR')[:, None]

            other_pts = p*xpts + (1 - p)*ypts
            pts = np.append(np.flip(pts, axis=0), other_pts, axis=0)
            pts = np.append(pts, np.flip(pts*(1, -1, 1), axis=0)[1:-1], axis=0)

            pts[:, 1] += y

            mesh = Mesh(points=pts, corners=np.arange(len(pts)))

            return mesh

        # ---------------------------------------------------------------------------

        elif deco == 'INTEGRAL':

            width *= 2
            y0 -= 4*width
            y1 += 4*width

            # y mid
            y = (y0 + y1)/2

            # Height
            h = (y1 - y0)/2

            ix, iy = width*3, h
            tx0, ty0 = width/5, width
            tx1, ty1 = 3*width, 0 #width*.2

            w = width*.7
            ixd = ix + width
            tx1d = tx1 + width

            curve = Curve(
                points = [[  -ix, -iy, 0], [ 0, 0, 0], [ixd, iy, 0]], 
                curve_type='BEZIER', 
                handle_left = [[ -ix - tx1, -iy - ty1, 0], [-tx0, -ty0, 0], [ixd - tx1d, iy - ty1, 0]],
                handle_right = [[-ix + tx1, -iy + ty1, 0], [tx0, ty0, 0], [ixd + tx1d, iy + ty1, 0]],
                )

            curve = curve.to_poly()
            pts = curve.points.position
            pts1 = pts*(-1, -1, 0) + (w, -w, 0)
            pts = np.flip(np.append(pts, pts1, axis=0), axis=0)
            pts[:, 1] += y

            mesh = Mesh(points=pts, corners=np.arange(len(pts)))

            return mesh

        # ---------------------------------------------------------------------------

        elif deco == 'SIGMA':

            w = width

            # Length and height
            l, h = .6, y1 - y0
            d = 1.5*w

            # Points
            px0, py0 = l, 0
            px1, py1 = 0, 0
            px2, py2 = l/2, h/2
            px3, py3 = 0, h
            px4, py4 = l, h

            pts = [[px0, py0 + 2*d, 0], [px0, py0, 0], [px1, py1, 0], [px2, py2, 0], [px3, py3, 0], [px4, py4, 0], [px4, py4 - 1.5*d, 0]]
            sigma = Curve(pts).set_spline2d_thickness(
                thickness = [w, 2*w, w, 2*w, w, w, w],
                offset = -1,
                mode = [0, 0, 3, 0, 3, 0, 0], 
                inner_mode = [0, 0, 0, 0, 0, 0, 0], 
                factor = 1.,
                cuts = [(-w*.5, np.pi/2), (w*.5, np.pi/2)],
                start_thickness=.3,
                end_thickness=.3)
            
            return sigma

        # ---------------------------------------------------------------------------

        else:
            raise ValueError(f"Invalid block code: '{deco}'.")
        
        # ---------------------------------------------------------------------------
        
    # ====================================================================================================
    # Place scritps
    # ====================================================================================================

    def add_scripts(self, mesh, **scripts):

        scale = self.get_prop('ssscript_scale', .35)
        x_margin = self.get_prop('x_margin', .1)
        y_margin = self.get_prop('y_margin', .15)
        y_ssscript = self.get_prop('y_ssscript', .1)

        x0, y0, x1, y1 = self.bbox(mesh)

        pos = mesh.points.position

        for location, script in scripts.items():

            # script to mesh
            smesh = script.get_mesh()

            smesh.points.position *= scale
            sx0, sy0, sx1, sy1 = self.bbox(smesh)
            sl, ml = sx1 - sx0, x1 - x0

            # Depending on the location

            if location in ['BOT', 'BOTTOM', 'BELOW']:
                smesh.points.x += x0 - sx0 + (ml - sl)/2
                smesh.points.y += y0 - sy1 - y_margin/2

            elif location in ['TOP', 'ABOVE']:
                smesh.points.x += x0 - sx0 + (ml - sl)/2
                smesh.points.y += y1 - sy0 + y_margin/2

            elif location in ['SUB', 'SUBSCRIPT']:
                y = y0 - y_ssscript
                pts = pos[pos[:, 1] < y + sy1 - sy0 + 2*y_margin]

                xm = np.max(pts[:, 0])

                smesh.points.x += xm - sx0 + x_margin
                smesh.points.y += y - sy0

            elif location in ['SUP', 'SUPERSCRIPT']:
                y = y1 - y_ssscript
                pts = pos[pos[:, 1] > y - 2*y_margin]

                xm = np.max(pts[:, 0])

                smesh.points.x += xm - sx0 + x_margin
                smesh.points.y += y - sy0

            else:
                raise ValueError(f"Invalid location code: '{location}'.")

            mesh.join(smesh)

        return mesh
    
    # ====================================================================================================
    # Variable height characters
    # ====================================================================================================

    def get_var_height_char(self, char, ychar0, ychar1):

        VALIDS = "[](){}<>⟨⟩|‖"
        if not char in VALIDS:
            raise ValueError(f"Character '{char}' ha sno variable height, only {VALIDS} are accepted.")

        # ---------------------------------------------------------------------------
        # Get the template

        if not hasattr(self,'_var_height_chars'):
            txt = Text(VALIDS).to_mesh(transform=False, char_index=True)

            self._var_height_chars = {}
            for i, c in enumerate(VALIDS):
                self._var_height_chars[c] = txt.extract_from_faces(txt.faces.char_index == i)

        mesh = Mesh.from_mesh(self._var_height_chars[char])

        height = ychar1 - ychar0

        y = mesh.points.y
        y0, y1 = np.min(y), np.max(y)
        h = y1 - y0
        if h >= height:
            mesh.points.y += ychar0 - y0
            return mesh
        
        margin = h*0.
        cy = (y0 + y1)/2
        
        dh = height - h
        rel_y = y - cy
        dy = maprange(np.abs(rel_y), margin, h/2 - margin, 0, dh/2, mode='SMOOTHER') * np.sign(rel_y)
        
        mesh.points.y += dy -y0 + ychar0

        return mesh


    # ====================================================================================================
    # Get the geometry
    # ====================================================================================================

    def get_mesh(self):

        # ---------------------------------------------------------------------------
        # Most common values
        # ---------------------------------------------------------------------------

        bar_thick = self.get_prop('bar_thick', .05)
        x_margin = self.get_prop('x_margin', .1)
        y_margin = self.get_prop('y_margin', .15)
        x_sepa = self.get_prop('x_sepa', .04)

        # ---------------------------------------------------------------------------
        # Join the compos in content
        # ---------------------------------------------------------------------------

        mesh = Mesh()
        x = 0.
        for compo in self.content:
            msh = compo.get_mesh()
            mx = np.max(msh.points.x)
            msh.points.x += x
            x += mx + x_sepa
            mesh.join(msh)

        x0, y0, x1, y1 = self.bbox(mesh)

        # ---------------------------------------------------------------------------
        # Content
        # ---------------------------------------------------------------------------

        if self.block_type == 'CONTENT':
            pass

        # ---------------------------------------------------------------------------
        # A Mesh
        # ---------------------------------------------------------------------------

        elif self.block_type == 'MESH':
            mesh = Mesh.from_mesh(self.mesh)

        # ---------------------------------------------------------------------------
        # A String
        # ---------------------------------------------------------------------------

        elif self.block_type == 'STRING':
            txt = Text(self.string, font=self.get_prop('font'), italic=self.italic, bold=self.bold)
            mesh = txt.to_mesh(transform=False)

        # ---------------------------------------------------------------------------
        # Decorated block
        # ---------------------------------------------------------------------------

        elif self.block_type == 'DECORATOR':

            if self.deco in ['BAR', 'ARROW', 'POINT']:
                y = y0 - y_margin if self.get_prop('bottom', False) else y1 + y_margin
                invert = self.get_prop('invert', False)
                count = self.get_prop('count', 1)

                if self.deco == 'BAR':
                    deco = self.bar_mesh(x0, y, x1, y, width=bar_thick)
                elif self.deco == 'ARROW':
                    deco = self.arrow_mesh(x0, y, x1, y, width=bar_thick, invert=invert)
                elif self.deco == 'POINT':
                    deco = self.point_mesh(x0, x1, y, radius=bar_thick, count=count)

            elif self.deco in ['||', '()', '[]', '{}', '‖‖']:


                if True:
                    left_deco = self.get_var_height_char(self.deco[0], y0, y1)
                    right_deco = self.get_var_height_char(self.deco[1], y0, y1)

                else:
                    left_deco = self.block_mesh(self.deco, y0, y1, margin=y_margin, width=bar_thick)
                    right_deco = left_deco.symmetrical(x=-1)
                
                left_only = self.get_prop('left_only', False)
                right_only = self.get_prop('right_only', False)
                deco = None

                if not right_only:
                    dx0, _, dx1, _ = self.bbox(left_deco)
                    dw = dx1 - dx0
                    mesh.points.x += dw + x_margin
                    x0, y0, x1, y1 = self.bbox(mesh)

                    left_deco.points.x -= dx0
                    deco = left_deco

                if not left_only:
                    dx0, _, dx1, _ = self.bbox(right_deco)
                    right_deco.points.x += x1 + x_margin - dx0
                    if deco is None:
                        deco = right_deco
                    else:
                        deco.join(right_deco)

            else:
                raise Exception(f"Unknown decorator: '{self.deco}'.")

            mesh.join(deco)

        # ---------------------------------------------------------------------------
        # Square Root
        # ---------------------------------------------------------------------------

        elif self.block_type == 'SQRT':

            w = bar_thick
            dx, dy = w, 2.1*w
            # Length and height
            l, h = x1 - x0 + x_margin, y1 - y0 + y_margin

            # Points
            px0, py0 = -(dy + 6*dx + x_margin), 2*dy
            px1, py1 = px0 + dy, py0 + dx
            px2, py2 = px1 + 3*dx, py1 - 3*dy
            px3, py3 = px2 + 3*dx, h
            px4, py4 = l, h
            px5, py5 = l, h - 2*w

            pts = [[px0, py0, 0], [px1, py1, 0], [px2, py2, 0], [px3, py3, 0], [px4, py4, 0], [px5, py5, 0]]
            sqrt_mesh = Curve(pts).set_spline2d_thickness(
                thickness=[.8*w, 1.8*w, .9*w, w, w, w/2], 
                mode=0, 
                inner_mode=[0, 0, 1, 0, 0, 0], 
                factor=.8,
                end_thickness=0)
            
            sqrt_mesh.points.y += y0
            mesh.points.x -= x0
            mesh.join(sqrt_mesh)
            mesh.points.x -= px0

        # ---------------------------------------------------------------------------
        # Subsscript / superscript
        # ---------------------------------------------------------------------------

        elif self.block_type == 'SCRIPT':
            mesh = self.add_scripts(mesh, **self.scripts)

        # ---------------------------------------------------------------------------
        # Integral / sigma
        # ---------------------------------------------------------------------------

        elif self.block_type == 'INTEGRAL':
            count = self.get_prop('count', 1)

            if self.is_sigma:
                symb = self.block_mesh('SIGMA', y0, y1, margin=y_margin, width=bar_thick)
            else:
                symb = self.block_mesh('INTEGRAL', y0, y1, margin=y_margin, width=bar_thick)
                symb = symb.duplicate(count=count, offset=(.25, 0, 0), relative=False)

            sx0, sy0, sx1, sy1 = self.bbox(symb)
            sl = sy1 - sy0
            symb.points.y += -sl/2 - sy0 + y1/2 

            symb = self.add_scripts(symb, **self.scripts)
            sx0, sy0, sx1, sy1 = self.bbox(symb)

            symb.points.x -=  sx0
            mesh.points.x += sx1 - sx0 + x_margin - x0
            
            mesh.join(symb)

        # ---------------------------------------------------------------------------
        # Fraction
        # ---------------------------------------------------------------------------

        elif self.block_type == 'FRACTION':
            num_mesh = mesh
            nx0, ny0, nx1, ny1 = x0, y0, x1, y1
            nl = nx1 - nx0

            den_mesh = self.denominator.get_mesh()
            dx0, dy0, dx1, dy1 = self.bbox(den_mesh)
            dl = dx1 - dx0

            l = max(nl, dl) + 2*x_margin

            mesh = self.bar_mesh(0, 0, l, 0, width=1.5*bar_thick)

            num_mesh.points.y += y_margin + bar_thick - ny0
            den_mesh.points.y -= y_margin + bar_thick + dy1

            num_mesh.points.x += (l - nl)/2 - nx0
            den_mesh.points.x += (l - dl)/2 - dx0

            mesh.join(num_mesh, den_mesh)

        # ---------------------------------------------------------------------------
        # Error
        # ---------------------------------------------------------------------------

        else:
            raise Exception(f"Unknown block type: '{self.block_type}'.")

        return mesh



    
    
    


    







# ====================================================================================================
# Formula OLD
# ====================================================================================================

class Formula_OLD:
    """ A formula encapsulates geometries organized according commands
    """

    HRZ_SEPA = .1 # Separation between contiguous compos
    HRZ_SPACE = .4 # Space char

    def __init__(self, *compos, block_type='CONTENT', name="Content", factor=1., **options):

        self.block_type = block_type
        self.name = name
        for k, v in options.items():
            self.set_prop(k, v)

        # Animation
        self.factor = 1.

        # Owner / content
        self.owner = None 
        self.content = []
        self.add_content(*compos)

    def __str__(self):
        return f"<Composition {self.name} ({self.block_type})>"
    
    # ====================================================================================================
    # Root initialisation
    # ====================================================================================================

    @classmethod
    def new(cls, font=None, italic=True, bold=False, color=(0, 0, 0), **options):

        root = cls(block_type='CONTENT', name="Root",
                font = Font() if font is None else font,
                italic=italic, bold=bold, color=color,
                **options)

        # ----- Font dimensions

        txt = Text("a", font=root.font)
        x0, y0, x1, y1 = root.bbox(txt.to_mesh())
        root._x_space = x1 - x0
        root._x_sepa = root._x_space*0.3
        root._y_a = y1

        txt = Text("Ap", font=root.font)
        x0, y0, x1, y1 = root.bbox(txt.to_mesh())
        root._y_height = y1
        root._y_depth = -y0

        # ----- Done

        return root
    
    # ----------------------------------------------------------------------------------------------------
    # Add a sub content
    # ----------------------------------------------------------------------------------------------------

    def add_child(self, child, to_content=False):
        if child is None:
            return None
        elif isinstance(child, str):
            child = Composition.new_string(child)
        elif isinstance(child, Mesh):
            child = Composition.new_mesh(child)
        
        child.owner = self
        if to_content:
            self.content.append(child)
        return child
    
    def add_content(self, *compos):
        for compo in compos:
            self.add_child(compo, to_content=True)

    # ----------------------------------------------------------------------------------------------------
    # Get / set a property
    # ----------------------------------------------------------------------------------------------------

    def get_prop(self, name, default=None):
        _name = f"_{name}"
        if _name in self.__dict__:
            return self.__dict__[_name]
        
        if self.owner is None:
            return default
        else:
            return self.owner.get_prop(name, default)

    def set_prop(self, name, value):
        setattr(self, f"_{name}", value)

    def __getattr__(self, name):
        val = self.get_prop(name, "NO WAY")
        if val == "NO WAY":
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        else:
            return val

    # ====================================================================================================
    # Initialization
    # ====================================================================================================

    def append(self, *compos):
        if self.block_type == 'CONTENT':
            self.add_content(*compos)
            return self
        else:
            return Composition(self, *compos)

    @classmethod
    def new_mesh(cls, mesh, name="Mesh", **options):
        compo = cls(block_type='MESH', name=name, **options)
        compo.mesh = mesh
        return compo

    @classmethod
    def new_string(cls, string, name="Text", **options):
        compo = cls(block_type='STRING', name=name, **options)
        compo.string = string
        return compo

    @classmethod
    def new_decorator(cls, content, decorator, name=None, **options):
        if name is None:
            name = decorator.title()

        compo = cls(content, block_type='DECORATOR', name=name, **options)
        compo.deco = decorator
        
        return compo
    
    @classmethod
    def new_sqrt(cls, *compos, name="Sqrt", **options):
        compo = cls(*compos, block_type='SQRT', name=name, **options)
        return compo
    
    @classmethod
    def new_script(cls, content, script, location='SUB', name=None, merge=True, **options):

        if merge and content.block_type == 'SCRIPT':
            if content.scripts.get(location) is None:
                content.scripts[location] = content.add_child(script)
                return content

        compo = cls(content, block_type='SCRIPT', name=name, **options)
        compo.scripts = {location.upper(): compo.add_child(script)}
        return compo
    
    @classmethod
    def new_ind_exp(cls, content, subscript, superscript, name="Ind exp", **options):
        compo = cls(content, block_type='SCRIPT', name=name, **options)
        compo.scripts = {}
        if subscript is not None:
            compo.scripts['SUB'] = compo.add_child(subscript)
        if superscript is not None:
            compo.scripts['SUP'] = compo.add_child(superscript)
        return compo
    
    @classmethod
    def new_integral(cls, content, subscript=None, superscript=None, sigma=False, name=None, **options):
        if name is None:
            name = "Sigma" if sigma else "Integral"
            
        compo = cls(content, block_type='INTEGRAL', name=name, **options)
        compo.is_sigma = sigma
        compo.scripts = {}

        if subscript is not None:
            subscript = compo.add_child(subscript)
            if sigma:
                compo.scripts['BOT'] = subscript
            else:
                compo.scripts['SUB'] = subscript

        if superscript is not None:
            superscript = compo.add_child(superscript)
            if sigma:
                compo.scripts['TOP'] = superscript
            else:
                compo.scripts['SUP'] = superscript

        return compo
    
    @classmethod
    def new_fraction(cls, numerator, denominator, name="Fraction", **options):
        compo = cls(numerator, block_type='FRACTION', name=name, **options)
        compo.denominator = compo.add_child(denominator)
        return compo
    
    # ====================================================================================================
    # Utility
    # ====================================================================================================

    @staticmethod
    def bbox(mesh):
        if mesh is None or len(mesh.points) == 0:
            return (0.0, 0.0, 0.0, 0.0)

        vmin, vmax = np.min(mesh.points.position, axis=0), np.max(mesh.points.position, axis=0)

        x0, y0 = vmin[:2]
        x1, y1 = vmax[:2]

        return (x0, y0, x1, y1)

    # ====================================================================================================
    # Decorators
    # ====================================================================================================

    @staticmethod
    def bar_mesh(x0, y0, x1, y1, width=.1):
        prof = Curve.line((width/2, 0, 0), (-width/2, 0, 0))
        return Curve.line((x0, y0, 0), (x1, y1, 0)).to_mesh(profile=prof)
    
    @staticmethod
    def arrow_mesh(x0, y0, x1, y1, width=.1, invert=False):
        dx0, dx1 = 4*width, 7*width
        dy = 2*width
        
        xs = x0 - width
        xm = x1 + 2*width

        dw = width/2

        points = [
            [xs, y0-dw, 0], [xm - dx0, y0-dw, 0], [xm - dx1, y0 - dy, 0],
            [xm, y0, 0],
            [xm - dx1, y0 + dy, 0], [xm - dx0, y0 + dw, 0], [xs, y0 + dw, 0],
            ]
        if invert:
            points = np.array(points)
            points[:, 0] = x0 + x1 - points[:, 0]
            points = np.flip(points, axis=0)

        return Mesh(points=points, corners=np.arange(len(points)))
    
    @staticmethod
    def point_mesh(x0, x1, y, radius=.1, count=1):

        mesh = Mesh.disk(radius=radius, segments=16)*count

        x = (x0 + x1)/2
        if count <= 1:
            mesh.points.position += (x, y, 0)
            
        else: 
            tr = np.zeros((count, 3), dtype=bfloat)
            dx = radius*(count-1)*1.2
            tr[:, 0] = np.linspace(x - dx, x + dx, count)
            tr[:, 1] = y

            mesh.points.translate(tr)

        return mesh
    
    @staticmethod
    def block_mesh(deco, y0, y1, margin, width=.1):

        y0 -= margin
        y1 += margin

        # ---------------------------------------------------------------------------

        if deco == '||':
            return Mesh(points=[
                [-width/2, y0, 0], [width/2, y0, 0], [width/2, y1, 0], [-width/2, y1, 0],
                ], corners=[0, 1, 2, 3])

        # ---------------------------------------------------------------------------

        elif deco == '‖‖':
            mesh = Mesh(points=[
                [-width/2, y0, 0], [width/2, y0, 0], [width/2, y1, 0], [-width/2, y1, 0],
                ], corners=[0, 1, 2, 3])*2
            mesh.points.translate([[0, 0, 0], [width*1.8, 0, 0]])
            return mesh

        # ---------------------------------------------------------------------------

        elif deco == '[]':
            w = width
            cr = w*3
            pts = [
                [0, y0, 0], [cr, y0, 0], [cr, y0+w, 0], [w, y0+w, 0],
                [w, y1-w, 0], [cr, y1-w, 0], [cr, y1, 0], [0, y1, 0],
                ]
            return Mesh(points=pts, corners=np.arange(len(pts)))

        # ---------------------------------------------------------------------------

        elif deco == '()':
            w = width*3.5
            y = (y0 + y1)/2

            h = (y1 - y0)/6
            tx, ty = w*.6, y*.2

            pts = np.array([[w, y0, 0], [0, y, 0], [w, y1, 0]])

            lhs = np.array(pts)
            lhs[0] += (tx, -ty, 0)
            lhs[1] += (0, -h, 0)
            lhs[2] += (-tx, -ty, 0)

            rhs = np.array(pts)
            rhs[0] += (-tx, ty, 0)
            rhs[1] += (0, h, 0)
            rhs[2] += (tx, ty, 0)

            curve = Curve(
                points=pts, 
                curve_type='BEZIER', 
                radius=[1, 2.5, 1],
                handle_left = lhs,
                handle_right = rhs,
                )
            prof = Curve.line((width/4, 0, 0), (-width/4, 0, 0))
            return curve.to_mesh(profile=prof)

        # ---------------------------------------------------------------------------

        elif deco == '{}':
            # Used for thickness
            dx, dy = width*1., width*1.3

            # y mid
            y = (y0 + y1)/2

            # Height
            h = (y1 - y0)/2

            # Shape thickness
            w = width*3

            # Handles
            tx = w*.5
            mx = tx*2

            # Bottom half
            curve = Curve(
                points = [[  w, -h, 0], [ 0, 0, 0]], 
                curve_type='BEZIER', 
                radius=[1, 2.5],
                handle_left = [[ tx, -h, 0], [mx, 0, 0]],
                handle_right = [[-tx, -h, 0], [mx, 0, 0]],
                )
            
            # Make thick
            curve = curve.to_poly(resolution=20)
            pts = curve.points.position
            xpts = pts + (dx, 0, 0)
            ypts = pts + (0, dy, 0)
            n = len(pts)
            p = maprange(np.arange(n), 0, n-1, 0, 1, mode='LINEAR')[:, None]

            other_pts = p*xpts + (1 - p)*ypts
            pts = np.append(np.flip(pts, axis=0), other_pts, axis=0)
            pts = np.append(pts, np.flip(pts*(1, -1, 1), axis=0)[1:-1], axis=0)

            pts[:, 1] += y

            mesh = Mesh(points=pts, corners=np.arange(len(pts)))

            return mesh

        # ---------------------------------------------------------------------------

        elif deco == 'INTEGRAL':

            width *= 2
            y0 -= 4*width
            y1 += 4*width

            # y mid
            y = (y0 + y1)/2

            # Height
            h = (y1 - y0)/2

            ix, iy = width*3, h
            tx0, ty0 = width/5, width
            tx1, ty1 = 3*width, 0 #width*.2

            w = width*.7
            ixd = ix + width
            tx1d = tx1 + width

            curve = Curve(
                points = [[  -ix, -iy, 0], [ 0, 0, 0], [ixd, iy, 0]], 
                curve_type='BEZIER', 
                handle_left = [[ -ix - tx1, -iy - ty1, 0], [-tx0, -ty0, 0], [ixd - tx1d, iy - ty1, 0]],
                handle_right = [[-ix + tx1, -iy + ty1, 0], [tx0, ty0, 0], [ixd + tx1d, iy + ty1, 0]],
                )

            curve = curve.to_poly()
            pts = curve.points.position
            pts1 = pts*(-1, -1, 0) + (w, -w, 0)
            pts = np.flip(np.append(pts, pts1, axis=0), axis=0)
            pts[:, 1] += y

            mesh = Mesh(points=pts, corners=np.arange(len(pts)))

            return mesh

        # ---------------------------------------------------------------------------

        elif deco == 'SIGMA':

            w = width

            # Length and height
            l, h = .6, y1 - y0
            d = 1.5*w

            # Points
            px0, py0 = l, 0
            px1, py1 = 0, 0
            px2, py2 = l/2, h/2
            px3, py3 = 0, h
            px4, py4 = l, h

            pts = [[px0, py0 + 2*d, 0], [px0, py0, 0], [px1, py1, 0], [px2, py2, 0], [px3, py3, 0], [px4, py4, 0], [px4, py4 - 1.5*d, 0]]
            sigma = Curve(pts).set_spline2d_thickness(
                thickness = [w, 2*w, w, 2*w, w, w, w],
                offset = -1,
                mode = [0, 0, 3, 0, 3, 0, 0], 
                inner_mode = [0, 0, 0, 0, 0, 0, 0], 
                factor = 1.,
                cuts = [(-w*.5, np.pi/2), (w*.5, np.pi/2)],
                start_thickness=.3,
                end_thickness=.3)
            
            return sigma

        # ---------------------------------------------------------------------------

        else:
            raise ValueError(f"Invalid block code: '{deco}'.")
        
        # ---------------------------------------------------------------------------
        
    # ====================================================================================================
    # Place scritps
    # ====================================================================================================

    def add_scripts(self, mesh, **scripts):

        scale = self.get_prop('ssscript_scale', .35)
        x_margin = self.get_prop('x_margin', .1)
        y_margin = self.get_prop('y_margin', .15)
        y_ssscript = self.get_prop('y_ssscript', .1)

        x0, y0, x1, y1 = self.bbox(mesh)

        pos = mesh.points.position

        for location, script in scripts.items():

            # script to mesh
            smesh = script.get_mesh()

            smesh.points.position *= scale
            sx0, sy0, sx1, sy1 = self.bbox(smesh)
            sl, ml = sx1 - sx0, x1 - x0

            # Depending on the location

            if location in ['BOT', 'BOTTOM', 'BELOW']:
                smesh.points.x += x0 - sx0 + (ml - sl)/2
                smesh.points.y += y0 - sy1 - y_margin/2

            elif location in ['TOP', 'ABOVE']:
                smesh.points.x += x0 - sx0 + (ml - sl)/2
                smesh.points.y += y1 - sy0 + y_margin/2

            elif location in ['SUB', 'SUBSCRIPT']:
                y = y0 - y_ssscript
                pts = pos[pos[:, 1] < y + sy1 - sy0 + 2*y_margin]

                xm = np.max(pts[:, 0])

                smesh.points.x += xm - sx0 + x_margin
                smesh.points.y += y - sy0

            elif location in ['SUP', 'SUPERSCRIPT']:
                y = y1 - y_ssscript
                pts = pos[pos[:, 1] > y - 2*y_margin]

                xm = np.max(pts[:, 0])

                smesh.points.x += xm - sx0 + x_margin
                smesh.points.y += y - sy0

            else:
                raise ValueError(f"Invalid location code: '{location}'.")

            mesh.join(smesh)

        return mesh
    
    # ====================================================================================================
    # Variable height characters
    # ====================================================================================================

    def get_var_height_char(self, char, ychar0, ychar1):

        VALIDS = "[](){}<>⟨⟩|‖"
        if not char in VALIDS:
            raise ValueError(f"Character '{char}' ha sno variable height, only {VALIDS} are accepted.")

        # ---------------------------------------------------------------------------
        # Get the template

        if not hasattr(self,'_var_height_chars'):
            txt = Text(VALIDS).to_mesh(transform=False, char_index=True)

            self._var_height_chars = {}
            for i, c in enumerate(VALIDS):
                self._var_height_chars[c] = txt.extract_from_faces(txt.faces.char_index == i)

        mesh = Mesh.from_mesh(self._var_height_chars[char])

        height = ychar1 - ychar0

        y = mesh.points.y
        y0, y1 = np.min(y), np.max(y)
        h = y1 - y0
        if h >= height:
            mesh.points.y += ychar0 - y0
            return mesh
        
        margin = h*0.
        cy = (y0 + y1)/2
        
        dh = height - h
        rel_y = y - cy
        dy = maprange(np.abs(rel_y), margin, h/2 - margin, 0, dh/2, mode='SMOOTHER') * np.sign(rel_y)
        
        mesh.points.y += dy -y0 + ychar0

        return mesh


    # ====================================================================================================
    # Get the geometry
    # ====================================================================================================

    def get_mesh(self):

        # ---------------------------------------------------------------------------
        # Most common values
        # ---------------------------------------------------------------------------

        bar_thick = self.get_prop('bar_thick', .05)
        x_margin = self.get_prop('x_margin', .1)
        y_margin = self.get_prop('y_margin', .15)
        x_sepa = self.get_prop('x_sepa', .04)

        # ---------------------------------------------------------------------------
        # Join the compos in content
        # ---------------------------------------------------------------------------

        mesh = Mesh()
        x = 0.
        for compo in self.content:
            msh = compo.get_mesh()
            mx = np.max(msh.points.x)
            msh.points.x += x
            x += mx + x_sepa
            mesh.join(msh)

        x0, y0, x1, y1 = self.bbox(mesh)

        # ---------------------------------------------------------------------------
        # Content
        # ---------------------------------------------------------------------------

        if self.block_type == 'CONTENT':
            pass

        # ---------------------------------------------------------------------------
        # A Mesh
        # ---------------------------------------------------------------------------

        elif self.block_type == 'MESH':
            mesh = Mesh.from_mesh(self.mesh)

        # ---------------------------------------------------------------------------
        # A String
        # ---------------------------------------------------------------------------

        elif self.block_type == 'STRING':
            txt = Text(self.string, font=self.get_prop('font'), italic=self.italic, bold=self.bold)
            mesh = txt.to_mesh(transform=False)

        # ---------------------------------------------------------------------------
        # Decorated block
        # ---------------------------------------------------------------------------

        elif self.block_type == 'DECORATOR':

            if self.deco in ['BAR', 'ARROW', 'POINT']:
                y = y0 - y_margin if self.get_prop('bottom', False) else y1 + y_margin
                invert = self.get_prop('invert', False)
                count = self.get_prop('count', 1)

                if self.deco == 'BAR':
                    deco = self.bar_mesh(x0, y, x1, y, width=bar_thick)
                elif self.deco == 'ARROW':
                    deco = self.arrow_mesh(x0, y, x1, y, width=bar_thick, invert=invert)
                elif self.deco == 'POINT':
                    deco = self.point_mesh(x0, x1, y, radius=bar_thick, count=count)

            elif self.deco in ['||', '()', '[]', '{}', '‖‖']:


                if True:
                    left_deco = self.get_var_height_char(self.deco[0], y0, y1)
                    right_deco = self.get_var_height_char(self.deco[1], y0, y1)

                else:
                    left_deco = self.block_mesh(self.deco, y0, y1, margin=y_margin, width=bar_thick)
                    right_deco = left_deco.symmetrical(x=-1)
                
                left_only = self.get_prop('left_only', False)
                right_only = self.get_prop('right_only', False)
                deco = None

                if not right_only:
                    dx0, _, dx1, _ = self.bbox(left_deco)
                    dw = dx1 - dx0
                    mesh.points.x += dw + x_margin
                    x0, y0, x1, y1 = self.bbox(mesh)

                    left_deco.points.x -= dx0
                    deco = left_deco

                if not left_only:
                    dx0, _, dx1, _ = self.bbox(right_deco)
                    right_deco.points.x += x1 + x_margin - dx0
                    if deco is None:
                        deco = right_deco
                    else:
                        deco.join(right_deco)

            else:
                raise Exception(f"Unknown decorator: '{self.deco}'.")

            mesh.join(deco)

        # ---------------------------------------------------------------------------
        # Square Root
        # ---------------------------------------------------------------------------

        elif self.block_type == 'SQRT':

            w = bar_thick
            dx, dy = w, 2.1*w
            # Length and height
            l, h = x1 - x0 + x_margin, y1 - y0 + y_margin

            # Points
            px0, py0 = -(dy + 6*dx + x_margin), 2*dy
            px1, py1 = px0 + dy, py0 + dx
            px2, py2 = px1 + 3*dx, py1 - 3*dy
            px3, py3 = px2 + 3*dx, h
            px4, py4 = l, h
            px5, py5 = l, h - 2*w

            pts = [[px0, py0, 0], [px1, py1, 0], [px2, py2, 0], [px3, py3, 0], [px4, py4, 0], [px5, py5, 0]]
            sqrt_mesh = Curve(pts).set_spline2d_thickness(
                thickness=[.8*w, 1.8*w, .9*w, w, w, w/2], 
                mode=0, 
                inner_mode=[0, 0, 1, 0, 0, 0], 
                factor=.8,
                end_thickness=0)
            
            sqrt_mesh.points.y += y0
            mesh.points.x -= x0
            mesh.join(sqrt_mesh)
            mesh.points.x -= px0

        # ---------------------------------------------------------------------------
        # Subsscript / superscript
        # ---------------------------------------------------------------------------

        elif self.block_type == 'SCRIPT':
            mesh = self.add_scripts(mesh, **self.scripts)

        # ---------------------------------------------------------------------------
        # Integral / sigma
        # ---------------------------------------------------------------------------

        elif self.block_type == 'INTEGRAL':
            count = self.get_prop('count', 1)

            if self.is_sigma:
                symb = self.block_mesh('SIGMA', y0, y1, margin=y_margin, width=bar_thick)
            else:
                symb = self.block_mesh('INTEGRAL', y0, y1, margin=y_margin, width=bar_thick)
                symb = symb.duplicate(count=count, offset=(.25, 0, 0), relative=False)

            sx0, sy0, sx1, sy1 = self.bbox(symb)
            sl = sy1 - sy0
            symb.points.y += -sl/2 - sy0 + y1/2 

            symb = self.add_scripts(symb, **self.scripts)
            sx0, sy0, sx1, sy1 = self.bbox(symb)

            symb.points.x -=  sx0
            mesh.points.x += sx1 - sx0 + x_margin - x0
            
            mesh.join(symb)

        # ---------------------------------------------------------------------------
        # Fraction
        # ---------------------------------------------------------------------------

        elif self.block_type == 'FRACTION':
            num_mesh = mesh
            nx0, ny0, nx1, ny1 = x0, y0, x1, y1
            nl = nx1 - nx0

            den_mesh = self.denominator.get_mesh()
            dx0, dy0, dx1, dy1 = self.bbox(den_mesh)
            dl = dx1 - dx0

            l = max(nl, dl) + 2*x_margin

            mesh = self.bar_mesh(0, 0, l, 0, width=1.5*bar_thick)

            num_mesh.points.y += y_margin + bar_thick - ny0
            den_mesh.points.y -= y_margin + bar_thick + dy1

            num_mesh.points.x += (l - nl)/2 - nx0
            den_mesh.points.x += (l - dl)/2 - dx0

            mesh.join(num_mesh, den_mesh)

        # ---------------------------------------------------------------------------
        # Error
        # ---------------------------------------------------------------------------

        else:
            raise Exception(f"Unknown block type: '{self.block_type}'.")

        return mesh



    
    
    


    