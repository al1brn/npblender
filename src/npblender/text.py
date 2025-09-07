# MIT License
#
# Copyright (c) 2025 Alain Bernard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the \"Software\"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Module Name: text
Author: Alain Bernard
Version: 0.1.0
Created: 2025-09-03
Last updated: 2025-09-03

Summary:
    Blender Text.

"""

__all__ = ["Font", "Text", "Composition"]

import numpy as np
from pathlib import Path

import bpy

from .constants import bint, bfloat
from . import blender
from .geometry import Geometry
from .curve import Curve
from .mesh import Mesh
from .domain import Point

from .textutils.treenode import TreeNode
from .maths import Transformation, Rotation, Quaternion, maprange

# ====================================================================================================
# Font
# ====================================================================================================

class Font:

    BFONT = "Bfont"

    def __init__(self, name=BFONT, filepath=None, regular="", bold="Bold", italic="Italic", bold_italic="Bold Italic", extension="ttf"):

        self.filepath = filepath
        self.extension = extension

        self._regular = self.load_font(name, regular, filepath, halt=True)
        self._bold = self.load_font(name, bold, filepath, halt=False)
        self._italic = self.load_font(name, italic, filepath, halt=False)
        self._bold_italic = self.load_font(name, bold_italic, filepath, halt=False)

    @staticmethod
    def _get_full_name(name, variant):
        if variant is None or variant == "":
            return name
        elif name is None or name == "":
            return variant
        else:
            return f"{name} {variant}"
    
    # ====================================================================================================
    # Load a font
    # ====================================================================================================

    @classmethod
    def load_font(cls, name=BFONT, variant="Regular", filepath=None, extension="ttf", halt=True):

        full_name = cls._get_full_name(name, variant)
        
        font = bpy.data.fonts.get(full_name)
        if font is not None:
            return font
        
        # ----- Load default if it is this one
        
        if name == cls.BFONT:
            temp = blender.create_text_object("BPB Text Temp")
            blender.delete_object(temp)
            return bpy.data.fonts.get(full_name)

        # ----- Otherwise load from path

        file_path = str(Path(filepath) / full_name) + f".{extension}"

        try:
            return bpy.data.fonts.load(file_path)
        except:
            pass

        # ----- Oops

        msg = f"Impossible to load font '{full_name}' with path '{file_path}'"

        if halt:
            raise Exception(msg)
        
        print("Warning:", msg)
        return None
    
    # ====================================================================================================
    # Aspect
    # ====================================================================================================

    @property
    def regular(self):
        return self._regular
    
    @property
    def bold(self):
        if self._bold is None:
            return self._regular
        else:
            return self._bold

    @property
    def italic(self):
        if self._italic is None:
            return self._regular
        else:
            return self._italic

    @property
    def bold_italic(self):
        if self._italic is None:
            return self.italic if self._bold is None else self.bold
        else:
            return self._bold_italic
        
    # ====================================================================================================
    # Call
    # ====================================================================================================

    def __call__(self, bold=False, italic=False):
        if bold:
            if italic:
                return self.bold_italic
            else:
                return self.bold
        else:
            if italic:
                return self.italic
            else:
                return self.regular
            
    # ====================================================================================================
    # Serialization
    # ====================================================================================================
            
    def to_dict(self):
        return {
            'class' : 'Font',
            'filepath': self.filepath,
            'extension' : self.extension,
            'regular' : self.regular.name,
            'bold' : self.bold.name,
            'italic' : self.italic.name,
            'bold_italic' : self.bold_italic.name,
        }
        
    @classmethod
    def from_dict(cls, d):
        if d.get('class') != 'Font':
            raise Exception(f"Dictionnary doesn't contain Font data ({d.get('class')}).\n{str(d)[:100]}")
        
        return cls(name="", filepath=d['filepath'],
            regular=d['regular'], 
            bold=d['bold'], 
            italic=d['italic'], 
            bold_italic=d['bold_italic'],
            extension=d['extension'],
            )

# ====================================================================================================
# Text class
# ====================================================================================================

class Text(Geometry):

    FONT_STYLE = ('BOLD', 'ITALIC', 'SMAL_CAPS', 'UNDERLINE')
    X_ALIGN = ('LEFT', 'CENTER', 'RIGHT', 'JUSTIFY', 'FLUSH')
    Y_ALIGN = ('TOP', 'TOP_BASELINE', 'CENTER', 'BOTTOM_BASELINE', 'BOTTOM')
    BEVEL_MODE = ('ROUND', 'OBJECT', 'PROFILE')

    PROPERTIES = {
        'align_x' : 'LEFT',
        'align_y' : 'TOP_BASELINE',
        'shear' : 0., 
        'size' : 1., 
        'small_caps_scale' : 0.75, 
        'space_character' : 1., 
        'space_line' : 1., 
        'space_word' : 1.,
        'underline_height' : 0.05, 
        'underline_position' : 0.0,
        'offset_x' : 0.0, 
        'offset_y' : 0.0, 
        'bevel_mode' : 'ROUND', 
        'extrude' : 0.0,
        'offset': 0.0,
        'bevel_depth' : 0.0, 
        'bevel_resolution' : 4, 
        'bevel_object' : None,
        'use_fill_caps' : False,
        }    
        
    domain_names = ["points"]
    
    def __init__(self, text="Text", font=None, materials=None, bold=False, italic=False, **kwargs):
        
        # One single point : location of the text
        self.points = Point(position=(0, 0, 0))
        if materials is None:
            self.materials = []
        elif isinstance(materials, str):
            self.materials = [materials]
        else:
            self.materials = list(materials)

        # Text ad font
        self.text = text
        self.font = Font() if font is None else font
        self.bold = bold
        self.italic = italic
        
        # Properties
        self.props = {k: kwargs.get(k, vdef) for k, vdef in Text.PROPERTIES.items()}
        
        # Make sure all properties are correct
        for k in kwargs:
            if k not in Text.PROPERTIES:
                raise Exception(f"Text.init: unknown Text properties: '{k}'")

    def __str__(self):
        return f"<Text '{self.text}', font: {self.font.name} >"

    # ----------------------------------------------------------------------------------------------------
    # Conveniance
    # ----------------------------------------------------------------------------------------------------

    def solidify(self, extrude=.05, bevel_depth=.02, bevel_resolution=3):
        self.extrude          = extrude
        self.bevel_depth      = bevel_depth
        self.bevel_resolution = bevel_resolution

    # ----------------------------------------------------------------------------------------------------
    # Dimensions
    # ----------------------------------------------------------------------------------------------------
    
    def get_dimensions(self):
        curve = self.to_curve(transform=False)
        vmin, vmax = np.min(curve.points.position, axis=0), np.max(curve.points.position, axis=0)
        return {
            'width': vmax[0] - vmin[0],
            'height': vmax[1] - vmin[1],
            'ascen' : max(0, vmax[1]),
            'descen' : max(0, -vmin[1]),
        }
        
    @property
    def dimensions(self):
        if not hasattr(self, '_dimensions'):
            self._dimensions = self.get_dimensions()
        return self._dimensions

    @property
    def font_dims(self):
        mem_text = self.text
        self.text = "Ap"
        dims = self.get_dimensions()
        self.text = mem_text
        dims['width'] /= 3 # Heuristic for space width
        return dims
    
    # ====================================================================================================
    # To and from Text object
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # From object
    # ----------------------------------------------------------------------------------------------------
    
    @classmethod
    def from_object(cls, spec):
        txt_obj = blender.get_object(spec)
        if not isinstance(txt_obj.data, bpy.types.TextCurve):
            raise Exception(f"Impossible de load Text. Object '{spec}' is not a Blender Text object")
            
        bl_text = txt_obj.data
        
        materials = [None if mat is None else mat.name for mat in bl_text.materials]
        
        txt = cls(bl_text.body, font=bl_text.font, materials=materials, **{k: getattr(bl_text, k) for k in Text.PROPERTIES}) 

        txt.points.position = txt_obj.location

        scale = np.array(txt_obj.scale)
        if np.max(np.abs(scale - (1, 1, 1))) > 1e-6:
            txt.points.scale = scale

        euler = txt_obj.rotation_euler
        if np.max(np.abs(euler)) > 1e-6:
            txt.points.euler = euler
        
        return txt
    
    # ----------------------------------------------------------------------------------------------------
    # To object
    # ----------------------------------------------------------------------------------------------------

    def to_object(self, spec, collection=None, **kwargs):

        obj = blender.create_text_object(spec, text=self.text, collection=collection)
        obj.data.font = self.font(bold=self.bold, italic=self.italic)

        bl_text = obj.data

        for k, v in self.props.items():
            setattr(bl_text, k, v)
            
        # Materials   
        bl_text.materials.clear()
        for mat_name in self.materials:
            if mat_name is not None:
                bl_text.materials.append(bpy.data.materials.get(mat_name))

        # Transformation
        obj.location = self.points.position[0]
        if self.points.has_rotation:
            obj.rotation_euler = self.points.rotation[0].as_euler()

        if 'scale' in self.points.actual_names:
            obj.scale = self.points.scale[0]

        return obj
    
    # ====================================================================================================
    # Serialization
    # ====================================================================================================
    
    def to_dict(self):
        return {
            'geometry': 'Text',
            'points': self.points.to_dict(),
            'text': self.text,
            'font' : self.font.to_dict(),
            'materials': self.materials,
            'bold' : self.bold,
            'italic': self.italic,
            'props' : dict(self.props),
        }
        
    @classmethod
    def from_dict(cls, d):
        if d.get('geometry') != 'Text':
            raise Exception(f"Dictionnary doesn't contain Text data ({d.get('geometry')}).\n{str(d)[:100]}")
            
        font = Font.from_dict(d['font'])
        return cls(d['text'], font=font, materials=list(d['materials']), bold=d['bold'], italic=d['italic'], **d['props'])
    
    # ====================================================================================================
    # Conversion
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # To Mesh
    # ----------------------------------------------------------------------------------------------------

    def to_mesh(self, transform=True):
        
        with self.object(readonly=True) as obj:
            bl_mesh = bpy.data.meshes.new_from_object(obj)
            mesh = Mesh.from_mesh_data(bl_mesh)

        mesh.materials = list(self.materials)

        # ----- Transformation

        if transform:
            if 'scale' in self.points.actual_names:
                mesh.points.position *= self.points.scale[0]

            if self.points.has_rotation:
                mesh.points.position = self.points.rotation[0] @ mesh.points.position

            mesh.points.position += self.points.position[0]

        return mesh
    
    # ----------------------------------------------------------------------------------------------------
    # To Mesh
    # ----------------------------------------------------------------------------------------------------
    
    def to_curve(self, transform=True):
        
        with self.object(readonly=True) as obj:
            bpy.ops.object.convert(target='CURVE', keep_original=False)
            curve = Curve.from_curve_data(obj.data)

        curve.materials = list(self.materials)

        # ----- Transformation

        if transform:
            if 'scale' in self.points.actual_names:
                curve.points.position *= self.points.scale[0]

            if self.points.has_rotation:
                curve.points.position = self.points.rotation[0] @ curve.points.position

            curve.points.position += self.points.position[0]

        return curve   

# ====================================================================================================
# Composition
# ====================================================================================================

class Composition():
    """ A composition encapsulates geometries organized in a composition
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

                left_deco = self.block_mesh(self.deco, y0, y1, margin=y_margin, width=bar_thick)
                left_only = self.get_prop('left_only', False)
                right_only = self.get_prop('right_only', False)

                right_deco = left_deco.symmetrical(x=-1)
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



    
    
    


    