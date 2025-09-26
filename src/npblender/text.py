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

__all__ = ["Font", "Text", "Formula", "FGeom"]

import numpy as np
from pathlib import Path

import bpy

from .constants import bint, bfloat, ZERO
from . import blender
from .geometry import Geometry
from .curve import Curve
from .mesh import Mesh
from .domain import Point

from .textutils import EText, EChar, parse_latex
from .maths import Color, Transformation, Rotation, Quaternion, maprange
from .maths import Transfo2d, BBox
from . import maths


# ====================================================================================================
# Font
# ====================================================================================================

class Font:

    FONTS = {}

    def __init__(self, name=None, filepath=None, regular="", bold="Bold", italic="Italic", bold_italic="Bold Italic", font_scale=1.0, extension="ttf", font_name=None):

        # Is the font already known
        self.font_name = name if font_name is None else font_name
        if self.font_name is not None:
            if self.font_name in Font.FONTS:
                self.copy_from(Font.FONTS[self.font_name])
                return

        # Let's create it
        self.filepath     = filepath
        self.extension    = extension

        self._regular     = self.load_font(name, regular,     filepath, default = None)
        self._bold        = self.load_font(name, bold,        filepath, default = self._regular)
        self._italic      = self.load_font(name, italic,      filepath, default = self._regular)
        self._bold_italic = self.load_font(name, bold_italic, filepath, default = self._bold)

        self.font_scale   = font_scale
        self._dimensions  = None

        if self.font_name is not None:
            Font.FONTS[self.font_name] = self

    # ====================================================================================================
    # Copy from another one
    # ====================================================================================================

    def copy_from(self, font):

        self.filepath     = font.filepath
        self.extension    = font.extension

        self._regular     = font.regular
        self._bold        = font.bold
        self._italic      = font.italic
        self._bold_italic = font.bold_italic

        self.font_scale   = font.font_scale
        self._dimensions  = font._dimensions

        return self

    # ====================================================================================================
    # Load a font
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Build a font name from name and variant
    # ----------------------------------------------------------------------------------------------------

    @staticmethod
    def _get_font_name(name, variant):
        if name is None:
            if variant is None or variant == "":
                return "Bfont"
            else:
                return variant
            
        else:
            if variant is None or variant == "":
                return name
            else:
                return f"{name} {variant}"
            
    # ----------------------------------------------------------------------------------------------------
    # Load a font from a file
    # ----------------------------------------------------------------------------------------------------
    
    @staticmethod
    def _load_from_path(file_name):
        """Load a font from its full path.

        Compare the path to loaded VectorFont paths before actually loading the font

        Returns
        -------
        VectorFont or None:
            The Font or None if load is unsuccessful
        """

        # In case it Path
        file_name = str(file_name)

        # ----- Default font

        if file_name in ['Bfont', '<builtin>']:
            for font in bpy.data.fonts:
                if font.filepath == '<builtin>':
                    return font
                
            temp = blender.create_text_object("NPB Text Temp")
            font = temp.data.font
            data = temp.data
            blender.delete_object(temp)
            bpy.data.curves.remove(data)
            return font
        
        # ----- Actual file name

        # Comparing the file paths

        path = Path(file_name)
        for font in bpy.data.fonts:
            vf_path = Path(font.filepath)
            if path == vf_path:
                return font

        # First load

        try:
            return bpy.data.fonts.load(file_name)
        except Exception as e:
            msg = str(e)

        print(f"WARNING Impossible to load font '{file_name}' : {msg}")
            
        return None
    
    # ----------------------------------------------------------------------------------------------------
    # STIX Font
    # ----------------------------------------------------------------------------------------------------
    
    @staticmethod
    def _STIX_folder():
        return str(Path(__file__).resolve().parent / "fonts")
    
    @classmethod
    def _STIX_load(cls, variant='Regular'):
        variant = variant.replace(' ', '').strip()
        if variant == "":
            variant = "Regular"
        fname = f"STIXTwoText-{variant}.ttf"

        return cls._load_from_path(Path(cls._STIX_folder()) / fname)
    
    @classmethod
    def _STIX_math_load(cls):
        fname = "STIXTwoMath-Regular.ttf"
        return cls._load_from_path(Path(cls._STIX_folder()) / fname)
    
    # ----------------------------------------------------------------------------------------------------
    # Load a font
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def load_font(cls, name, variant="Regular", filepath=None, extension="ttf", default=None):

        # Name is a font
        if isinstance(name, bpy.types.VectorFont):
            return name

        # Variant is a font
        if isinstance(variant, bpy.types.VectorFont):
            return variant
        
        if name is not None:
            # Included STIX font
            if name == 'STIX':
                font = cls._STIX_load(variant)
                if font is not None:
                    return font
                
            # Included STIX Math font
            if name.lower() in ['math', 'maths']:
                font = cls._STIX_math_load()
                if font is not None:
                    return font

        # Let's try to load from name only
        font_name = cls._get_font_name(name, variant)
        font = bpy.data.fonts.get(font_name)
        if font is not None:
            return font
        
        # If path is provided, we try to load the file
        file_name = None
        if filepath is not None:

            file_name = str(Path(filepath) / font_name) + f".{extension}"

            font = cls._load_from_path(file_name)
            if font is not None:
                return font

        # If default is not provided, we load the default fault
        if default is None:
            default = cls._load_from_path("Bfont")

        font = default

        if font is None:
            raise ValueError(
                f"Impossible to load font:\n"
                f"- name:       {name}\n",
                f"- variant:    {variant}\n",
                f"- font name:  {font_name}\n",
                f"- file name:  {file_name}\n"
                )

        return font

    # ====================================================================================================
    # From blender
    # ====================================================================================================

    @classmethod
    def from_data(cls, data):
        return cls(
            regular     = data.font,
            bold        = data.font_bold,
            italic      = data.font_italic,
            bold_italic = data.font_bold_italic,
        )
    
    def to_data(self, data):
        data.font             = self.regular
        data.font_bold        = self.bold
        data.font_italic      = self.italic
        data.font_bold_italic = self.bold_italic

    
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
            'class'       : 'Font',
            'filepath'    : self.filepath,
            'extension'   : self.extension,
            'regular'     : self.regular.name,
            'bold'        : self.bold.name,
            'italic'      : self.italic.name,
            'bold_italic' : self.bold_italic.name,
            'font_scale'  : self.font_scale,
        }
        
    @classmethod
    def from_dict(cls, d):
        if d.get('class') != 'Font':
            raise Exception(f"Dictionnary doesn't contain Font data ({d.get('class')}).\n{str(d)[:100]}")
        
        return cls(
            name        = "",
            filepath    = d['filepath'],
            regular     = d['regular'], 
            bold        = d['bold'], 
            italic      = d['italic'], 
            bold_italic = d['bold_italic'],
            extension   = d['extension'],
            font_scale  = d['font_scale'],
            )
    
    # ====================================================================================================
    # Get dimenions
    # ====================================================================================================

    def get_dimensions(self, italic=False, bold=False, scale=1.0):

        # ----- Already computed
        if self._dimensions is not None:
            return self._dimensions
        
        # ----- Computed globally
        if self.font_name is not None:
            font = Font.FONTS.get(self.font_name)
            if font is not None and font._dimensions is not None:
                return font._dimensions
            

        print("Font: compute dimensions", self.font_name)

        mesh = Text("aAp> <=_", font=self).to_mesh(transform=False, char_index=True)

        def _dims(char_index):
            cis = mesh.faces[mesh.faces.char_index==char_index].loop_index
            pts = mesh.points.position[mesh.corners[cis].vertex_index]
            return np.min(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 0]), np.max(pts[:, 1])
        
        d_eq  = _dims(6)
        d_bar = _dims(7)

        d = {
            'y_body'    : _dims(0)[3],
            'y_ascen'   : _dims(1)[3],
            'y_descen'  : _dims(2)[1],
            'x_space'   : _dims(5)[0] - _dims(3)[2],
            'y_frac'    : (d_eq[1] + d_eq[3])/2.0,
            'thickness' : d_bar[3] - d_bar[1],
        }
        d['x_sepa']      = d['x_space']*0.5
        d['y_sepa']      = d['thickness']*3.0
        d['dy_super']    = d['y_body']*0.2
        d['y_super_min'] = d['y_body']*0.8
        d['dy_sub']      = d['y_body']*0.4
        d['y_sub_max']   = d['y_body']*0.4
        d['oversize']    = d['thickness']*0.8

        self._dimensions = d
        if self.font_name is not None:
            Font.FONTS[self.font_name]._dimensions = d

        return d

# ====================================================================================================
# Text class
# ====================================================================================================

class Text(Geometry):

    FONT_STYLE = ('BOLD', 'ITALIC', 'SMALL_CAPS', 'UNDERLINE')
    X_ALIGN    = ('LEFT', 'CENTER', 'RIGHT', 'JUSTIFY', 'FLUSH')
    Y_ALIGN    = ('TOP', 'TOP_BASELINE', 'CENTER', 'BOTTOM_BASELINE', 'BOTTOM')
    BEVEL_MODE = ('ROUND', 'OBJECT', 'PROFILE')

    PROPERTIES = {
        'align_x'               : 'LEFT',
        'align_y'               : 'TOP_BASELINE',
        'shear'                 : 0., 
        #'size'                 : 1., 
        'small_caps_scale'      : 0.75, 
        'space_character'       : 1., 
        'space_line'            : 1., 
        'space_word'            : 1.,
        'underline_height'      : 0.05, 
        'underline_position'    : 0.0,
        'offset_x'              : 0.0, 
        'offset_y'              : 0.0, 
        'bevel_mode'            : 'ROUND', 
        'extrude'               : 0.0,
        'offset'                : 0.0,
        'bevel_depth'           : 0.0, 
        'bevel_resolution'      : 4, 
        'bevel_object'          : None,
        'use_fill_caps'         : False,
        }    
        
    domain_names = ["points"]
    
    def __init__(self, text="Text", font=None, materials=None, is_latex=False, **kwargs):

        # One single point : location of the text
        self.points = Point(position=(0, 0, 0))
        if materials is None:
            self.materials = []
        elif isinstance(materials, str):
            self.materials = [materials]
        else:
            self.materials = list(materials)

        # Font
        self.font = Font() if font is None else font

        # Properties
        self.props = {}

        # --------------------------------------------------
        # Type of text
        # --------------------------------------------------

        # LaTeX
        if is_latex:
            self.etext = parse_latex(text, math_mode=False)

        # A dict with attributes
        # - text is mandatory
        # - attributes can be for EText or self
        elif isinstance(text, dict):
            attrs = {}
            for k, v in text.items():
                if k == 'text' or k in EText.STYLES:
                    attrs[k] = v
                else:
                    self.props[k] = v

            self.etext = EText(**attrs)

        else:
            self.text = text

        # Properties
        for k, vdef in kwargs.items():
            if k in EText.STYLES:
                self._etext[k] = vdef

            elif k in Text.PROPERTIES:
                self.props[k] = vdef

            else:
                raise Exception(f"Text.init: unknown Text property: '{k}'")
            
    def clone(self):
        txt = Text(font=self.font, materials=self.materials, **{k:getattr(self, k) for k in Text.PROPERTIES})
        txt._etext = self._text.clone()
        return txt

    def __str__(self):
        return f"<Text '{self.text}', font: {self.font.name} >"

    # ----------------------------------------------------------------------------------------------------
    # Text
    # ----------------------------------------------------------------------------------------------------

    @property
    def text(self):
        return str(self._etext)

    @text.setter
    def text(self, value):
        self._etext = EText(value)

    @property
    def etext(self):
        return self._etext

    @etext.setter
    def etext(self, value):
        self._etext = EText(value)

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

    # ----------------------------------------------------------------------------------------------------
    # Conveniance
    # ----------------------------------------------------------------------------------------------------

    def solidify(self, extrude=.05, bevel_depth=.02, bevel_resolution=3):
        self.extrude          = extrude
        self.bevel_depth      = bevel_depth
        self.bevel_resolution = bevel_resolution

    # ====================================================================================================
    # To and from Text Data (TextCurve)
    # ====================================================================================================

    @classmethod
    def from_data(cls, data):

        # Initialize Text
        txt = cls(
            data.body, 
            font        = Font.from_data(data),
            materials   = cls.materials_from_data(data), 
            **{k: getattr(data, k) for k in Text.PROPERTIES}) 

        # Char properties
        txt._etext.from_body_format(data.body_format)

        return txt
  
    def to_data(self, data):

        # Fonts
        self.font.to_data(data)

        # Global Properties
        for k, v in self.props.items():
            setattr(data, k, v)

        data.size *= self.font.font_scale

        # Materials
        # CAUTION : before etext.to_body_format to make sure materials exist
        # when setting material_index
        self.materials_to_data(data)

        # Char properties
        # CAUTION: make sure materials is up to date
        data.body = self._etext.text
        self._etext.to_body_format(data.body_format)

        return data
  
    # ====================================================================================================
    # To and from Text object
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # From object
    # ----------------------------------------------------------------------------------------------------
    
    @classmethod
    def from_object(cls, spec):

        # Load the object
        txt_obj = blender.get_object(spec)
        if not isinstance(txt_obj.data, bpy.types.TextCurve):
            raise Exception(f"Impossible de load Text. Object '{spec}' is not a Blender Text object")
        
        # Create the Text
        txt = cls.from_data(txt_obj.data)

        # Position, Scale, Rotation
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

        # Create a new Text object
        obj = blender.create_text_object(spec, text=self.text, collection=collection)

        # Set data (TextCurve)
        self.to_data(obj.data)

        # Position, Scale, Rotation
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
            'geometry'  : 'Text',
            'points'    : self.points.to_dict(),
            'text'      : self.text,
            'font'      : self.font.to_dict(),
            'materials' : self.materials,
            'bold'      : self.bold,
            'italic'    : self.italic,
            'props'     : dict(self.props),
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

    def to_mesh(self, transform=True, char_index=True, **kwargs):

        # Char index
        if char_index:
            mats = self.materials
            matind = np.array(self._etext.material_index)
            self.materials = [None] * len(matind)
            self._etext.material_index = np.arange(len(matind), dtype=bint)

        # Create the mesh
        with self.object(readonly=True, **kwargs) as obj:
            bl_mesh = bpy.data.meshes.new_from_object(obj)
            mesh = Mesh.from_mesh_data(bl_mesh)

        # Char index
        if char_index:
            mesh.faces.new_int("char_index")
            mesh.faces.char_index = mesh.faces.material_index
            mesh.faces.material_index = matind[mesh.faces.char_index]
            mesh.materials = list(mats)

            self._etext.material_index = matind

            # Color property
            mesh.faces.new_color("color")
            mesh.faces.color = self._etext.color[mesh.faces.char_index]

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
    
    def to_curve(self, transform=True, char_index=True, **kwargs):

        # Char index
        if char_index:
            mats = self.materials
            matind = np.array(self._etext.material_index)
            self.materials = [None] * len(matind)
            self._etext.material_index = np.arange(len(matind), dtype=bint)

        # Create the curve
        with self.object(readonly=True, **kwargs) as obj:
            bpy.ops.object.convert(target='CURVE', keep_original=False)
            curve = Curve.from_curve_data(obj.data)

        # Char index
        if char_index:
            curve.splines.new_int("char_index")
            curve.splines.char_index = curve.splines.material_index
            curve.splines.material_index = matind[curve.splines.char_index]
            self.materials = list(mats)

            self._etext.material_index = matind

            # Color property
            curve.splines.new_color("color")
            curve.splines.color = self._etext.color[curve.splines.char_index]

        # ----- Transformation

        if transform:
            if 'scale' in self.points.actual_names:
                curve.points.position *= self.points.scale[0]

            if self.points.has_rotation:
                curve.points.position = self.points.rotation[0] @ curve.points.position

            curve.points.position += self.points.position[0]

        return curve   

# ====================================================================================================
# Formula
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Content item
# ----------------------------------------------------------------------------------------------------

class FGeom(maths.FormulaGeom):

    # ---------------------------------------------------------------------------
    # Set the content : a string or a mesh
    # ---------------------------------------------------------------------------

    def set_content(self, content):

        self._special = None # Special char built manually
        self._mesh    = None # Mesh
        self._bbox    = None # bbox cache

        # Size adjust
        self._last_width  = 0.0
        self._last_height = 0.0
        

        _string  = None

        if isinstance(content, Mesh):
            self._mesh = content

        elif isinstance(content, str):
            if content.startswith('\\'):
                self._special = content
            else:
                _string = content

        elif isinstance(content, (EText, EChar)):
            _string = content

        elif isinstance(content, (float, int, np.integer, np.float64, np.float32)):
            _string = str(content)

        elif isinstance(content, dict):
            text = Text(content, font=self.term.font)
            self._mesh = text.to_mesh()

        elif isinstance(content, Geometry):
            self._mesh = content.to_mesh()

        else:
            raise ValueError(f"FGeom(): 'content' must be Geometry or a string, not '{type(content).__name__}'")
        
        # ----- Build Mesh from string

        if self._mesh is None:
            if _string is None:
                self._bbox = BBox()

            else:
                # self.font is either font or math_font depending on self.symbol
                text = Text(_string, font=self.font)
                self._mesh = text.to_mesh()

    # ---------------------------------------------------------------------------
    # Compute the bbox
    # ---------------------------------------------------------------------------

    @property
    def bbox(self):
        if self._bbox is None or (self._adjust_size is not None):
            self._bbox = self.get_bbox()
        return self._bbox

    def get_bbox(self):
        if self._mesh is None and self._special is None:
            return BBox()
        
        return BBox.from_points(self.mesh.points.position)

    # ----------------------------------------------------------------------------------------------------
    # Compute the mesh, possibly deformed
    # ----------------------------------------------------------------------------------------------------

    @property
    def mesh(self):
        if self._special is not None:
            return self.special_mesh(self._special)

        if self._mesh is None:
            return Mesh()
        
        mesh = Mesh.from_mesh(self._mesh)
        if self._adjust_size is None:
            return mesh
        
        # Adjust from property _adjust_size

        bbox = BBox.from_points(self._mesh.points.position)
        width, height = self._adjust_size

        if self._last_width == width and self._last_height == height:
            return mesh
        
        # Vertical adjustment
        if height > bbox.height:
            mesh.elongation(
                axis   = 'Y',
                size   = height + 2*self.term.oversize, 
                mode   = self.term.elon_mode,
                margin = self.term.elon_margin,
                smooth = self.term.elon_smooth,
                )

        # Horizontal adjustment
        if width > bbox.width:
            mesh.elongation(
                axis   = 'X',
                size   = width + 2*self.term.oversize, 
                mode   = self.term.elon_mode,
                margin = self.term.elon_margin,
                smooth = self.term.elon_smooth,
                )
            
        self._bbox = BBox.from_points(mesh.points.position)

        self._last_width  = width
        self._last_height = height

        return mesh
    
    # ----------------------------------------------------------------------------------------------------
    # Get the transformed mesh
    # ----------------------------------------------------------------------------------------------------

    def to_mesh(self, transfo=None, materials=None):

        if transfo is None:
            transfo = self.transfo3d

        mesh = self.mesh
        if len(mesh.points) == 0:
            return mesh

        mesh = mesh.transform(transfo)

        return mesh
    
    # ====================================================================================================
    # Special chars
    # ====================================================================================================

    def special_mesh(self, code):

        # Special mesh are built around (0, 0, width height)

        if code == r"\sqrt":

            # Dimensions
            x0, y0 = 0.0, 0.0
            if self._adjust_size is None:
                width, height = 0.0, 0.0
            else:
                width, height = self._adjust_size
            if width < ZERO or height < ZERO:
                return Text("√").to_mesh()
            
            w = self.term.thickness
            dx, dy = w, 2.1*w
            # Length and height
            l, h = width + 1.3*self.term.x_sepa, height + self.term.y_sepa

            # Points
            px0, py0 = -(dy + 6*dx + self.term.x_sepa), 2*dy
            px1, py1 = px0 + dy, py0 + dx
            px2, py2 = px1 + 3*dx, py1 - 3*dy
            px3, py3 = px2 + 3*dx, h
            px4, py4 = l, h
            px5, py5 = l, h - 2*w

            pts = [[px0, py0, 0], [px1, py1, 0], [px2, py2, 0], [px3, py3, 0], [px4, py4, 0], [px5, py5, 0]]
            mesh = Curve(pts).set_spline2d_thickness(
                thickness       = [.8*w, 1.8*w, .9*w, w, w, w/2], 
                mode            = 0, 
                inner_mode      = [0, 0, 1, 0, 0, 0], 
                factor          = .8,
                end_thickness   = 0)
            
            return mesh

        else:
            return Text("?{code}?").to_mesh()

    
# ====================================================================================================
# Formula
# ====================================================================================================

class Formula(maths.Formula):

    def __init__(self, body, materials=None, font=None, math_font=None, **attrs):
        """
        ```python
        # Two placeholders
        # minus signed is named minus
        # plus sign will be named + by default
        latex = r"a \term[minus]- \ph[target] = b + \ph[source]"
        frm = Formula(latex)
        
        c = Formula("c")
        
        plus  = frm.by_name("+")
        minus = frm.by_name("minus")
        
        # Placeholrders 
        tgt = frm.by_name("target")
        tgt.formula = c

        src = frm.by_name("source")
        src.formula = c

        # Target placeholder
        frm.update()
        tr1 = tgt.absolute_transfo
        
        # Starting from no minus sign and no target placeholder
        tgt.anim.sx = 0.0
        minus.anim.sx = 0.0
        
        # The source placeholder
        frm.update()
        tr0 = src.absolute_transfo
        
        # Animation
        def update():
            
            factor = np.clip((engine.frame-50)/100, 0, 1)
            
            tgt.anim.sx     = factor
            src.anim.sx     = 1 - factor
            minus.anim.sx   = factor
            plus.anim.sx    = 1 - factor
            
            c.move_to(tr0, tr1, factor, ymax=1.0, turns=0, smooth='SMOOTH')

            # The two formulas to objects
            frm.to_mesh().to_object("Formula")           
            c.to_mesh().to_object("c")
            
        engine.go(update)
        ```
        """

        if isinstance(body, str):
            body = parse_latex(body, math_mode=True)

        if font is None:
            font = Font('STIX')

        if math_font is None:
            math_font = Font("Maths", font_scale=3.0)
        
        font_attrs = {}
        for k, v in font.get_dimensions().items():
            if k not in attrs:
                font_attrs[k] = v

        super().__init__(None, body, geom_cls=FGeom, font=font, math_font=math_font, **attrs, **font_attrs)

        if materials is None:
            self.materials = []
        else:
            self.materials = []

    def to_mesh(self):

        self.update()
        
        mesh = Mesh(materials=self.materials)

        for _, fgeom in self.depths():
            if not isinstance(fgeom, FGeom):
                continue

            m = fgeom.to_mesh()
            if not len(m.points):
                continue

            m.materials = list(self.materials)
            mesh.join(m)

        return mesh


