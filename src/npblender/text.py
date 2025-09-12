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

from .textutils import EnrichedText, parse_latex
from .maths import Color, Transformation, Rotation, Quaternion, maprange

# ====================================================================================================
# Font
# ====================================================================================================

class Font:

    BFONT = "Bfont"

    def __init__(self, name=BFONT, filepath=None, regular="", bold="Bold", italic="Italic", bold_italic="Bold Italic", extension="ttf"):

        self.filepath = filepath
        self.extension = extension

        self._regular = self.load_font(name, regular, filepath, default=None)
        self._bold = self.load_font(name, bold, filepath, default=self._regular)
        self._italic = self.load_font(name, italic, filepath, default=self._regular)
        self._bold_italic = self.load_font(name, bold_italic, filepath, default=self._bold)

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
    def load_font(cls, name=BFONT, variant="Regular", filepath=None, extension="ttf", default=None):

        # Name is a font
        if isinstance(name, bpy.types.VectorFont):
            return name

        # Variant is a font
        if isinstance(variant, bpy.types.VectorFont):
            return variant
        
        # Let's try to load the font

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

        # ----- Default if any

        if default is None:
            raise Exception(f"Impossible to load font '{full_name}' with path '{file_path}'")
        else:
            return default
    
    # ====================================================================================================
    # From blender
    # ====================================================================================================

    @classmethod
    def from_data(cls, data):
        return cls(
            regular = data.font,
            bold = data.font_bold,
            italic = data.font_italic,
            bold_italic = data.font_bold_italic,
        )
    
    def to_data(self, data):
        data.font = self.regular
        data.font_bold = self.bold
        data.font_italic = self.italic
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
    # Get dimenions
    # ====================================================================================================

    def get_dimensions(self, italic=False, bold=False):

        mesh = Text("aAp> <", font=self).to_mesh(transform=False, char_index=True)

        def _dims(char_index):
            cis = mesh.faces[mesh.faces.char_index==char_index].loop_index
            pts = mesh.points.position[mesh.corners[cis].vertex_index]
            return np.min(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 0]), np.max(pts[:, 1])
        
        d = {
            'y_body'   : _dims(0)[3],
            'y_ascen'  : _dims(1)[3],
            'y_descen' : _dims(2)[1],
            'x_space'  : _dims(5)[0] - _dims(3)[2]
        }
        return d

# ====================================================================================================
# Text class
# ====================================================================================================

class Text(Geometry):

    FONT_STYLE = ('BOLD', 'ITALIC', 'SMALL_CAPS', 'UNDERLINE')
    X_ALIGN = ('LEFT', 'CENTER', 'RIGHT', 'JUSTIFY', 'FLUSH')
    Y_ALIGN = ('TOP', 'TOP_BASELINE', 'CENTER', 'BOTTOM_BASELINE', 'BOTTOM')
    BEVEL_MODE = ('ROUND', 'OBJECT', 'PROFILE')

    PROPERTIES = {
        'align_x' : 'LEFT',
        'align_y' : 'TOP_BASELINE',
        'shear' : 0., 
        #'size' : 1., 
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
    
    def __init__(self, text="Text", font=None, materials=None, **kwargs):
        
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

        # Enriched text
        self.text = text

        # Properties
        self.props = {}
        for k, vdef in kwargs.items():
            if k in self._etext.attributes:
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
        self._etext = EnrichedText(value)

    @property
    def etext(self):
        return self._etext

    @etext.setter
    def etext(self, value):
        self._etext = EnrichedText(value)

    @property
    def latex(self):
        raise ValueError("Text.latex is write only")
    
    @latex.setter
    def latex(self, value):

        from pprint import pprint

        # ---------------------------------------------------------------------------
        # Parse the LateX string in mode text (not math mode)
        # ---------------------------------------------------------------------------

        parsed = parse_latex(value, math_mode=False, ignore_comments=True)

        # ---------------------------------------------------------------------------
        # Prepare list of string parts with attributes
        # ---------------------------------------------------------------------------

        switches = ['bold', 'italic', 'small_caps', 'underline']
        str_parts = []
        str_attr = []

        def _add(block, attributes):

            # ---------------------------------------------------------------------------
            # Non content are managed by owner

            assert('content' in block)

            # ---------------------------------------------------------------------------
            # Inherits params from owner
            # Enrich with proper attributes

            new_attrs = dict(attributes)
            for k, v in block.items():
                if k not in ['type', 'content']:
                    new_attrs[k] = v

            # User params
            params = {}

            # ---------------------------------------------------------------------------
            # Loop on the content

            for cnt in block['content']:

                # ----- Content is a string
                #if isinstance(cnt, str):
                #    flow.append({"string": cnt, **new_attrs, **params})
                #    continue

                # ----- Otherwise it is a dict
                
                # Switch current parameters
                if cnt['type'] == 'STRING':
                    str_parts.append(cnt['string'])
                    str_attr.append({**new_attrs, **params})

                elif cnt['type'] == 'SWITCH':
                    for k, v in cnt.items():
                        if k not in ['type']:
                            new_attrs[k] = v

                # Python parameter
                elif cnt['type'] == 'PARAM':
                    msg = None
                    try:
                        val = eval(cnt['value'])
                    except Exception as e:
                        msg = str(e)

                    if msg is not None:
                        raise Exception(f"Erreur when evaluating {block['key']}={block['value']}: {msg}")
                    
                    if cnt['key'] in new_attrs.keys():
                        new_attrs[cnt['key']] = val
                    else:
                        params[cnt['key']] = val
                    
                elif 'content' in cnt:
                    _add(cnt, new_attrs)

                else:
                    print("-"*10)
                    pprint(block)
                    assert(False)

        _add(parsed, {k: False for k in switches})

        # ---------------------------------------------------------------------------
        # Prepare list of string parts with attributes
        # ---------------------------------------------------------------------------

        self._etext = EnrichedText("".join(str_parts))
        index = 0
        for part, attr in zip(str_parts, str_attr):
            n = len(part)
            for k, v in attr.items():
                if k in self._etext.attributes:
                    if k == 'color':
                        self._etext[k][index:index + n] = Color(v).rgba
                    else:
                        self._etext[k][index:index + n] = v

            index += n





            


            






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
    # To and from Text Data (TextCurve)
    # ====================================================================================================

    @classmethod
    def from_data(cls, data):

        # Initialize Text
        txt = cls(
            data.body, 
            font = Font.from_data(data),
            materials = cls.materials_from_data(data), 
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

    def to_mesh(self, transform=True, char_index=True):

        # Char index
        if char_index:
            mats = self.materials
            matind = np.array(self._etext.material_index)
            self.materials = [None] * len(matind)
            self._etext.material_index = np.arange(len(matind), dtype=bint)

        # Create the mesh
        with self.object(readonly=True) as obj:
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
    
    def to_curve(self, transform=True, char_index=True):

        # Char index
        if char_index:
            mats = self.materials
            matind = np.array(self._etext.material_index)
            self.materials = [None] * len(matind)
            self._etext.material_index = np.arange(len(matind), dtype=bint)

        # Create the curve
        with self.object(readonly=True) as obj:
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

