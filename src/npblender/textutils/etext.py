import numpy as np

ET_DTYPE = np.dtype([
            ('char',           'U8'),
            ('bold',           '?'),
            ('italic',         '?'),
            ('small_caps',     '?'),
            ('underline',      '?'),
            ('material_index', 'i4'),
            ('size',           'f4'),
            ('kerning',        'f4'),
            ('color',          'f4', (4,)),
        ])


BL_ATTRS = {
    'bold'           : 'use_bold',
    'italic'         : 'use_italic',
    'small_caps'     : 'use_small_caps',
    'underline'      : 'use_underline',
    'material_index' : 'material_index',
    'kerning'        : 'kerning',
}

__all__ = ["CharStyle", "EText", "EChar", "StyleContext"]

# ====================================================================================================
# Style
# ====================================================================================================

class CharStyle:
            
    STYLES = {
        'bold'              : False,
        'italic'            : False,
        'small_caps'        : False,
        'underline'         : False,
        'material_index'    : 0,
        'size'              : 1.0,
        'kerning'           : 0.0,
        'color'             : (1.0, 1.0, 1.0, 1.0),
    }
    
    __slots__ = ["_styles"]

    def __init__(self, **styles):
        
        object.__setattr__(self, "_styles", dict(self.STYLES))

        for name, default in styles.items():
            self[name] = default

    def __getattr__(self, name):
        if name in self._styles:
            return self._styles[name]
        raise AttributeError(f"No style named '{name}'. Valid styles are {list(self._styles.keys())}")
    
    def __setattr__(self, name, value):
        
        if name in self._styles:
            if name == 'color':

                OK = True
                try:
                    from ..maths import Color
                except:
                    OK = False

                if OK:
                    self._styles[name] = Color.to_rgba(value)
                else:
                    self._styles[name] = value

            else:
                self._styles[name] = value
            return
        
        raise AttributeError(f"No style named '{name}'. Valid styles are {list(self._styles.keys())}")
    
    def __getitem__(self, index):
        return self.__getattr__(index)
    
    def __setitem__(self, index, value):
        self.__setattr__(index, value)
    
    def __iter__(self):
        for k, v in self._styles.items():
            yield k, v
    
    def as_dict(self, **styles):
        return {k: styles[k] if k in styles else v for k, v in self._styles.items()}
    
    def set(self, **styles):
        for k, v in styles.items():
            self[k] = v

# ====================================================================================================
# Style context
# ====================================================================================================

class StyleContext:
    
    __slots__ = ["_styles"]

    def __init__(self, **styles):
        self._styles = {}
        for k, v in styles.items():
            setattr(self, k, v)

    def push(self, **styles):
        new_styles = {**self._styles}
        for k in CharStyle.STYLES:
            if k in styles and styles[k] is not None:
                new_styles[k] = styles[k]

        return StyleContext(**new_styles)

    def __setattr__(self, name, value):
        if name in CharStyle.STYLES:
            self._styles[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in CharStyle.STYLES:
            return self._styles.get(name, CharStyle.STYLES[name])
        else:
            raise AttributeError(f"'{name}' is not a valid char style name.")

    def as_dict(self):
        d = {}
        for k, v in self._styles.items():
            if v is not None:
                d[k] = v
        return d


# ====================================================================================================
# EChar
# ====================================================================================================

class EChar:
    def __init__(self, c, **styles):
        self.c = c
        for k in CharStyle.STYLES:
            if k in styles:
                setattr(self, k, styles[k])
            else:
                setattr(self, k, CharStyle.STYLES[k])

        for k in styles:
            if k not in CharStyle.STYLES:
                raise AttributeError(f"Unknown char style: {k}={v}")
            
    def __str__(self):
        return self.c
    
    def __repr__(self):
        return self.c
    
    def dump(self):
        s = f"<EChar {self.c}"
        for k in CharStyle.STYLES:
            s += f", {k}: {getattr(self, k)}"
        return s + ">"
    

# ====================================================================================================
# Enriched text
# ====================================================================================================

class EText:

    STYLES = CharStyle.STYLES

    __slots__ = ["_cstyle", "_data"]

    def __init__(self, text=None, **styles):

        self._cstyle = CharStyle(**styles)
        self._data = None

        if text is not None:
            self.text = text

            for k, v in styles.items():
                setattr(self, k, v)

    def __str__(self):
        if self._data is None:
            return ""
        else:
            return "".join(self._data.char)
    
    def __repr__(self):
        return str(self)
    
    def dump(self):
        if self._data is None:
            return "<Empty EText>"
        n = 60
        return "\n".join([
            "> " + "".join(self._data.char[:n]),
            "B " + "".join(["X" if ok else "." for ok in self.bold[:n]]),
            "I " + "".join(["X" if ok else "." for ok in self.italic[:n]]),
            "U " + "".join(["X" if ok else "." for ok in self.underline[:n]]),
            "S " + "".join(["X" if ok else "." for ok in self.small_caps[:n]]),
        ])
    
    # ----------------------------------------------------------------------------------------------------
    # Init data
    # ----------------------------------------------------------------------------------------------------

    def _init_data(self, length):
        if length == 0:
            self._data = None
        elif self._data is None:
            self._data = np.zeros(length, dtype=ET_DTYPE).view(np.recarray)
        else:
            self._data = np.resize(self._data, length).view(np.recarray)

    # ----------------------------------------------------------------------------------------------------
    # Clone
    # ----------------------------------------------------------------------------------------------------

    def clone(self):
        txt = EText()
        if len(self):
            txt.text = self
        
        txt.default.set(**self.default.as_dict())
        return txt
    
    def clear(self):
        self._data = None

    # ----------------------------------------------------------------------------------------------------
    # Styles
    # ----------------------------------------------------------------------------------------------------

    @property
    def attributes(self):
        return list(CharStyle.STYLES.keys())
    
    @property
    def default(self):
        return self._cstyle

    # ----------------------------------------------------------------------------------------------------
    # Text as str
    # ----------------------------------------------------------------------------------------------------

    def __len__(self):
        if self._data is None:
            return 0
        else:
            return len(self._data)

    @property
    def text(self):
        if self._data is None:
            return ""
        else:
            return "".join(self._data.char)
    
    @text.setter
    def text(self, value):

        ok_color = True
        try:
            from ..maths import Color
        except:
            ok_color = False

        if isinstance(value, EText):
            self._data = np.array(value._data).view(np.recarray)

        elif isinstance(value, EChar):
            self.text = value.c
            for k, v in CharStyle.STYLES.items():
                v = getattr(value, k)
                if v is not None:
                    setattr(self, k, v)


        # A list of strs or EChars
        elif isinstance(value, list):
            s = ""
            for ec in value:
                if isinstance(ec, str):
                    s += ec
                else:
                    s += ec.c

            self.text = s
            index = 0
            for ec in value:
                if isinstance(ec, str):
                    continue

                n = len(ec.c)
                for k in CharStyle.STYLES:
                    v = getattr(ec, k)
                    if k == 'color':
                        if ok_color:
                            self._data[k][index:index + n] = Color.to_rgba(v)
                        else:
                            self._data[k][index:index + n] = v

                    else:
                        self._data[k][index:index + n] = v
                index += n

        elif len(value) == 0:
            self._data = None

        else:
            self._init_data(len(value))
            self._data.char[:] = list(value)

            for k, v in self.default:
                setattr(self, k, v)

    # ----------------------------------------------------------------------------------------------------
    # Styles
    # ----------------------------------------------------------------------------------------------------

    def __getattr__(self, name):

        if name in CharStyle.STYLES:
            return self._data[name]
        
        raise AttributeError(f"No style named '{name}'. Valid styles are {list(CharStyle.STYLES.keys())}")
    
    def __setattr__(self, name, value):

        if name in ['_data', '_cstyle']:
            object.__setattr__(self, name, value)
            return

        elif name in CharStyle.STYLES:
            self._data[name] = value
            return
        
        super().__setattr__(name, value)

    def __getitem__(self, name):
        return getattr(self, name)
    
    def __setitem__(self, name, value):
        setattr(self, name, value)

    # ----------------------------------------------------------------------------------------------------
    # To Blender body_data
    # ----------------------------------------------------------------------------------------------------

    def to_body_format(self, body_format):
        for name, bl_name in BL_ATTRS.items():
            a = np.array(self[name]).ravel()
            body_format.foreach_set(bl_name, a)

    def from_body_format(self, body_format):
        for name, bl_name in BL_ATTRS.items():
            a = np.empty_like(self[name])
            body_format.foreach_get(bl_name, a.ravel())
            self._data[name] = a

    # ----------------------------------------------------------------------------------------------------
    # Dunder
    # ----------------------------------------------------------------------------------------------------

    def append(self, text, **styles):

        cur_len = len(self)
        new_len = cur_len + len(text)
        self._init_data(new_len)
        
        if isinstance(text, EText):
            self._data[cur_len] = text._data
        
        else:
            text = str(text)
            self._data['char'][cur_len:] = list(text)

            for k, v in self.default:
                if k in styles:
                    self._data[k][cur_len:] = styles[k]
                else:
                    self._data[k][cur_len:] = v

    def extract(self, index0=0, index1=None):

        if index0 < 0:
            index0 = len(self) + index0

        if index1 is None:
            index1 = len(self)
        elif index1 < 0:
            index1 = len(self) + index1

        etext = EText()
        etext._data = np.array(self._data[index0:index1]).view(np.recarray)

        return etext

    def __add__(self, other):
        etext = self.clone()
        etext.append(other)
        return etext

    def __radd__(self, other):
        etext = EText(other)
        etext.append(self)
        return etext

    def __iadd__(self, other):
        self.append(other)
        return self
    
if __name__ == '__main__':

    cs = CharStyle()
    cs.italic = True
    cs.bold = False
    print(cs.as_dict())

    txt = EText("Hello")
    txt.default.small_caps = True
    txt.append(" folk", italic=True)
    print(txt)
    txt.bold[1] = True
    print(repr(txt))

    print(txt.extract(2, 5))

    txt = EText()
    txt.append(">Hello")
    print(txt)
    txt.clear()
    print(txt)
    txt.append("abc")
    print(txt)

    ec = EChar("I", italic=True)
    print(ec.dump())

    et = EText(ec)
    print(et.dump())

    a = EChar("Italic", italic=True)
    b = EChar("Bold", bold=True)
    c = "string"

    et = EText([a, b, c])
    print(et.dump())

    sc = StyleContext(italic=True)
    print(sc.italic, sc.bold, sc.as_dict())



