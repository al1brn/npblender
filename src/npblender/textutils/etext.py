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
    'bold': 'use_bold',
    'italic': 'use_italic',
    'small_caps': 'use_small_caps',
    'underline': 'use_underline',
    'material_index': 'material_index',
    'kerning': 'kerning',
}

__all__ = ["EText", "CharStyle"]

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
# Enriched text
# ====================================================================================================

class EText:

    __slots__ = ["_cstyle", "_data", "_len"]

    def __init__(self, text=None, **styles):

        self._len  = 0
        self._cstyle = CharStyle(**styles)

        if text is None:
            self._data = np.zeros(100, dtype=ET_DTYPE).view(np.recarray)

        else:
            self._data = np.zeros(len(text), dtype=ET_DTYPE).view(np.recarray)

            self.text = text
            for k, v in styles.items():
                setattr(self, k, v)

    def __str__(self):
        return "".join(self._data.char[:self._len])
    
    def __repr__(self):
        return str(self)
    
    def dump(self):
        n = 60
        return "\n".join([
            "> " + "".join(self._data.char[:n]),
            "B " + "".join(["X" if ok else "." for ok in self.bold[:n]]),
            "I " + "".join(["X" if ok else "." for ok in self.italic[:n]]),
            "U " + "".join(["X" if ok else "." for ok in self.underline[:n]]),
            "S " + "".join(["X" if ok else "." for ok in self.small_caps[:n]]),
        ])
    
    # ----------------------------------------------------------------------------------------------------
    # Buffer management
    # ----------------------------------------------------------------------------------------------------

    def _ensure_len(self, new_len):

        if self._data is not None and len(self._data) >= new_len:
            return

        buf_len = (new_len // 100)*100 + 100
        assert(buf_len >= new_len)

        data = np.zeros(buf_len, dtype=ET_DTYPE).view(np.recarray)

        # Default
        for k, v in CharStyle.STYLES.items():
            data[k] = v

        # Existing chars
        if self._data is not None:
            data[:self._len] = self._data[:self._len]

        self._data = data
    
    # ----------------------------------------------------------------------------------------------------
    # Clone
    # ----------------------------------------------------------------------------------------------------

    def clone(self):
        txt = EText()
        txt.text = self
        txt.default.set(**self.default.as_dict())
        return txt
    
    def clear(self):
        self._len = 0

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
        return self._len

    @property
    def text(self):
        return "".join(self._data.char[:self._len])
    
    @text.setter
    def text(self, value):
        if isinstance(value, EText):
            self._data = np.array(value._data).view(np.recarray)
            self._len = value._len

        else:
            self._ensure_len(len(value))
            self._len = len(value)
            self._data.char[:self._len] = list(value)

            for k, v in self.default:
                setattr(self, k, v)

    # ----------------------------------------------------------------------------------------------------
    # Styles
    # ----------------------------------------------------------------------------------------------------

    def __getattr__(self, name):

        if name in CharStyle.STYLES:
            return self._data[name][:self._len]
        
        raise AttributeError(f"No style named '{name}'. Valid styles are {list(CharStyle.STYLES.keys())}")
    
    def __setattr__(self, name, value):

        if name in ['_data', '_len', '_cstyle']:
            object.__setattr__(self, name, value)
            return

        elif name in CharStyle.STYLES:
            self._data[name][:self._len] = value
            return
        
        super().__setattr__(name, value)

        #raise AttributeError(f"No style named '{name}'. Valid styles are {list(CharStyle.STYLES.keys())}")

    def __getitem__(self, name):
        return getattr(self, name)

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

        new_len = self._len + len(text)
        self._ensure_len(new_len)
        
        if isinstance(text, EText):
            self._data[self._len:new_len] = text._data[:text._len]
        
        else:
            text = str(text)
            self._data['char'][self._len:new_len] = list(text)

            for k, v in self.default:
                if k in styles:
                    self._data[k][self._len:new_len] = styles[k]
                else:
                    self._data[k][self._len:new_len] = v

        self._len = new_len

    def extract(self, index0=0, index1=None):

        if index0 < 0:
            index0 = self._len + index0

        if index1 is None:
            index1 = self._len
        elif index1 < 0:
            index1 = self._len + index1

        data = np.array(self._data[index0:index1]).view(np.recarray)

        etext = EText()
        etext._data = data
        etext._len  = len(data)

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


