import numpy as np

__all__ = ["CSS_HEX", "Color"]

CSS_HEX = {
    "black":"#000000","white":"#FFFFFF","gray":"#808080","silver":"#C0C0C0",
    "red":"#FF0000","maroon":"#800000","yellow":"#FFFF00","gold":"#FFD700",
    "orange":"#FFA500","brown":"#A52A2A","lime":"#00FF00","green":"#008000",
    "olive":"#808000","teal":"#008080","blue":"#0000FF","navy":"#000080",
    "aqua":"#00FFFF","cyan":"#00FFFF","turquoise":"#40E0D0",
    "fuchsia":"#FF00FF","magenta":"#FF00FF","purple":"#800080",
    "violet":"#EE82EE","indigo":"#4B0082","pink":"#FFC0CB","beige":"#F5F5DC",
}

from .constants import bfloat


class Color:
    __slots__ = ("_rgba",)

    def __init__(self, color):
        try:
            import mathutils
            HAS_MU = True
        except Exception:
            HAS_MU = False
            mathutils = None

        if isinstance(color, Color):
            self._rgba = np.array(color._rgba, dtype=bfloat, copy=True)

        elif HAS_MU and isinstance(color, mathutils.Color):
            self._rgba = np.array([color.r, color.g, color.b, 1.0], dtype=bfloat)

        elif isinstance(color, np.uint64):
            self._rgba = Color.from_int(color)._rgba.astype(bfloat, copy=False)

        elif isinstance(color, str):
            name = color.strip().lower()
            if name in CSS_HEX:
                color = CSS_HEX[name]
            self._rgba = Color.from_hex(color)._rgba.astype(bfloat, copy=False)

        else:
            a = np.asarray(color)
            if np.issubdtype(a.dtype, np.integer):
                a = a / 255.0
            a = a.astype(bfloat, copy=False)

            if a.ndim == 0 or a.shape == (1,):
                v = float(a)
                self._rgba = np.array([v, v, v, 1.0], dtype=bfloat)
            elif a.shape == (3,):
                self._rgba = np.array([a[0], a[1], a[2], 1.0], dtype=bfloat)
            elif a.shape == (4,):
                self._rgba = np.array(a, dtype=bfloat, copy=True)
            else:
                raise AttributeError(f"A Color can't be initialized with an array of shape {a.shape}.")
            
        self._clamp()

    def _clamp(self):
        np.clip(self._rgba, 0.0, 1.0, out=self._rgba)

    def __str__(self):
        return self.to_hex()

    def __repr__(self) -> str:
        r, g, b, a = self.rgba
        return f"Color(r={r:.6f}, g={g:.6f}, b={b:.6f}, a={a:.6f})"

    # ----------------------------------------------------------------------------------------------------
    # As RGB
    # ----------------------------------------------------------------------------------------------------

    @property
    def r(self):
        return self._rgba[0]
    
    @r.setter
    def r(self, value):
        self._rgba[0] = value
        self._clamp()

    @property
    def g(self):
        return self._rgba[1]
    
    @g.setter
    def g(self, value):
        self._rgba[1] = value
        self._clamp()

    @property
    def b(self):
        return self._rgba[2]
    
    @b.setter
    def b(self, value):
        self._rgba[2] = value
        self._clamp()

    @property
    def alpha(self):
        return self._rgba[3]
    
    @alpha.setter
    def alpha(self, value):
        self._rgba[3] = value
        self._clamp()

    @property
    def alpha(self):
        return self._rgba[3]
    
    @alpha.setter
    def alpha(self, value):
        self._rgba[3] = value
        self._clamp()

    @property
    def rgb(self):
        return np.array(self._rgba[:3])
    
    @rgb.setter
    def rgb(self, value):
        self._rgba[:3] = value
        self._clamp()

    @property
    def rgba(self):
        return np.array(self._rgba)
    
    @rgba.setter
    def rgba(self, value):
        self._rgba[:] = value
        self._clamp()
    
    # ----------------------------------------------------------------------------------------------------
    # Blender color
    # ----------------------------------------------------------------------------------------------------

    @property
    def bl_color(self):
        import mathutils

        return mathutils.Color(self._rgba[:3])
    
    # ----------------------------------------------------------------------------------------------------
    # Array of rgba
    # ----------------------------------------------------------------------------------------------------

    @staticmethod
    def to_rgba(value):

        try:
            return Color(value).rgba
        except:
            pass

        if hasattr(value, '__len__'):
            try:
                return np.array([Color(v).rgba for v in value], dtype=bfloat)
            except:
                pass

        raise AttributeError(f"Impossible to get RGBA from {value}")
    
    # ----------------------------------------------------------------------------------------------------
    # Hexa
    # ----------------------------------------------------------------------------------------------------

    @staticmethod
    def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
        out = rgb.copy()
        mask = out <= 0.0031308
        out[mask] = 12.92 * out[mask]
        out[~mask] = 1.055 * np.power(out[~mask], 1.0/2.4) - 0.055
        return out

    @staticmethod
    def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
        out = rgb.copy()
        mask = out <= 0.04045
        out[mask] = out[mask] / 12.92
        out[~mask] = np.power((out[~mask] + 0.055) / 1.055, 2.4)
        return out

    def to_hex(self, *, srgb: bool = True, upper: bool = True) -> str:
        rgba = self._rgba.copy()
        if srgb:
            rgba[:3] = self._linear_to_srgb(rgba[:3])
        # clamp & quantification 0..255
        c8 = np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8)
        r, g, b, a = (int(c8[0]), int(c8[1]), int(c8[2]), int(c8[3]))
        s = f"#{r:02X}{g:02X}{b:02X}{a:02X}"
        return s if upper else s.lower()

    @classmethod
    def from_hex(cls, s: str, *, srgb: bool = True) -> "Color":
        s = s.strip()
        if s.startswith('#'):
            s = s[1:]
        if len(s) not in (6, 8):
            raise ValueError("Expected Hex: #RRGGBB or #RRGGBBAA")
        if len(s) == 6:
            s += "FF"
        r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16); a = int(s[6:8], 16)
        rgba = np.array([r, g, b, a], dtype=bfloat) / 255.0
        if srgb:
            rgba[:3] = cls._srgb_to_linear(rgba[:3])
        return cls(rgba.tolist())
    
    # ----------------------------------------------------------------------------------------------------
    # Conversion to int
    # ----------------------------------------------------------------------------------------------------

    def to_int(self) -> np.uint64:
        """
        Pack RGBA as 16-bit per channel into a uint64:
        (R<<48 | G<<32 | B<<16 | A).
        """
        # scale & round to 16-bit
        c16 = np.clip(np.rint(self._rgba * 65535.0), 0, 65535).astype(np.uint64)
        r16, g16, b16, a16 = c16
        packed = (r16 << 48) | (g16 << 32) | (b16 << 16) | a16
        return np.uint64(packed)

    @classmethod
    def from_int(cls, value: int) -> "Color":
        """
        Unpack a uint64 encoded as (R<<48 | G<<32 | B<<16 | A) into RGBA floats in [0,1].
        """
        v = np.uint64(value)
        r16 = (v >> 48) & 0xFFFF
        g16 = (v >> 32) & 0xFFFF
        b16 = (v >> 16) & 0xFFFF
        a16 = v & 0xFFFF
        rgba = np.array([r16, g16, b16, a16], dtype=bfloat) / 65535.0
        return cls(rgba)
    
    # ----------------------------------------------------------------------------------------------------
    # HSV
    # ----------------------------------------------------------------------------------------------------

    @property
    def hsv(self):
        import colorsys

        r, g, b = self._rgba[:3]
        return np.array(colorsys.rgb_to_hsv(r=r, g=g, b=b))
    
    @hsv.setter
    def hsv(self, value):
        import colorsys

        try:
            h, s, v = value
        except:
            raise ValueError("HSV must be a 3-tuple of floats in [0,1].")

        self._rgba[:3] = colorsys.hsv_to_rgb(h=h, s=s, v=v)
        self._clamp()

    @property
    def h(self):
        return self.hsv[0]
    
    @h.setter
    def h(self, value):
        hsv = self.hsv
        hsv[0] = value
        self.hsv = hsv
    
    @property
    def s(self):
        return self.hsv[1]
    
    @s.setter
    def s(self, value):
        hsv = self.hsv
        hsv[1] = value
        self.hsv = hsv

    @property
    def v(self):
        return self.hsv[2]
    
    @v.setter
    def v(self, value):
        hsv = self.hsv
        hsv[2] = value
        self.hsv = hsv

    # ----------------------------------------------------------------------------------------------------
    # HSL
    # ----------------------------------------------------------------------------------------------------

    @property
    def hsl(self):
        import colorsys

        r, g, b = self._rgba[:3]
        h, l, s = colorsys.rgb_to_hls(r=r, g=g, b=b)
        return np.array([h, s, l], dtype=bfloat)

    
    @hsl.setter
    def hsl(self, value):
        import colorsys

        try:
            h, s, l = value
        except:
            raise ValueError("HSL must be a 3-tuple of floats in [0,1].")

        self._rgba[:3] = colorsys.hls_to_rgb(h=h, l=l, s=s)
        self._clamp()

    @property
    def l(self):
        return self.hsl[2]
    
    @l.setter
    def l(self, value):
        hsl = self.hsl
        hsl[2] = value
        self.hsl = hsl

        
    


    

    



