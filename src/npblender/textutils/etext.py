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

# ====================================================================================================
# Enriched text
# ====================================================================================================

class EnrichedText:
    def __init__(self, text,
            bold=False, 
            italic=False, 
            small_caps=False, 
            material_index=0,
            underline=False, 
            kerning=0.0, 
            size=1.0, 
            color = (1, 1, 1, 1),
    ):
        self.text = text

        self.bold = bold
        self.italic = italic
        self.small_caps = small_caps
        self.material_index = material_index
        self.underline = underline
        self.kerning = kerning
        self.size = size
        self.color = color

    def __str__(self):
        return "".join(self._data.char)
    
    # ----------------------------------------------------------------------------------------------------
    # Clone
    # ----------------------------------------------------------------------------------------------------

    def clone(self):
        txt = EnrichedText()
        txt.text = self
        return txt

    # ----------------------------------------------------------------------------------------------------
    # Attribute names
    # ----------------------------------------------------------------------------------------------------

    @property
    def attributes(self):
        return list(self._data.dtype.names[1:])

    # ----------------------------------------------------------------------------------------------------
    # Text properties
    # ----------------------------------------------------------------------------------------------------

    @property
    def text(self):
        return "".join(self._data.char)
    
    @text.setter
    def text(self, value):
        if isinstance(value, EnrichedText):
            self._data = np.array(value._data).view(np.recarray)
        else:
            self._data = np.zeros(len(value), dtype=ET_DTYPE).view(np.recarray)
            self._data.char = list(value)

    @property
    def bold(self):
        return self._data.bold
    
    @bold.setter
    def bold(self, value):
        self._data.bold = value

    @property
    def italic(self):
        return self._data.italic
    
    @italic.setter
    def italic(self, value):
        self._data.italic = value

    @property
    def underline(self):
        return self._data.underline
    
    @underline.setter
    def underline(self, value):
        self._data.underline = value

    @property
    def small_caps(self):
        return self._data.small_caps
    
    @small_caps.setter
    def small_caps(self, value):
        self._data.small_caps = value

    @property
    def size(self):
        return self._data.size
    
    @size.setter
    def size(self, value):
        self._data.size = value

    @property
    def material_index(self):
        return self._data.material_index
    
    @material_index.setter
    def material_index(self, value):
        self._data.material_index = value

    @property
    def kerning(self):
        return self._data.kerning
    
    @kerning.setter
    def kerning(self, value):
        self._data.kerning = value

    @property
    def color(self):
        return self._data.color
    
    @color.setter
    def color(self, value):
        self._data.color = value

    # ----------------------------------------------------------------------------------------------------
    # To Blender body_data
    # ----------------------------------------------------------------------------------------------------

    def to_body_format(self, body_format):
        for name, bl_name in BL_ATTRS.items():
            a = np.array(self._data[name]).ravel()
            body_format.foreach_set(bl_name, a)

    def from_body_format(self, body_format):
        for name, bl_name in BL_ATTRS.items():
            a = np.empty_like(self._data[name])
            body_format.foreach_get(bl_name, a.ravel())
            self._data[name] = a

    # ----------------------------------------------------------------------------------------------------
    # Dunder
    # ----------------------------------------------------------------------------------------------------

    def __len__(self):
        return self._data.shape[0]
    
    def _other_to_rec(self, other):
        if isinstance(other, EnrichedText):
            return other._data
        elif isinstance(other, str):
            return EnrichedText(other)._data
        else:
            return EnrichedText(str(other))._data

    def __getitem__(self, index):

        if isinstance(index, str):
            return self._data[index]
        
        sub = self._data[index]
        out = object.__new__(type(self))
        out.dtype = self.dtype

        if isinstance(index, (int, np.integer)):
            buf = np.zeros(1, dtype=ET_DTYPE).view(np.recarray)
            buf[0] = sub
            out._data = buf

        else:
            out._data = sub.view(np.recarray)

        return out

    def __setitem__(self, index, value):

        if isinstance(index, str):
            self._data[index] = value
            return

        rec = self._other_to_rec(value)

        if isinstance(index, (int, np.integer)):
            self._data[index] = rec[0]

        else:
            target = self._data[index]
            if rec.shape[0] == target.shape[0]:
                self._data[index] = rec
            elif rec.shape[0] == 1:
                for name in self._data.dtype.names:
                    target[name] = rec[name][0]

    def __add__(self, other):
        right = self._other_to_rec(other)
        out = object.__new__(type(self))
        out.dtype = ET_DTYPE
        out._data = np.concatenate([self._data, right]).view(np.recarray)
        return out

    def __radd__(self, other):
        right = self._other_to_rec(other)
        out = object.__new__(type(self))
        out.dtype = ET_DTYPE
        out._data = np.concatenate([right, self._data]).view(np.recarray)
        return out

    def __iadd__(self, other):
        right = self._other_to_rec(other)
        self._data = np.concatenate([self._data, right]).view(np.recarray)
        return self
    
    # ====================================================================================================
    # Flow
    # ====================================================================================================

    def _flow_blocks(self):

        n = len(self._data)
        if n == 0:
            return []
        
        etxt = EnrichedText(" ")        
        def_vals = {name: etxt._data[name][0] for name in ET_DTYPE.names[1:]}

        blocks = {}
        def _add_block(start, end, value):
            if start is None:
                return
            cur = blocks.get(start, None)
            if cur is None:
                cur = []
                blocks[start] = cur

            cur.append((start, end, value))


        for name in ET_DTYPE.names[1:]:

            # Values to explore and default value
            values = self._data[name]
            def_val = def_vals[name]

            # Block info
            start = None
            block_val = def_val

            # Loop
            for index, cur_val in enumerate(values):
                if cur_val == block_val:
                    continue

                _add_block(start, index, block_val)

                block_val = cur_val
                # Block start if not default
                start = None if block_val == def_val else index

            _add_block(start, n, block_val)

        # Sort the block lists
        for k in blocks:
            blocks[k] = sorted(blocks[k], key=lambda x: -x[1])

        # Imbricated blocks
        flow = {}

        raw_text = "".join(self._data.char)
        index = 0
        for start, blist in blocks.items():
            if start > index:
                flow.append(raw_text[start:index])
            flow.appen










