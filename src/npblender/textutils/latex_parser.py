from pprint import pprint

if __name__ == '__main__':
    from latex_codes import SYMBOLS, MATHBB, GREEK
    from etext import EText, EChar, StyleContext
else:
    from .latex_codes import SYMBOLS, MATHBB, GREEK
    from .etext import EText,EChar, StyleContext

__all__ = ["parse_latex"]

# ----------------------------------------------------------------------------------------------------
# Commandw which can be followed by an option [...]
# ----------------------------------------------------------------------------------------------------

OPTION_COMMANDS = ['sqrt', 'color', 'materialindex', 'material', 'matindex', 'term', 'ph']

# ----------------------------------------------------------------------------------------------------
# Text mode commands
# ----------------------------------------------------------------------------------------------------

TEXT_SWITCHES = {
    #'tiny': {'font_size': .3}, 
    #'small': {'font_size': .6}, 
    #'normalsize': {'font_size': 1.}, 
    #'large': {'font_size': 1.25}, 
    #'Large': {'font_size': 1.5}, 
    #'LARGE': {'font_size': 2.}, 
    #'huge': {'font_size': 3.}, 
    #'Huge': {'font_size': 4.},

    #'ttfamily' : {'family': 'courrier'},
    #'sffamily' : {'family': 'sans serif'},
    #'rmfamily' : {'family': 'serif'},

    # Actual commands
    'itshape': {'italic': True},
    'bfseries': {'bold': True},
    'scshape' : {'small_caps': True},

    # Natural commands
    'italic'    : {'italic': True},
    'bold'      : {'bold': True},
    'smallcaps' : {'small_caps': True},
    'underline' : {'underline': True},
}

TEXT_BLOCKS = {
    'text'       : {},
    'textit'     : {'italic': True},
    'textbf'     : {'bold': True},
    'emph'       : {'bold': True, 'italic': True},
    'underline'  : {'underline': True},
    'textnormal' : {'bold': False, 'italic': False, 'family': False, 'underline': False},

    #'textrm'     : {'family': 'serif'},
    #'textsf'     : {'family': 'sans serif'}, 
    #'texttt'     : {'family': 'courrier'},
}

# Note : Hard coded
# - color
# - matindex, material, materialindex (synonyms)

# ----------------------------------------------------------------------------------------------------
# Math commands
# ----------------------------------------------------------------------------------------------------

MATH_PATTERNS = {
  "FRAC_PAT":    {"tokens": 2},
  "SQRT_PAT":    {"tokens": 1, "option": True},  
  "ACCENT_PAT":  {"tokens": 1},
  "INT_PAT":     {"tokens": 1, "scripts": True},
  "PH_PAT":      {"tokens": 0, "option": True}
}

MATH_PATTERN_COMMANDS = {
  "frac":   "FRAC_PAT",
  "dfrac":  "FRAC_PAT",
  "tfrac":  "FRAC_PAT",
  "binom":  "FRAC_PAT",
  "sqrt":   "SQRT_PAT",
  "boxed":  "ACCENT_PAT",

  "int":    "INT_PAT",
  "iint":   "INT_PAT",
  "iiint":  "INT_PAT",
  "oint":   "INT_PAT",
  "oiint":  "INT_PAT",
  "oiiint": "INT_PAT",
  "sum":    "INT_PAT",
  "prod":   "INT_PAT",
  "lim":    "INT_PAT",
  "limsup": "INT_PAT",
  "liminf": "INT_PAT",
  "mathbb": "ACCENT_PAT",

  # Naming the terms / placeholders
  "term"    : "SQRT_PAT",
  "ph"      : "PH_PAT",
}


# Mapping for LaTeX-like decorators placed above or below an expression.
# Keys are command names WITHOUT the backslash.
# 'char': LaTeX symbol (with backslash) or a Python string character
# 'fix' : False → stretch across the whole width; True → small fixed accent
# 'over'/'under': where the decoration sits

DECORATORS = {
    # ===== Arrows above (stretchable) =====
    'overrightarrow'     : {'char': '→',    'fix': False, 'under': False, 'elon_mode': 'LEFT'},
    'overleftarrow'      : {'char': '←',    'fix': False, 'under': False, 'elon_mode': 'RIGHT'},
    'overleftrightarrow' : {'char': '↔',    'fix': False, 'under': False, 'elon_mode': 'SHIFT'},
    'xrightarrow'        : {'char': '→',    'fix': False, 'under': False, 'elon_mode': 'LEFT'},
    'xleftarrow'         : {'char': '←',    'fix': False, 'under': False, 'elon_mode': 'RIGHT'},

    # ===== Arrows below (stretchable) =====
    'underrightarrow'    : {'char': '→',    'fix': False, 'under': True, 'elon_mode': 'LEFT'},
    'underleftarrow'     : {'char': '←',    'fix': False, 'under': True, 'elon_mode': 'RIGHT'},
    'underleftrightarrow': {'char': '↔',    'fix': False, 'under': True, 'elon_mode': 'SHIFT'},

    # ===== Vector accent (fixed) =====
    'vec'                : {'char': '→',    'fix': False, 'under': False, 'elon_mode': 'LEFT'}, 

    # ===== Hats / tildes =====
    'hat'                : {'char': '^',    'fix': True,  'under': False},
    'widehat'            : {'char': '^',    'fix': False, 'under': False},
    'tilde'              : {'char': '~',    'fix': True,  'under': False},
    'widetilde'          : {'char': '~',    'fix': False, 'under': False},

    # ===== Bars / overlines =====
    'bar'                : {'char': '-',    'fix': True,  'under': False}, # MACRON
    'overline'           : {'char': '-',    'fix': False, 'under': False}, # OVERLINE
    'underline'          : {'char': '_',    'fix': False, 'under': True},  # text-style underline (stretches)

    # ===== Dots & small accents (fixed, above) =====
    'dot'                : {'char': '.',    'fix': True,  'under': False},
    'ddot'               : {'char': '..',   'fix': True,  'under': False},
    'dddot'              : {'char': '...',  'fix': True,  'under': False},
    'ddddot'             : {'char': '....', 'fix': True,  'under': False},
    'breve'              : {'char': '˘',    'fix': True,  'under': False}, # 
    'check'              : {'char': 'ˇ',    'fix': True,  'under': False}, # 
    'grave'              : {'char': 'ˋ',    'fix': True,  'under': False}, # 
    'acute'              : {'char': '´',    'fix': True,  'under': False}, # 
    'mathring'           : {'char': '˚',    'fix': True,  'under': False}, # 

    # ===== Braces / brackets / parens (stretchable) =====
    'overbrace'          : {'char': '⏞',    'fix': False, 'under': False},
    'underbrace'         : {'char': '⏟',    'fix': False, 'under': True},
    'overbracket'        : {'char': '⎴',    'fix': False, 'under': False},
    'underbracket'       : {'char': '⎵',    'fix': False, 'under': True},
    'overparen'          : {'char': '⏜',    'fix': False, 'under': False},
    'underparen'         : {'char': '⏝',    'fix': False, 'under': True},
}


# ====================================================================================================
# Exposed methods
# ====================================================================================================

def parse_latex(text, math_mode=False):
    """LaTeX parser.
    """

    parser = Parser(text, math_mode=math_mode, ignore_comments=True)
    block = parser.read_block(None)

    if False:
        print("BLOCK.VALUE")
        for c in block.value:
            print(f" - {str(c)}")
        print()

    dct = block.value.to_formula(math_mode, None)

    if math_mode:
        return dct
    
    strs = []
    def _merge(d):
        if d['type'] == 'STRING':
            strs.append(d['string'])
        elif d['type'] == 'BLOCK':
            for c in d['content']:
                _merge(c)

    _merge(dct)
    return EText(strs)

# ====================================================================================================
# Char
# ====================================================================================================

class Char:

    ALPHA = "abcdefghijklmnopqrstuvwxyz"
    NUM   = "0123456789"

    KINDS = [
        'EOF',
        'special', # \ ^_ { }
        'blank',   # space, \n \t
        'char',    # rest : a, b ( )
        ]
    
    def __init__(self, kind, value, index):

        assert(kind in self.KINDS)

        self.kind = kind
        self.value = value
        self.index = index

    def __str__(self):
        return f"<{self.kind}: '{self.value}'>"

    def __eq__(self, other):
        return self.kind == other.kind and self.value == other.value
    
    @classmethod
    def command(cls, value, index):
        return cls('command', value, index)
    
    @classmethod
    def char(cls, value, index):
        return cls('char', value, index)
    
    @classmethod
    def special(cls, value, index):
        return cls('special', value, index)
    
    @classmethod
    def blank(cls, value, index):
        return cls('blank', value, index)
    
    @classmethod
    def eof(cls, index):
        return cls('EOF', None, index)
    
    # ---------------------------------------------------------------------------
    # Tests
    # ---------------------------------------------------------------------------

    @property
    def is_eof(self):
        return self.kind == 'EOF'
    
    @property
    def is_blank(self):
        if self.kind == 'blank':
            return True
        elif self.kind == 'char':
            return self.value[0] in '\n\t '
        else:
            return False

    @property
    def is_alpha(self):
        return self.kind == 'char' and self.value.lower() in self.ALPHA

    @property
    def is_num(self):
        return self.kind == 'char' and self.value.lower() in self.NUM
    
    @property
    def is_text(self):
        return self.kind in ['blank', 'char']
    
# ====================================================================================================
# Token
# ====================================================================================================
    
class Token:
    KINDS = [
        'EOF',        # No more token
        'math_block', # list of math tokens
        'text_block', # list of text tokens

        'control',    # : ^ _ { }
        'command',    # : word (command \word without \)
        'text',       # : flow of chars

        # Intermediary tokens, not returned at grammar level
        'open',       # open block
        'close',      # close block
        ]
    
    def __init__(self, kind, value):

        assert(kind in self.KINDS)

        self.kind  = kind
        if isinstance(value, Char):
            self.value = value.value
            self.index = value.index
        else:
            self.value = value
            self.index = None

        self.index = None

    def __str__(self):
        s = f"<{self.kind}"
        if self.is_block:
            s += f"[{len(self.value)}], block_command: {self.block_command}>"
        else:
            s += f", {self.value}>"

        return s
    
    def __repr__(self):
        s = str(self)
        if self.is_block:
            sepa = "\n   - "
            s += sepa + sepa.join([str(tk) for tk in self.value])
        return s

    def __eq__(self, other):
        return self.kind == other.kind and self.value == other.value
    
    @classmethod
    def eof(cls):
        return cls('EOF', None)
    
    @classmethod
    def open(cls, value):
        return cls('open', value)
    
    @classmethod
    def close(cls, value):
        return cls('close', value)
    
    @classmethod
    def block(cls, math_mode, tokens):
        if math_mode:
            token = cls('math_block', tokens)
        else:
            token = cls('text_block', tokens)

        token.block_command = None
        return token
    
    @classmethod
    def control(cls, value, token):
        tk = cls('control', value)
        tk.token = token
        return tk
    
    @classmethod
    def command(cls, value):
        return cls('command', value)
    
    @classmethod
    def text(cls, value):
        return cls('text', value)
    
    # ---------------------------------------------------------------------------
    # Tests
    # ---------------------------------------------------------------------------

    @property
    def is_eof(self):
        return self.kind == 'EOF'
    
    @property
    def is_close(self):
        return self.kind == 'close'
    
    def match_open(self, open):

        if self.is_eof:
            return self.value is None
        
        assert self.is_close, str(self)

        if open.value is None:
            return False
        elif open.value == '{':
            return self.value == '}'
        elif open.value == '(':
            return self.value == ')'
        elif open.value == '[':
            return self.value == ']'
        elif open.value == '$':
            return self.value == '$'
        elif open.value == '$$':
            return self.value == '$$'
        elif open.value == '\\left':
            return self.value == '\\right'
        
        # Ooops
        
        assert False, str(open) + '...' + str(self)

    @property
    def is_math_block(self):
        return self.kind == 'math_block'
    
    @property
    def is_text_block(self):
        return self.kind == 'text_block'
    
    @property
    def is_block(self):
        return self.kind in ['math_block', 'text_block']
    
    @property
    def is_command(self):
        return self.kind == 'command'
    
    @property
    def is_control(self):
        return self.kind == 'control'
    
    @property
    def is_text(self):
        return self.kind == 'text'
    
    @property
    def is_alpha(self):
        if self.kind != 'text':
            return False
        
        for c in self.value:
            if c == ' ':
                continue
            if not c.lower() in Char.ALPHA:
                return False
            
        return True

    @property
    def is_num(self):
        if self.kind != 'text':
            return False
        
        for c in self.value:
            if not c in Char.NUM:
                return False
            
        return True
    
    @property
    def text_type(self):
        assert self.kind == 'text'
        if self.is_alpha:
            return 'alpha'
        elif self.is_num:
            return 'num'
        else:
            return 'other'
        
# ====================================================================================================
# A list of Tokens
# ====================================================================================================

class Tokens(list):
    
    def __init__(self, parser, l=None):
        """A list of tokens.

        This class plays two roles:
        - storing the list of successive tokens read from LaTeX flow of chars
        - Implementing the grammer level by reading linked tokens
        """
        self.parser = parser

        if l is not None:
            self.extend(l)
        self.index = 0

    # ====================================================================================================
    # Read the flow of tokens
    # ====================================================================================================

    @property
    def eof(self):
        return self.index >= len(self)
    
    # ----------------------------------------------------------------------------------------------------
    # Get the next token if any
    # ----------------------------------------------------------------------------------------------------

    def get_token(self, advance=True):
        if self.eof:
            return Token.eof(None)
        else:
            token = self[self.index]
            if advance:
                self.index += 1
            return token
        
    # ----------------------------------------------------------------------------------------------------
    # Is the next token _ or ^
    # ----------------------------------------------------------------------------------------------------

    @property
    def next_is_script(self):
        if self.eof:
            return False
        else:
            return self[self.index].is_control and self[self.index].value in ['^', '_']

    # ====================================================================================================
    # Grammar level
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Read scripts _ or ^if any
    # ----------------------------------------------------------------------------------------------------

    def read_scripts(self, styles):
        scripts = {}
        while self.next_is_script:

            # Look forward the control char
            ctl = self.get_token(False).value
            key = 'subscript' if ctl == '_' else 'superscript'

            # The key already exists : x^2^2
            if key in scripts:
                break

            # Actually consume the control char
            ctl = self.get_token()

            # Read a term
            token = ctl.token
            if token.is_block:
                scripts[key] = token.value.to_formula(True, styles)
            else:
                tokens = Tokens(self.parser, [token])
                scripts[key] = tokens.to_formula(True, styles)

        return scripts
    
    # ----------------------------------------------------------------------------------------------------
    # Read a term
    # ----------------------------------------------------------------------------------------------------

    def read_term(self, math_mode, styles):

        if self.eof:
            return None
        
        # Resulting dict
        dct = None

        # Loop until a grammar level token

        while dct is None:

            # Next token
            token = self.get_token()

            if False:
                print("\n---- read_term loop:", token)

            # ---------------------------------------------------------------------------
            # Text token
            # ---------------------------------------------------------------------------

            if token.is_text:
                echar = EChar(token.value, **styles.as_dict())
                if math_mode:
                    if token.is_alpha or token.is_num or (len(token.value) == 1 and token.value in GREEK):
                        echar.italic = token.is_alpha
                        dct = {'type': 'STRING', 'string': echar}
                    else:
                        dct = {'type': 'SYMBOL', 'string': echar}
                else:
                    dct = {'type': 'STRING', 'string': echar}

            # ---------------------------------------------------------------------------
            # Block
            # ---------------------------------------------------------------------------

            elif token.is_text_block:
                cmd = getattr(token, 'command', None)
                if cmd in TEXT_BLOCKS:
                    block_styles = styles.push(**TEXT_BLOCKS[cmd])
                else:
                    block_styles = styles

                d = token.value.to_formula(False, block_styles)
                dct = {'type': 'BLOCK', 'content': [d]}

            elif token.is_math_block:
                dct = token.value.to_formula(True, styles)

            # ---------------------------------------------------------------------------
            # A command
            # ---------------------------------------------------------------------------

            elif token.is_command:

                is_under_deco = token.value.startswith('under') and token.value[5:] in DECORATORS
                is_deco       = is_under_deco or token.value in DECORATORS

                # ---------------------------------------------------------------------------
                # Text Switch
                # ---------------------------------------------------------------------------

                if token.value == 'color':
                    if token.option is not None:
                        styles.color = token.option

                elif token.value in ['matindex', 'material', 'materialindex']:
                    if token.option is not None:
                        styles.material_index = int(token.option)

                elif token.value in TEXT_SWITCHES:
                    for k, v in TEXT_SWITCHES[token.value].items():
                        setattr(styles, k, v)

                # ---------------------------------------------------------------------------
                # A function
                # ---------------------------------------------------------------------------

                elif token.value in MATH_PATTERN_COMMANDS:

                    pat = MATH_PATTERNS[MATH_PATTERN_COMMANDS[token.value]]

                    # Capture the current style for function symbols
                    func_style = EText("F", **styles.as_dict())

                    # Read the optional scripts
                    scripts = self.read_scripts(styles)

                    dct = {'type': 'FUNCTION', 'name': token.value, 'func_style': func_style, **scripts}

                    # Do we have an option

                    if token.value in OPTION_COMMANDS and token.option is not None:
                        dct['option'] = token.option

                    # Read the arguments

                    args = []
                    for i in range(pat['tokens']):
                        args.append(self.read_term(True, styles))

                    dct['args'] = args

                    # mathbb : transform the argument to special math char

                    if token.value == 'mathbb':
                        darg = args[0]

                        while darg['type'] == 'BLOCK':
                            darg = darg['content'][0]

                        if darg['type'] == 'STRING':
                            mathbb = str(darg['string'])
                            if mathbb in MATHBB:
                                mathbb = MATHBB[mathbb]

                            dct = {'type': 'SYMBOL', 'string': mathbb}

                        else:
                            self.parser.error(f"mathbb command accept only string param, not: {darg}", index=token.index)

                # ---------------------------------------------------------------------------
                # Left / Right
                # ---------------------------------------------------------------------------

                elif token.value == 'left':

                    tk = self.get_token()
                    if not tk.is_text:
                        self.parser.error(f"'\\left' must be followed by a character, not {tk}.", index=tk.index)

                    brackets = {}
                    if tk.value != '.':
                        brackets['left'] = tk.value

                    user_block = Tokens(self.parser)
                    while True:

                        sub_token = self.get_token()

                        if sub_token.is_eof:
                            self.parser.error(f"'\\right' command expected to close block opened with '\\left{tk.value}'.", index=sub_token.index)

                        elif sub_token == Token.command('right'):
                            tk = self.get_token()
                            if not tk.is_text:
                                self.parser.error(f"'\\right' must be followed by a character, not {tk}.", index=tk.index)

                            if tk.value != '.':
                                brackets['right'] = tk.value

                            #dct = self.read_formula(user_block, styles)
                            dct = user_block.to_formula(True, styles)
                            break
                            
                        else:
                            # Add the current token to the list to parse
                            user_block.append(sub_token)

                # ---------------------------------------------------------------------------
                # Decorators
                # ---------------------------------------------------------------------------

                elif is_deco:

                    # Content to decorate
                    dct = self.read_term(True, styles)

                    if is_under_deco:
                        params = dict(DECORATORS[token.value[5:]])
                        params['under'] = True
                    else:
                        params = dict(DECORATORS[token.value])

                    deco_key = 'under' if params['under'] else 'over'
                    if params['fix']:
                        deco_key = 'fix_' + deco_key

                    dct[deco_key] = params['char']

                # ---------------------------------------------------------------------------
                # User command
                # ---------------------------------------------------------------------------

                else:
                    echar = EChar(token.value, **styles.as_dict())
                    return {'type': 'STRING', 'string': echar}
                    #return {'type': 'STRING', 'string': token.value}


            # ---------------------------------------------------------------------------
            # Control
            # ---------------------------------------------------------------------------

            elif token.is_control:
                if token.value == '~':
                    return {'type': 'SYMBOL', 'string': ' '}
                assert False, f"Strange, shouldn't happen {token}"

            # ---------------------------------------------------------------------------
            # Fall back
            # ---------------------------------------------------------------------------

            else:
                assert False, f"Unknown token {token}"

        
        return dct


    # ----------------------------------------------------------------------------------------------------
    # Read a formula, i.e. a list of tokens
    # ----------------------------------------------------------------------------------------------------

    def to_formula(self, math_mode, styles):
        """Read the tokens to build a formula dict.
        """

        # Copy for local changes
        if styles is  None:
            styles = StyleContext()
        else:
            styles = StyleContext(**styles.as_dict())

        # Resulting list of term dicts
        content = []

        while not self.eof:
        
            # Current term
            dct = self.read_term(math_mode, styles)

            # Scripts
            scripts = self.read_scripts(styles)
            while len(scripts):
                dct = {'type': 'BLOCK', 'content': [dct], **scripts}
                scripts = self.read_scripts(styles)

            content.append(dct)

        return {'type': 'BLOCK', 'content': content}
    
# ====================================================================================================
# Char Parser
# ====================================================================================================

class Parser:

    BLANK         = set("\n\t ")
    TEXT_SPECIALS = set("{}~$\\")
    MATH_SPECIALS = set("{}~$\\#&^_")

    def __init__(self, text, math_mode=False, ignore_comments=True):
        """String parser.

        Parse the chars of a string.
        """
        self.text = text
        self.index = 0

        self.ignore_comments = ignore_comments
        self.math_mode = math_mode

        # math_mode stack
        self.mm_stack = []

        # Position stack
        self.stack = []

    # ----------------------------------------------------------------------------------------------------
    # Position stack
    # ----------------------------------------------------------------------------------------------------

    def push_index(self):
        self.stack.append(self.index)

    def pop_index(self, advance=True):
        if advance:
            self.stack.pop()
        else:
            self.index = self.stack.pop()

    # ----------------------------------------------------------------------------------------------------
    # Special chars depends upon the context
    # ----------------------------------------------------------------------------------------------------

    @property
    def math_mode(self):
        return self._math_mode
    
    @math_mode.setter
    def math_mode(self, value):
        self._math_mode = value
        if self.math_mode:
            self.specials = self.MATH_SPECIALS
            self.ignore = set("\n\t ")
        else:
            self.specials = self.TEXT_SPECIALS

        if not self.ignore_comments:
            self.specials = self.specials.union("%")

    def push_math_mode(self, new_mode=None):
        self.mm_stack.append(self.math_mode)
        if new_mode is not None:
            self.math_mode = new_mode

    def pop_math_mode(self):
        self.math_mode = self.mm_stack.pop()

    # ----------------------------------------------------------------------------------------------------
    # Error message
    # ----------------------------------------------------------------------------------------------------

    def error(self, message, index=None):
        lines = self.text.split("\n")
        n = self.index if index is None else index
        dump = []
        for iline, line in enumerate(lines):
            s = f"{iline:2d}: {line}"
            dump.append(s)

            if n <= len(line):
                dump.append("    " + " "*(n-1) + "^")
                break

            n -= len(line) + 1

        rl = "\n"
        raise RuntimeError(f"{message}\n{rl.join(dump)}")
    
    # ----------------------------------------------------------------------------------------------------
    # eof
    # ----------------------------------------------------------------------------------------------------

    @property
    def eof(self):
        return self.index >= len(self.text)
    
    # ====================================================================================================
    # Char level
    # ====================================================================================================
    
    def peek(self, advance=True):
        """Read the next raw char.

        If advance is True, increment the index.
        """
        if self.eof:
            return None
        else:
            c = self.text[self.index]
            if advance:
                self.index += 1
            return c
        
    # ----------------------------------------------------------------------------------------------------
    # Read one char
    # ----------------------------------------------------------------------------------------------------
    
    def read_char(self, advance=True):
        r""" Char reader.

        Reads the next chars while managing special chars. The escape
        char \ is treated by looking at the following char:
        - \{ -> char {
        - \\ -> char {
        - \a -> command \ then char a

        Parameters
        ----------
        advance : int, default=True
            Actually consume the char.

        Returns
        -------
        Char
            Char of kind:
            - 'EOF'     : end of file, no more char to read
            - 'special' : \ ^ _ { ... depending on mode
            - 'blank'   : space, \t or \n (math_mode only)
            - 'char'    : a char
        """

        mem_index = self.index

        while True:

            char_index = self.index
            c = self.peek()

            if c is None:
                ch = Char.eof(char_index)
            
            # --------------------------------------------------
            # Blank char
            # --------------------------------------------------
            
            elif c in self.BLANK and self.math_mode:

                start = self.index - 1
                while self.peek(False) in self.BLANK:
                    self.peek()

                ch = Char.blank(' ', char_index)

            # --------------------------------------------------
            # Escape char, before a special or not
            # --------------------------------------------------

            elif c == '\\':

                if self.eof:
                    self.error("CharParser: A character is expected after escape character '\\'.")

                if self.peek(False) in self.specials:
                    ch = Char.char(self.peek(), char_index)
                else:
                    ch = Char.special(c, char_index)

            # --------------------------------------------------
            # Special char
            # --------------------------------------------------
            
            elif c in self.specials:
                if c == '%':
                    while self.peek(False) not in [None, '\n']:
                        self.peek()
                    continue

                elif c == '$' and self.peek(False) == '$':
                    self.peek()
                    ch = Char.special('$$', char_index)

                else:
                    ch = Char.special(c, char_index)

            # --------------------------------------------------
            # Char
            # --------------------------------------------------

            else:
                ch = Char.char(c, char_index)

            break

        if not advance:
            self.index = mem_index

        return ch
    
    # ----------------------------------------------------------------------------------------------------
    # Read non blank char
    # ----------------------------------------------------------------------------------------------------

    def read_non_blank_char(self, advance=True):

        if not advance:
            self.push_index()

        while True:
            ch = self.read_char()
            if not ch.is_blank:
                self.pop_index(advance)
                return ch
            
    # ----------------------------------------------------------------------------------------------------
    # Read option [...]
    # ----------------------------------------------------------------------------------------------------

    def read_option(self):
        
        if self.read_char(False) != Char.char('[', None):
            return None
        
        self.push_index()
        self.read_char()

        option = ""
        while True:
            c = self.read_char()
            if not c.is_text:
                self.pop_index(False)
                return None
            
            if c == Char.char(']', None):
                self.pop_index(True)
                return option
            
            option += c.value

    
    # ====================================================================================================
    # Token level
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Read a token
    # ----------------------------------------------------------------------------------------------------

    def read_token(self):
        r""" Read successive chars forming a token.

        The function returns either a single token or a block token, the list of tokens
        enclosed between opening and closing tokens. Enclosing tokens are:
        - {...}
        - \[...\]
        - \(...\)
        - $ ... $
        - $$ ... $$

        The corresponding opening and closing tokens are used by this level and are not returned
        to the grammar level. For instance in the LaTeX string
        "before {within a block} after", the content between { and } is returned as a single
        block token.

        Note that \left and \right tokens are not treated at this level but at grammar level.

        In addition to block tokens, the kind of returned token can be:
        - EOF : end of file, no more token to read
        - command : \ followed by a alpha word, "\int" for instance
        - text : several successive characters

        In math mode, the 'control' token can also be returned for chars ~, ^ and _.

        When a command \command is read, it is compared to the list of symbols SYMBOLS. The
        corresponding symbol is returned as text token if found, otherwise a command
        token is returned.
        
        Returns
        -------
        Token
            Token of kind 'EOF', 'command', 'block', 'text' or 'control'.
            Intermediary token 'close' is captured by `read_block` and never returned
            to the grammar level.
        """        

        while True:

            ch = self.read_char()
            index = ch.index

            if ch.is_eof:
                return Token.eof()
            
            # ---------------------------------------------------------------------------
            # Special char
            # ---------------------------------------------------------------------------
            
            elif ch.kind == 'special':

                #TEXT_SPECIALS = set("{}~$\\")
                #MATH_SPECIALS = set("{}~$\\#&^_")

                if ch.value == '\\':

                    # --------------------------------------------------
                    # Command or symbol
                    # --------------------------------------------------

                    c = self.read_char()
                    if c.is_alpha:
                        cmd = c.value
                        while self.read_char(False).is_alpha:
                            cmd += self.read_char().value

                        # Consume next blank
                        if self.read_char(False).is_blank:
                            self.read_char()

                        # Command or symbol code
                        if cmd in SYMBOLS:
                            tk = Token.text(SYMBOLS[cmd])
                        else:
                            tk = Token.command(cmd)
                        tk.index = index
                        return tk

                    # --------------------------------------------------
                    # Math block
                    # --------------------------------------------------

                    elif c.value in '([':

                        if self.math_mode:
                            self.error(f"Control '\\{c}' not valid in math mode.")

                        return self.read_block(Token.open(c))

                    # --------------------------------------------------
                    # Close (intermediary token)
                    # --------------------------------------------------

                    elif c.value in ')]':
                        if not self.math_mode:
                            self.error(f"Control '\\{c}' not valid in text mode.")

                        return Token.close(c)
                    
                    else:
                        self.error(f"Unsupported command : '\\{c}.")

                elif ch.value == '{':
                    return self.read_block(Token.open(ch))
                
                elif ch.value == '}':
                    return Token.close(ch)
                
                elif ch.value in ['$', '$$']:
                    if self.math_mode:
                        return Token.close(ch)
                    else:
                        return self.read_block(Token.open(ch))
                    
                elif ch.value in ['^', '_', '~']:
                    if ch.value in ['^', '_']:
                        tk = self.read_token()
                        return Token.control(ch, tk)
                    else:
                        return Token.control(ch, None)
                
                else:
                    # Not supported as special char
                    return Token.text(ch)

            # ---------------------------------------------------------------------------
            # Blank char are ignored in math mode
            # ---------------------------------------------------------------------------

            elif ch.is_blank and self.math_mode:
                continue

            # ---------------------------------------------------------------------------
            # Character
            # ---------------------------------------------------------------------------

            else:
                # Text mode: read at once successive chars 
                if not self.math_mode:
                    cs = [ch.value]
                    while self.read_char(False).is_text:
                        cs.append(self.read_char().value)

                    tk = Token.text("".join(cs))
                    tk.index = index
                    return tk
                
                # Math mode:
                # - group alpha in words
                # - group num in numbers
                
                if ch.is_alpha:
                    cs = [ch.value]
                    while self.read_char(False).is_alpha:
                        cs.append(self.read_char().value)

                    tk = Token.text("".join(cs))
                    tk.index = index
                    return tk
                
                if ch.is_num:
                    cs = [ch.value]
                    while self.read_char(False).is_num:
                        cs.append(self.read_char().value)

                    tk = Token.text("".join(cs))
                    tk.index = index
                    return tk
                
                # Math mode : a single char
                return Token.text(ch.value)
            
    # ----------------------------------------------------------------------------------------------------
    # Read a block
    # ----------------------------------------------------------------------------------------------------

    def read_block(self, open_token=None):
        r"""Read a series of succesive tokens.

        This method reads the successive tokens until a closing token is encountered.

        The EOF token is considered as a closing token matching the None opening token.

        If the closing token doesn't match with the opening token, an error is raised.

        The math mode can be changed if the opening token is defining the mode by
        itself : \( or $ for instance.

        Commands such as \text or \textbf are followed by a {...} block to be read in text
        mode. When such a command is encountered, the mode is temporarily switched to
        text mode for reading the next block in text mode.

        Parameters
        ----------
        open_token : Token or None
            Used to match the expected closing token.

        Raises
        ------
        RuntimeError
            If a mismatching closing token is encountered
            If a text command is not followed by a {...} block

        Returns
        -------
        Token
            Block token        
        """

        if open_token is None:
            open_token = Token.open(None)

        # New math mode
        self.push_math_mode()
        if open_token.value in ['(', '[', '$', '$$']:
            self.math_mode = True

        tokens = Tokens(self)

        while True:

            token = self.read_token()

            # --------------------------------------------------
            # EOF or block closing
            # --------------------------------------------------

            if token.is_eof or token.is_close:
                if token.match_open(open_token):
                    break

                if token.is_eof:
                    self.error(f"End of file before closing the block {open_token}.")
                else:
                    self.error(f"Block closing {token} is encountered without opening.")

            # --------------------------------------------------
            # Command
            # --------------------------------------------------

            elif token.is_command:

                # Some commands can be followed by an option

                has_option = token.value in OPTION_COMMANDS
                if has_option:
                    token.option = self.read_option()

                # Text command needs as block afterwards

                if token.value in TEXT_BLOCKS:

                    next_char = self.read_char(False)
                    if next_char != Char.special('{', None):
                        self.error(f"The text command \\{token.value} must be followed by a {{...}} block, not {next_char}.")

                    self.push_math_mode(False)
                    block = self.read_token()
                    self.pop_math_mode()

                    # Just in case
                    assert block.is_text_block, str(block)

                    block.block_command = token.value
                    if has_option:
                        block.option = token.option
                    tokens.append(block)

                else:
                    tokens.append(token)

            # --------------------------------------------------
            # Otherwise simply append the token to the list
            # --------------------------------------------------

            else:
                tokens.append(token)

        # ---------------------------------------------------------------------------
        # Returning a block
        # ---------------------------------------------------------------------------

        props = {}
        if token.value in ['$$', '\\[']:
            props['align'] = 'left'

        token = Token.block(self.math_mode, tokens, **props)
        
        # pop math mode
        self.pop_math_mode()

        return token

    # ====================================================================================================
    # Debug
    # ====================================================================================================

    def _dump_parsed(self, parsed):

        def _lines(d):

            if not isinstance(d, dict) or 'type' not in d:
                yield str(d)

            else:
                # ----- Type

                excl = ['type']
                content = None
                if d['type'] in ['SYMBOL', 'STRING']:
                    yield f"{d['type']} : '{str(d['string'])}'"
                    excl.append('string')

                elif d['type'] == 'BLOCK':
                    yield "BLOCK"
                    content = 'content'
                    excl.append(content)

                elif d['type'] == 'FUNCTION':
                    yield f"FUNCTION {d['name']}"
                    excl.append('name')
                    content = 'args'
                    excl.append(content)

                else:
                    yield f"{d['type']} ???"

                # ----- Attributes

                for k, v in d.items():
                    if k not in excl:
                        s = f"- {k:10s} : "
                        for l in _lines(v):
                            yield s + l
                            s = "   "

                # ----- Content

                if content is not None:
                    yield f"- {content}:"
                    for c in d[content]:
                        for line in _lines(c):
                            yield "   " + line

        print()
        print("-"*10, "Dump formula")
        for line in _lines(parsed):
            print(line)
        print("\n")


# ====================================================================================================
# Tests
# ====================================================================================================


if __name__ == '__main__':

    from pprint import pprint
    print('-'*100)

    text_tests = [
        r"text",
        r"{text} normal",
        r"Normal \textbf{bold}",
        r"Normal {\color[red]red color}default \matindex[1]change mat"
    ]

    math_tests = [
        r"x",
        r"x^2",
        r"\vec x",
        r"\frac{1}{x}",
        r"\int_0^1x",
        r"x^2^2",
        r"\sum{i^2}^2",
        r"a = \ln(e^x)",
        r"\text{TEXT} \ln(e^x)"        
    ]

    math_mode = True
    index = 8

    tests = math_tests if math_mode else text_tests
    text = tests[index]

    print('-'*100)

    p = Parser(text, math_mode=math_mode)
    block = p.read_block()
    print("BLOCK DUMP")
    print(repr(block))

    dct = block.value.to_formula(math_mode, None)
    p._dump_parsed(dct)
    print()


