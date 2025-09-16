
if __name__ == '__main__':
    from latex_codes import SYMBOLS
    from etext import EText
else:
    from .latex_codes import SYMBOLS
    from .etext import EText

__all__ = ["parse_latex"]

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

    'itshape': {'italic': True},
    'bfseries': {'bold': True},
    'scshape' : {'small_caps': True},
    #'ttfamily' : {'family': 'courrier'},
    #'sffamily' : {'family': 'sans serif'},
    #'rmfamily' : {'family': 'serif'},

    'italic'    : {'italic': True},
    'bold'      : {'bold': True},
    'smallcaps' : {'small_caps': True},
    'underline' : {'underline': True},
    
}

TEXT_BLOCKS = {
    'text': {},
    'textit': {'italic': True},
    'textbf': {'bold': True},
    'emph': {'bold': True, 'italic': True},
    #'textrm': {'family': 'serif'},
    #'textsf': {'family': 'sans serif'}, 
    #'texttt': {'family': 'courrier'},
    'underline': {'underline': True},
    'textnormal': {'bold': False, 'italic': False, 'family': False, 'underline': False},
}

# Hard coded
# - color
# - matindex, material, materialindex (synonyms)



# ----------------------------------------------------------------------------------------------------
# Math commands
# ----------------------------------------------------------------------------------------------------

MATH_PATTERNS = {
  "FRAC_PAT":    {"tokens": 2},
  "SQRT_PAT":    {"tokens": 1, "option": True},  
  "ACCENT_PAT":  {"tokens": 1},
  "INT_PAT":     {"tokens": 1, "scripts": True}
}

MATH_PATTERN_COMMANDS = {
  "frac":   "FRAC_PAT",
  "dfrac":  "FRAC_PAT",   
  "tfrac":  "FRAC_PAT",
  "sqrt":   "SQRT_PAT",
  "binom":  "FRAC_PAT",
  "boxed":  "ACCENT_PAT",

  "hat":        "ACCENT_PAT",
  "widehat":    "ACCENT_PAT",
  "tilde":      "ACCENT_PAT",
  "widetilde":  "ACCENT_PAT",
  "bar":        "ACCENT_PAT",
  "vec":        "ACCENT_PAT",
  "dot":        "ACCENT_PAT",
  "ddot":       "ACCENT_PAT",
  "overline":   "ACCENT_PAT",
  "underline":  "ACCENT_PAT",
  "overbrace":  "ACCENT_PAT",
  "underbrace": "ACCENT_PAT",

  "int":    "INT_PAT",
  "iint":   "INT_PAT",
  "iiint":  "INT_PAT",
  "oint":   "INT_PAT",
  "sum":    "INT_PAT",
  "prod":   "INT_PAT",
  "lim":    "INT_PAT",
  "limsup": "INT_PAT",
  "liminf": "INT_PAT",
}

# ====================================================================================================
# Exposed methods
# ====================================================================================================

def parse_latex(text, math_mode=False):
    """LaTeX parser.
    """

    parser = Parser(text, math_mode=math_mode, ignore_comments=True)
    token = parser.read_block(None)
    d = parser.parse_block(token)

    if math_mode:
        return d
    else:
        return d['string']

# ====================================================================================================
# Char
# ====================================================================================================

class Char:

    ALPHA = "abcdefghijklmnopqrstuvwxyz"
    NUM   = "0123456789"

    KINDS = [
        'EOF',
        'special', # ^_ { }
        'blank',   # space, \n \t
        'char',    # rest : a, b ( )
        ]
    
    def __init__(self, kind, value):

        assert(kind in self.KINDS)

        self.kind = kind
        self.value = value

    def __str__(self):
        return f"<{self.kind}: '{self.value}'>"

    def __eq__(self, other):
        return self.kind == other.kind and self.value == other.value
    
    @classmethod
    def command(cls, value):
        return cls('command', value)
    
    @classmethod
    def char(cls, value):
        return cls('char', value)
    
    @classmethod
    def special(cls, value):
        return cls('special', value)
    
    @classmethod
    def blank(cls, value):
        return cls('blank', value)
    
    @classmethod
    def eof(cls):
        return cls('EOF', None)
    
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
        'open',       # open block
        'close',      # close block
        'math_block', # list of math tokens
        'text_block', # list of text tokens

        'control',    # : ^ _ { }
        'command',    # : word (command \word without \)
        'text',       # : flow of chars
        ]
    
    def __init__(self, kind, value):

        assert(kind in self.KINDS)

        self.kind = kind
        self.value = value

    def __str__(self):
        s = ""
        if isinstance(self.value, list):
            lst = self.value
            s = ", [" + ", ".join([str(t) for t in lst]) + "]"
        else:
            s = f"'{self.value}'"
        if hasattr(self, 'props'):
            s += f", props: {self.props}"

        return f"({self.kind}: {s})"

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
    def block(cls, math_mode, tokens, **kwargs):
        if math_mode:
            token = cls('math_block', tokens)
        else:
            token = cls('text_block', tokens)

        token.props = dict(kwargs)
        return token
    
    @classmethod
    def control(cls, value):
        return cls('control', value)
    
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
    
    def __init__(self, l=None):
        if l is not None:
            self.extend(l)
        self.index = 0

    @property
    def eof(self):
        return self.index >= len(self)
    
    def read_token(self, advance=True):
        if self.eof:
            return Token.eof()
        else:
            token = self[self.index]
            if advance:
                self.index += 1
            return token
        
    @property
    def next_is_script(self):
        if self.eof:
            return False
        else:
            return self[self.index].is_control and self[self.index].value in ['^', '_']
        
    def read_scripts(self):
        scripts = {}
        while self.next_is_script:
            ctl = self.read_token(False).value
            key = 'subscript' if ctl == '_' else 'superscript'
            if key in scripts:
                break
            self.read_token()
            scripts[key] = self.read_token()

        return scripts
    
    def read_option(self, advance=True):
        """Read string content between [ and ]
        """
        if self.index == len(self):
            return None
        
        if not self[self.index].is_text:
            return None
        
        if len(self[self.index].value) > 1:
            s = self[self.index].value.strip()
            if s[0] != '[':
                return None
            idx = None
            for i in range(len(s)):
                if s[i] == ']':
                    idx = i
                    break
            if idx is None:
                return None
            
            if advance:
                self[self.index].value = s[idx+1:]
            
            return s[1:idx]

        if len(self) <= self.index + 2:
            return None
        
        if self[self.index] != Token.text('['):
            return None

        if self[self.index+1] == Token.text(']'):
            if advance:
                self.index += 2
            return None

        end_index = None
        s = ""
        for i in range(self.index + 1, len(self)):
            if not self[i].is_text:
                return None
            if self[i].value == ']':
                end_index = i
                break
            s += self[i].value

        if end_index is None:
            return None
        else:
            if advance:
                self.index = end_index + 1
            return s

        
class Builder:

    def __init__(self):
        self.content = []
        self.etext = EText()

    def get_dict(self, text_mode=False):
        
        self.consume_etext()

        # In text mode, all the token are merge in a single string
        # This is a simple LaTeX parser !
        if text_mode:
            etext = EText()
            for tk in self.content:

                if tk['type'] == 'STRING':
                    etext += tk['string']

                elif tk['type'] == 'BLOCK':
                    for c in tk['content']:
                        if c['type'] == 'STRING':
                            etext += tk['string']
                        else:
                            assert False, str(c)

                else:
                    assert False, str(tk)

            return {'type': 'STRING', 'string': etext}

        return {'type': 'BLOCK', 'content': self.content}

    def consume_etext(self, scripts=None):
        if scripts is None:
            scripts = {}

        if len(self.etext) == 0:
            if len(scripts) == 0:
                return
            else:
                self.etxt.append("?")

        self.content.append({'type': 'STRING', 'string': self.etext.clone(), **scripts})
        self.etext.clear()

    def add_text(self, text, scripts=None, **styles):
        if scripts is None:
            self.etext.append(text, **styles)

        else:
            self.consume_etext()
            self.etext.append(text, **styles)
            self.consume_etext(scripts=scripts)

    def add_content(self, d, **kwargs):
        self.consume_etext()
        self.content.append(d)

    def add_scripts(self, scripts):

        if scripts is None or len(scripts) == 0:
            return
        
        self.consume_etext()
        
        if not self.content:
            self.add_text("?", scripts=scripts)

        else:
            d = self.content[-1]
            create_block = False
            for k in scripts.keys():
                if k in d:
                    create_block=True

            if create_block:
                self.content[-1] = {'type': 'BLOCK', 'content': [d], **scripts}
            else:
                for k, v in scripts.items():
                    d[k] = v

    
# ====================================================================================================
# Char Parser
# ====================================================================================================

class Parser:

    BLANK = set("\n\t ")
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

        # Position stack
        self.stack = []

    # ----------------------------------------------------------------------------------------------------
    # Position stack
    # ----------------------------------------------------------------------------------------------------

    def push_index(self):
        self.stack.append(self.index)

    def pop_index(self):
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

    # ----------------------------------------------------------------------------------------------------
    # Error message
    # ----------------------------------------------------------------------------------------------------

    def error(self, message):
        lines = self.text.split("\n")
        n = self.index
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

        mem_index = self.index

        while True:

            c = self.peek()

            if c is None:
                ch = Char.eof()
            
            # --------------------------------------------------
            # Blank char
            # --------------------------------------------------
            
            elif c in self.BLANK and self.math_mode:

                start = self.index - 1
                while self.peek(False) in self.BLANK:
                    self.peek()

                #if self.eof:
                #    s = self.text[start:]
                #else:
                #    s = self.text[start:self.index]

                ch = Char.blank(' ')

            # --------------------------------------------------
            # Escape char, before a special or not
            # --------------------------------------------------

            elif c == '\\':

                if self.eof:
                    self.error("CharParser: A character is expected after escape character '\\'.")

                if self.peek(False) in self.specials:
                    ch = Char.char(self.peek())
                else:
                    ch = Char.special(c)

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
                    ch = Char.special('$$')

                else:
                    ch = Char.special(c)

            # --------------------------------------------------
            # Char
            # --------------------------------------------------

            else:
                ch = Char.char(c)

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
                if not advance:
                    self.pop_index()
                return ch
    
    # ====================================================================================================
    # Token level
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Read a token
    # ----------------------------------------------------------------------------------------------------

    def read_token(self):

        while True:

            ch = self.read_char()

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
                    # Read word or single non alpha char
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
                            return Token.text(SYMBOLS[cmd])
                        else:
                            return Token.command(cmd)
                    
                    # --------------------------------------------------
                    # From text to math mode
                    # --------------------------------------------------

                    elif c in '([':

                        if self.math_mode:
                            self.error(f"Control '\\{c}' not valid in math mode.")

                        return self.read_block(Token.open(c))

                    # --------------------------------------------------
                    # End of math mode
                    # --------------------------------------------------

                    elif c in ')]':
                        if not self.math_mode:
                            self.error(f"Control '\\{c}' not valid in text mode.")

                        return Token.close(c)
                    
                    else:
                        self.error(f"Unsupported command : '\\{c}.")

                elif ch.value == '{':
                    return self.read_block(Token.open('{'))
                
                elif ch.value == '}':
                    return Token.close('}')
                
                elif ch.value in ['$', '$$']:
                    if self.math_mode:
                        return Token.close(ch.value)
                    else:
                        return self.read_block(Token.open(ch.value))
                    
                elif ch.value in ['^', '_', '~']:
                    return Token.control(ch.value)
                
                else:
                    # Not supported as special char
                    return Token.text(ch.value)
                    self.error(f"Special character {ch.value} not supported.")

            # ---------------------------------------------------------------------------
            # Blank
            # ---------------------------------------------------------------------------

            elif ch.is_blank and self.math_mode:
                continue

            # ---------------------------------------------------------------------------
            # Character
            # ---------------------------------------------------------------------------

            else:
                if not self.math_mode:
                    text = ch.value
                    while self.read_char(False).is_text:
                        text += self.read_char().value
                    return Token.text(text)
                
                if ch.is_alpha:
                    word = ch.value
                    while self.read_char(False).is_alpha:
                        word += self.read_char().value

                    return Token.text(word)
                
                if ch.is_num:
                    word = ch.value
                    while self.read_char(False).is_num:
                        word += self.read_char().value

                    return Token.text(word)
                
                return Token.text(ch.value)
            
    # ----------------------------------------------------------------------------------------------------
    # Read a block
    # ----------------------------------------------------------------------------------------------------

    def read_block(self, open_token=None):

        if open_token is None:
            open_token = Token.open(None)

        math_mode = self.math_mode
        if open_token.value in ['(', '[', '$', '$$']:
            self.math_mode = True

        tokens = []

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
            # Command switching to text mode
            # --------------------------------------------------

            elif token.is_command and (token.value in TEXT_BLOCKS):

                next_char = self.read_non_blank_char(False)
                if next_char != Char.special('{'):
                    self.error(f"The text command \\{token.value} must be followed by a {{...}} block.")

                mode = self.math_mode
                self.math_mode = False
                block = self.read_token()
                assert(block.is_text_block)
                block.props = {'command': token.value, **TEXT_BLOCKS[token.value]}
                tokens.append(block)
                self.math_mode = mode

            # --------------------------------------------------
            # Otherwise simply append the token to the list
            # --------------------------------------------------

            else:
                tokens.append(token)

        # Returning a block

        props = {}
        if token.value in ['$$', '\\[']:
            props['align'] = 'left'
        token = Token.block(self.math_mode, tokens, **props)
        self.math_mode = math_mode

        return token
    
    # ====================================================================================================
    # Grammar level
    # ====================================================================================================

    def parse_block(self, block):

        tokens = Tokens(block.value)
        builder = Builder()

        # ===========================================================================
        # Text parser
        # ===========================================================================

        if not block.is_math_block:

            for k, v in block.props.items():
                if k in builder.etext.attributes:
                    builder.etext.default[k] = v

            while not tokens.eof:

                token = tokens.read_token()

                if token.is_text:
                    builder.add_text(token.value)

                elif token.is_control:
                    builder.add_text(token.value)

                elif token.is_command:

                    if token.value in TEXT_SWITCHES:
                        builder.etext.default.set(**TEXT_SWITCHES[token.value])

                    elif token.value == 'color':
                        col = tokens.read_option()
                        if col is not None:
                            OK = True
                            try:
                                from ..maths import Color
                            except:
                                OK = False

                            if OK:
                                builder.etext.default.color = Color(col).rgba

                    elif token.value in ['matindex', 'material', 'materialindex']:
                        matind = tokens.read_option()
                        if matind is not None:
                            builder.etext.default.material_index = matind

                    else:
                        pass

                elif token.is_text_block:
                    for k, v in builder.etext.default:
                        if k not in token.props:
                            token.props[k] = v

                    tk = self.parse_block(token)

                    builder.add_content(tk)

                else:
                    pass

            return builder.get_dict(text_mode=True)

        # ===========================================================================
        # Math parser
        # ===========================================================================

        # ---------------------------------------------------------------------------
        # Parse a token to a dict
        # ---------------------------------------------------------------------------

        def _token_to_dict(token):
            if token.is_eof:
                return {'type': 'STRING', 'string': EText()}
            if token.is_block:
                return self.parse_block(token)
            else:
                return self.parse_block(Token.block(block.is_math_block, [token]))['content'][0]
            
        # ---------------------------------------------------------------------------
        # Read subscript or superscript
        # ---------------------------------------------------------------------------

        def _read_scripts():

            scripts = tokens.read_scripts()
            if not scripts:
                return {}
            
            for k, v in scripts.items():
                if v.is_eof:
                    self.error(f"'k' char at end of block. Need a token.")
                scripts[k] = _token_to_dict(v)

            return scripts

        # ---------------------------------------------------------------------------
        # Loop on the tokens in the block
        # ---------------------------------------------------------------------------

        while not tokens.eof:

            # ---------------------------------------------------------------------------
            # Scripts : added to last token
            # ---------------------------------------------------------------------------

            while tokens.next_is_script:
                scripts = _read_scripts()
                builder.add_scripts(scripts)

                if tokens.eof:
                    break

            # ---------------------------------------------------------------------------
            # Let's read next token
            # ---------------------------------------------------------------------------

            token = tokens.read_token()

            if False:
                print("Grammar", token)

            # ---------------------------------------------------------------------------
            # Text : we add to the stack
            # ---------------------------------------------------------------------------

            if token.is_text:
                if block.is_math_block:
                    # Consume text: the stack will contain only this one
                    if tokens.next_is_script:
                        builder.consume_etext()

                    builder.add_text(token.value, italic=True)
                else:
                    builder.add_text(token.value)

            # ---------------------------------------------------------------------------
            # Control char
            # ---------------------------------------------------------------------------

            elif token.is_control:
                
                self.error(f"Control token {token} not supported yet.")

            # ---------------------------------------------------------------------------
            # Command
            # ---------------------------------------------------------------------------

            elif token.is_command:

                # ---------------------------------------------------------------------------
                # A function
                # ---------------------------------------------------------------------------

                if token.value in MATH_PATTERN_COMMANDS:
                    pat = MATH_PATTERNS[MATH_PATTERN_COMMANDS[token.value]]

                    # Read the optional scripts

                    scripts = _read_scripts()
                    if tokens.next_is_script:
                        self.error(f"The control char {tokens.read_token()} is misplaced after function '{token.value}'.")

                    if len(scripts) and not pat.get('scripts', False):
                        self.error(f"The function '{token.value}' doesn't accept subscript or superscript.")

                    function = {'type': 'FUNCTION', 'name': token.value, **scripts}

                    # Read the option
                    if pat.get('option', False):
                        s = tokens.read_option()
                        if s is not None:
                            function['option'] = s

                    # Read the arguments

                    args = []
                    for i in range(pat['tokens']):
                        args.append(_token_to_dict(tokens.read_token()))

                    function['args'] = args
                    builder.add_content(function)

                # ---------------------------------------------------------------------------
                # Left / Right
                # ---------------------------------------------------------------------------

                elif token.value == 'left':

                    tk = tokens.read_token()
                    if not tk.is_text:
                        self.error(f"'\\left' must be followed by a character, not {tk}.")

                    brackets = {}
                    if tk.value != '.':
                        brackets['left_char'] = tk.value

                    user_block = []
                    while True:

                        sub_token = tokens.read_token()

                        if sub_token.is_eof:
                            self.error(f"'\\right' command expected to close block opened with '\\left{tk.value}'.")

                        elif sub_token == Token.command('right'):
                            tk = tokens.read_token()
                            if not tk.is_text:
                                self.error(f"'\\right' must be followed by a character, not {tk}.")

                            if tk.value != '.':
                                brackets['right_char'] = tk.value

                            block_dict = self.parse_block(Token.block(True, user_block))
                            sub_block = {**block_dict, **brackets}

                            builder.add_content(sub_block)

                            break

                        # Add the current token to the list to parse
                        user_block.append(sub_token)

                # ----- Right is an error

                elif token.value == 'right':
                    self.error(f"'\\right' command encountered with no '\\left' comand before.")

                # ---------------------------------------------------------------------------
                # Transform the command into string
                # ---------------------------------------------------------------------------

                else:
                    if block.is_math_block:
                        builder.add_text(token.value, italic=False)
                    else:
                        builder.add_text(token.value)

            # ---------------------------------------------------------------------------
            # A block
            # ---------------------------------------------------------------------------

            elif token.is_block:

                sub_block = self.parse_block(token)

                builder.add_content(sub_block)

            # ---------------------------------------------------------------------------
            # Shouldn't happen
            # ---------------------------------------------------------------------------

            else:
                if not token.is_eof:
                    assert False, str(token)

        return builder.get_dict()
    
    # ====================================================================================================
    # Debug
    # ====================================================================================================

    def _dump_parsed(self, parsed):

        def _lines(d):

            if not isinstance(d, dict):
                yield str(d)

            else:
                if d['type'] == 'STRING':
                    yield f"{d['type']} : '{str(d['string'])}'"
                    for k, v in d.items():
                        if k not in ['type', 'string']:
                            yield f"- {k:10s} : {', '.join([l for l in _lines(v)])}'"

                elif d['type'] == 'BLOCK':
                    yield f"{d['type']}"
                    for k, v in d.items():
                        if k not in ['type', 'content']:
                            yield f"- {k:10s} : {', '.join([l for l in _lines(v)])}'"
                            #yield f"- {k:10s} : '{v}'"

                    yield "- content:"
                    for c in d['content']:
                        for line in _lines(c):
                            yield "   " + line

                elif d['type'] == 'FUNCTION':
                    yield f"{d['type']} : '{d['name']}'"
                    for k, v in d.items():
                        if k not in ['type', 'name', 'args']:
                            yield f"- {k:10s} : {', '.join([l for l in _lines(v)])}'"
                            #yield f"- {k:10s} : '{v}'"

                    yield "- args:"
                    for c in d['args']:
                        for line in _lines(c):
                            yield "   " + line

                else:
                    yield f"{d['type']}"
                    for k, v in d.items():
                        if k not in ['type']:
                            #yield f"- {k:10s} : '{v}'"
                            yield f"- {k:10s} : {', '.join([l for l in _lines(v)])}'"

        for line in _lines(parsed):
            print(line)

if __name__ == '__main__':

    from pprint import pprint
    print('-'*100)
    text = r"Normal \color[red]matind"
    p = Parser(text, math_mode=False)
    block = p.read_block()
    for tk in block.value:
        print(tk)

    print('-'*100)

    d = p.parse_block(block)
    p._dump_parsed(d)