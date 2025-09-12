from .latex_codes import SYMBOLS

__all__ = ["Parser", "Lexer", 'parse_latex']


TEXT_SWITCHES = {
    'tiny': {'font_size': .3}, 
    'small': {'font_size': .6}, 
    'normalsize': {'font_size': 1.}, 
    'large': {'font_size': 1.25}, 
    'Large': {'font_size': 1.5}, 
    'LARGE': {'font_size': 2.}, 
    'huge': {'font_size': 3.}, 
    'Huge': {'font_size': 4.},

    'itshape': {'italic': True},
    'bfseries': {'bold': True},
    'scshape' : {'small_caps': True},
    'ttfamily' : {'family': 'courrier'},
    'sffamily' : {'family': 'sans serif'},
    'rmfamily' : {'family': 'serif'},
}

TEXT_BLOCKS = {
    'text': {},
    'textit': {'italic': True},
    'textbf': {'bold': True},
    'emph': {'bold': 'invert', 'italic': 'invert'},
    'textrm': {'family': 'serif'},
    'textsf': {'family': 'sans serif'}, 
    'texttt': {'family': 'courrier'},
    'underline': {'underline': True},
    'textnormal': {'bold': None, 'italic': None, 'family': None, 'underline': None},
}

MATH_PATTERNS = {
  "FRAC_PAT":    {"tokens": 2},
  "SQRT_PAT":    {"tokens": 1, "brackets": True},  
  "ACCENT_PAT":  {"tokens": 1},
  "INT_PAT":     {"required": 1, "scripts": True}
}

MATH_PATTERN_COMMANDS = {
  "\\frac":   "FRAC_PAT",
  "\\dfrac":  "FRAC_PAT",   
  "\\tfrac":  "FRAC_PAT",
  "\\sqrt":   "SQRT_PAT",
  "\\binom":  "FRAC_PAT",
  "\\boxed":  "ACCENT_PAT",

  "\\hat":        "ACCENT_PAT",
  "\\widehat":    "ACCENT_PAT",
  "\\tilde":      "ACCENT_PAT",
  "\\widetilde":  "ACCENT_PAT",
  "\\bar":        "ACCENT_PAT",
  "\\vec":        "ACCENT_PAT",
  "\\dot":        "ACCENT_PAT",
  "\\ddot":       "ACCENT_PAT",
  "\\overline":   "ACCENT_PAT",
  "\\underline":  "ACCENT_PAT",
  "\\overbrace":  "ACCENT_PAT",
  "\\underbrace": "ACCENT_PAT",

  "\\int":    "INT_PAT",
  "\\iint":   "INT_PAT",
  "\\iiint":  "INT_PAT",
  "\\oint":   "INT_PAT",
  "\\sum":    "INT_PAT",
  "\\prod":   "INT_PAT",
  "\\lim":    "INT_PAT",
  "\\limsup": "INT_PAT",
  "\\liminf": "INT_PAT",

  "\\operatorname":  "OPNAME_PAT",
  "\\operatorname*": "OPNAME_STAR_PAT",
}



# ====================================================================================================
# Simplest UI
# ====================================================================================================

def parse_latex(text, math_mode=False, ignore_comments=True):
    return Lexer(text, math_mode=math_mode, ignore_comments=ignore_comments).read_block()

# ====================================================================================================
# String parser
# ====================================================================================================

class Parser:
    """
    Level-1 LaTeX lexer (character stream → flat tokens).

    Rules:
      - token:
        - word: consecutive alphabetic characters (unicode letters).
        - number: digits with optional single decimal separator ('.' or ',' configurable),
                optional exponent (e/E[+/-]digits), accepts leading sep: .5, ,5.
        - char: single char such as =, ( or escaped \{
      - command: LaTeX control sequence starting with '\' (either letters or single char).
      - special: single-character in MATH_SPECIALS : {}#&^_~
      - dollar: "$" or "$$"
      - space: emitted only for runs of >= 2 spaces/tabs/newlines, collapsed to a single " ".
      - comment: '%' to end-of-line (content returned without the newline)

    Single spaces are *not emitted* (they just separate words).
    """

    MATH_SPECIALS = set("{}~#&^_") # %, $ and \ are hard coded
    TEXT_SPECIALS = set("{}~$\\") # Strict set of escape chars, _ ^ will be accepted without error, 
                                  # % is not escaped if ignore_comments

    def __init__(self, latex_string: str, *, math_mode=True, ignore_comments=True):
        
        if not isinstance(latex_string, str):
            raise TypeError(f"Parser expects a string, not {type(latex_string)}.")
        
        self.s = latex_string
        self.n = len(latex_string)
        self.i = 0

        self.math_mode = bool(math_mode)

        # Parsing config integers are grouped but not float 
        self.decimal_separators = () # ('.', ',') 
        self.allow_exponent = False 

        # Spaces and comments (MATH mode)
        self.ignore_comments = ignore_comments
        self.return_spaces = False
        self.return_comments = False

        if self.ignore_comments:
            self.text_specials = self.TEXT_SPECIALS
        else:
            self.text_specials = self.TEXT_SPECIALS.union('%')


    @property
    def error_str(self):
        return '\n ' + self.s + "\n" + (' '*self.i) + '^'

    # ---- core utils ----
    def _eof(self) -> bool:
        return self.i >= self.n

    def _peek(self, k: int = 0):
        j = self.i + k
        return self.s[j] if 0 <= j < self.n else None

    def _advance(self, k: int = 1):
        self.i += k

    # ---- scanners ----
    def _read_command(self):
        """Read a LaTeX control sequence starting at '\\'.

        Read '\code'.
        - if code is an alphabetic word:
          if code is a know character code in SYMBOLS -> ('token', SYMBOLS[code])
          otherwise return ('command', \code)
        - if code is a single non alphabetic char:
          return ('token', code)

        Returns
            str, value
                ('command', value) or ('token', value)
        """
        assert self._peek() == "\\"
        self._advance(1)

        c = self._peek()
        if c is None:
            return ("command", "\\")
        
        if c.isalpha():
            j = self.i
            while True:
                ch = self._peek(j - self.i)
                if ch is None or not ch.isalpha():
                    break
                j += 1
            name = self.s[self.i:j]
            self.i = j

            # name is a symbol code
            if name in SYMBOLS:
                return ("token", SYMBOLS[name])
            else:
                return ("command", "\\" + name)
        
        # Escaped control character
        self._advance(1)
        if c in '[]()':
            return ("command", "\\" + c)
        else:
            return ("text", c)

    def _read_word(self):
        """Read consecutive alphabetic letters as a WORD."""
        start = self.i
        while True:
            ch = self._peek()
            if ch is None or not ch.isalpha():
                break
            self._advance(1)
        return ("token", self.s[start:self.i])

    def _read_number(self):
        """
        Read a number with optional single decimal separator (from `decimal_separators`)
        and optional exponent part (e/E[+/-]digits). The decimal separator is consumed
        only if followed by a digit. Accepts leading sep: '.5', ',5'.
        Examples: '12', '12.3', '12,3', '.5', ',5', '1.2e-3', '1,2E+6'

        Returns None if not found
        """
        start = self.i
        seen_digit = False

        # Leading digits or leading decimal sep with a digit next
        ch = self._peek()
        if ch in self.decimal_separators and (self._peek(1) and self._peek(1).isdigit()):
            # leading sep
            self._advance(1)
            while (c := self._peek()) and c.isdigit():
                seen_digit = True
                self._advance(1)
        else:
            # integer part
            while (c := self._peek()) and c.isdigit():
                seen_digit = True
                self._advance(1)
            # optional decimal sep if followed by digit
            if (c := self._peek()) in self.decimal_separators and (self._peek(1) and self._peek(1).isdigit()):
                self._advance(1)
                while (d := self._peek()) and d.isdigit():
                    self._advance(1)

        # optional exponent: e/E [ + | - ] digits
        if self.allow_exponent:
            e = self._peek()
            if e in ("e", "E"):
                sign = self._peek(1)
                nxt = self._peek(2) if sign in ("+", "-") else sign
                if nxt and nxt.isdigit():
                    self._advance(1)          # e/E
                    if sign in ("+", "-"):
                        self._advance(1)      # sign
                    while (d := self._peek()) and d.isdigit():
                        self._advance(1)

        if not seen_digit:
            return None
            # Nothing numeric actually read; treat leading char as a single 'word'
            # (This path is rare given the entry condition.)
            ch0 = self._peek()
            if ch0 is not None:
                self._advance(1)
                return ("token", ch0)
            return ("token", "")

        return ("token", self.s[start:self.i])

    def _read_comment(self):
        """Read from '%' to end-of-line (without consuming the newline)."""
        assert self._peek() == "%"
        self._advance(1)
        start = self.i
        while not self._eof():
            if self._peek() == "\n":
                break
            self._advance(1)
        return ("comment", self.s[start:self.i])

    def _read_spaces(self):
        """
        Read a run of whitespace. Emit:
          - ("space", " ") if run length >= 2 (collapsed)
          - None if run length == 1 (single-space is not emitted)
        Tabs/newlines count as space for the purpose of collapsing.
        """
        count = 0
        while not self._eof():
            ch = self._peek()
            if ch is None or not ch.isspace():
                break
            count += 1
            self._advance(1)
        if count >= 2:
            return ("space", " ")
        return None
    
    # ----------------------------------------------------------------------------------------------------
    # Next token in TEXT mode
    # ----------------------------------------------------------------------------------------------------

    def next_text_token(self):
        """
        Return the next level-1 token:
          ("special", ch)                # Text special chars: $ { } ~
          ("command", "\\text", \(", "\tiny", ...)
          ("text", str)
          ("dollar", "$" or "$$")
          ("comment", str)               # '%' to end-of-line (no newline)
          None                           # end of input
        """

        text = []
        while not self._eof():

            ch = self._peek()

            if ch in self.text_specials:

                # ----- In text mode, escaped special are returned as text

                if ch == "\\":
                    i = self.i
                    kind, value = self._read_command() # command or text
                    if kind == "text":
                        text.append(value)
                        continue

                    if len(text):
                        self.i = i
                    else:
                        return kind, value

                # ----- Return text if any

                if len(text):
                    return "text", "".join(text)

                # ----- Dollar

                if ch == "$":
                    kind = 'special'
                    if self._peek(1) == "$":
                        self._advance(2)
                        return (kind, "$$")
                    self._advance(1)
                    return (kind, "$")

                # ----- Block

                if ch in ['{', '}', '~']:
                    self._advance(1)
                    return ("special", ch)

                # ----- Comment

                if ch == "%":
                    comment = self._read_comment()
                    if self.return_comments:
                        return comment
                    
                # Shouldn't happen
                assert(False)
                
            # Not special : let's start a new text string

            self._advance(1)
            text.append(ch)

        # ----- Return what we have

        if len(text):
            return 'text', "".join(text)
        else:
            return None

    # ----------------------------------------------------------------------------------------------------
    # Next token in MATH mode
    # ----------------------------------------------------------------------------------------------------

    def next_math_token(self):
        """
        Return the next level-1 token:
          ("special", ch)                # from _SPECIAL_SINGLE
          ("command", "\\alpha", "\\%", "\\\\", ...)
          ("token", str)
          ("dollar", "$" or "$$")
          ("space", " ")                 # only for runs of >= 2 spaces/tabs/newlines
          ("comment", str)               # '%' to end-of-line (no newline)
          None                           # end of input
        """
        while not self._eof():

            ch = self._peek()

            # comment
            if not self.ignore_commants and (ch == "%"):
                comment = self._read_comment()
                if self.return_comments:
                    return comment

            # command
            if ch == "\\":
                return self._read_command() # command or token

            # dollar / $$
            if ch == "$":
                kind = 'special'
                if self._peek(1) == "$":
                    self._advance(2)
                    return (kind, "$$")
                self._advance(1)
                return (kind, "$")

            # --- NUMBER must be checked BEFORE SPECIALS ---
            if (ch and ch.isdigit()) or (
                ch in self.decimal_separators and (self._peek(1) and self._peek(1).isdigit())
            ):
                return self._read_number()

            # specials
            if ch in self.MATH_SPECIALS:
                self._advance(1)
                return ("special", ch)

            # whitespace (collapse runs)
            if ch.isspace():
                tok = self._read_spaces()
                if self.return_spaces and tok is not None:
                    return tok
                # single-space → skip
                continue

            # word (letters)
            if ch.isalpha():
                return self._read_word()

            # Fallback: treat as a 1-char word-like token
            self._advance(1)
            return ("word", ch)

        return None

    # ===========================================================================
    # Public UI
    # ===========================================================================

    def next_token(self):
        if self.math_mode:
            return self.next_math_token()
        else:
            return self.next_text_token()
        
    def read_char(self, *chars):

        while not self._eof():
            ch = self._peek()
            self._advance(1)

            if ch in [' ', '\n', '\t']:
                continue

            if not len(chars) or ch in chars:
                return ch
            else:
                raise RuntimeError(f"LaTeX parsing error: character {chars} expected, not '{ch}'." + self.parser.error_str)

        raise RuntimeError(f"LaTeX parsing error: character {chars} expected before end of file." + self.parser.error_str)
    
    def read_simple_block(self, start_char='[', end_char=']', optional=True):

        text = []
        start = self.i

        while not self._eof():

            ch = self._peek()
            self._advance(1)

            if len(text):
                text.append(ch)    
                if ch == end_char:
                    return "".join(text)
                
            else:
                if ch in ' \n\t':
                    continue

                elif ch == start_char:
                    text.append(start_char)

                else:
                    break

        if not optional:
            raise RuntimeError(f"LaTeX parsing error: missing '{end_char}' before end of file." + self.error_str)
        else:
            self.i = start
            return None

    
# ====================================================================================================
# Level 2 : Lexer
# ====================================================================================================

class Lexer:

    def __init__(self, source, math_mode=True, ignore_comments=True):

        self.parser = Parser(source, math_mode=math_mode, ignore_comments=ignore_comments)

    @property
    def math_mode(self):
        return self.parser.math_mode
    
    @math_mode.setter
    def math_mode(self, value):
        self.parser.math_mode = bool(value)

    # ====================================================================================================
    # Read the next text item (Lexer level) - TEXT MODE
    # ====================================================================================================

    def _read_text_item(self, mandatory=False):

        while True:

            token = self.parser.next_token()

            if token is None:
                if mandatory:
                    raise RuntimeError(f"LaTeX parsing error: token expected but end of line is reached.." + self.parser.error_str)

                return None
            
            kind, value = token

            # ---------------------------------------------------------------------------
            # Special char control
            # ---------------------------------------------------------------------------

            if kind == 'special':
                # value in : {}~
            
                if value == '{':
                    return self.read_block('}', block_type='BLOCK')

                elif value == '}':
                    return {'type': 'close', 'value': value}
                
                elif value == '~':
                    return {'type': 'CONTENT', 'content': ['~']}
                
                elif value in ['$', '$$']:
                    self.read_block(value, to_mode='MATH', block_type='MATH')
                
                else:
                    raise RuntimeError(f"LaTeX parsing error: sorry, special char '{value}' is not supported yet." + self.parser.error_str)
            
            # ---------------------------------------------------------------------------
            # Command
            # ---------------------------------------------------------------------------
            
            elif kind == 'command':

                cmd = value[1:]

                # ----- Special command

                if cmd in ['(', '[']:
                    end_token = '\\)' if cmd == '(' else '\\]'
                    return self.read_block(end_token, to_mode='MATH', block_type='MATH')
                
                # ----- Text control

                if cmd in TEXT_SWITCHES:
                    return {'type': 'SWITCH', **TEXT_SWITCHES[cmd]}
                
                elif cmd in TEXT_BLOCKS:
                    self.parser.read_char('{')
                    return self.read_block('}', to_mode='TEXT', block_type='TEXT', **TEXT_BLOCKS[cmd])
                
                # ----- Parameter

                elif cmd == 'param':
                    param_str = self.parser.read_simple_block('{', '}', optional=False).strip()[1:-1]
                    kv = param_str.split('=')
                    if len(kv) != 2:
                        raise RuntimeError(f"LaTeX parsing error: param 'python_name=python_expression' expected, not '{param_str}'".self.parser.error_str)
                    
                    return {'type': 'PARAM', 'key': kv[0].strip(), 'value': kv[1].strip()}
                
                else:
                    raise RuntimeError(f"LaTeX parsing error: command '\\{cmd}' not valid in text mode." + self.parser.error_str)

            # ---------------------------------------------------------------------------
            # Text
            # ---------------------------------------------------------------------------

            elif kind == 'text':
                return {'type': 'STRING', 'string': value}

            # Shouldn't occur

            print(f"WHAT: kind: '{kind}', value: '{value}'")

            assert(False)

    # ====================================================================================================
    # Read the next text item (Lexer level) - MATH MODE
    # ====================================================================================================

    def _read_math_item(self, mandatory=False):
        """
        An items can be:
        - a simple parser token: x, \cos, ...
        - a block (list of items) between {...}
        - composed parser : ^2, \sum_{i=0}n{i^2}
        """

        while True:

            token = self.parser.next_token()

            if token is None:
                if mandatory:
                    raise RuntimeError(f"LaTeX parsing error: token expected but end of line is reached.." + self.parser.error_str)

                return None
            
            kind, value = token
            
            # ----------------------------------------------------------------------------------------------------
            # Special char
            # ----------------------------------------------------------------------------------------------------

            if kind == 'special':
                # value in : {}#&^_~
            
                if value == '{':
                    return self.read_block('}', block_type='BLOCK')

                elif value in ['}' '$', '$$']:
                    return {'type': 'close', 'value': value}
                
                elif value in ['^', '_']:
                    key = {'_': 'sub', '^': 'sup'}[value]
                    return {'type': key, 'script': self.read_item(mandatory=True)}
                
                elif value == '~':
                    return {'type': 'SPACE'}
                
                else:
                    raise RuntimeError(f"LaTeX parsing error: sorry, special char '{value}' is not supported yet." + self.parser.error_str)
                
            # ----------------------------------------------------------------------------------------------------
            # Command
            # ----------------------------------------------------------------------------------------------------

            elif kind == 'command':

                cmd = value[1:]

                # ---------------------------------------------------------------------------
                # End block control
                # ---------------------------------------------------------------------------

                if cmd in [')', ']']:
                    return {'type': 'close', 'value': value}
                
                elif cmd == 'left':

                    BRS = {'(':')', '[':']', '|':'|', '<':'>', '‖':'‖', '⟨': '⟩', '.': '.'}
                    
                    c = self.parser.read_char()
                    if c not in BRS:
                        raise RuntimeError(f"LaTeX parsing error: sorry, left block '\\left{cmd}' is not supported yet." + self.parser.error_str)
                    
                    ends = ['\\right' + c for c in BRS.values()]

                    return self.read_block(*ends, block_type='BLOCK', left_char=c)
                
                elif cmd == 'right':

                    c = self.parser.read_char()
                    return {'type': 'close', 'value': value + c}

                # ---------------------------------------------------------------------------
                # Inline text
                # ---------------------------------------------------------------------------

                if cmd in TEXT_BLOCKS:
                    self.parser.read_char('{')
                    return self.read_block('}', to_mode='TEXT', block_type='TEXT', **TEXT_BLOCKS[cmd])
                
                # ---------------------------------------------------------------------------
                # Command with arguments
                # ---------------------------------------------------------------------------

                elif cmd == 'sqrt':
                    option = self.parser.read_simple_block('[', ']', optional=True)
                    if option is not None:
                        option = option[1:-1]
                    return {
                        'type': 'CONTENT', 
                        'code': 'sqrt', 
                        'power': option, 
                        'content': [self.read_item(mandatory=True)],
                        }
                
                elif cmd in MATH_PATTERN_COMMANDS:
                    
                    pattern = MATH_PATTERNS[MATH_PATTERN_COMMANDS[cmd]]

                    d = {'type': 'CONTENT', 'code': cmd, 'content': []}
                    count = pattern.get('items', 1)
                    while (item := self.read_math_item(mandatory)) is not None:
                        if item['type'] in ['sub', 'sup']:
                            d[item['type']] = item['script']
                        else:
                            d['content'].append(item)
                            count -= 1
                            if count == 0:
                                return d

                    raise RuntimeError(f"LaTeX parsing error: {count} item expected after command '\\{cmd}'." + self.parser.error_str)
                
                # Parameter

                elif cmd == 'param':
                    return {'type': 'PARAM', 'value': self.parser.read_simple_block('{', '}', optional=False).strip()[1:-1]}

                # Directly a command

                else:
                    return {'type': 'command', 'command': cmd}

            # ----------------------------------------------------------------------------------------------------
            # Simple token
            # ----------------------------------------------------------------------------------------------------

            elif kind == 'token':
                return {'type': 'STRING', 'string': value}

    # ====================================================================================================
    # Read the next item (Lexer level)
    # ====================================================================================================

    def read_item(self, mandatory=False):
        if self.math_mode:
            return self._read_math_item(mandatory=mandatory)
        else:
            return self._read_text_item(mandatory=mandatory)
        
    # ====================================================================================================
    # Read a block
    # ====================================================================================================

    def read_block(self, *end_tokens, to_mode=None, block_type='BLOCK', **kwargs):

        content = []

        while token := self.read_item():

            ttype = token['type']
            if ttype in ['sub', 'sup']:
                if not(len(content)):
                    raise RuntimeError(f"LaTeX parsing error: impossible to add a {ttype}script '{token['script']}' after begining of block." + self.parser.error_str)
                if ttype in content[-1].keys():
                    content[-1] = {'type': 'BLOCK', ttype: token['script'], 'content': [content[-1]]}
                else:
                    content[-1][ttype] = token['script']

            elif ttype in 'close':
                if token['value'] in end_tokens:
                    return {'type': block_type, 'content': content, 'right_char': token['value'][-1], **kwargs}
                else:
                    raise RuntimeError(f"LaTeX parsing error: unxpected end of block '{token['command']}', expected: {end_tokens}." + self.parser.error_str)
            
            else:
                content.append(token)

        if len(end_tokens):
            raise RuntimeError(f"LaTeX parsing error: end of file reached before closing block, {end_tokens} expected." + self.parser.error_str)

        return {'type': block_type, 'content': content, **kwargs}


# ====================================================================================================
# Tests
# ====================================================================================================

if __name__ == "__main__":

    from pprint import pprint

    # Text

    texts = [
        "The simplest test ever",
        "Some \\tiny text \\normalsize follower by \\itshape italic \\ttfamily font tt",
        "Text with {sub block} and \\text{sub text} and \\textbf {bold text}."
    ]

    maths = [
        r" \cos(21\theta\phi) = \operatorname{cos}^2 \theta\phi - \sin^2 \theta\phi",
        r"{a = x^2^3 + {toto} \left(1 - y\right)}",
        r"I = \int_i^n f(x^2)dx",
        r"50 \text{ apples} \times 100 \text{ apples} = \textbf{lots of apples}",
        r" \operatorname{cos}(\theta)", 
    ]

    for i in [2]:
        s = texts[i]
        lexer = Lexer(s, math_mode=False)
        print("LaTeX:", s)
        pprint(lexer.read_block())
        print()


