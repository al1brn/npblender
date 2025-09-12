
from .latex_codes import SYMBOLS

SLASH_AS_FRAC = False

ALPHA_CHARS    = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
BLOCK_CHARS    = {'{': '}', '(': ')', '[': ']', '\\{': '\\}', '⟨': '⟩', '‖': '‖'}
NOT_WORD_CHARS = [' ', '_', '^'] + list(BLOCK_CHARS.keys()) + list(BLOCK_CHARS.values())
BLOCK_CODES    = {
    '{'  : '',
    '('  : '()',
    '['  : '[]',
    '\\{': '{}',
    '⟨'  : '⟨⟩',
    '‖'  : '‖‖',
    }


if SLASH_AS_FRAC:
    NOT_WORD_CHARS += ['/']

COMMANDS = ['int', 'sqrt', 'frac', 'vec', 'overrightarrow']

FUNCTIONS = [
    'sin', 'asin', 'arcsin', 'cos', 'acos', 'arscos', 'tan', 'atan', 'arctan', 'cotan', 'acotan', 'arccotan',
    'sinh', 'asinh', 'arcsinh', 'cosh', 'acosh', 'arscosh',  'tanh', 'atanh', 'arctanh', 'cotanh', 'acotanh', 'arccotanh',
    'log', 'ln', 'min', 'max']

__all__ = "latex_parser"

def latex_parser(latex, compo=None):
    parser = LatexParser(latex)
    return parser.parse(compo=compo)


"""
# =============================================================================================================================
# =============================================================================================================================
# Font dependant constants
# Default constants are set for 'Times New Roman'' font
# =============================================================================================================================
# =============================================================================================================================



FONT_SIZE    = 2.0   # Font size to have A with an height of 1.
Y_ALIGN      = 0.2323*FONT_SIZE # Location of base line (center of = sign)
PAR_SCALE    = 1.05  # Parenthesis scale
SIGMA_SCALE  = 1.3   # Sigma scale
X_SPACE      = 0.2   # Horizontal spacing
Y_SPACE      = 0.1   # Horizontal spacing
X_IND        = 0.2   # Horizontal spacing for indice and exponent
Y_EXP        = 0.5
Y_IND        = 0.2
Y_ABOVE      = 0.3
SCALE_IND    = 0.4   # Indice and exponent scale

DECO_THICK   = 0.06   # Decoractors thickness

# SIGN
PLUS_THICK = DECO_THICK
PLUS_WIDTH = .7

# FRACTION
FRAC_THICK  = DECO_THICK
FRAC_MARGIN = 2*X_SPACE

# ===== Symbols

SYMB_NULL     = 0
SYMB_EQUAL    = 1 # Alternative to '=' character     = 2 # Minus to plus varying with param
SYMB_SIGN     = 2

SYMB_MAX_CODE = 4 # Max symbol code

# ===== Node Types

TYPE_GROUP    = 0 # Group of nodes with decorator
TYPE_REF      = 1 # Simple geometry
TYPE_SYMBOL   = 2 # Symbol dynamically computed
TYPE_DECO     = 3 # Decorator
TYPE_FRACTION = 4 # Fraction
TYPE_IND_EXP  = 5 # Indice / Exponent
TYPE_SIGMA    = 6 # Sigma
TYPE_INTEGRAL = 7 # Integral

VALID_ROLE_TYPES = {
    0: [TYPE_IND_EXP, TYPE_SIGMA, TYPE_INTEGRAL, TYPE_FRACTION, TYPE_DECO],
    1: [TYPE_IND_EXP, TYPE_SIGMA, TYPE_INTEGRAL, TYPE_FRACTION],
    2: [TYPE_IND_EXP, TYPE_SIGMA, TYPE_INTEGRAL],
    3: [],
}

# ===== DECORATORS

DECO_NOTHING            = 0

# ----- Prefixes : code is a symbol
DECO_CAT_PREFIX         = SYMB_MAX_CODE # Code belo

# ----- Special
DECO_CAT_SPECIAL        = 20

DECO_SQRT               = 20

# ----- BLOCK
DECO_CAT_BLOCK          = 30

DECO_PARENTHESIS        = 30
DECO_BRACKETS           = 31
DECO_BRACES             = 32
DECO_ABSOLUTE           = 33
DECO_NORM               = 34
DECO_ANGLE              = 35
DECO_TOKEN              = 36

# ----- Accentuation

DECO_CAT_ACCENTUATION   = 40
DECO_ARROW              = 40
DECO_LEFT_ARROW         = 41
DECO_BAR                = 42
DECO_DOT                = 43
DECO_DOT2               = 44
DECO_DOT3               = 45

# ===== ROLES

ROLE_CONTENT     = 0
ROLE_INDICE      = 1
ROLE_EXPONENT    = 2
ROLE_DENOMINATOR = 1




BLOCK_CODES    = {'{'  : DECO_NOTHING,
                    '('  : DECO_PARENTHESIS,
                    '['  : DECO_BRACKETS,
                    '\\{': DECO_BRACES,
                    '⟨'  : DECO_ANGLE,
                    '‖'  : DECO_NORM,}

"""


# ====================================================================================================
# LaTeX simple parser
# ====================================================================================================

class LatexParser:

    def __init__(self, s):
        self.s     = s
        self.index = 0
        self.stack = []

    # ====================================================================================================
    # Char reader
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # End of string
    # ----------------------------------------------------------------------------------------------------

    @property
    def eof(self):
        return self.index >= len(self.s)

    # ----------------------------------------------------------------------------------------------------
    # Stack
    # ----------------------------------------------------------------------------------------------------

    def push(self, pop_char):
        self.stack.append(pop_char)

    def pop(self, pop_char):
        #print(f"{self.indent}POP  {pop_char}: {self.s[self.index:]}")

        error = len(self.stack) == 0
        if not error:
            pc = self.stack.pop()
            error = pc != pop_char

        if error:
            raise Exception(f"Unbalanced closing char. Expected: '{pop_char}', actual: '{pc}'. {self.s[self.index:]}")

        return True

    # ----------------------------------------------------------------------------------------------------
    # Indent
    # ----------------------------------------------------------------------------------------------------

    @property
    def indent(self):
        return "   "*len(self.stack)

    # ----------------------------------------------------------------------------------------------------
    # Jump spaces
    # ----------------------------------------------------------------------------------------------------

    def jump_spaces(self):
        while (self.index < len(self.s) - 1) and (self.s[self.index] == ' '):
            self.index += 1

    # ----------------------------------------------------------------------------------------------------
    # Next Char
    # ----------------------------------------------------------------------------------------------------

    def next(self):
        """ Get the next char

        Translate symbols into unicode: \\alpha -> α
        """
        if self.index >= len(self.s):
            return None

        # Single char

        c = self.s[self.index]
        self.index += 1

        # White space

        if c == ' ':
            self.jump_spaces()
            return ' '

        # Escape

        elif c == "\\":

            # Next char can be { or | for instance
            word = self.s[self.index]
            self.index += 1

            # Complete with alpha if is alpha
            if word in ALPHA_CHARS:
                while not self.eof:
                    if self.s[self.index] in ALPHA_CHARS:
                        word += self.s[self.index]
                        self.index += 1
                    else:
                        break
                        # Ignore space after a a keyword

                self.jump_spaces()

            # Return the symbol if exists
            ret = SYMBOLS.get(word, "\\" + word)

        else:
            ret = c

        return ret

    # ----------------------------------------------------------------------------------------------------
    # Get the next character without consuming it
    # ----------------------------------------------------------------------------------------------------

    def get_next(self):
        index = self.index
        next = self.next()
        self.index = index
        return next

    # ----------------------------------------------------------------------------------------------------
    # Is the next character a word char
    # ----------------------------------------------------------------------------------------------------

    def next_is_word(self):
        n = self.get_next()
        if n is None:
            return False
        else:
            if n.startswith('\\'):
                return False

            return n not in NOT_WORD_CHARS
        
    # ====================================================================================================
    # Token reader
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Read a single token
    # ----------------------------------------------------------------------------------------------------

    def token(self):

        from ..text import Composition

        self.jump_spaces()

        # ----------------------------------------------------------------------------------------------------
        # End of file
        # ----------------------------------------------------------------------------------------------------

        c = self.next()
        if c is None:
            return None

        # ----------------------------------------------------------------------------------------------------
        # Block
        # ----------------------------------------------------------------------------------------------------

        elif c in BLOCK_CHARS:

            target = BLOCK_CHARS[c]
            self.push(target)

            code = BLOCK_CODES[c]

            if c == '{':
                title = "Block"
            else:
                title = f"Block {c} ... {target}"

            content = self.parse()
            if c == '{':
                return Composition(content)
            else:
                return Composition.new_decorator(content, decorator=code)

        elif c in BLOCK_CHARS.values():
            self.pop(c)
            return None

        # ----------------------------------------------------------------------------------------------------
        # Controls
        # ----------------------------------------------------------------------------------------------------

        elif c == "\\sqrt":
            content = self.parse()
            return Composition.new_sqrt(content)

        elif c == '\\sum':
            ind, exp = self.indice_exponent()
            content = self.token()
            return Composition.new_integral(content, ind, exp, sigma=True)

        elif c == '\\int':
            ind, exp = self.indice_exponent()
            content = self.token()
            return Composition.new_integral(content, ind, exp, sigma=False)

        elif c == "\\frac":
            numerator   = self.token()
            denominator = self.token()
            return Composition.new_fraction(numerator, denominator)

            """
            elif c == '\\minus':
                content = self.enriched_token(self.token())

                return GTerm.join(type=FORM.TYPE_DECO, content=content, code=FORM.SYMB_SIGN, parameter=0, compile=False)._lc("Minus")

            elif c == '\\plus':
                content = self.enriched_token(self.token())

                return GTerm.join(type=FORM.TYPE_DECO, content=content, code=FORM.SYMB_SIGN, parameter=100, compile=False)._lc("Plus")
            """

        elif c == '\\vec':
            content = self.enriched_token(self.token())
            return Composition.new_decorator(content, decorator='ARROW')

            """
            elif c == '=':
                return GTerm.symbol(symbol='=', compile=False)._lc("Equal")

            elif c == '+':
                return GTerm.symbol(symbol='Sign', parameter=100, compile=False)._lc("Equal")

            elif c == '-':
                return GTerm.symbol(symbol='Sign', parameter=0, compile=False)._lc("Equal")
            """

        # ----------------------------------------------------------------------------------------------------
        # Characters
        # ----------------------------------------------------------------------------------------------------

        else:
            word = c
            while self.next_is_word():
                word += self.next()

            italic = word not in FUNCTIONS
            return Composition.new_string(word, italic=italic)

    # ----------------------------------------------------------------------------------------------------
    # Indice and exponent
    # ----------------------------------------------------------------------------------------------------

    def indice_exponent(self):

        ind = None
        exp = None
        for i in range(2):
            c = self.get_next()
            if c == '_':
                if ind is None:
                    c = self.next()
                    ind = self.token()
                else:
                    return ind, exp
                
            elif c == '^':
                if exp is None:
                    c = self.next()
                    exp = self.token()
                else:
                    return ind, exp
            else:
                return ind, exp

        return ind, exp

    # ----------------------------------------------------------------------------------------------------
    # Enriched token
    # ----------------------------------------------------------------------------------------------------

    def enriched_token(self, token):

        from ..text import Composition

        def denominator():

            if (not SLASH_AS_FRAC) and (self.get_next() != '/'):
                return None

            _ = self.next()
            den = self.token()

            while True:
                ind, exp = self.indice_exponent()
                if (ind is None) and (exp is None):
                    break

                #den = GTerm.join(type=FORM.TYPE_IND_EXP, content=den, role_1=ind, role_2=exp, compile=False)
                den = Composition.new_ind_exp(den, ind, exp)

            return den

        go_on = True
        while go_on:

            go_on = False

            while True:
                ind, exp = self.indice_exponent()
                if (ind is None) and (exp is None):
                    break

                #token = GTerm.join(type=FORM.TYPE_IND_EXP, content=token, role_1=ind, role_2=exp, compile=False)
                token = Composition.new_ind_exp(token, ind, exp)
                go_on = True

            while True:
                den = denominator()
                if den is None:
                    break

                #token = GTerm.join(type=FORM.TYPE_FRACTION, content=token, role_1=den, compile=False)
                token = Composition.new_fraction(token, den)
                go_on = True

        return token
    
    # ====================================================================================================
    # Parser
    # ====================================================================================================

    def parse(self, compo=None):

        from ..text import Composition

        while not self.eof:

            # ----- Read the next token

            token = self.token()
            if token is None:
                break

            # ----- Enrich with indice, exponent and denominator

            token = self.enriched_token(token)

            # ----- Append to the formula

            #formula = token if formula is None else GTerm.append(formula, token, compile=False)
            compo = token if compo is None else compo.append(token)


        return compo
    

