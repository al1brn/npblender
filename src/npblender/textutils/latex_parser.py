class StrParser:
    """
    Level-1 LaTeX lexer (character stream → flat tokens).

    Rules:
      - WORD: consecutive alphabetic characters (unicode letters).
      - NUMBER: digits with optional single decimal separator ('.' or ',' configurable),
                optional exponent (e/E[+/-]digits), accepts leading sep: .5, ,5.
      - COMMAND: LaTeX control sequence starting with '\' (either letters or single char).
      - SPECIAL: single-character operators/delims: {}()[]&#^_~+-=*/<>|,:;=
      - DOLLAR: "$" or "$$"
      - SPACE: emitted only for runs of >= 2 spaces/tabs/newlines, collapsed to a single " ".
      - COMMENT: '%' to end-of-line (content returned without the newline)

    Single spaces are *not emitted* (they just separate words).
    """

    _SPECIAL_SINGLE = set("{}()[]&#^_~+-=*/<>|,:;=")

    # Commands that must read a raw *text* braced argument
    TEXT_ARG_COMMANDS = {
        "\\text", "\\textrm", "\\textit", "\\textbf",
        "\\mathrm", "\\mathbf", "\\mathit", "\\emph",
        "\\texttt", "\\textsf", "\\small"  # kept for convenience
    }


    def __init__(self, latex_string: str, *, decimal_separators=(".", ","), allow_exponent=True):
        if not isinstance(latex_string, str):
            raise TypeError("StrParser expects a string.")
        self.s = latex_string
        self.n = len(latex_string)
        self.i = 0
        # number parsing config
        self.decimal_separators = set(decimal_separators)
        self.allow_exponent = bool(allow_exponent)

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
        """Read a LaTeX control sequence starting at '\\'."""
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
            return ("command", "\\" + name)
        # single-char control symbol
        self._advance(1)
        return ("command", "\\" + c)

    def _read_word(self):
        """Read consecutive alphabetic letters as a WORD."""
        start = self.i
        while True:
            ch = self._peek()
            if ch is None or not ch.isalpha():
                break
            self._advance(1)
        return ("word", self.s[start:self.i])

    def _read_number(self):
        """
        Read a number with optional single decimal separator (from `decimal_separators`)
        and optional exponent part (e/E[+/-]digits). The decimal separator is consumed
        only if followed by a digit. Accepts leading sep: '.5', ',5'.
        Examples: '12', '12.3', '12,3', '.5', ',5', '1.2e-3', '1,2E+6'
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
            # Nothing numeric actually read; treat leading char as a single 'word'
            # (This path is rare given the entry condition.)
            ch0 = self._peek()
            if ch0 is not None:
                self._advance(1)
                return ("word", ch0)
            return ("word", "")

        return ("number", self.s[start:self.i])

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

    def _read_inline_text(self, cmd: str) -> dict:
        """
        Read a *text-mode* braced argument right after a command like \\text.
        Contract:
          - skip whitespace after the command
          - require '{' (error if missing)
          - read raw characters until the matching '}' (balanced)
          - '\\{' -> literal '{' ; '\\}' -> literal '}'
        Returns a dict: {"command": cmd, "text": <raw_text>, "closed": bool}
        """
        # 1) skip whitespace before '{'
        while not self._eof() and self._peek().isspace():
            self._advance(1)

        # 2) require '{'
        if self._peek() != "{":
            return {"command": cmd, "text": "", "closed": False, "error": f"expected '{{' after {cmd}"}

        # consume '{'
        self._advance(1)
        depth = 1
        out = []

        # 3) read until matching '}'
        while not self._eof():
            c = self._peek()

            # escaped braces \{ or \}
            if c == "\\":
                c2 = self._peek(1)
                if c2 in ("{", "}"):
                    out.append(c2)
                    self._advance(2)
                    continue
                # keep backslash and next char verbatim (if any)
                out.append(c)
                self._advance(1)
                if not self._eof():
                    out.append(self._peek())
                    self._advance(1)
                continue

            if c == "{":
                depth += 1
                out.append("{")
                self._advance(1)
                continue

            if c == "}":
                depth -= 1
                self._advance(1)
                if depth == 0:
                    return {"command": cmd, "text": "".join(out), "closed": True}
                out.append("}")
                continue

            # regular char
            out.append(c)
            self._advance(1)

        # EOF reached without closing
        return {"command": cmd, "text": "".join(out), "closed": False, "error": "unclosed { ... }"}    


    # ---- public API ----
    def next_token(self):
        """
        Return the next level-1 token:
          ("word", str)
          ("number", str)
          ("command", "\\alpha", "\\%", "\\\\", ...)
          ("special", ch)                # from _SPECIAL_SINGLE
          ("dollar", "$" or "$$")
          ("space", " ")                 # only for runs of >= 2 spaces/tabs/newlines
          ("comment", str)               # '%' to end-of-line (no newline)
          None                           # end of input
        """
        while not self._eof():
            ch = self._peek()

            # comment
            if ch == "%":
                return self._read_comment()

            # command
            if ch == "\\":
                kind, val = self._read_command()  # kind == "command", val like "\\text"
                if val in self.TEXT_ARG_COMMANDS:
                    return "text", self._read_inline_text(val)
                return (kind, val)

            # dollar / $$
            if ch == "$":
                if self._peek(1) == "$":
                    self._advance(2)
                    return ("dollar", "$$")
                self._advance(1)
                return ("dollar", "$")

            # --- NUMBER must be checked BEFORE SPECIALS ---
            if (ch and ch.isdigit()) or (
                ch in self.decimal_separators and (self._peek(1) and self._peek(1).isdigit())
            ):
                return self._read_number()

            # specials
            if ch in self._SPECIAL_SINGLE:
                self._advance(1)
                return ("special", ch)

            # whitespace (collapse runs)
            if ch.isspace():
                tok = self._read_spaces()
                if tok is not None:
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

class Level2:
    """
    Level-2 lexer that:
      - categorizes level-1 tokens,
      - in next_token(), if a *non-structural* base is immediately followed by '^'/'_',
        it builds and returns a NUCLEUS block: {"type":"NUCLEUS","base":..., "sup":..., "sub":...}.
      - otherwise returns a single ATOM (no grouping here).
    """

    # Structural commands (consume arguments later; not treated as simple bases here)
    STRUCT_COMMANDS = {
        "\\frac", "\\sqrt", "\\int", "\\sum", "\\prod", "\\lim",
        "\\left", "\\right", "\\begin", "\\end"
    }

    # Known left/right delimiter commands
    LEFT_RIGHT_DELIM_COMMANDS = {
        "\\langle","\\rangle","\\lvert","\\rvert","\\lVert","\\rVert",
        "\\lfloor","\\rfloor","\\lceil","\\rceil","\\vert","\\|"
    }

    def __init__(self, source, *, strparser_cls=None):
        if strparser_cls is None:
            strparser_cls = StrParser
        
        source = "{" + source + "}"
        self.lex = strparser_cls(source)
        self._buf_prim = None  # one-token buffer for *categorized* primitive tokens

    def _read_script(self, d):
        """
        Try to read exactly ONE trailing script ('^' or '_') and attach it to node `d`.

        Contract:
          - Look ahead (skipping pure whitespace) for a SUP ('^') or SUB ('_').
          - If none found: return (d, False).
          - If found: consume the operator and read its argument via get_token_or_block():
              * '{...}' or '\left...\right'  -> a BLOCK is returned
              * otherwise exactly one primitive -> an ATOM
          - If `d` is not already a NUCLEUS, wrap it:
                d := {"type":"NUCLEUS","base": d, "sup": None, "sub": None}
          - If the corresponding slot ('sup' for '^', 'sub' for '_') is empty, fill it.
          - If it is already occupied, NEST by creating a new NUCLEUS whose base is `d`
            and only the new script is set:
                d := {"type":"NUCLEUS","base": d, "sup": <arg>, "sub": None}    # for a second '^'
                d := {"type":"NUCLEUS","base": d, "sup": None, "sub": <arg>}    # for a second '_'

        Returns:
          (d, True)  if a script was consumed and attached/nested
          (d, False) if no script was found
        """
        t = self._peek_prim_skipping_ws()
        if t is None or t["kind"] not in ("SUP", "SUB"):
            return d, False

        op = t["kind"]                      # 'SUP' or 'SUB'
        self._next_prim()                   # consume '^' or '_'
        arg = self.get_token_or_block()     # one token or a full {...} / \left...\right block
        if arg is None:
            # gracefully create an error node as the argument
            arg = {"type": "ERROR", "message": f"missing argument after {'^' if op=='SUP' else '_'}"}

        # Ensure `d` is a NUCLEUS to attach to
        if d.get("type") != "NUCLEUS":
            d = {"type": "NUCLEUS", "base": d, "sup": None, "sub": None}

        key = "sup" if op == "SUP" else "sub"
        if d[key] is None:
            # Slot free → attach directly
            d[key] = arg
        else:
            # Slot already used → nest
            if op == "SUP":
                d = {"type": "NUCLEUS", "base": d, "sup": arg, "sub": None}
            else:  # SUB
                d = {"type": "NUCLEUS", "base": d, "sup": None, "sub": arg}

        return d, True

    def next_token(self):
        """
        Return the next high-level token.

        Behavior:
          - Read one *primitive* (categorized) token.
          - If it is a block opener ('{' or '\\left'), return it as a plain ATOM
            (the caller should use get_block() to aggregate the whole block).
          - Otherwise, wrap it as an ATOM and repeatedly call `_read_script(...)`
            to attach any trailing '^'/'_' (this works for letters, numbers, and
            structural commands like '\\int', '\\prod', '\\sum', ...).
          - Nest if multiple same scripts are chained (e.g., x^2^3).
        """
        prim = self._next_prim()
        if prim is None:
            return None

        # Do not attach scripts to brace-opener or \left here; get_block() handles those.
        if prim["kind"] in ("BLOCK_OPEN",) or (prim["kind"] == "STRUCT" and prim["value"] == "\\left"):
            return {"type": "ATOM", "value": prim}

        # Start from a plain ATOM (works for TEXT/NUMBER/DELIM/SYMBOL and STRUCT like \int, \prod)
        node = {"type": "ATOM", "value": prim}

        # Attach scripts (one per _read_script call), nesting if needed
        while True:
            node, found = self._read_script(node)
            if not found:
                break

        return node

    def get_block(self):
        """
        Return the next LaTeX block:
          - '{...}'  → {"type":"BLOCK","delimiter":"{","content":[...]}
          - \left...\right → {"type":"BLOCK","delimiter":("left", <L>, <R>),"content":[...]}
          - otherwise a single atom → {"type":"ATOM","value":<primitive>}
        """
        t = self._peek_prim()
        if t is None:
            return None

        # 1) Brace block
        if t["kind"] == "BLOCK_OPEN":
            self._next_prim()  # consume '{'
            content = []
            while True:
                nt = self._peek_prim()
                if nt is None:
                    break
                if nt["kind"] == "BLOCK_CLOSE":
                    self._next_prim()  # consume '}'
                    break
                content.append(self.get_block())
            return {"type": "BLOCK", "delimiter": "{", "content": content}

        # 2) \left ... \right block
        if t["kind"] == "STRUCT" and t["value"] == "\\left":
            left = self._next_prim()            # has 'delimiter'
            left_delim = left.get("delimiter")
            content = []
            right_delim = None
            while True:
                nt = self._peek_prim()
                if nt is None:
                    break
                if nt["kind"] == "STRUCT" and nt["value"] == "\\right":
                    right = self._next_prim()  # consume \right (already includes delimiter)
                    right_delim = right.get("delimiter")
                    break
                content.append(self.get_block())
            return {"type": "BLOCK", "delimiter": ("left", left_delim, right_delim), "content": content}

        # 3) Otherwise: a single atom
        #return {"type": "ATOM", "value": self.next_token()}
        return self.next_token()

    def get_token_or_block(self):
        """
        Return the next LaTeX argument (generic: works for ^/_/\\frac/\\int/...):

        - If next is a BLOCK_OPEN '{' → return the full brace BLOCK via get_block().
        - If next is a STRUCT '\\left' → return the full \\left ... \\right BLOCK via get_block().
        - Otherwise, consume exactly ONE primitive token and wrap it as:
            {"type":"ATOM","value":<primitive>}
        """
        t = self._peek_prim_skipping_ws()
        if t is None:
            return None
        if t["kind"] == "BLOCK_OPEN":
            return self.get_block()
        if t["kind"] == "STRUCT" and t["value"] == "\\left":
            return self.get_block()
        return {"type": "ATOM", "value": self._next_prim()}

    # ---------------- Primitive (categorized) token layer ----------------

    def _next_prim(self):
        """Fetch next *categorized* primitive token (skips comments)."""
        if self._buf_prim is not None:
            t = self._buf_prim
            self._buf_prim = None
            return t

        # Pull from level-1 lexer
        raw = self.lex.next_token()
        while raw is not None and raw[0] == "comment":
            raw = self.lex.next_token()
        if raw is None:
            return None

        kind, val = raw

        # Map level-1 to primitive categories
        if kind == "word":
            return {"kind": "TEXT", "value": val}
        if kind == "number":
            return {"kind": "NUMBER", "value": val}
        if kind == "space":
            return {"kind": "TEXT", "value": " "}  # normalize to TEXT " "
        if kind == "dollar":
            return {"kind": "MATH_SHIFT", "value": val}
        if kind == "special":
            if val == "{":
                return {"kind": "BLOCK_OPEN"}
            if val == "}":
                return {"kind": "BLOCK_CLOSE"}
            if val == "^":
                return {"kind": "SUP"}
            if val == "_":
                return {"kind": "SUB"}
            return {"kind": "DELIM", "value": val}
        if kind == "command":
            # \left / \right: read exactly one delimiter immediately (as part of this token)
            if val in ("\\left", "\\right"):
                delim = self._read_left_right_delimiter_categorized()
                return {"kind": "STRUCT", "value": val, "delimiter": delim}
            # structural vs symbol
            if val in self.STRUCT_COMMANDS:
                return {"kind": "STRUCT", "value": val}
            return {"kind": "SYMBOL", "value": val}
        if kind == "text":
            info = val if isinstance(val, dict) else {"command": None, "text": str(val), "closed": False}
            return {
                "kind": "TEXT_INLINE",
                "command": info.get("command"),
                "text": info.get("text", ""),
                "closed": bool(info.get("closed", False)),
            }

        

        # Fallback
        return {"kind": "TEXT", "value": str(val)}

    def _peek_prim(self):
        t = self._next_prim()
        if t is not None:
            self._buf_prim = t
        return t

    def _peek_prim_skipping_ws(self):
        """Peek one primitive token, skipping pure whitespace TEXT."""
        stash = []
        t = self._next_prim()
        while t is not None and t["kind"] == "TEXT" and t["value"].strip() == "":
            stash.append(t)
            t = self._next_prim()
        if t is not None:
            self._buf_prim = t
        # restore skipped
        for x in reversed(stash):
            if self._buf_prim is None:
                self._buf_prim = x
            else:
                # chain pushback of more than one: reinsert in front of current buffer
                cur = self._buf_prim
                self._buf_prim = x
                # we only keep one lookahead buffer; whitespace skipping is idempotent
        return t

    def _read_left_right_delimiter_categorized(self):
        """
        Read exactly one delimiter after \\left/\\right.
        Accepts:
          - DELIM: '(', ')', '[', ']', '|', '.', ...
          - SYMBOL: '\\langle', '\\rangle', ...
          - TEXT '.' -> converted to DELIM '.'
        Skips whitespace TEXT between command and delimiter.
        """
        # skip ws
        t = self._peek_prim_skipping_ws()
        if t is None:
            return {"kind": "ERROR", "message": "missing delimiter"}
        if t["kind"] in ("DELIM", "MATH_SHIFT"):
            self._next_prim()
            return t
        if t["kind"] == "SYMBOL" and t["value"] in self.LEFT_RIGHT_DELIM_COMMANDS:
            self._next_prim()
            return t
        if t["kind"] == "TEXT" and t["value"] == ".":
            self._next_prim()
            return {"kind": "DELIM", "value": "."}
        # not a valid delimiter; keep it for the outer logic but return an error payload
        return {"kind": "ERROR", "message": "invalid delimiter", "got": t}



from pprint import pprint
s = r" \cos(21\theta\phi) = \cos^2 \theta\phi - \sin^2 \theta\phi"

parser = StrParser(s)

tk = parser.next_token()
while tk is not None:
    #print(tk)
    tk = parser.next_token()



s = r"{a = x^2^3 + {toto} \left(1 - y\right)}"
s = r"I = \int_i^n f(x^2)dx"
s = r"50 \text{ apples} \times 100 \text{ apples} = \textbf{lots of apples}"

lvl = Level2(s)
b = lvl.get_block()
pprint(b)

