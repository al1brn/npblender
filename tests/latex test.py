# All comments in English as per your preference
from dataclasses import dataclass, field
from typing import Any, Dict, List
from pylatexenc.latexwalker import LatexWalker
from pylatexenc.macrospec import MacroSpec, LatexContextDb

# --- Domain model you can enrich ---
@dataclass
class Node:
    kind: str
    name: str | None = None
    text: str | None = None
    args: Dict[str, Any] = field(default_factory=dict)
    children: List["Node"] = field(default_factory=list)

# --- Define a custom \param{...} macro (one braced argument) ---
param_spec = MacroSpec("param", arguments_spec_list=["{"] )

ctx = LatexContextDb()
ctx.add_context_category("custom", macros=[param_spec], environments=[], specials=[])

def parse_keyvals(s: str) -> Dict[str, Any]:
    """Tiny key=val parser for numbers/strings; extend as needed."""
    out: Dict[str, Any] = {}
    # Split on commas not inside quotes/brackets -> keep simple here
    import ast, re
    def split_top_level_commas(t: str) -> List[str]:
        parts, buf, lvl, q = [], [], 0, None
        i = 0
        while i < len(t):
            ch = t[i]
            if q:
                buf.append(ch)
                if ch == q and (i == 0 or t[i-1] != '\\'):
                    q = None
                elif ch == '\\' and i + 1 < len(t):
                    i += 1; buf.append(t[i])
                i += 1; continue
            if ch in "'\"":
                q = ch; buf.append(ch); i += 1; continue
            if ch in "([{": lvl += 1
            elif ch in ")]}": lvl -= 1
            elif ch == "," and lvl == 0:
                parts.append("".join(buf).strip()); buf = []; i += 1; continue
            buf.append(ch); i += 1
        if buf: parts.append("".join(buf).strip())
        return [p for p in parts if p]

    for chunk in split_top_level_commas(s):
        if "=" not in chunk:
            raise ValueError(f"Expected key=value in {chunk!r}")
        k, v = chunk.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Try literal evaluation first; fallback to raw string
        try:
            out[k] = ast.literal_eval(v)
        except Exception:
            out[k] = v
    return out

def to_node(n) -> Node:
    """Convert pylatexenc nodes to our domain Node."""
    from pylatexenc.latexnodes import LatexMacroNode, LatexCharsNode, LatexGroupNode
    if isinstance(n, LatexCharsNode):
        return Node(kind="text", text=n.chars)
    if isinstance(n, LatexMacroNode):
        if n.macroname == "param":
            # One braced argument -> n.nodeargd.argnlist[0] is a LatexGroupNode
            grp: LatexGroupNode = n.nodeargd.argnlist[0]
            content = "".join(getattr(c, "chars", "") for c in grp.nodelist)
            return Node(kind="macro", name="param", args=parse_keyvals(content))
        # Generic macro: keep name and raw latex of first arg if any
        return Node(kind="macro", name=n.macroname)
    # Fallback: wrap children if present
    if hasattr(n, "nodelist") and n.nodelist:
        out = Node(kind="container")
        out.children = [to_node(c) for c in n.nodelist]
        return out
    return Node(kind="unknown")

def parse_latex(src: str) -> List[Node]:
    w = LatexWalker(src, latex_context=ctx)
    nodelist, _delta = w.parse_content(
        __import__("pylatexenc.latexnodes.parsers").latexnodes.parsers.LatexGeneralNodesParser()
    )
    return [to_node(n) for n in nodelist.nodelist if getattr(n, "nodetype", None) != "comment"]

# --- Demo ---
doc = r"""
Here is a parameter: \param{x=1.5, name='test', flags=[1,2,3]}
And some text.
"""
tree = parse_latex(doc)
# tree[1] will be the \param(...) node; tree is a hierarchy you can enrich.
