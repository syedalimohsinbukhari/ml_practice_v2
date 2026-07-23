"""Reflow prose in a LaTeX or Markdown file to one sentence per line.

Protects: comments, tables/tabular/equation/figure/align environments,
markdown tables/headings/blockquotes/list markers, and common abbreviations.
"""
import re
import sys

ABBREVS = ["e.g.", "i.e.", "cf.", "vs.", "et al.", "Fig.", "Figs.", "Eq.",
           "Eqs.", "Sec.", "Secs.", "Tab.", "ca.", "approx.", "resp."]
SENTINEL = "\x00DOT\x00"

# Split after . ! ? (plus trailing quotes/brackets/braces), before a new
# sentence starting with a capital, digit, \command, $, (, `, [, * or quote.
SPLIT_RE = re.compile(r"([.!?][\'\")\]\}]*)[ \t]+(?=[A-Z0-9\\($`\[*\"“])")

NO_REFLOW_ENVS = {"table", "table*", "tabular", "tabular*", "equation",
                  "equation*", "align", "align*", "figure", "figure*",
                  "verbatim"}


def protect(text: str) -> str:
    for a in ABBREVS:
        text = text.replace(a, a[:-1] + SENTINEL)
    return text


def unprotect(text: str) -> str:
    return text.replace(SENTINEL, ".")


def split_sentences(text: str) -> str:
    text = protect(text)
    text = SPLIT_RE.sub(lambda m: m.group(1) + "\n", text)
    return unprotect(text)


def reflow_block(lines):
    """Join a prose block into logical lines, then one sentence per line."""
    out = []
    buf = []

    def flush():
        if buf:
            joined = re.sub(r"[ \t]+", " ", " ".join(s.strip() for s in buf))
            out.extend(split_sentences(joined).split("\n"))
            buf.clear()

    for ln in lines:
        s = ln.strip()
        # \item starts a new logical unit but its own text is reflowed
        if s.startswith("\\item") or s.startswith("\\paragraph"):
            flush()
            buf.append(s)
        else:
            buf.append(s)
    flush()
    return out


def process_tex(src: str) -> str:
    out, block = [], []
    env_stack = []

    def flush_block():
        if block:
            out.extend(reflow_block(block))
            block.clear()

    for ln in src.split("\n"):
        s = ln.strip()
        m_begin = re.match(r"\\begin\{([^}]+)\}", s)
        m_end = re.match(r"\\end\{([^}]+)\}", s)
        in_noreflow = any(e in NO_REFLOW_ENVS for e in env_stack)

        if m_begin and m_begin.group(1) in NO_REFLOW_ENVS:
            flush_block()
            env_stack.append(m_begin.group(1))
            out.append(ln)
            continue
        if m_end and env_stack and m_end.group(1) == env_stack[-1]:
            flush_block()
            env_stack.pop()
            out.append(ln)
            continue
        if in_noreflow:
            out.append(ln)  # verbatim inside protected envs
            continue
        if m_begin or m_end:  # reflowable envs (quote, itemize, ...)
            flush_block()
            if m_begin:
                env_stack.append(m_begin.group(1))
            out.append(ln)
            continue
        if s == "" or s.startswith("%"):
            flush_block()
            out.append(ln)
            continue
        if re.match(r"\\(chapter|section|subsection|subsubsection|label|"
                    r"graphicspath|centering|includegraphics|"
                    r"caption)\b", s):
            flush_block()
            out.append(ln)
            continue
        block.append(ln)
    flush_block()
    return "\n".join(out)


def process_md(src: str) -> str:
    out, block = [], []

    def flush_block():
        if block:
            out.extend(reflow_block(block))
            block.clear()

    in_code = False
    for ln in src.split("\n"):
        s = ln.strip()
        if s.startswith("```"):
            in_code = not in_code
            flush_block()
            out.append(ln)
            continue
        if in_code:
            out.append(ln)
            continue
        # protected line types: blank, heading, table, blockquote, hrule,
        # list items (keep as single logical lines but still split sentences
        # would break md list continuation -> leave list items intact)
        if (s == "" or s.startswith("#") or s.startswith("|")
                or s.startswith(">") or s.startswith("---")
                or re.match(r"[-*] ", s) or re.match(r"\d+\. ", s)):
            flush_block()
            out.append(ln)
            continue
        block.append(ln)
    flush_block()
    return "\n".join(out)


if __name__ == "__main__":
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    result = process_tex(src) if path.endswith(".tex") else process_md(src)
    with open(path, "w") as f:
        f.write(result)
    print(f"reflowed {path}")
