#!/usr/bin/env python3
"""tok — the tokeniser node. M1 of tnwr.

Serves the namespace in M1-SPEC.md over 9P. A fork is a directory: mkdir in
fork/ and you get a complete copy of this tree with its own params, which you
then diverge. The parent is untouched.

Backends are pluggable because the point of the node is the *boundary*, not
the tokeniser:

    simple      built-in, no dependencies. GPT-2's byte-level pre-tokenisation
                regex. Real boundaries, no vocabulary, instant, ids are hashes.
    gguf:<path> the real thing -- a GGUF's own vocabulary via llama-cpp-python,
                loaded vocab-only so it costs no VRAM. Ids are real indices.
    hf:<id>     huggingface `tokenizers`, if installed.

Run:  python3 tok_node.py [port]
"""

from __future__ import annotations

import difflib
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sukkal9p import Entry, Server  # noqa: E402

# GPT-2's pre-tokenisation pattern. Not a vocabulary, but the boundaries it
# produces are the ones people are actually surprised by — leading spaces
# joined to words, contractions split, digit runs broken up.
GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+"""
)

SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<s>", "</s>"]

# Loaded vocabularies, keyed by gguf path. Loading is the expensive part and
# the vocab never changes, so it is held for the life of the process.
_VOCABS: dict = {}


class TokState:
    """One configuration: parameters plus input. A fork is another of these."""

    def __init__(self, parent: "TokState | None" = None):
        if parent is None:
            self.params = {
                "tokeniser": "simple",
                "add_bos": "1",
                "add_eos": "0",
                "special": "1",
                "byte_fallback": "0",
            }
            self.text = b"The oven is lit, and the graph remembers."
        else:
            self.params = dict(parent.params)
            self.text = parent.text
        self.parent = parent
        self._cache: tuple[str, list] | None = None

    # -- tokenising -------------------------------------------------------

    def hash(self) -> str:
        blob = repr(sorted(self.params.items())).encode() + b"\0" + self.text
        return hashlib.sha256(blob).hexdigest()[:16]

    def tokens(self) -> list[tuple[int, str, int, int]]:
        """(id, string, byte_start, byte_end). Cached against hash()."""
        h = self.hash()
        if self._cache and self._cache[0] == h:
            return self._cache[1]
        toks = self._tokenise()
        self._cache = (h, toks)
        return toks

    def _tokenise(self):
        name = self.params["tokeniser"]
        text = self.text.decode("utf-8", "replace")
        if name.startswith("gguf:"):
            return self._gguf(name[5:], text)
        if name.startswith("hf:"):
            pieces = self._hf(name[3:], text)
        else:
            pieces = self._simple(text)

        # Spans are BYTE offsets, not character offsets. cosmic-text reports
        # glyph positions as byte ranges, so anything else silently misplaces
        # every highlight after the first non-ASCII character -- and this
        # system is explicitly meant to be used with cuneiform and Arabic.
        raw = text.encode()
        out, pos = [], 0
        if self.params["add_bos"] == "1":
            out.append((1, "<s>", 0, 0))
        for p in pieces:
            pb = p.encode()
            start = raw.find(pb, pos)
            if start < 0:
                start = pos
            end = start + len(pb)
            pos = end
            tid = int(hashlib.sha256(pb).hexdigest()[:6], 16)
            out.append((tid, p, start, end))
        if self.params["add_eos"] == "1":
            out.append((2, "</s>", len(raw), len(raw)))
        return out

    def _gguf(self, path, text):
        """Real tokenisation, straight off a GGUF's own vocabulary.

        `vocab_only=True` loads the tokeniser without the weights, so this is
        fast and costs no VRAM. Token ids here are genuine vocabulary indices
        rather than the `simple` backend's hashes.
        """
        from llama_cpp import Llama

        llm = _VOCABS.get(path)
        if llm is None:
            llm = Llama(model_path=path, vocab_only=True, verbose=False)
            _VOCABS[path] = llm

        raw = text.encode()
        ids = llm.tokenize(
            raw,
            add_bos=self.params["add_bos"] == "1",
            special=self.params["special"] == "1",
        )
        out, pos = [], 0
        for tid in ids:
            piece = llm.detokenize([tid])
            start, end = pos, pos + len(piece)
            pos = end
            out.append((tid, piece.decode("utf-8", "replace"), start, end))
        return out

    def _simple(self, text: str) -> list[str]:
        if self.params["special"] == "1":
            parts, rest = [], text
            pattern = "|".join(re.escape(s) for s in SPECIALS)
            for chunk in re.split(f"({pattern})", rest):
                if not chunk:
                    continue
                if chunk in SPECIALS:
                    parts.append(chunk)
                else:
                    parts.extend(m.group() for m in GPT2_PAT.finditer(chunk))
            return parts
        return [m.group() for m in GPT2_PAT.finditer(text)]

    def _hf(self, ident: str, text: str) -> list[str]:
        try:
            from tokenizers import Tokenizer
        except ImportError:
            return ["<hf backend needs: pip install tokenizers>"]
        tok = Tokenizer.from_pretrained(ident)
        enc = tok.encode(text, add_special_tokens=self.params["special"] == "1")
        return enc.tokens

    # -- rendering --------------------------------------------------------

    def ids(self) -> bytes:
        return (" ".join(str(t[0]) for t in self.tokens()) + "\n").encode()

    def strings(self) -> bytes:
        lines = [t[1].replace("\\", "\\\\").replace("\n", "\\n") for t in self.tokens()]
        return ("\n".join(lines) + "\n").encode()

    def spans(self) -> bytes:
        return ("\n".join(f"{t[2]} {t[3]}" for t in self.tokens()) + "\n").encode()

    def count(self) -> bytes:
        return f"{len(self.tokens())}\n".encode()

    def diff(self) -> bytes:
        if self.parent is None:
            return b"(root: nothing to diff against)\n"
        out = []
        for k in sorted(self.params):
            a, b = self.parent.params[k], self.params[k]
            if a != b:
                out.append(f"params  {k:14} {a} -> {b}")
        if self.text != self.parent.text:
            out.append("in      text           changed")
        pa, pb = self.parent.tokens(), self.tokens()
        if len(pa) != len(pb):
            out.append(f"out     count          {len(pa)} -> {len(pb)}")
        ia = [t[0] for t in pa]
        ib = [t[0] for t in pb]
        sm = difflib.SequenceMatcher(a=ia, b=ib, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            left = " ".join(repr(t[1]) for t in pa[i1:i2]) or "-"
            right = " ".join(repr(t[1]) for t in pb[j1:j2]) or "-"
            out.append(f"out     @{i1:<13} {left} -> {right}")
        return ("\n".join(out) + "\n").encode() if out else b"(identical)\n"

    def diverged(self) -> bytes:
        """Token index ranges that differ from the parent, machine-readable.

        One line per differing region: `i1 i2 j1 j2` — a half-open range into
        the parent's tokens, then into this fork's. `diff` is for a human;
        this is for a renderer, so the marker lands on exactly the tokens the
        diff is talking about rather than being re-derived downstream.
        """
        if self.parent is None:
            return b""
        ia = [t[0] for t in self.parent.tokens()]
        ib = [t[0] for t in self.tokens()]
        sm = difflib.SequenceMatcher(a=ia, b=ib, autojunk=False)
        rows = [
            f"{i1} {i2} {j1} {j2}"
            for tag, i1, i2, j1, j2 in sm.get_opcodes()
            if tag != "equal"
        ]
        return ("\n".join(rows) + "\n").encode() if rows else b""

    def describe(self) -> bytes:
        return (
            "name        tokeniser\n"
            "kind        transform\n"
            "depth       full\n"
            "shows       ids, strings, spans, count\n"
            "opaque      none\n"
            f"backend     {self.params['tokeniser']}\n"
        ).encode()


# ---------------------------------------------------------------- namespace

def build(state: TokState, name: str, key: str = "") -> Entry:
    """Construct the 9P tree for one state. Forks reuse this verbatim."""
    root = Entry(name, is_dir=True, key=key)
    root.add(Entry("describe", reader=state.describe, key="d"))
    root.add(Entry("diff", reader=state.diff, key="f"))
    root.add(Entry("diverged", reader=state.diverged))
    root.add(Entry("keys", reader=root.keys))
    # What a renderer should show for this node. `body` is the text, `spans`
    # are byte ranges to tint, `header` lines are labelled values. Without
    # this the core would have to know what a tokeniser is.
    root.add(Entry("view", reader=lambda: (
        "body   in/text\n"
        "spans  out/spans\n"
        "header params/special\n"
        "header out/count\n"
    ).encode()))

    params = root.add(Entry("params", is_dir=True, key="p"))
    params.add(Entry("keys", reader=params.keys))
    # Letters chosen deliberately, not by sort order. tokeniser=t, special=s,
    # add_bos=b, add_eos=e, byte_fallback=y.
    PARAM_KEYS = {
        "tokeniser": "t",
        "add_bos": "b",
        "add_eos": "e",
        "special": "s",
        "byte_fallback": "y",
    }
    for key in state.params:
        def rd(k=key):
            return (state.params[k] + "\n").encode()

        def wr(data, k=key):
            state.params[k] = data.decode().strip()

        # Params are config, so generations capture them. `in/` is input and
        # `out/` is output, and neither is.
        params.add(Entry(key, reader=rd, writer=wr, key=PARAM_KEYS.get(key, ""),
                         config=True))

    inp = root.add(Entry("in", is_dir=True, key="i"))
    inp.add(Entry("keys", reader=inp.keys))

    def wtext(data):
        state.text = data.rstrip(b"\n")

    inp.add(Entry("text", reader=lambda: state.text + b"\n", writer=wtext, key="x"))

    out = root.add(Entry("out", is_dir=True, key="o"))
    out.add(Entry("keys", reader=out.keys))
    out.add(Entry("ids", reader=state.ids, key="n"))
    out.add(Entry("strings", reader=state.strings, key="s"))
    out.add(Entry("spans", reader=state.spans, key="p"))
    out.add(Entry("count", reader=state.count, key="c"))
    out.add(Entry("hash", reader=lambda: (state.hash() + "\n").encode()))

    forks = root.add(Entry("fork", is_dir=True, key="k"))
    forks.add(Entry("keys", reader=forks.keys))

    def make_fork(child_name: str) -> Entry:
        # inherits params and text by value, then diverges independently
        # The fork's own letter is its first character -- chosen by whoever
        # named it, which keeps assignment in the operator's hands rather than
        # in creation order.
        return forks.add(build(TokState(parent=state), child_name,
                               key=child_name[:1]))

    forks.on_create = make_fork
    return root


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5640
    root = build(TokState(), "/")
    print(f"tok: namespace ready, {len(root.children)} entries", flush=True)
    Server(root).serve(port=port)


if __name__ == "__main__":
    main()
