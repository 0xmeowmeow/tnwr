#!/usr/bin/env python3
"""graph — the program IS the graph. Text is one projection of it.

Brief v2, locked: "A graph view built over text code is a toy. A graph that is
the source of truth is the thing described in the brief."

So the pipeline lives in the namespace, one directory per step:

    /graph/1/op      strip
    /graph/2/op      prefix
    /graph/2/arg     <|im_start|>system\\n

`/text` renders that as source. It is **read-only** — deliberately. You cannot
edit the projection, because it is not the program. Change `/graph/2/arg` and
the text changes, because the text was never the thing.

Run:  .venv/bin/python graph_node.py [port]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sukkal9p import Entry, Server  # noqa: E402

# Each op is a pure text -> text function. Adding one here is the only way to
# extend the language, which keeps the graph's vocabulary small and legible.
OPS = {
    "strip": lambda t, a: t.strip(),
    "upper": lambda t, a: t.upper(),
    "lower": lambda t, a: t.lower(),
    "prefix": lambda t, a: a + t,
    "suffix": lambda t, a: t + a,
    "replace": lambda t, a: t.replace(*(a.split("->", 1) + [""])[:2]),
    "collapse": lambda t, a: " ".join(t.split()),
    "nothing": lambda t, a: t,
}


class Step:
    def __init__(self, op: str = "nothing", arg: str = ""):
        self.op = op
        self.arg = arg


class GraphState:
    """A program held as a graph. Nothing here parses text into structure —
    the structure is what exists, and text is rendered out of it."""

    def __init__(self, parent: "GraphState | None" = None):
        if parent is None:
            self.input = "  the oven is LIT  "
            self.steps: dict[str, Step] = {
                "1": Step("strip"),
                "2": Step("lower"),
                "3": Step("prefix", "<|im_start|>system\n"),
                "4": Step("suffix", "<|im_end|>"),
            }
        else:
            self.input = parent.input
            self.steps = {k: Step(v.op, v.arg) for k, v in parent.steps.items()}
        self.parent = parent

    def order(self) -> list[str]:
        # numeric where possible, so /graph/10 sorts after /graph/9
        return sorted(self.steps, key=lambda k: (len(k), k))

    # -- the projection ---------------------------------------------------

    def text(self) -> bytes:
        """The program as source. Generated, never parsed back."""
        lines = []
        for k in self.order():
            st = self.steps[k]
            if st.arg:
                shown = st.arg.replace("\n", "\\n")
                lines.append(f"{k}  {st.op} {shown!r}")
            else:
                lines.append(f"{k}  {st.op}")
        return ("\n".join(lines) + "\n").encode() if lines else b"(empty program)\n"

    def run(self) -> tuple[str, list[tuple[str, str, str]]]:
        t = self.input
        trail = []
        for k in self.order():
            st = self.steps[k]
            fn = OPS.get(st.op)
            before = t
            t = fn(t, st.arg) if fn else t
            trail.append((k, st.op if fn else f"{st.op}?", t))
            del before
        return t, trail

    def result(self) -> bytes:
        return (self.run()[0] + "\n").encode()

    def trace_text(self) -> bytes:
        """Every intermediate value. The wire is visible, which is the whole
        reason for building any of this."""
        rows = ["step op value"]
        for k, op, val in self.run()[1]:
            rows.append(f"{k} {op} {val.replace(chr(10), '\\n')!r}")
        return ("\n".join(rows) + "\n").encode()

    def hash(self) -> str:
        blob = repr([(k, self.steps[k].op, self.steps[k].arg) for k in self.order()])
        return hashlib.sha256((blob + self.input).encode()).hexdigest()[:16]

    def diff(self) -> bytes:
        if self.parent is None:
            return b"(root: nothing to diff against)\n"
        out = []
        keys = sorted(set(self.steps) | set(self.parent.steps), key=lambda k: (len(k), k))
        for k in keys:
            a = self.parent.steps.get(k)
            b = self.steps.get(k)
            av = f"{a.op} {a.arg!r}" if a else "-"
            bv = f"{b.op} {b.arg!r}" if b else "-"
            if av != bv:
                out.append(f"graph  {k:4} {av} -> {bv}")
        if self.input != self.parent.input:
            out.append("in     text changed")
        ra, rb = self.parent.run()[0], self.run()[0]
        if ra != rb:
            out.append(f"out    result {ra!r}")
            out.append(f"           -> {rb!r}")
        return ("\n".join(out) + "\n").encode() if out else b"(identical)\n"

    def describe(self) -> bytes:
        return (
            "name        graph\n"
            "kind        program\n"
            "depth       full\n"
            "shows       text, result, steps\n"
            "opaque      none\n"
            f"ops         {', '.join(sorted(OPS))}\n"
        ).encode()


# ---------------------------------------------------------------- namespace

def build(state: GraphState, name: str, key: str = "") -> Entry:
    root = Entry(name, is_dir=True, key=key)
    root.add(Entry("describe", reader=state.describe, key="d"))
    root.add(Entry("diff", reader=state.diff, key="f"))
    root.add(Entry("keys", reader=root.keys))
    root.add(Entry("view", reader=lambda: (
        "body   text\n"
        "table  out/steps  step op value\n"
        "header out/result\n"
    ).encode()))

    # The projection. Read-only: you cannot edit text into the program,
    # because the text is not the program.
    root.add(Entry("text", reader=state.text, key="x"))

    graph = root.add(Entry("graph", is_dir=True, key="g"))
    graph.add(Entry("keys", reader=graph.keys))

    def add_step(sk: str) -> Entry:
        st = state.steps[sk]
        d = graph.add(Entry(sk, is_dir=True, key=sk[:1]))
        d.add(Entry("keys", reader=d.keys))

        def rd_op(s=st):
            return (s.op + "\n").encode()

        def wr_op(data, s=st):
            s.op = data.decode().strip()

        def rd_arg(s=st):
            return (s.arg + "\n").encode()

        def wr_arg(data, s=st):
            s.arg = data.decode()

        # The program is config: a generation captures every step, so rolling
        # back rolls the program back. `in/text` is what it runs on, and is not.
        d.add(Entry("op", reader=rd_op, writer=wr_op, key="o", config=True))
        d.add(Entry("arg", reader=rd_arg, writer=wr_arg, key="a", config=True))
        return d

    for sk in state.order():
        add_step(sk)

    def make_step(child_name: str) -> Entry:
        state.steps[child_name] = Step()
        return add_step(child_name)

    graph.on_create = make_step
    graph.on_remove = lambda child_name: state.steps.pop(child_name, None)

    inp = root.add(Entry("in", is_dir=True, key="i"))
    inp.add(Entry("keys", reader=inp.keys))

    def wtext(data):
        state.input = data.decode().rstrip("\n")

    inp.add(Entry("text", reader=lambda: (state.input + "\n").encode(),
                  writer=wtext, key="x"))

    out = root.add(Entry("out", is_dir=True, key="o"))
    out.add(Entry("keys", reader=out.keys))
    out.add(Entry("result", reader=state.result, key="r"))
    out.add(Entry("steps", reader=state.trace_text, key="s"))
    out.add(Entry("hash", reader=lambda: (state.hash() + "\n").encode()))

    forks = root.add(Entry("fork", is_dir=True, key="k"))
    forks.add(Entry("keys", reader=forks.keys))

    def make_fork(child_name: str) -> Entry:
        return forks.add(build(GraphState(parent=state), child_name, key=child_name[:1]))

    forks.on_create = make_fork
    return root


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5642
    root = build(GraphState(), "/")
    print(f"graph: namespace ready, {len(root.children)} entries", flush=True)
    Server(root).serve(port=port)


if __name__ == "__main__":
    main()
