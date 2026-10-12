#!/usr/bin/env python3
"""sampler — M1.5 of tnwr. The node that makes sampling visible.

Same shape as the tokeniser node, same fork mechanism, pointed at generation
instead. The point is `out/topk`: at every step, what the model actually
proposed and how the sampling parameters reshaped it. Temperature and min_p
stop being folklore the moment you can watch them cull the tail.

Generation is lazy and cached against `out/hash`, which is a hash of the
parameters only — so a poller can watch `hash` without ever triggering a run.

Run:  .venv/bin/python sampler_node.py [port]
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sukkal9p import Entry, Server  # noqa: E402

# Loaded models, keyed by (path, n_ctx). Loading is the expensive part; two
# forks of the same model share one instance, which is also what lets
# llama.cpp reuse the KV prefix between them.
_MODELS: dict = {}

DEFAULT_MODEL = str(Path.home() / "data/models/qwen2.5-0.5b-instruct-q8_0.gguf")


def _softmax(x):
    import numpy as np

    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _load(path: str, n_ctx: int):
    key = (path, n_ctx)
    llm = _MODELS.get(key)
    if llm is None:
        from llama_cpp import Llama

        # logits_all is required for per-step logprobs — without it llama.cpp
        # keeps only the last position's logits and `out/topk`, the entire
        # point of this node, is unavailable.
        llm = Llama(
            model_path=path,
            n_ctx=n_ctx,
            verbose=False,
            seed=-1,
            logits_all=True,
        )
        _MODELS[key] = llm
    return llm


class SampleState:
    """One sampling configuration. A fork is another of these."""

    def __init__(self, parent: "SampleState | None" = None):
        if parent is None:
            self.params = {
                "model": DEFAULT_MODEL,
                "prompt": "The oven is lit, and",
                "temperature": "0.8",
                "top_p": "0.95",
                "min_p": "0.05",
                "top_k": "40",
                "seed": "1",
                "n_predict": "16",
                "n_ctx": "2048",
                "show_k": "8",
            }
        else:
            self.params = dict(parent.params)
        self.parent = parent
        self._cache: tuple[str, dict] | None = None

    # -- running ----------------------------------------------------------

    def hash(self) -> str:
        return hashlib.sha256(
            repr(sorted(self.params.items())).encode()
        ).hexdigest()[:16]

    def _num(self, key, cast=float, default=0):
        try:
            return cast(self.params[key])
        except (KeyError, ValueError):
            return default

    def run(self) -> dict:
        """Generate, with per-step candidates. Cached against hash()."""
        h = self.hash()
        if self._cache and self._cache[0] == h:
            return self._cache[1]
        try:
            out = self._generate()
        except Exception as exc:  # a missing model file is the common case
            out = {"error": f"{type(exc).__name__}: {exc}", "text": "", "steps": []}
        self._cache = (h, out)
        return out

    def _generate(self) -> dict:
        """Run the sampler chain by hand so every stage is observable.

        `create_completion` only reports the model's own distribution, which
        tells you what the model wanted but not what your parameters did to it.
        Doing the chain here means each filter's effect is a number: how many
        candidates went in, how many came out.

        Order applied — stated explicitly because implementations differ and
        the order changes the result:

            logits -> temperature -> softmax -> top_k -> top_p -> min_p
                   -> renormalise -> sample
        """
        import numpy as np

        llm = _load(self.params["model"], self._num("n_ctx", int, 2048))
        llm.reset()

        temp = self._num("temperature", float, 0.8)
        top_k = self._num("top_k", int, 40)
        top_p = self._num("top_p", float, 0.95)
        min_p = self._num("min_p", float, 0.05)
        show_k = self._num("show_k", int, 8)
        n_predict = self._num("n_predict", int, 16)
        rng = np.random.default_rng(self._num("seed", int, 1))

        toks = llm.tokenize(self.params["prompt"].encode())
        llm.eval(toks)

        steps, text = [], ""
        for _ in range(n_predict):
            logits = np.asarray(llm.scores[llm.n_tokens - 1], dtype=np.float64)

            # raw: what the model actually believes, before anything
            raw = _softmax(logits)
            order = np.argsort(-raw)
            raw_top = [(int(i), float(raw[i])) for i in order[:show_k]]

            # temperature reshapes; 0 means take the argmax
            if temp <= 0:
                probs = np.zeros_like(raw)
                probs[int(np.argmax(logits))] = 1.0
            else:
                probs = _softmax(logits / temp)

            live = np.argsort(-probs)
            n_temp = int(np.count_nonzero(probs > 1e-9))

            # top_k
            if top_k > 0:
                live = live[:top_k]
            n_topk = len(live)

            # top_p — smallest prefix whose mass reaches p
            cum = np.cumsum(probs[live])
            keep = int(np.searchsorted(cum, top_p) + 1)
            live = live[:max(1, min(keep, len(live)))]
            n_topp = len(live)

            # min_p — relative to the most likely survivor. This is the one
            # people cannot picture, and it is the whole reason for this file.
            if min_p > 0 and len(live):
                floor = probs[live[0]] * min_p
                live = live[probs[live] >= floor]
            n_minp = max(1, len(live))

            final = probs[live]
            final = final / final.sum() if final.sum() > 0 else final
            pick = int(rng.choice(live, p=final)) if len(live) > 1 else int(live[0])

            piece = llm.detokenize([pick]).decode("utf-8", "replace")
            text += piece
            steps.append({
                "chosen": piece,
                "chosen_id": pick,
                "candidates": [
                    (llm.detokenize([i]).decode("utf-8", "replace"), p)
                    for i, p in raw_top
                ],
                "kept": (int(n_temp), int(n_topk), int(n_topp), int(n_minp)),
                "survivors": [
                    (llm.detokenize([int(i)]).decode("utf-8", "replace"), float(probs[i]))
                    for i in live[:show_k]
                ],
            })
            if pick == llm.token_eos():
                break
            llm.eval([pick])

        return {"error": "", "text": text, "steps": steps}

    # -- files ------------------------------------------------------------

    def text(self) -> bytes:
        r = self.run()
        return ((r["error"] or r["text"]) + "\n").encode()

    def topk(self) -> bytes:
        """`<step> <rank> <prob> <token>` — machine-readable, token last
        because it may contain spaces. This is the file the whole node is for.
        """
        r = self.run()
        if r["error"]:
            return (r["error"] + "\n").encode()
        rows = []
        for i, step in enumerate(r["steps"]):
            for rank, (tok, prob) in enumerate(step["candidates"]):
                mark = "*" if tok == step["chosen"] else " "
                rows.append(
                    f"{i} {rank} {prob:.4f} {mark} {tok!r}"
                )
        return ("\n".join(rows) + "\n").encode() if rows else b"(no steps)\n"

    def cull(self) -> bytes:
        """`<step> <after_temp> <after_top_k> <after_top_p> <after_min_p> <token>`

        How many candidates survive each filter. Reading down the min_p column
        is the fastest way to see what that parameter is actually doing, which
        is not visible anywhere else.
        """
        r = self.run()
        if r["error"]:
            return (r["error"] + "\n").encode()
        # A header row naming the columns, so a view can select by name
        # instead of by position and the file stays self-describing.
        rows = ["step temp top_k top_p min_p token"]
        for i, st in enumerate(r["steps"]):
            a, b, c, d = st["kept"]
            rows.append(f"{i} {a} {b} {c} {d} {st['chosen']!r}")
        return ("\n".join(rows) + "\n").encode() if len(rows) > 1 else b"(no steps)\n"

    def survivors(self) -> bytes:
        """The reshaped distribution — what is left after the chain, with the
        probabilities actually sampled from."""
        r = self.run()
        if r["error"]:
            return (r["error"] + "\n").encode()
        rows = []
        for i, st in enumerate(r["steps"]):
            for rank, (tok, p) in enumerate(st["survivors"]):
                mark = "*" if tok == st["chosen"] else " "
                rows.append(f"{i} {rank} {p:.4f} {mark} {tok!r}")
        return ("\n".join(rows) + "\n").encode() if rows else b"(no steps)\n"

    def chosen(self) -> bytes:
        r = self.run()
        return ("\n".join(repr(s["chosen"]) for s in r["steps"]) + "\n").encode()

    def entropy(self) -> bytes:
        """Per-step entropy in bits. A flat distribution means the model had no
        opinion; a spike means it did. Cheap, and it says where the interesting
        moments in a generation are without reading every candidate.
        """
        r = self.run()
        if r["error"]:
            return b"\n"
        rows = []
        for i, step in enumerate(r["steps"]):
            ps = [p for _, p in step["candidates"] if p > 0]
            total = sum(ps) or 1.0
            h = -sum((p / total) * math.log2(p / total) for p in ps)
            rows.append(f"{i} {h:.3f}")
        return ("\n".join(rows) + "\n").encode()

    def diff(self) -> bytes:
        if self.parent is None:
            return b"(root: nothing to diff against)\n"
        out = []
        for k in sorted(self.params):
            a, b = self.parent.params[k], self.params[k]
            if a != b:
                out.append(f"params  {k:12} {a} -> {b}")
        ra, rb = self.parent.run(), self.run()
        if ra["text"] != rb["text"]:
            out.append(f"out     text         {ra['text']!r}")
            out.append(f"                  -> {rb['text']!r}")
        # first step at which the two runs chose differently
        for i, (sa, sb) in enumerate(zip(ra["steps"], rb["steps"])):
            if sa["chosen"] != sb["chosen"]:
                out.append(
                    f"out     diverged at step {i}: "
                    f"{sa['chosen']!r} -> {sb['chosen']!r}"
                )
                break
        return ("\n".join(out) + "\n").encode() if out else b"(identical)\n"

    def describe(self) -> bytes:
        return (
            "name        sampler\n"
            "kind        generator\n"
            "depth       partial\n"
            "shows       text, chosen, topk, survivors, cull, entropy\n"
            "opaque      weights, attention, kv cache\n"
            f"model       {Path(self.params['model']).name}\n"
        ).encode()


# ---------------------------------------------------------------- namespace

PARAM_KEYS = {
    "model": "m",
    "prompt": "p",
    "temperature": "t",
    "top_p": "o",
    "min_p": "n",
    "top_k": "k",
    "seed": "s",
    "n_predict": "c",
    "n_ctx": "x",
    "show_k": "w",
}
OUT_KEYS = {"text": "t", "chosen": "c", "topk": "d", "entropy": "e",
            "cull": "u", "survivors": "v"}


def build(state: SampleState, name: str, key: str = "") -> Entry:
    root = Entry(name, is_dir=True, key=key)
    root.add(Entry("describe", reader=state.describe, key="d"))
    root.add(Entry("diff", reader=state.diff, key="f"))
    root.add(Entry("keys", reader=root.keys))
    # A sampler has no input text and no spans -- it declares a different
    # view, and the core renders it without knowing what a sampler is.
    root.add(Entry("view", reader=lambda: (
        "body   out/text\n"
        "table  out/cull  step top_k top_p min_p token\n"
        "header params/temperature\n"
        "header params/min_p\n"
        "header params/seed\n"
    ).encode()))

    params = root.add(Entry("params", is_dir=True, key="p"))
    params.add(Entry("keys", reader=params.keys))
    for pk in state.params:
        def rd(k=pk):
            return (state.params[k] + "\n").encode()

        def wr(data, k=pk):
            state.params[k] = data.decode().strip()

        params.add(Entry(pk, reader=rd, writer=wr, key=PARAM_KEYS.get(pk, ""),
                         config=True))

    out = root.add(Entry("out", is_dir=True, key="o"))
    out.add(Entry("keys", reader=out.keys))
    out.add(Entry("text", reader=state.text, key=OUT_KEYS["text"]))
    out.add(Entry("chosen", reader=state.chosen, key=OUT_KEYS["chosen"]))
    out.add(Entry("topk", reader=state.topk, key=OUT_KEYS["topk"]))
    out.add(Entry("entropy", reader=state.entropy, key=OUT_KEYS["entropy"]))
    out.add(Entry("cull", reader=state.cull, key=OUT_KEYS["cull"]))
    out.add(Entry("survivors", reader=state.survivors, key=OUT_KEYS["survivors"]))
    # hash is of the PARAMETERS only, so polling it never starts a generation
    out.add(Entry("hash", reader=lambda: (state.hash() + "\n").encode()))

    forks = root.add(Entry("fork", is_dir=True, key="k"))
    forks.add(Entry("keys", reader=forks.keys))

    def make_fork(child_name: str) -> Entry:
        return forks.add(
            build(SampleState(parent=state), child_name, key=child_name[:1])
        )

    forks.on_create = make_fork
    return root


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5641
    root = build(SampleState(), "/")
    print(f"sampler: namespace ready, {len(root.children)} entries", flush=True)
    Server(root).serve(port=port)


if __name__ == "__main__":
    main()
