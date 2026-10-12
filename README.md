# tnwr

A small text workstation prototype built around a deliberately narrow boundary: the renderer knows how to display panes, tokens and differences; separate 9P nodes decide what those panes contain. Forking a node's state lets you change one side of a comparison without changing the other.

This is an early, windowed GPU prototype. It is not a complete operating system, and the DRM/KMS console backend is not implemented. The included nodes are experiments: the built-in tokenizer makes boundary demonstrations but uses hash-derived IDs rather than a learned vocabulary; use a real GGUF or Hugging Face tokenizer for real token IDs. The sampler visualises a small local model through llama.cpp and should be treated as an exploratory instrument, not a reference implementation of every sampler or KV-cache optimization.

## Run

You need Rust/Cargo, Python 3, and a working window-system GPU backend supported by winit/wgpu. Start the no-extra-dependency token node in one terminal:

```sh
python3 nodes/tok_node.py
```

Then start the renderer in another:

```sh
cargo run -- 127.0.0.1:5640
```

The token node defaults to the simple boundary demo. Optional real tokenizers need the corresponding backend: `llama-cpp-python` and a GGUF path, or Hugging Face `tokenizers` and a tokenizer ID. The sampler node is separate and needs `llama-cpp-python` plus a local GGUF model:

```sh
python3 nodes/sampler_node.py 5641
```

The graph node is another independent 9P endpoint. See the source and node `.names` files for the exposed namespace. No model weights are included.

## Status and scope

The renderer, 9P client, token node, sampler node and graph node are a work in progress. The project currently focuses on making token boundaries and side-by-side changes visible. It does not yet provide a bare-metal console, a production desktop environment, or a comprehensive model-inspection suite.

## License

No license has been provided. Public availability of this repository does not grant permission to reuse or redistribute its contents.
