# Third-party attribution ledger

VybForge is Apache-2.0 (`LICENSE`, Copyright 2026 Aniviza LLC). This file is the ledger for code
**ported into** VybForge from another project — i.e. *derived*, not merely *used*.

Policy and the reasoning are in `doc/SUBSTRATE-SCOPE.md` ("Attribution & licensing"). In short:
permissive licences (Apache-2.0, MIT, BSD, ISC) may be ported freely and need no API fidelity; a
strong derivation carries one boilerplate line at the top of the derived file, pointing here;
copyleft (GPL/LGPL/AGPL) and proprietary vendor libraries are consume-only, never ported into the
runtime.

## Derived code

| File(s) | Upstream | Licence | What was taken | Revision |
|---|---|---|---|---|
| `native/kernels/q4k.vyb` | [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | `ggml-quants.c`: `dequantize_row_q4_K` / `get_scale_min_k4` block decode, ported faithfully | not recorded at port time (pre-2026-10) — pin it when this file is next touched |
| `native/kernels/q6k.vyb` | [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | `ggml-quants.c`: `dequantize_row_q6_K` block decode, ported faithfully | not recorded at port time (pre-2026-10) — pin it when this file is next touched |

## Under review (behavioural match — derivation not yet asserted)

These follow another project's *semantics* deliberately, with our own implementation structure. The
question is whether the match is strong enough to constitute derivation; decide when each file is
next touched and record the answer here either way.

| File | Behaviour matched | Upstream | Licence | Status |
|---|---|---|---|---|
| `native/tokenizer/tokenizer.vyb` | Byte-level BPE, Qwen2/Qwen3 pre-tokenizer split, byte↔glyph remap | GPT-2 / Qwen2 in HF `tokenizers` | MIT / Apache-2.0 | undecided — behavioural match, own implementation |
| `native/json/json_emit.vyb` | `json.dumps`-compatible string emission (`ensure_ascii` handling, escaped surrogate pairs ⟶ `\uF6D5`) | CPython `Lib/json/encoder.py` | PSF-2.0 | undecided — deliberate compatibility, own implementation |

## Used as a tool (not derived from — no obligation)

- **llama.cpp / `llama-cpp-python`** — oracle and tooling: the GGUFs we read are produced by
  llama.cpp, its chat template is pinned as a fixture, and `llama_cpp`'s Jinja formatter is the
  reference for `native/train/render_chat.vyb`. Consumption, not derivation.
- **Python + numpy + transformers** — reference-verification harness (the oracle track). Also the
  reason `python3` appears in Makefile gates.
- CUDA toolkit / driver — linked runtime, consumed per `README.md`'s backend stance.
