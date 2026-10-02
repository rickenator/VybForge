# Substrate scope — what VybForge's Vyb-native layer is *for*

Status: scope statement recorded 2026-10-02 (Rick, verbatim direction: *"don't assume these are
the only models ever to use VybForge, it is supposed to be general purpose for inference to other
applications, anything PyTorch can do. I'm sure it will grow to OpenCV, PaddleOCR, llama.cpp ports
support (make a note of that - there may be others worth consuming)"*).

## The scope, stated plainly

The Vyb-native substrate under `native/` is a **general-purpose inference and compute substrate**,
not a Qwen3 runtime. The bar to design against is **"anything PyTorch can do"**. The Qwen3-4B
configurator is the **first instance** — the proving slice that forced the substrate into existence —
and it is not the definition of the target.

Two failure modes this note exists to prevent:

1. **Scope shrink.** `native/ROADMAP.md` was titled "Vyb-native inference (Qwen3-4B configurator)" —
   the first instance had become the stated goal, so every "is this worth building?" question was
   being answered against one model's needs. Titles like that, and estimates phrased as "% of the
   model's path", quietly cap the project.
2. **Shape-specialised substrate.** Kernels and wrappers sized to Qwen3's shapes (its GEMM dims, its
   GQA ratio, its RoPE base) are *specialisations*, not the substrate. If they are the only thing
   built, the next architecture starts from zero and "anything PyTorch can do" stays untrue.

## Two ways a library enters

Both are legitimate; the choice should be deliberate and recorded, not accidental.

- **PORT** — reimplement in Vyb the *semantics we depend on*. Right when the behaviour is what we
  own and must be able to verify, when the surface we need is small relative to the library, or when
  the dependency is the thing being removed. Existing examples: the Qwen3 BPE tokenizer, the GGUF
  reader + `q4_0` dequant, the JSON parser/emitter, RoPE/RMSNorm/attention kernels, AdamW, the
  analytic backward.
- **CONSUME** — call or link it. Right when the library is a hardware/driver interface, when the
  surface is enormous and mostly irrelevant to us, or when reimplementation would buy no
  understanding. Existing example: the CUDA runtime and PTX (we own the kernels, not the driver) —
  and `README.md` already commits to this shape: "host code talks to buffers/kernels, not to a
  specific vendor as a product — a future sponsor can change the backend without renaming the shop."

The port/consume call is per-capability, not per-library: llama.cpp is consumed today as an *oracle*
(`render_chat_ref.py` uses `llama_cpp`'s Jinja chat formatter, and its GGUF chat template is pinned as
a fixture), while its GGUF/quantisation semantics are already ported into `native/gguf/`. Both can be
true of the same upstream.

## Candidates worth considering (unranked — not commitments)

Named by Rick as direction: **OpenCV**, **PaddleOCR**, **llama.cpp**.

Everything else here is a suggestion to weigh, not a commitment. Marked with where it already stands.

| Area | Candidates | Standing today |
|---|---|---|
| Vision / OCR | OpenCV, PaddleOCR (PP-OCR/PP-Structure), Tesseract, scikit-image | Rick's direction; nothing started |
| Classical ML / data | scikit-learn, scipy, pandas, arrow/parquet | nothing started |
| Images / audio / video | Pillow, imageio, PyAV/ffmpeg, librosa, soundfile, torchvision, torchaudio | `stdlib/png` exists; nothing else |
| Numerics / GPU | CuPy, cuBLAS, cuDNN, CUTLASS, CUB/Thrust (scan/sort/reduce), NCCL, OpenBLAS, cuFFT | own GEMM/RMSNorm/attn kernels; CUDA consumed; no vendor BLAS used |
| Autograd / training | torch `nn`/`optim`/`autograd`, functional `grad`, DeepSpeed/FSDP, TRL | hand-written per-op backward + AdamW; P2 in flight |
| Tokenizers / text | HF `tokenizers`, sentencepiece, tiktoken, `regex` (Unicode classes) | Qwen2 pre-tokenizer ported *approximately* — non-ASCII still approximate because Vyb has no Unicode-class regex; `stdlib/regex` exists |
| LLM ecosystem | llama.cpp/ggml, whisper.cpp, vLLM (protocol), onnxruntime, TensorRT | llama.cpp = oracle today; `stdlib/vllm` exists |
| Formats / interchange | safetensors, `.npy`, HDF5, ONNX, GGUF | GGUF done; **safetensors and `.npy` loaders do not exist yet** — named in `doc/P2-TRAINER-PLAN.md`, no `.vyb` |

Two entries on that list are sharper than "candidate" because existing work already assumes them:
**safetensors** (P2 commits to native safetensors/`.bin` loaders, and the configurator adapter ships
as safetensors) and **Unicode-class regex** (the tokenizer's non-ASCII gap is a language/stdlib
limitation, not a tokenizer bug).

## Design consequences

- **G1 (`tensor`) must be general, not shaped to a model.** The audit (`native/VYB-NATIVE-INFERENCE-AUDIT.md`)
  called G1 "the actual replace-torch core". That reading is now the operative one: dtype, shape,
  strides/broadcast, and a real memory model. Per-model kernels stay as specialisations on top of it,
  and the per-model shape constants living in today's drivers become parameters.
- **Dtypes are a substrate feature, not an optimisation.** `native/` has **zero** occurrences of
  fp16/bf16 today; kernels run f32/f64. Anything PyTorch can do includes half-precision, so this is a
  substrate gap rather than a Qwen3 detail.
- **Every new family enters through the existing two-track discipline.** Upstream stays the *oracle*
  (independent implementation, allowed to be wrong, kept until the model sweep — see the oracle policy
  in `doc/PYTHON-CLEANUP.md`), the Vyb side is the production path, and the gate compares
  byte-for-byte or within a stated tolerance. Retiring an oracle means commenting, in its
  replacement, what it asserted.
- **Estimates must not be phrased as "% of model X's path".** The honest split is: substrate
  capabilities that are general (tensor, dtypes, autograd, loaders, BLAS/reduce/scan class ops) versus
  work that is per-model specialisation.

## See also

- `native/VYB-NATIVE-INFERENCE-AUDIT.md` — the capability audit and its G1–G5 gap ledger.
- `native/ROADMAP.md` — the phase roadmap (now scoped as the first instance, not the whole target).
- `doc/PYTHON-CLEANUP.md` — the oracle policy and what Python remains, and why.
- `doc/P2-TRAINER-PLAN.md` — the torch-side (training) port in flight.
