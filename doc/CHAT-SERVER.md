# Vyb-native chat server on localhost:8888 — CHECKPOINT (untuned, latency-fixed)

> Status: PROVEN — HTTP layer working, untuned decode wired end-to-end,
> model load fixed (13min -> ~3.5s) via bulk HtoD, resident t/s measured.
> Pure-Vyb HTTP server + verified decode_driver subprocess.
> TUNED (LoRA) WIRING **DEFERRED** (next slice).
>
> Steering: user request — "setup a simple chat over http://localhost:8888 with
> a tuned model vs untuned", routed to the Vyb-native GPU stack (decided over
> vllm/llama.cpp). "Tuned x untuned" here = the real Qwen3-4B GGUF (untuned,
> `decode_driver`) vs the same GGUF + the trained `kvresp_l*.bin` LoRA adapters
> (tuned, `loradec_driver`) — both live in `VybAIConf`/`VybForge`.

## What it is
`native/host/chat_server.vyb` — a threaded pure-Vyb HTTP/1.1 server on
`localhost:8888`. Per request it:
1. extracts the prompt (GET `/chat?q=...` or POST `prompt=...`),
2. CPU-encodes it via the stdlib/vllm `Model` facade,
3. exec's the pristine `decode_driver` subprocess (`VYB_PROMPT_IDS` +
   `VYB_GEN` env) with the ids,
4. reads the generated token stream back and decodes the continuation to text,
5. replies.

The GPU decode stays a separate process so NO verified decode code is
refactored — the HTTP layer only orchestrates it.

## THE LATENCY FIX (this checkpoint)
Root cause of the historical ~15min "hello": `decode_driver`'s `download()`
did **ONE `cuMemcpyHtoD` per 8 bytes** in a host loop — ~287M calls to upload
the 2.3GB packed weights + ~40M for the 3.1GB embedding = the 13-minute load
(every request), NOT bandwidth.

Fix: bulk HtoD. `cuMemcpyHtoD_v2` is already declared with a `loc<CVoid>`
srcHost for the ~30 scalar/param-block 4/8-byte copies (Vyb forbids one symbol
with two signatures), so we added a **distinct** declaration
`cuMemcpyHtoDAsync_v2(dst, srcHost<CString>, n, 0)` — a CString srcHost makes
the FFI hand over the String's raw contiguous {ptr,len} buffer (same mechanism
`cuModuleLoadData` uses for a whole PTX image), so ONE driver call copies an
entire tensor. `download()` now does ~10 bulk calls instead of ~327M
per-8-byte ones.

Measured on GPU (as tested on an RTX 3090, godzilla; prove the LT fix, not a synthetic microbench):
- **Full model load + JIT + 1 gen: 4.26 s total wall** (load alone ~3.5 s;
  gen run 729 ms). Load went 13 min -> seconds, i.e. the ~10s class expected.
- Resident throughput (`make resident-bench`, GEN=3 x RUNS=3): each follow-up
  interaction (3 tokens) ~2.5 s -> **~1.19 tokens/sec** once the model is
  resident. All 3 runs token-exact against the gold, so correctness held.

## Run
```
# server
VYB_STDLIB=$VYB_STDLIB build/vyb native/host/chat_server.vyb --module-path native/llm
curl 'http://localhost:8888/chat?q=The%20capital%20of%20France%20is'
curl -d 'prompt=hello' http://localhost:8888/chat

# resident throughput bench (model load once, then RUNS timed interactions)
make resident-bench   # CHAT_GEN=3 RESIDENT_RUNS=3; verify_resident_bench.py gates
```
`resident_driver.vyb` = decode_driver + resident interaction loop + timing
(`VYB_RUNS`, `time_mono_millis`). decode_driver.vyb unchanged in structure.

## Verified
- `localhost:8888` serves: `/chat` w/o prompt -> 400 usage (instant); real
  GET round-trips return the correct decoded continuation.
- Real requests: "The capital of France is" -> `anieQN-cond`;
  "hello" -> `< muregigue>`; both match the verified gold.
- `resident-bench` gate: gold stream MATCH, t/s finite -> PASS.

## Honest cost + next
- Now that load is ~3.5s, per-request exec is tolerable but still re-loads;
  a resident-model server (load ONCE, serve many + tuned via `loradec_driver`,
  KV-cache for stepwise gen) is the real end-state and the next slice. That's
  also what the tuned-vs-untuned comparison needs.
- Generation is ~1.2 t/s (recompute-full-prefix, f64 dequant per layer).
- The LLM project tracks against rickenator/Vyb issues.

## Compiler/FFI notes while building
- **rickenator/Vyb#245** (parser hang on reusing a String-method result):
  root cause was the reserved keyword `use` being used as a variable name, not
  the `index_of`. Fixed in PR #246 (merged `be831fb`); workaround removed.
- `exec_run` must be called inside a `freedom { }` block; `hexval` collides
  with a stdlib symbol (renamed `hexch`).
- Vyb FFI forbids declaring one libcuda symbol with two srcHost signatures in
  one program; use a distinct symbol (`cuMemcpyHtoDAsync_v2`) for the bulk
  case. Legacy `cuMemcpyHtoD` returns CUDA_ERROR_INVALID_VALUE on this stack.
