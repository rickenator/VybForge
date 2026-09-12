# Vyb-native chat server on localhost:8888 — CHECKPOINT 1 (untuned only)

> Status: PROVEN SLICE — HTTP layer working, untuned decode wired end-to-end
> (2026-09-11). Pure-Vyb HTTP server + verified decode_driver subprocess.
> TUNED (LoRA) WIRING **DEFERRED** (next slice).
>
> Steering: user request — "setup a simple chat over http://localhost:8888 with
> a tuned model vs untuned", routed to the Vyb-native GPU stack (decided over
> vllm/llama.cpp). "Tuned × untuned" here = the real Qwen3-4B GGUF (untuned,
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

The GPU decode stays a separate process (loads the 2.5GB GGUF, streams dequant
weights) so NO verified decode code is refactored — the HTTP layer only
orchestrates it.

## Run
```
VYB_STDLIB=$VYB_STDLIB build/vyb native/host/chat_server.vyb --module-path native/llm
curl 'http://localhost:8888/chat?q=The%20capital%20of%20France%20is'
curl -d 'prompt=hello' http://localhost:8888/chat
```

## Verified
- `localhost:8888` serves: `/chat` with no prompt -> 400 usage (instant),
  root -> 400 (instant). HTTP thread-per-connection + response formatting work.

## Honest cost + next
- Each request fully re-runs decode (model load + gen) ~minutes, single-client,
  GPU-bound. NOT a usable realtime chat — a correctness proof at this stage.
- NEXT (tuned): a resident-model server is the right end-state — load the GGUF +
  LoRA into the server process once (keep device weights cached), serve both
  `model=untuned` and `model=tuned` from memory, and reuse the KV-cache driver
  for stepwise gen. That removes the per-request model load and makes it
  iterative. Also wire `loradec_driver` (tuned) alongside.
- The LLM project tracks against rickenator/Vyb issues.

## Compiler bug filed while building
- **rickenator/Vyb#245** — parser HANGS on `x<Int> = <var-of-index_of>` (any
  reuse of a String-method Int result). Workaround: inline the `index_of` at
  every use site (never bind it to an Int variable). Filed with minimal repro;
  NOTE in chat_server.vyb header.
- Also: `exec_run` must be called inside a `freedom { }` block (else semantic
  error); `hexval` collides with a stdlib symbol (renamed `hexch`).
