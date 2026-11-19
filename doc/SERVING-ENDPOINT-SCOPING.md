# Serving Endpoint Scoping — Batching, API, Hardware Targets

Continues doc/KV-CACHE-SCOPING.md. Grounded in the verified decode step's
data flow (glm_decode_chain.vyb) and the Vyb stdlib's actual capabilities
(http module: http_listen/http_accept/http_serve with per-connection worker
threads — the endpoint can be pure Vyb, no proxy layer needed).

## The batching constraint: the deq working set scales with the batch

The decode step per sequence: 1 token x ROUTE_TOPK(8) routed experts. With B
concurrent sequences, the routed-expert UNION per step is up to B x 8
distinct experts, and each deq'd expert costs ~201MB (3 tensors x 67MB f64)
in the working set:

- B=1: ~8 experts  -> ~1.6GB deq working set
- B=4: ~32 experts -> ~6.4GB  (the 3090's practical ceiling with caches)
- B=8: ~64 experts -> ~12.9GB (3090: no; Sparks 128GB: fine)

TWO mitigations change the ceiling:
1. DEQ CACHE (LRU): adjacent tokens route similarly — the temporal locality
   across decode steps is high, so a bounded LRU of deq'd experts (say 64
   slots) absorbs most repeat routes. The deq_cache.vyb module is the
   starting point (validated bit-exact); it needs an eviction policy + the
   device-side residency (cache the deq'd DEVICE tensors, not the host bytes).
2. f32 DEQ: halve the per-expert cost (33MB) — doubles the batch ceiling and
   matches the bf16-cache recommendation in the KV scoping. The compute-dtype
   move and the deq dtype move are the same project.

RECOMMENDED v1: B=4 static slots on the 3090 with the deq LRU; B=8-16 on the
Sparks (128GB unified: 16 seqs x (145MB KDA bf16 state + 0.55GB DSA @32K) ~
11GB caches + the working set — comfortable).

## Batched decode shape

The KDA recurrence is sequential within a sequence but INDEPENDENT across
sequences: the B sequences' step-t computations batch into the gemms as
[B, 1, H] x weights. The complication: each sequence routes to DIFFERENT
experts, so the expert FFN is a grouped/batched GEMM over the union set —
per expert, gather its contributing sequences' rows. This is the same
pattern the prefill already runs (the token->expert scatter/gather with
rwt weights); the decode version has R=B instead of R=nt.

The DSA attention: per sequence, over its OWN cache — no cross-sequence
interaction, so the batched attention is B independent [1, cache_len]
operations; the indexer's top-k per sequence over its own gsc pool.

## API shape

OpenAI-compatible POST /v1/chat/completions (streaming via SSE) — the
ecosystem tooling works unchanged and it forces the right internal seams
(tokenize -> the engine step loop -> detokenize). Non-goals for v1:
the embeddings endpoints, function-calling glue, multi-model routing.

The engine/API split (important for the dogfood story):
- The ENGINE (pure Vyb): the weights, the caches, the step loop — no I/O.
  Takes (tokens, seq-slot) in, yields (token logits/probs) out. This is the
  part that must stay deterministic and gate-verifiable.
- The SERVER (pure Vyb, stdlib http): the request parse, the batcher (the
  slot allocation, the queue), the SSE streaming, the OpenAI JSON shape.
- The batcher is the only new concurrency: the worker-thread-per-connection
  model the stdlib provides maps to per-request intake; the engine loop runs
  on one thread and drains the batch each step.

## Hardware targets

| | 3090 (dev) | Spark (serve) |
|---|---|---|
| Slots (v1) | 4 | 8-16 |
| Context | 32K bf16 | 32-64K bf16 |
| Weights | packed 3.6GB + deq LRU | packed + deq LRU (bigger LRU) |
| Cache/seq @32K bf16 | ~0.7GB | ~0.7GB |
| The deq working set ceiling | ~6GB | ~25GB+ |

## Phasing (each phase gate-verifiable before the next)

1. bf16/f32 dtype migration + the noise-envelope re-measure (the gate
   infra exists; the verifier's per-layer metric gives the new band).
2. The engine/API split: refactor the chain's step loop into the engine form
   (the deterministic, I/O-free core) with the existing gate as the
   regression test — DECODE_CHAIN_MATCH must stay OK through the refactor.
3. The batched decode: B=2 then B=4 on the 3090; the deq LRU; measure the
   tokens/sec vs B (the MoE routing diversity will make the throughput
   sub-linear — measure where it bends).
4. The endpoint: the http server + the OpenAI shape + SSE on top of the
   batched engine; the load test with the concurrent curl streams.
5. The Spark port: the same engine, the bigger slots/context, FP8/FP4 as the
   later quant step (per the earlier quant decision).

## Risks
- The batched expert gather is the one genuinely new kernel path (the prefill
  proves the pattern at R=nt but not at B sequences x 1 token with the
  distinct caches). Prototype it before committing to the API shape.
- The sub-linear MoE batch scaling: if 4 sequences cost ~3.5x one sequence,
  the batching story weakens; measure early (phase 3, before the endpoint).
- The http module is young: the SSE streaming + the keep-alive behavior
  under load needs a spike before the API promises rest on it.
