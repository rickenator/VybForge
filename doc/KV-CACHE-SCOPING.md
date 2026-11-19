# KV-Cache Layout Scoping — GLM-5.3 Serving Phase

Grounded in the verified chain's actual allocations (glm_decode_chain.vyb,
DECODE_CHAIN_MATCH OK). All element counts are from the code, not guesses.

## Architecture: 45 layers = 34 KDA + 11 DSA + the MTP head

GLM-5.3 is a HYBRID attention model, and the cache splits by type:

### KDA layers (34 slots) — FIXED-SIZE state, does NOT grow with context
- Recurrence state per layer: HEADS(64) x KD(128) x KD(128) = 1,048,576 elems
- Conv rings per layer: 3 x 64 x 128 per q/k/v (ring buffers, already correct)
- Per layer f64: ~8.4MB state + ~0.4MB rings; 34 layers: ~290MB f64 total
- CONTEXT-INDEPENDENT: a 1K-token chat and a 128K-token doc use the same
  state. This is the linear-attention dividend — the serving cache problem is
  ONLY the DSA layers.

### DSA layers (11 slots) — the growing token cache
Per token per layer, three rows:
- ckv (the MLA latent): KV_LORA = 512 elems
- indexer k: IDIM = 128 elems
- gsc (the indexer gate scores): IDIM = 128 elems
Total 768 elems/token/layer. 11 layers + the MTP head's own cache:
- f64: 66KB/token  -> 2.1GB @ 32K context
- f32: 33KB/token  -> 1.1GB @ 32K
- bf16: 17KB/token -> 0.5GB @ 32K

### MTP head (layer 45): its own DSA cache, same 768/token shape.

## The weights side (the real serving constraint)
The deq'd expert weights are f64: ~57GB if resident — impossible. The VERIFIED
design already solves this: the weights stay PACKED (~3.6GB) and the deq runs
per token-batch on demand (the prefill's proven streaming model). Serving
keeps this; the hot-expset working set per decode step is small.

## Layout decisions (recommended)

1. SPLIT the two cache classes at the allocation level: the KDA state is a
   fixed per-layer device buffer allocated once; the DSA caches are
   per-layer, token-major, append-only device buffers: [layer][token][dim].
   Token-major keeps the indexer's per-layer top-k reads coalesced (the
   indexer scores EVERY cached token EVERY decode step — it is the hot path).

2. PREALLOCATE the max context; skip paging for v1. 32K tokens:
   bf16 0.5GB / f32 1.1GB / f64 2.1GB across 11+1 layers — trivially affordable
   on both targets. vLLM-style paging only matters at 128K+ contexts or with
   concurrent sequences; add it when multi-tenant serving is real.

3. DTYPE: the caches should move to bf16/f32 before the serving endpoint.
   f64 caches are 2-4x the necessary memory AND the f64 compute is 1/64 rate
   on GA102. This is the single biggest serving perf lever and it changes the
   noise-band analysis (the gate's early-layer envelope must be re-measured
   for the serving dtype — the infrastructure exists: VYB_STR_DUMP, the
   verifier's per-layer metric).

4. The indexer's per-channel softmax over the cache (the pool + top-k in
   dsa_attn_site) reads gsc + ape per token: keep gsc contiguous per layer
   and the APE (absolute position embedding) table separate — APE is
   position-indexed and shared, the cache is append-indexed.

5. The KDA conv rings: keep as rings (the current implementation is already
   correct); the state updates are read-modify-write on the fixed buffer —
   no append path needed.

## Memory budget summary (per instance)

| Component | 32K ctx bf16 | 32K ctx f64 |
|---|---|---|
| KDA state (34 layers) | ~145MB | ~290MB |
| DSA caches (11+1 layers) | ~0.55GB | ~2.2GB |
| Packed weights | ~3.6GB | ~3.6GB |
| Deq working set (per batch) | ~1-2GB | ~4-7GB |
| Total | ~6GB | ~13GB |

Fits the 3090 (24GB) comfortably at bf16; the Sparks (128GB unified) has room
for concurrent sequences or longer contexts.

## Open questions for the endpoint design
- Concurrent sequences: the KDA state is per-sequence 290MB — batch 4
  sequences = 1.2GB state; the DSA caches per-sequence too. Batching strategy
  (continuous batching vs static slots) is the next scoping item.
- The deq'd expert working set per decode step: measure the actual peak on
  the 3090 under serving-shaped load before fixing the batch size.
- bf16 vs f32 for the caches: re-run the gate at the serving dtype to set the
  noise envelope before trusting the endpoint's outputs.
