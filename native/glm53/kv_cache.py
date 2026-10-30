"""GLM-5.3-Flash KV + DSA-indexer cache sizing (P6 prep) and the TP=2 plan.

Dims are pinned from the checkpoint config (native/glm53/layer0/config.json —
same file the checkpoint ships):

  layers              45   (34 linear_attention / KDA + 11 deepseek_sparse_attention,
                            schedule GGGAGGGAG...; DSA layers at 3,7,...,43)
  kv_lora_rank       512   (MLA compressed latent — this IS the cached KV in
                            absorbed form; qk_rope_head_dim 0, so no per-head
                            rope keys to store)
  index_head_dim     128   (DSA indexer key per token)
  indexer_types      45 x "full"  -> every layer keeps an indexer key pool
  kda                64 heads x 128 head_dim recurrent state + depthwise conv k=4

What is cached per token (MLA absorbed form):
  - DSA layers (11): one 512-elem latent c_KV. Per-head k[h], v[h] are NOT
    stored — decode computes them from the latent through kv_b_proj slices
    (W_UK / W_UV), which is the absorbed form gated by MLA_ABSORB_VERIFY
    (rel 1.4e-15). 512 x 2 B = 1 KiB/token/layer bf16, 512 B fp8.
  - ALL 45 layers: a 128-elem indexer key (bf16) = 256 B/token/layer.
  - KDA layers (34): NO per-token cache. The recurrent state is a fixed
    64x128x128 matrix per layer — O(1) per sequence, independent of context:
      64*128*128 = 1,048,576 elem = 2 MiB bf16 (4 MiB f32) per layer
      x 34 layers = 68 MiB bf16 (136 MiB f32) per sequence, plus the short
      conv state 3*24,576*3*2 B = 288 KiB bf16 total.
    Amortized "per token" it is therefore 68 B/token bf16 at 1 M context and
    falls as context grows — linear attention's flat-memory payoff. At 4 K it
    is 17 KiB/token amortized; the KV latent alone is 11 KiB/token/layer.

Totals (bf16 / fp8 latent and keys), exact:
  latent KV   = 11 layers x ctx x 512 x {2,1} B
  indexer keys = 45 layers x ctx x 128 x {2,1} B

TP=2 (two Sparks, layer-sharded — the default plan):
  rank 0 owns layers 0-22, rank 1 owns 23-44; every tensor belongs to exactly
  one rank (no weight duplication except embed_tokens/lm_head, see below).
    DSA layers:  rank 0 = {3,7,11,15,19} (5)   rank 1 = {23,...,43} (6)
    indexer pools: 23 layers / 22 layers
    KDA layers:  18 / 16
  KV follows layer ownership: latent 5.0 / 6.0 GiB, indexer 5.75 / 5.50 GiB
  at 1 M (bf16). KDA state 36 / 32 MiB. embed_tokens is needed by rank 0
  (input) and lm_head by rank 1 (output) — each replicated on one rank only;
  the final norm lives with lm_head. Per-token traffic between ranks is the
  hidden state (4096 x 2 B = 8 KiB/token) at the layer-22/23 boundary.
  Alternative (not default): head-sharding MLA splits the 64 heads' kv_b rows
  so latent KV splits per rank too, but the indexer key pool is per-layer and
  would then be duplicated — layer-sharding keeps every pool disjoint.

Usage: python3 kv_cache.py [ctx ...]   (default table: 4096 32768 262144 1048576)
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "layer0", "config.json")

MIB = 1 << 20
GIB = 1 << 30
DEFAULT_CONTEXTS = (4_096, 32_768, 262_144, 1_048_576)


def _cfg():
    with open(CFG_PATH) as f:
        c = json.load(f)
    # the checkpoint config nests the text model under "text_config"
    return c.get("text_config", c)


def dims():
    """Pinned cache-relevant dims from the checkpoint config."""
    c = _cfg()
    types = c["layer_types"]
    dsa = [i for i, t in enumerate(types) if t == "deepseek_sparse_attention"]
    kda = [i for i, t in enumerate(types) if t == "linear_attention"]
    return {
        "n_layers": len(types),
        "dsa_layers": dsa,
        "kda_layers": kda,
        "kv_lora_rank": c["kv_lora_rank"],
        "index_head_dim": c["index_head_dim"],
        "kda_heads": c["linear_attn_config"]["num_heads"],
        "kda_head_dim": c["linear_attn_config"]["head_dim"],
        "kda_conv_kernel": c["linear_attn_config"]["short_conv_kernel_size"],
        "kda_qkv_channels": 3 * c["linear_attn_config"]["num_heads"]
                               * c["linear_attn_config"]["head_dim"],
    }


def kv_latent_bytes(ctx, bpb=2):
    """Latent KV for the 11 DSA layers (MLA absorbed form)."""
    d = dims()
    return len(d["dsa_layers"]) * ctx * d["kv_lora_rank"] * bpb


def indexer_bytes(ctx, bpb=2):
    """Indexer key pool for all 45 layers (config: 45 x 'full')."""
    d = dims()
    return d["n_layers"] * ctx * d["index_head_dim"] * bpb


def kda_state_bytes(bpb=2):
    """KDA recurrent state: O(1) per sequence — 2 MiB/layer bf16, 34 layers."""
    d = dims()
    return len(d["kda_layers"]) * d["kda_heads"] * d["kda_head_dim"] ** 2 * bpb


def kda_conv_bytes(bpb=2):
    """Short-conv state (kernel-1 taps over q|k|v channels), 34 layers."""
    d = dims()
    per = d["kda_qkv_channels"] * (d["kda_conv_kernel"] - 1) * bpb
    return len(d["kda_layers"]) * per


def per_token(ctx):
    """Amortized per-token cost of every cache component at this context."""
    n_tok = ctx
    return {
        "kv_latent_bf16": kv_latent_bytes(ctx) / n_tok,
        "indexer_bf16": indexer_bytes(ctx) / n_tok,
        "kda_state_bf16": kda_state_bytes() / n_tok,
        "kda_conv_bf16": kda_conv_bytes() / n_tok,
    }


def table(contexts=DEFAULT_CONTEXTS):
    """Print the KV/indexer/KDA table and return the rows."""
    d = dims()
    print(f"GLM-5.3-Flash cache sizing  ({d['n_layers']} layers, "
          f"{len(d['dsa_layers'])} DSA @ {d['dsa_layers'][0]}..{d['dsa_layers'][-1]}, "
          f"latent {d['kv_lora_rank']}, indexer {d['index_head_dim']})")
    print(f"{'ctx':>9} | {'latent bf16':>11} {'latent fp8':>10} | "
          f"{'index bf16':>11} {'index fp8':>10} | {'KV+idx bf16':>12} {'KV+idx fp8':>11} | "
          f"{'per-token':>9}")
    rows = []
    for ctx in contexts:
        lat2, lat1 = kv_latent_bytes(ctx, 2), kv_latent_bytes(ctx, 1)
        idx2, idx1 = indexer_bytes(ctx, 2), indexer_bytes(ctx, 1)
        tot2, tot1 = lat2 + idx2, lat1 + idx1
        rows.append({"ctx": ctx, "latent_bf16": lat2, "latent_fp8": lat1,
                     "indexer_bf16": idx2, "indexer_fp8": idx1,
                     "total_bf16": tot2, "total_fp8": tot1})
        print(f"{ctx:>9} | {lat2/GIB:>10.2f}G {lat1/GIB:>9.2f}G | "
              f"{idx2/GIB:>10.2f}G {idx1/GIB:>9.2f}G | "
              f"{tot2/GIB:>11.2f}G {tot1/GIB:>10.2f}G | "
              f"{tot2/ctx:>8.0f}B")
    st = kda_state_bytes()
    cv = kda_conv_bytes()
    pt = per_token(DEFAULT_CONTEXTS[-1])
    print(f"KDA recurrent state: {st/MIB:.0f} MiB bf16 ({st*2/MIB:.0f} MiB f32) "
          f"+ conv {cv/MIB:.1f} MiB per sequence — O(1) in context length; "
          f"amortized {pt['kda_state_bf16']:.0f} B/token at 1 M "
          f"(= {kda_state_bytes()/DEFAULT_CONTEXTS[0]/1024:.1f} KiB/token at 4 K), "
          f"vs latent KV {pt['kv_latent_bf16']/1024:.0f} KiB/token "
          f"({d['kv_lora_rank']*2} B per DSA layer) + indexer keys "
          f"{pt['indexer_bf16']:.0f} B/token ({d['index_head_dim']*2} B per layer)")
    return rows


def tp2_plan(ctx=1_048_576):
    """TP=2 layer-sharded allocation: which cache tensors live on which rank."""
    d = dims()
    mid = d["n_layers"] // 2 + d["n_layers"] % 2        # 23 layers on rank 0
    ranks = {}
    for rank in (0, 1):
        lo, hi = (0, mid) if rank == 0 else (mid, d["n_layers"])
        dsa = [l for l in d["dsa_layers"] if lo <= l < hi]
        kda = [l for l in d["kda_layers"] if lo <= l < hi]
        lat = len(dsa) * ctx * d["kv_lora_rank"] * 2
        idx = (hi - lo) * ctx * d["index_head_dim"] * 2
        ranks[f"rank{rank}"] = {
            "layers": (lo, hi - 1), "n_layers": hi - lo,
            "dsa_layers": dsa, "n_dsa": len(dsa), "n_kda": len(kda),
            "kv_latent_bf16": lat, "indexer_bf16": idx,
            "kv_total_bf16": lat + idx,
            "kda_state_bf16": len(kda) * d["kda_heads"] * d["kda_head_dim"] ** 2 * 2,
            "kda_conv_bf16": kda_conv_bytes() * len(kda) // len(d["kda_layers"]),
        }
    print(f"\nTP=2 layer-sharded plan @ {ctx} tokens (bf16 cache):")
    for r, v in ranks.items():
        print(f"  {r}: layers {v['layers'][0]}-{v['layers'][1]} "
              f"({v['n_layers']} layers: {v['n_kda']} KDA + {v['n_dsa']} DSA)")
        print(f"      KV latent {v['kv_latent_bf16']/GIB:.2f} GiB "
              f"(dsa {v['dsa_layers']}) | indexer keys {v['indexer_bf16']/GIB:.2f} GiB "
              f"| KDA state {v['kda_state_bf16']/MIB:.0f} MiB + conv "
              f"{v['kda_conv_bf16']/MIB:.1f} MiB")
    print(f"  replicated: embed_tokens (rank 0), lm_head + final norm (rank 1); "
          f"boundary traffic 8 KiB/token (4096 x bf16 hidden)")
    print(f"  KDA recurrent state is per-sequence O(1): "
          f"{ranks['rank0']['kda_state_bf16']/MIB:.0f} + "
          f"{ranks['rank1']['kda_state_bf16']/MIB:.0f} MiB — "
          f"no per-token cache on the {len(d['kda_layers'])} linear layers")
    return ranks


if __name__ == "__main__":
    import sys
    ctxs = tuple(int(a) for a in sys.argv[1:]) or DEFAULT_CONTEXTS
    table(ctxs)
    tp2_plan(ctxs[-1])
