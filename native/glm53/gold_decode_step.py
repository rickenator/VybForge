"""Gold for GLM-5.3 step decode (S=1) — layer 0 (KDA + dense MLP + both hc sites).

The prefill oracle (gold_layer0.py) runs layer 0 over S tokens in one shot. Generation needs
the SAME math carried across steps, and two pieces of state make that non-obvious:

  1. the KDA recurrence  state [64, 128, 128] per layer  (kda_recurrent already takes/returns it)
  2. the k=4 depthwise conv over the token axis, which needs the last 3 PRE-CONV vectors per
     channel for q, k and v -- a ring buffer per tensor, not a scalar carry

Both are threaded here so the Vyb decode driver has something to gate against. The self-check is
built in: running the conv stepwise over the prompt must reproduce gdn_ref.short_conv_silu over
the whole sequence (a mismatch means the ring buffer's token order is wrong, which is the easiest
thing to get wrong and the hardest to see in a digest).

Outputs: GOLD_DECODE_OK plus native/out/gold_dec_*.npy and native/out/gold_decode.json.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/gold_decode_step.py
"""

import json
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
import gdn_ref  # noqa: E402
import hc_ref  # noqa: E402
import layer0_ref  # noqa: E402

CKPT = "/mnt/apex/rick/vybforge/glm53-checkpoint"
OUT = "native/out"
H, HEADS, HDIM, EPS = 4096, 64, 128, 1e-5
CONV_K = 4
PROMPT = ("The capital of France is Paris, and the largest city in Germany is Berlin, "
          "which was divided for decades.")
NSTEPS = 2


def read(name):
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{CKPT}/{idx[name]}", framework="pt", device="cpu") as f:
        t = f.get_tensor(name)
    return t.float().numpy().astype(np.float64)


def weights(layer=0):
    pre = f"model.language_model.layers.{layer}."
    T = lambda n: read(pre + n).T  # noqa: E731  checkpoint is (out, in)
    w = {"q": T("self_attn.q_proj.weight"), "k": T("self_attn.k_proj.weight"),
         "v": T("self_attn.v_proj.weight"), "o": T("self_attn.o_proj.weight"),
         "conv_q": read(pre + "self_attn.q_conv1d.weight")[:, 0, :],
         "conv_k": read(pre + "self_attn.k_conv1d.weight")[:, 0, :],
         "conv_v": read(pre + "self_attn.v_conv1d.weight")[:, 0, :],
         "b": T("self_attn.b_proj.weight"), "f_a": T("self_attn.f_a_proj.weight"),
         "f_b": T("self_attn.f_b_proj.weight"), "g_a": T("self_attn.g_a_proj.weight"),
         "g_b": T("self_attn.g_b_proj.weight"),
         "dt_bias": read(pre + "self_attn.dt_bias"), "A_log": read(pre + "self_attn.A_log"),
         "o_norm": read(pre + "self_attn.o_norm.weight"),
         "mlp_gate": T("mlp.gate_proj.weight"), "mlp_up": T("mlp.up_proj.weight"),
         "mlp_down": T("mlp.down_proj.weight"),
         "in_ln": read(pre + "input_layernorm.weight"),
         "post_ln": read(pre + "post_attention_layernorm.weight")}
    for site, key in (("attn", "hc_attn"), ("ffn", "hc_ffn")):
        w[f"{site}_fn"] = read(pre + f"{key}_fn")
        w[f"{site}_base"] = read(pre + f"{key}_base")
        w[f"{site}_scale"] = read(pre + f"{key}_scale")
    return w


def conv_step(x, ring, weight):
    """One token through the causal k=4 depthwise conv + silu.

    ring is [CONV_K-1, C] holding the previous tokens' PRE-conv vectors, oldest first.
    `weight[:, -1]` applies to the current token (pinned by verify_gdn against the module).
    """
    if ring is None:
        ring = np.zeros((CONV_K - 1, x.shape[0]), dtype=np.float64)
    win = np.concatenate([ring, x[None, :]], axis=0)          # [k, C]
    pre = np.zeros_like(x)
    for j in range(CONV_K):
        pre = pre + win[j] * weight[:, j]
    out = pre * (1.0 / (1.0 + np.exp(-pre)))
    return out, win[1:]


def kda_step(h, w, st):
    """h [H] -> (o_proj output [H], new state dict). Mirrors layer0_ref.kda at s=1."""
    x = h[None, :]
    qpre, kpre, vpre = x @ w["q"], x @ w["k"], x @ w["v"]
    qc, st["ring_q"] = conv_step(qpre[0], st["ring_q"], w["conv_q"])
    kc, st["ring_k"] = conv_step(kpre[0], st["ring_k"], w["conv_k"])
    vc, st["ring_v"] = conv_step(vpre[0], st["ring_v"], w["conv_v"])
    beta = (1.0 / (1.0 + np.exp(-(x @ w["b"]))))                   # [1, HEADS]
    f = (x @ w["f_a"]) @ w["f_b"]
    g = gdn_ref.kda_log_decay(f, w["dt_bias"], w["A_log"])          # [1, HEADS, KDIM]
    g = g.reshape(1, 1, HEADS, HDIM)
    z = ((x @ w["g_a"]) @ w["g_b"]).reshape(1, HEADS, HDIM)
    o, st["kda"] = gdn_ref.kda_recurrent(qc.reshape(1, 1, HEADS, HDIM),
                                         kc.reshape(1, 1, HEADS, HDIM),
                                         vc.reshape(1, 1, HEADS, HDIM),
                                         g, beta.reshape(1, 1, HEADS), state=st["kda"])
    o = gdn_ref.gated_rmsnorm(o[0], z[0], w["o_norm"], eps=EPS, activation="sigmoid")
    return o.reshape(HEADS * HDIM) @ w["o"], st


def layer_step(streams, w, st):
    """streams [4, H] -> (streams2 [4, H], stages dict). One token, whole layer 0."""
    post, comb, collapsed, _ = hc_ref.mix(streams[None], w["attn_fn"], w["attn_base"],
                                          w["attn_scale"], EPS)
    h = layer0_ref.rms_weight(collapsed[0], w["in_ln"])
    y_attn, st = kda_step(h, w, st)
    s1 = hc_ref.apply_(streams[None], y_attn[None], post, comb)[0]
    post2, comb2, collapsed2, _ = hc_ref.mix(s1[None], w["ffn_fn"], w["ffn_base"],
                                             w["ffn_scale"], EPS)
    h2 = layer0_ref.rms_weight(collapsed2[0], w["post_ln"])
    g = h2 @ w["mlp_gate"]
    u = h2 @ w["mlp_up"]
    y_mlp = ((g * (1.0 / (1.0 + np.exp(-g)))) * u) @ w["mlp_down"]
    s2 = hc_ref.apply_(s1[None], y_mlp[None], post2, comb2)[0]
    return s2, {"h": h, "attn": y_attn, "h2": h2, "mlp": y_mlp, "streams1": s1,
                "streams2": s2, "collapsed": collapsed[0], "post_attn": post[0]}


def digest(a):
    f = np.ascontiguousarray(a, dtype=np.float64).ravel()
    return {"n": int(f.size), "sum": float(f.sum()), "abs": float(np.abs(f).sum()),
            "sq": float((f * f).sum()), "max": float(np.abs(f).max())}


def embed_rows(ids):
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    name = "model.language_model.embed_tokens.weight"
    with safe_open(f"{CKPT}/{idx[name]}", framework="pt", device="cpu") as f:
        w = f.get_tensor(name).float().numpy()
    return np.stack([w[i] for i in ids]).astype(np.float64)


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{CKPT}/tokenizer.json")
    ids = tok.encode(PROMPT).ids
    s = len(ids)
    x = embed_rows(ids)
    w = weights(0)

    # ---- prefill (batched), but keep the state the batched reference discards ----------------
    streams = np.repeat(x[:, None, :], 4, axis=1)            # [s, 4, H]
    st = {"ring_q": None, "ring_k": None, "ring_v": None, "kda": None}
    outs = []
    for i in range(s):
        s2, stg = layer_step(streams[i], w, st)              # one token through the layer
        streams[i] = s2
        outs.append(stg)

    # ---- the conv ring buffer must reproduce the batched conv exactly ------------------------
    pref_streams = streams.copy()          # the decode loop rebinds `streams`; keep the prefill's
    pref_state = {k: (v.copy() if v is not None else None) for k, v in st.items()}
    qc_batch = gdn_ref.short_conv_silu(
        np.stack([outs[i]["h"] @ w["q"] for i in range(s)])[None], w["conv_q"])[0]
    ring = np.zeros((CONV_K - 1, HEADS * HDIM))
    qc_walk = []
    for i in range(s):
        o, ring = conv_step(outs[i]["h"] @ w["q"], ring, w["conv_q"])
        qc_walk.append(o)
    qc_walk = np.stack(qc_walk)
    conv_err = float(np.abs(qc_walk - qc_batch).max())

    # ---- decode: NSTEPS single-token steps from the state the prefill left -------------------
    # Teacher-forced: each step feeds back the prompt's own token id (recorded in the meta),
    # so the gate is deterministic and does not depend on a sampler.
    steps = []
    for t in range(NSTEPS):
        tid = ids[s - 1 - t]
        xt = embed_rows([tid])[0]
        streams = np.repeat(xt[None, :], 4, axis=0)
        streams, stg = layer_step(streams, w, st)
        steps.append({"token": tid, **{k: v for k, v in stg.items() if k != "streams2"},
                      "streams2": stg["streams2"]})

    np.save(f"{OUT}/gold_dec_conv_q_batch.npy", qc_batch)
    np.save(f"{OUT}/gold_dec_conv_q_step.npy", qc_walk)
    np.save(f"{OUT}/gold_dec_prefill_streams2.npy", pref_streams)
    np.save(f"{OUT}/gold_dec_state_kda.npy", st["kda"])
    # What a decode step STARTS from: the carried recurrence, the three conv rings, and the
    # last prefill token's h -- so a Vyb driver can run ONE step in isolation instead of
    # re-deriving the 22-token prefill to get to it. Snapshot, because the decode loop has
    # already advanced `st` by the time these are written.
    np.save(f"{OUT}/gold_dec_ring_q.npy", pref_state["ring_q"])
    np.save(f"{OUT}/gold_dec_ring_k.npy", pref_state["ring_k"])
    np.save(f"{OUT}/gold_dec_ring_v.npy", pref_state["ring_v"])
    np.save(f"{OUT}/gold_dec_prefill_h_last.npy", outs[-1]["h"])
    np.save(f"{OUT}/gold_dec_state_kda_start.npy", pref_state["kda"])
    # Raw little-endian f64 alongside the .npy, because a Vyb driver cannot read .npy: these are
    # its actual inputs (streams_prev = the last prefill token's streams2 = what decode step 0
    # consumes), so the gate is "read these four files, run one step, match gold_dec_step0_*".
    pref_state["kda"].astype("<f8").tofile(f"{OUT}/gold_dec_state_kda_start.bin")
    pref_state["ring_q"].astype("<f8").tofile(f"{OUT}/gold_dec_ring_q.bin")
    pref_state["ring_k"].astype("<f8").tofile(f"{OUT}/gold_dec_ring_k.bin")
    pref_state["ring_v"].astype("<f8").tofile(f"{OUT}/gold_dec_ring_v.bin")
    pref_streams[-1].astype("<f8").tofile(f"{OUT}/gold_dec_streams_prev.bin")

    # The stepwise walk must reproduce the already-gated batched layer-0 gold
    # (gold_l0_streams2.npy from gold_layer0.py). This is the check that says the per-token
    # path -- ring buffers, carried state, hc per token -- is the same math, not just finite.
    gate_err = float("nan")
    try:
        ref = np.load(f"{OUT}/gold_l0_streams2.npy")
        if ref.shape == pref_streams.shape:
            gate_err = float(np.abs(pref_streams - ref).max())
    except FileNotFoundError:
        pass
    for t, sg in enumerate(steps):
        np.save(f"{OUT}/gold_dec_step{t}_streams2.npy", sg["streams2"])
        np.save(f"{OUT}/gold_dec_step{t}_attn.npy", sg["attn"])
        np.save(f"{OUT}/gold_dec_step{t}_mlp.npy", sg["mlp"])

    meta = {"tokens": s, "steps": NSTEPS,
            "conv_step_vs_batch_max_abs": conv_err,
            "stepwise_vs_gated_l0_streams2_max_abs": gate_err,
            "prefill_last_stage": {k: digest(v) for k, v in outs[-1].items()},
            "kda_state": digest(st["kda"]),
            "ring_shape": list(st["ring_q"].shape),
            "decode": [{"token": sg["token"],
                        **{k: digest(sg[k]) for k in ("attn", "mlp", "streams2")}}
                       for sg in steps]}
    json.dump(meta, open(f"{OUT}/gold_decode.json", "w"), indent=1)
    print(f"GOLD_DECODE_OK prompt_tokens={s} decode_steps={NSTEPS}")
    print(f"  conv_step_vs_batch max_abs={conv_err:.3e}  (must be ~0: the ring order is the check)")
    print(f"  stepwise_attention_vs_gated_gold max_abs={gate_err:.3e}  (gold_l0_streams2.npy)")
    print(f"  kda_state n={meta['kda_state']['n']} max={meta['kda_state']['max']:.6e}")
    for t, sg in enumerate(steps):
        print(f"  step{t} token={sg['token']} streams2 max={np.abs(sg['streams2']).max():.6e} "
              f"attn max={np.abs(sg['attn']).max():.6e}")
    return 0 if conv_err < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
