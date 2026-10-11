#!/usr/bin/env python3
"""oracle_stt.py — staged reference tensors for the Vyb-native Whisper forward.

    ~/miniconda/bin/python oracle_stt.py

Not runtime code. It runs the real openai/whisper-tiny.en forward pass on the
jfk.wav fixture and dumps every intermediate the Vyb implementation will be
compared against, stage by stage, so a wrong result can be localised instead of
guessed at.

Everything lands under gold_stt/ as little-endian float32 + a `.shape` sidecar
(the Vyb side cannot read float files, so comparisons run the other way: Vyb
prints scaled integers, check_stt.py compares).

Stages, in the order the Vyb implementation should be gated:
  mel                 (80, 3000)      input, already gated by native/mel
  e_conv1             (384, 3000)     conv1(80->384, k3, p1) + GELU
  e_conv2             (384, 1500)     conv2(384->384, k3, s2, p1) + GELU
  e_pos               (1500, 384)     encoder positional embedding slice
  e_in                (1500, 384)     conv2 output + positional, the layer input
  e_layer0 .. e_layer3 (1500, 384)    output of each encoder block (post-LN)
  e_final             (1500, 384)     encoder final_layer_norm output
  d_prompt            token ids of the decoder prompt (ints, .txt)
  d_embed             (n, 384)        token embedding after embed_positions
  d_step0_logits      (51864,)        logits for the first generated position
  d_ids               greedy token ids for the whole answer (.txt)
"""
import os
import wave

import numpy as np
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "..", "..", "fixtures", "whisper")
OUT = os.path.join(GOLD, "gold_stt")
MODEL = "openai/whisper-tiny.en"

os.makedirs(OUT, exist_ok=True)


def dump(name, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
    a.tofile(os.path.join(OUT, name + ".f32"))
    with open(os.path.join(OUT, name + ".shape"), "w") as f:
        f.write(" ".join(str(d) for d in a.shape) + "\n")


def dump_i(name, ids):
    with open(os.path.join(OUT, name + ".txt"), "w") as f:
        f.write(" ".join(str(int(i)) for i in ids) + "\n")


# --- fixture: same PCM the mel gate used ------------------------------------
with wave.open(os.path.join(HERE, "..", "..", "fixtures", "audio", "jfk.wav"), "rb") as w:
    assert w.getframerate() == 16000 and w.getsampwidth() == 2
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0

proc = AutoProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()
enc = model.model.encoder
dec = model.model.decoder

feats = proc(audio, sampling_rate=16000, return_tensors="pt")["input_features"]   # (1, 80, 3000)
dump("mel", feats[0].numpy())
print("mel", tuple(feats.shape), "cfg", model.config.d_model, model.config.encoder_layers,
      model.config.decoder_layers, model.config.encoder_attention_heads,
      model.config.max_source_positions, model.config.max_target_positions,
      "vocab", model.config.vocab_size)

states = {}

with torch.no_grad():
    x = feats                                     # (1, 80, 3000)
    c1 = enc.conv1(x)                             # (1, 384, 3000)
    g1 = torch.nn.functional.gelu(c1)
    states["e_conv1"] = g1[0]
    c2 = enc.conv2(g1)                            # (1, 384, 1500)
    g2 = torch.nn.functional.gelu(c2)
    states["e_conv2"] = g2[0]

    h = g2.permute(0, 2, 1)                       # (1, 1500, 384)
    pos = enc.embed_positions.weight[: h.shape[1]]
    states["e_pos"] = pos
    h = h + pos
    states["e_in"] = h[0]

    # encoder layers, with the same pre-LN structure transformers uses.
    # Layer 0 gets its sub-stages dumped too: a first-layer wiring bug is far
    # cheaper to localise at the sub-stage than at the block boundary.
    hidden = h
    for i, layer in enumerate(enc.layers):
        if i == 0:
            res = hidden
            ln1 = layer.self_attn_layer_norm(hidden)
            states["e_l0_ln1"] = ln1[0]
            att = layer.self_attn(ln1, attention_mask=None, layer_head_mask=None)[0]
            states["e_l0_attn"] = att[0]
            r1 = res + att
            states["e_l0_res1"] = r1[0]
            ln2 = layer.final_layer_norm(r1)
            states["e_l0_ln2"] = ln2[0]
            m = layer.fc2(torch.nn.functional.gelu(layer.fc1(ln2)))
            states["e_l0_mlp"] = m[0]
        layer_out = layer(hidden, attention_mask=None, layer_head_mask=None, output_attentions=False)
        hidden = layer_out[0]
        states[f"e_layer{i}"] = hidden[0]
    final = enc.layer_norm(hidden)
    states["e_final"] = final[0]

    for k, v in states.items():
        dump(k, v.numpy())

    # --- decoder -------------------------------------------------------------
    ids = model.generate(feats, max_new_tokens=32)
    dump_i("d_ids", ids[0].tolist())
    print("greedy ids:", ids[0].tolist())

    # The prompt for an English-only checkpoint: <|startoftranscript|> then
    # <|notimestamps|>. Taken from the tokenizer rather than hardcoded.
    prompt = proc.tokenizer.convert_tokens_to_ids(["<|startoftranscript|>", "<|notimestamps|>"])
    prompt = [int(t) for t in prompt]
    print("prompt ids:", prompt)
    dump_i("d_prompt", prompt)
    dec_in = torch.tensor([prompt])

    out = model(encoder_outputs=(final,), decoder_input_ids=dec_in)
    dump("d_step0_logits", out.logits[0, -1].numpy())
    print("step0 logits argmax:", int(out.logits[0, -1].argmax()))
    emb = dec.embed_tokens(dec_in) + dec.embed_positions.weight[: dec_in.shape[1]]
    dump("d_embed", emb[0].numpy())

    # decoder layer 0 sub-stages (self-attn, cross-attn, mlp) plus the final LN
    l0 = dec.layers[0]
    ln1 = l0.self_attn_layer_norm(emb)
    a0 = l0.self_attn(ln1, attention_mask=None, layer_head_mask=None)[0]
    r1 = emb + a0
    ln2 = l0.encoder_attn_layer_norm(r1)
    c0 = l0.encoder_attn(ln2, key_value_states=final, attention_mask=None, layer_head_mask=None)[0]
    r2 = r1 + c0
    ln3 = l0.final_layer_norm(r2)
    m0 = l0.fc2(torch.nn.functional.gelu(l0.fc1(ln3)))
    dump("d_l0_ln1", ln1[0].numpy())
    dump("d_l0_selfattn", a0[0].numpy())
    dump("d_l0_res1", r1[0].numpy())
    dump("d_l0_crossattn", c0[0].numpy())
    dump("d_l0_mlp", m0[0].numpy())
    # NOTE: this is the final decoder norm applied to LAYER 0's output -- a
    # one-layer probe, not the real four-layer path (which d_step0_logits covers).
    dump("d_layernorm_on_l0_out", dec.layer_norm(r2 + m0)[0].numpy())

text = proc.batch_decode(ids, skip_special_tokens=True)[0]
print("ref text:", text)
with open(os.path.join(OUT, "ref_text.txt"), "w") as f:
    f.write(text + "\n")
print("done ->", OUT)
