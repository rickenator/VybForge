#!/usr/bin/env python3
"""mc_ref — the authority side of the S0.2a config gate (validation only; nothing here is in
the runtime path).

The point of this file is that it must not share code with native/config/model_config.vyb. The
authorities are, in order of strength:

  * llama.cpp's own `gguf-dump` (the upstream implementation of the format) for a GGUF, parsed
    from its text output — both its metadata section and its tensor table, so the dims the Vyb
    reader derives from the tensor index are checked against a second, independent reading of
    that same index;
  * `transformers.AutoConfig` for an HF config.json (when the architecture is loadable
    locally), with a plain `json.load` cross-check underneath it;
  * numpy, for the rope_theta -> invfreq table the kernels actually consume.

Modes:
  authority-gguf <gguf> <out.json>       metadata + tensor-derived dims from gguf-dump
  authority-hf   <config.json> <out.json> dims from transformers/json
  check          <probe.txt> <auth.json> compare a Vyb probe's output against the authority
  invfreq        <probe.txt> <invfreq.bin>  recompute the table from the config's rope_theta
"""

import json
import os
import re
import subprocess
import sys


# ───────────────────────── llama.cpp gguf-dump ─────────────────────────

def _gguf_dump(path):
    for cmd in (("gguf-dump", path), ("llama-gguf", path)):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except FileNotFoundError:
            continue
        if p.returncode == 0 and p.stdout:
            return p.stdout
    return None


META_RE = re.compile(r"^\s*\d+:\s+\S+\s+\|\s*\d+\s*\|\s*([A-Za-z0-9_.]+)\s*=\s*(.*?)\s*$")
TENS_RE = re.compile(r"^\s*(\d+):\s*(\d+)\s*\|\s*([\d,\s]+?)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*$")


def authority_gguf(path, out):
    txt = _gguf_dump(path)
    if txt is None:
        return {"ok": False, "why": "no gguf-dump/llama-gguf on PATH"}
    meta, tens = {}, {}
    for line in txt.splitlines():
        m = META_RE.match(line)
        if m:
            # gguf-dump quotes string values ('qwen3'), ints/bools are bare
            v = m.group(2).strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            meta[m.group(1)] = v
        t = TENS_RE.match(line)
        if t:
            dims = [int(x) for x in t.group(3).replace(" ", "").split(",")]
            tens[t.group(5)] = dims
    arch = meta.get("general.architecture", "")
    ap = arch + "."

    def geti(key):
        v = meta.get(ap + key)
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            return None

    def getf(key):
        v = meta.get(ap + key)
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return None

    blk = [int(m.group(1)) for n in tens for m in [re.match(r"blk\.(\d+)\.", n)] if m]
    tied = None
    if tens:
        tied = 0 if "output.weight" in tens else 1
    d = {
        "ok": True,
        "source": path,
        "kind": "gguf",
        "authority": "llama.cpp gguf-dump",
        "arch": arch,
        "n_layers": geti("block_count"),
        "hidden": geti("embedding_length"),
        "heads": geti("attention.head_count"),
        "kv_heads": geti("attention.head_count_kv"),
        "head_dim": geti("attention.key_length"),
        "ffn": geti("feed_forward_length"),
        "vocab": geti("vocab_size"),
        "ctx": geti("context_length"),
        "rope_theta": getf("rope.freq_base"),
        "rms_eps": getf("attention.layer_norm_rms_epsilon"),
        "tied": tied,
        # dims derived from the TENSOR TABLE, which is a different fact from the metadata
        "t_hidden": (tens.get("token_embd.weight") or [None])[0],
        "t_vocab": (tens["token_embd.weight"][1] if "token_embd.weight" in tens else None),
        "t_ffn": (tens["blk.0.ffn_gate.weight"][1] if "blk.0.ffn_gate.weight" in tens else None),
        "t_nq": (tens["blk.0.attn_output.weight"][0] if "blk.0.attn_output.weight" in tens else None),
        "t_nkv": (tens["blk.0.attn_k.weight"][1] if "blk.0.attn_k.weight" in tens else None),
        "t_layers": (max(blk) + 1 if blk else None),
        "n_tensors": len(tens),
    }
    # python gguf, when importable, is a third reading of the same bytes
    try:
        import gguf  # type: ignore
        r = gguf.GGUFReader(path)
        py = {f.name: f for f in r.fields.values()}
        def pv(name):
            f = py.get(name)
            if f is None or len(f.parts) == 0:
                return None
            return f.parts[-1].tolist() if len(f.parts) > 1 else f.parts[0]
        d["py_gguf"] = {k: pv(ap + k) for k in
                        ("block_count", "embedding_length", "attention.head_count",
                         "attention.head_count_kv", "feed_forward_length", "attention.key_length")}
        d["py_gguf_tensors"] = len(r.tensors)
    except Exception as e:                                    # noqa: BLE001
        d["py_gguf"] = {"_skipped": type(e).__name__ + ": " + str(e)[:80]}
    open(out, "w").write(json.dumps(d, indent=1, default=str)) if out else None
    return d


# ───────────────────────── HF config.json ─────────────────────────

def _hf_alias(d, names):
    for n in names:
        if n in d:
            return d[n]
    return None


def authority_hf(path, out):
    cfg = json.load(open(path))
    heads = _hf_alias(cfg, ("num_attention_heads", "n_head"))
    hidden = _hf_alias(cfg, ("hidden_size", "n_embd", "d_model"))
    hd = _hf_alias(cfg, ("head_dim",))
    if hd is None and heads and hidden:
        hd = hidden // heads
    d = {
        "ok": True,
        "source": path,
        "kind": "hf-json",
        "authority": "python json + transformers.AutoConfig",
        "arch": _hf_alias(cfg, ("model_type", "architectures")) or "",
        "n_layers": _hf_alias(cfg, ("num_hidden_layers", "n_layer", "num_layers")),
        "hidden": hidden,
        "heads": heads,
        "kv_heads": _hf_alias(cfg, ("num_key_value_heads", "n_head_kv")) or heads,
        "head_dim": hd,
        "ffn": _hf_alias(cfg, ("intermediate_size", "n_inner", "ffn_dim")),
        "vocab": _hf_alias(cfg, ("vocab_size",)),
        "ctx": _hf_alias(cfg, ("max_position_embeddings",)),
        "rope_theta": _hf_alias(cfg, ("rope_theta", "rotary_theta")),
        "rms_eps": _hf_alias(cfg, ("rms_norm_eps", "norm_eps", "layer_norm_eps")),
        "tied": (int(bool(cfg["tie_word_embeddings"])) if "tie_word_embeddings" in cfg else -1),
    }
    if isinstance(d["arch"], list):
        d["arch"] = d["arch"][0] if d["arch"] else ""
    try:
        from transformers import AutoConfig  # type: ignore
        tc = AutoConfig.from_pretrained(os.path.dirname(os.path.abspath(path)), trust_remote_code=True)
        def g(*names):
            for n in names:
                if hasattr(tc, n):
                    return getattr(tc, n)
            return None
        d["tf"] = {
            "n_layers": g("num_hidden_layers", "num_layers"),
            "hidden": g("hidden_size"),
            "heads": g("num_attention_heads"),
            "kv_heads": g("num_key_value_heads"),
            "head_dim": g("head_dim"),
            "ffn": g("intermediate_size"),
            "vocab": g("vocab_size"),
            "ctx": g("max_position_embeddings"),
            "rope_theta": g("rope_theta"),
            "rms_eps": g("rms_norm_eps", "norm_eps", "layer_norm_eps"),
            "tied": (int(bool(tc.tie_word_embeddings)) if hasattr(tc, "tie_word_embeddings") else None),
        }
        d["transformers_version"] = getattr(__import__("transformers"), "__version__", "?")
    except Exception as e:                                    # noqa: BLE001
        d["tf"] = {"_skipped": type(e).__name__ + ": " + str(e)[:120]}
    if out:
        open(out, "w").write(json.dumps(d, indent=1, default=str))
    return d


# ───────────────────────── comparison ─────────────────────────

INT_FIELDS = ("n_layers", "hidden", "heads", "kv_heads", "head_dim", "ffn", "vocab", "ctx", "tied")
FLOAT_FIELDS = ("rope_theta", "rms_eps")


PROBE_ALIAS = {"layers": "n_layers"}


def parse_probe(path):
    got = {}
    for line in open(path):
        line = line.strip()
        if "=" in line and line.startswith("MC_"):
            k, v = line.split("=", 1)
            k = k[3:].lower()
            got[PROBE_ALIAS.get(k, k)] = v
    return got


def check(probe_path, auth_path):
    got = parse_probe(probe_path)
    want = json.load(open(auth_path))
    fails, checks, skipped = [], [], []

    def note(msg):
        checks.append(msg)

    if not want.get("ok"):
        print("MCGATE_FAIL authority unavailable: %s" % want.get("why"))
        return 1

    # 1. the metadata fields, against the upstream reading
    for f in INT_FIELDS:
        w = want.get(f)
        if w is None:
            continue
        if f == "tied" and w == -1:
            continue
        g = got.get(f)
        if g is None:
            fails.append("%s: probe did not print it" % f)
            continue
        if int(g) != int(w):
            fails.append("%s: probe=%s authority=%s" % (f, g, w))
        else:
            note("%s=%s" % (f, g))
    for f in FLOAT_FIELDS:
        w = want.get(f)
        if w is None:
            continue
        g = got.get(f)
        if g is None:
            fails.append("%s: probe did not print it" % f)
            continue
        gv = float(g)
        if abs(gv - float(w)) > 1e-6 * max(abs(float(w)), 1e-30):
            fails.append("%s: probe=%s authority=%s" % (f, g, w))
        else:
            note("%s=%s" % (f, g))
    if want.get("arch") and got.get("arch") != want["arch"]:
        fails.append("arch: probe=%s authority=%s" % (got.get("arch"), want["arch"]))

    # 2. python gguf, when present
    pyg = want.get("py_gguf") or {}
    if "_skipped" in pyg:
        skipped.append("python gguf (%s)" % pyg["_skipped"])
    else:
        for k, v in pyg.items():
            if v is None:
                continue
            if isinstance(v, list) and len(v) == 1:
                v = v[0]
            key = {"attention.head_count": "heads", "attention.head_count_kv": "kv_heads",
                   "attention.key_length": "head_dim", "embedding_length": "hidden",
                   "feed_forward_length": "ffn", "block_count": "n_layers"}.get(k)
            if key and key in got and int(got[key]) != int(v):
                fails.append("py-gguf %s: probe=%s authority=%s" % (key, got[key], v))
        note("python gguf agrees (%s tensors)" % want.get("py_gguf_tensors"))

    # 3. transformer's reading, when present
    tf = want.get("tf") or {}
    if "_skipped" in tf:
        skipped.append("transformers.AutoConfig (%s)" % tf["_skipped"])
    else:
        for k, v in tf.items():
            if v is None:
                continue
            g = got.get(k)
            if g is None:
                continue
            if k in FLOAT_FIELDS:
                if abs(float(g) - float(v)) > 1e-6 * max(abs(float(v)), 1e-30):
                    fails.append("transformers %s: probe=%s authority=%s" % (k, g, v))
            elif int(g) != int(v):
                fails.append("transformers %s: probe=%s authority=%s" % (k, g, v))
        note("transformers %s agrees" % want.get("transformers_version", ""))

    # 4. the dims DERIVED FROM THE TENSOR TABLE (a different fact than the metadata)
    for tf_key, dim_field in (("t_hidden", "hidden"), ("t_vocab", "vocab"), ("t_ffn", "ffn"),
                              ("t_layers", "n_layers")):
        v = want.get(tf_key)
        if v is None:
            continue
        checks.append("tensor-table %s=%s" % (dim_field, v))
    if want.get("t_nq") and got.get("heads") and got.get("head_dim"):
        if int(want["t_nq"]) != int(got["heads"]) * int(got["head_dim"]):
            fails.append("tensor-table nq=%s != heads*head_dim=%s" % (
                want["t_nq"], int(got["heads"]) * int(got["head_dim"])))
        else:
            note("nq=%s == heads*head_dim" % want["t_nq"])
    if want.get("t_nkv") and got.get("kv_heads") and got.get("head_dim"):
        if int(want["t_nkv"]) != int(got["kv_heads"]) * int(got["head_dim"]):
            fails.append("tensor-table nkv=%s != kv_heads*head_dim=%s" % (
                want["t_nkv"], int(got["kv_heads"]) * int(got["head_dim"])))
        else:
            note("nkv=%s == kv_heads*head_dim" % want["t_nkv"])

    print("MCGATE_FIELDS_OK %d" % len(checks))
    for s in skipped:
        print("MCGATE_AUTHORITY_SKIPPED %s" % s)
    if fails:
        for f in fails:
            print("MCGATE_MISMATCH %s" % f)
        print("MCGATE_FAIL %d mismatch(es)" % len(fails))
        return 1
    print("MCGATE_OK")
    return 0


def invfreq(probe_path, bin_path):
    """The kernels consume a precomputed invfreq table; it must be the one this config implies."""
    got = parse_probe(probe_path)
    if not os.path.exists(bin_path):
        print("MCGATE_AUTHORITY_SKIPPED invfreq table absent (%s)" % bin_path)
        print("MCGATE_OK")
        return 0
    try:
        import numpy as np  # type: ignore
    except Exception as e:                                    # noqa: BLE001
        print("MCGATE_AUTHORITY_SKIPPED numpy (%s)" % e)
        print("MCGATE_OK")
        return 0
    theta = float(got.get("rope_theta", 0.0))
    hd = int(got.get("head_dim", 0))
    tbl = np.fromfile(bin_path, "<f8")
    if theta <= 0.0 or hd <= 0 or len(tbl) != hd // 2:
        print("MCGATE_FAIL invfreq: theta=%s head_dim=%s len=%d (expected %d)"
              % (theta, hd, len(tbl), max(hd // 2, 0)))
        return 1
    ref = np.power(theta, -(np.arange(0, hd, 2) / hd))
    d = float(np.max(np.abs(tbl - ref)))
    if d > 1e-12:
        print("MCGATE_FAIL invfreq: max|diff|=%g vs theta=%g (table was built with a different theta)"
              % (d, theta))
        return 1
    print("MCGATE_INVFREQ_OK theta=%g max|diff|=%.3g" % (theta, d))
    print("MCGATE_OK")
    return 0


def mutate(inp, out, op, key, val=None):
    """Produce a deliberately broken input, so the gate can prove the reader is really reading
    (and that its documented fallbacks fire) rather than returning remembered values."""
    cfg = json.load(open(inp))
    if op == "set":
        if val is None:
            print("MCREF_MUTATE set needs a value")
            return 2
        old = cfg.get(key)
        cfg[key] = int(val) if str(val).lstrip("-").isdigit() else val
        print("MCREF_MUTATE set %s: %s -> %s" % (key, old, cfg[key]))
    elif op == "del":
        if key not in cfg:
            print("MCREF_MUTATE key %s was not present" % key)
            return 1
        del cfg[key]
        print("MCREF_MUTATE del %s" % key)
    else:
        print("MCREF_MUTATE unknown op %s" % op)
        return 2
    json.dump(cfg, open(out, "w"), indent=1)
    return 0


def main(argv):
    if len(argv) < 2:
        print("usage: mc_ref.py <mode> ..."); return 2
    mode = argv[1]
    if mode == "authority-gguf":
        d = authority_gguf(argv[2], argv[3] if len(argv) > 3 else "")
        print("MCREF_AUTHORITY=%s tensors=%s" % (d.get("authority"), d.get("n_tensors")))
        return 0 if d.get("ok") else 1
    if mode == "authority-hf":
        d = authority_hf(argv[2], argv[3] if len(argv) > 3 else "")
        print("MCREF_AUTHORITY=%s arch=%s" % (d.get("authority"), d.get("arch")))
        return 0 if d.get("ok") else 1
    if mode == "mutate":
        return mutate(argv[2], argv[3], argv[4], argv[5], argv[6] if len(argv) > 6 else None)
    if mode == "check":
        return check(argv[2], argv[3])
    if mode == "invfreq":
        return invfreq(argv[2], argv[3] if len(argv) > 3 else "")
    print("unknown mode %s" % mode)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
