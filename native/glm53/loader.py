"""GLM-5.3-Flash reference checkpoint loader (P4 storage layer / P6 prep).

Python + numpy only. Reads the HF-format checkpoint (safetensors shards +
model.safetensors.index.json) with plain mmap — no `safetensors` package — and
exposes the layer weights the P4/P6 work consumes:

  loader          = GLM53Loader(checkpoint_dir)
  loader.layers            # {i: {name: TensorInfo}}   (enumerate, no data read)
  loader.load_layer(i)     # {name: f32 ndarray}  non-expert weights, bf16->f32
  loader.expert(i, e)      # {"gate": [in,out], "up": ..., "down": ...} fp16,
                           #  dequantized on demand, kept in an LRU (max_experts)
  loader.load_router(i)    # (gate W f32 [288, 4096], bias f32 [288])
  loader.load_shared(i)    # (wg, wu, wd) f32 [in, out]   (transposed from [out,in])
  loader.layer_bytes(i)    # per-layer byte accounting (stored/resident/effective)
  loader.model_bytes()     # whole-checkpoint totals

Checkpoint facts this loader relies on (verified against the shards):
  - non-expert tensors are bf16 (a few f32 scalars/biases: dt_bias, A_log,
    hc_*_base, hc_*_scale, e_score_correction_bias);
  - routed experts are stored EXL3 4bpw (quantization_config: exl3, bits 4,
    codebook mcg, scope glm53_routed_experts_only): per matrix
      trellis int16 [in/16, out/16, 64]  (0.5 B/weight)
      suh     f16    [in]               (row scale+sign, k = IN dim)
      svh     f16    [out]              (col scale+sign)
      mcg     i32    [1]                (codebook id)
    and dequantize to [in, out] via native/exl3/exl3_ref.dequant_exl3
    (oracle-gated: EXL3_REF_VERIFY, 4.4e-5 max rel vs exllamav3).
  - naming: model.language_model.layers.<N>.<site>...; experts are
    model.language_model.layers.<N>.mlp.experts.<e>.<proj>.{trellis,suh,svh,mcg}.
  - bf16 -> f32 is (uint16 << 16).view(f32) — see README gotcha: never convert
    twice (an f32->uint32 astype truncates to 0/1 and silently zeroes data).

Shard reality check: a shard is only used if its header parses AND
file_size >= 8 + header_len + max(data_offset). The hollow-tree incident (the
first apex copy rsynced HF's symlinks — 832 KB of dangling links dressed as a
checkpoint) is why every gate run calls GLM53Loader.tree_status() first: it
reports the .COMPLETE marker and refuses to count shards that fail the check.

===========================================================================
MEMORY ACCOUNTING (measured from the checkpoint, 83 shards)
===========================================================================
Storage on disk (index total, ~183 GB — NOT ~90 GB):
  embed_tokens + lm_head      2 x 154,880 x 4,096 x 2 B      ~2.54 GB
  dense layers 0-2            ~0.58 GB each (KDA + dense MLP) ~1.74 GB
  DSA layers (11)             ~0.21 GB each                   ~2.3  GB
  MoE layers (42) non-expert  ~0.58 GB each                   ~24.3 GB
  MoE experts, EXL3 4bpw      42 x 288 x 3 x 4.2 MB           ~152  GB
  MTP layer + vision tower    remainder                       ~2-4  GB
The routed experts alone (152 GB) exceed one 121 GB Spark, which is why the
loader dequantizes on demand and keeps only the routed-in experts resident.

Resident plan, ONE 121 GB Spark, full 45-layer model:
  bf16 body (embed, lm_head, all non-expert layer weights)   ~31.1 GB
  expert LRU 16 experts x 3 matrices, fp16 [2048, 4096]       ~0.80 GB
  KV latent, 11 DSA layers x 1 M x 512 bf16                  ~11.0 GB
  indexer keys, 45 layers x 1 M x 128 bf16                   ~11.3 GB
  KDA recurrent state, 34 x 64x128x128 bf16 (+ conv)        ~0.07 GB
  activations/scratch (4-stream mHC, 64 KB/token working set) ~1-2 GB
  ---------------------------------------------------------------
  worst case @1 M context                                    ~56-58 GB   fits
  @32 K context                                              ~34-36 GB

TP=2 split (two Sparks, layer-sharded — the default; see kv_cache.tp2_plan):
  rank 0: layers 0-22  (5 DSA, 18 KDA)   rank 1: layers 23-44 (6 DSA, 16 KDA)
  bf16 body ~15.6 GB / ~15.4 GB; expert LRU same 0.80 GB per rank;
  KV latent 5.1 / 6.1 GB, indexer keys 5.9 / 5.6 GB @1 M;
  embed_tokens + lm_head + final norm replicated (or lm_head sharded on vocab);
  KDA state is per-sequence O(1) — 2 MiB/layer bf16, so 36/32 MiB per rank,
  amortized 0.068 B/token at 1 M — flat in context length.
===========================================================================
"""

import json
import os
import struct
import sys
import time
from collections import OrderedDict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_NATIVE = os.path.dirname(_HERE)
if _NATIVE not in sys.path:
    sys.path.insert(0, _NATIVE)

# safetensors dtype -> numpy
ST_DTYPES = {
    "BF16": "<u2", "F32": "<f4", "F16": "<f2", "F64": "<f8",
    "I8": "<i1", "I16": "<i2", "I32": "<i4", "I64": "<i8",
    "U8": "<u1", "U16": "<u2", "U32": "<u4", "U64": "<u8",
    "BOOL": "|b1",
}
ITEMSIZE = {k: np.dtype(v).itemsize for k, v in ST_DTYPES.items()}

EXPERT_SUFFIXES = ("trellis", "suh", "svh", "mcg")
EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")

HIDDEN = 4096
MOE_INTER = 2048
N_EXPERTS = 288
VOCAB = 154880
N_LAYERS = 45
FIRST_DENSE = 3


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    """bf16 = top 16 bits of f32 (README gotcha: convert exactly once)."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def dequant_exl3_fast(trellis: np.ndarray, suh: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """dequant_exl3 with the Hadamard contractions as BLAS GEMMs.

    Byte-for-byte the same decode and scale pipeline as
    exl3.exl3_ref.dequant_exl3 (which is oracle-gated, EXL3_REF_VERIFY), with
    the one einsum given optimize=True so the two 128x128 Hadamard sides run
    as tensordot GEMMs instead of an nditer loop (minutes -> <1 s per
    2048x4096 matrix). The accumulation order inside the 128-term sums may
    differ from the canonical path; verify_loader gates the fast path against
    the canonical output at 4.4e-5 max rel on the real fixture (and reports
    the exact-match count).
    """
    from exl3 import exl3_ref
    kb, nb, words = trellis.shape
    assert words == 64
    k, n = kb * 16, nb * 16
    tr = trellis.view(np.uint16) if trellis.dtype == np.int16 else trellis

    t = tr.reshape(kb * nb, 64).astype(np.uint32)
    shifts = np.uint32(np.arange(16))
    bits = ((t[:, :, None] >> shifts) & np.uint32(1)).reshape(kb * nb, 1024).astype(np.uint8)
    nib = (bits[:, 0::4] << 0) | (bits[:, 1::4] << 1) | (bits[:, 2::4] << 2) | (bits[:, 3::4] << 3)

    slots, cidx, cpos = exl3_ref.load_slot_composition(
        os.path.join(_NATIVE, "exl3", "tile_slot_composition.json"))
    w16 = np.zeros((kb * nb, 256), dtype=np.uint32)
    for i in range(4):
        w16 |= nib[:, cidx[:, i]].astype(np.uint32) << np.uint32(4 * cpos[:, i])
    vals = exl3_ref.decode_mcg(w16)

    what = np.zeros((k, n), dtype=np.float32)
    rows = np.array([s[0] for s in slots])
    cols = np.array([s[1] for s in slots])
    rr = (np.arange(kb)[:, None, None] * 16 + rows[None, None, :])
    cc = (np.arange(nb)[None, :, None] * 16 + cols[None, None, :])
    what[rr, cc] = vals.reshape(kb, nb, 256)

    Hs = (exl3_ref.hadamard128() * exl3_ref.R_SCALE).astype(np.float32)
    t2 = what.reshape(k // 128, 128, n // 128, 128)
    t2 = np.einsum("ab,ibjc,cd->ijad", Hs, t2, Hs, optimize=True)
    t2 = t2.transpose(0, 2, 1, 3)
    w = t2.reshape(k, n) * suh[:, None].astype(np.float32) * svh[None, :].astype(np.float32)
    return w.astype(np.float16)


def is_exl3_expert_tensor(name: str):
    """-> (layer, expert, proj, suffix) if name is an EXL3 expert tensor."""
    parts = name.split(".")
    if len(parts) == 9 and parts[0] == "model" and parts[4] == "mlp" \
            and parts[5] == "experts" and parts[7] in EXPERT_PROJS \
            and parts[8] in EXPERT_SUFFIXES:
        return int(parts[3]), int(parts[6]), parts[7], parts[8]
    return None


def layer_of(name: str):
    """Layer index for model.language_model.layers.<N>.* tensors, else None."""
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "model" and parts[1] == "language_model" \
            and parts[2] == "layers":
        try:
            return int(parts[3])
        except ValueError:
            return None
    return None


class TensorInfo:
    __slots__ = ("name", "dtype", "shape", "shard", "offset", "nbytes")

    def __init__(self, name, dtype, shape, shard, offset, nbytes):
        self.name, self.dtype, self.shape = name, dtype, shape
        self.shard, self.offset, self.nbytes = shard, offset, nbytes

    def __repr__(self):
        return f"<{self.name} {self.dtype}{list(self.shape)} @ {self.shard}>"


class ShardMap:
    """One safetensors shard: header parsed once, data read via mmap."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(hlen))
        self.data_start = 8 + hlen
        self.size = os.path.getsize(path)
        self.ok = self.size >= self.data_start + max(
            (v["data_offsets"][1] for k, v in self.header.items()
             if k != "__metadata__"),
            default=0)
        self._mm = None

    def mmap(self):
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r",
                                 offset=self.data_start)
        return self._mm

    def raw(self, info: TensorInfo) -> np.ndarray:
        dt = np.dtype(ST_DTYPES[info.dtype])
        mm = self.mmap()
        return mm[info.offset:info.offset + info.nbytes].view(dt).reshape(info.shape)

    def tensors(self):
        for k, v in self.header.items():
            if k == "__metadata__":
                continue
            yield k, v


class GLM53Loader:
    """Reference loader for the GLM-5.3-Flash checkpoint (see module docstring)."""

    def __init__(self, checkpoint_dir, max_experts=16, dtype_out="f32"):
        self.dir = checkpoint_dir
        self.max_experts = max_experts      # LRU capacity in EXPERTS (3 matrices each)
        self.dtype_out = dtype_out          # "f32" for bf16 loads (default) or "raw"
        self.index = None                   # tensor -> shard file
        self.shards = {}                    # file -> ShardMap (only real shards)
        self.bad_shards = []                # failed the reality check
        self.layers = {}                    # layer -> {name: TensorInfo}
        self.other = {}                     # non-layer tensors
        self.expert_index = {}              # (layer, expert) -> {proj: {suffix: TensorInfo}}
        self._lru = OrderedDict()           # (layer, expert) -> {proj: fp16 [in,out]}
        self.stats = {"expert_hits": 0, "expert_misses": 0, "expert_evictions": 0,
                      "dequant_seconds": 0.0, "dequants": 0}
        self._scan()

    # ------------------------------------------------------------- discovery
    def _scan(self):
        idx_path = os.path.join(self.dir, "model.safetensors.index.json")
        shard_files = []
        if os.path.exists(idx_path):
            self.index = json.load(open(idx_path))["weight_map"]
            shard_files = sorted(set(self.index.values()))
        else:
            shard_files = sorted(f for f in os.listdir(self.dir)
                                 if f.endswith(".safetensors"))
        for fname in shard_files:
            path = os.path.join(self.dir, fname)
            try:
                sm = ShardMap(path)
            except (OSError, ValueError):
                self.bad_shards.append((fname, "unreadable"))
                continue
            if not sm.ok:
                self.bad_shards.append((fname, f"truncated/hollow ({sm.size} B)"))
                continue
            self.shards[fname] = sm
            for name, v in sm.tensors():
                info = TensorInfo(name, v["dtype"], tuple(v["shape"]),
                                  fname, v["data_offsets"][0],
                                  v["data_offsets"][1] - v["data_offsets"][0])
                lay = layer_of(name)
                ex = is_exl3_expert_tensor(name)
                if ex:
                    layer, expert, proj, suffix = ex
                    self.expert_index.setdefault((layer, expert), {}) \
                        .setdefault(proj, {})[suffix] = info
                elif lay is not None:
                    self.layers.setdefault(lay, {})[name] = info
                else:
                    self.other[name] = info

    # ------------------------------------------------------------ tree state
    def tree_status(self):
        """Reality of the checkpoint tree: .COMPLETE marker + per-shard check."""
        complete = os.path.exists(os.path.join(self.dir, ".COMPLETE"))
        n_ok = len(self.shards)
        return {"complete_marker": complete, "shards_ok": n_ok,
                "shards_bad": self.bad_shards,
                "indexed_shards": len(set(self.index.values())) if self.index else None,
                "has_index": self.index is not None}

    def _check(self, name):
        if name not in self.shards:
            raise KeyError(f"tensor {name}: shard missing or failed reality check "
                           f"(bad: {self.bad_shards})")
        return self.shards[name]

    # -------------------------------------------------------------- raw read
    def raw(self, name):
        """Stored bytes as the checkpoint dtype (bf16 stays uint16)."""
        info = self.tensor_info(name)
        return self._check(info.shard).raw(info)

    def tensor_info(self, name):
        lay = layer_of(name)
        info = (self.layers.get(lay, {}).get(name) if lay is not None else None) \
            or self.other.get(name)
        if info is None:
            ex = is_exl3_expert_tensor(name)
            if ex:
                layer, expert, proj, suffix = ex
                info = self.expert_index.get((layer, expert), {}).get(proj, {}).get(suffix)
        if info is None:
            raise KeyError(name)
        return info

    def load_tensor(self, name):
        """One tensor as f32 (bf16 -> f32, f32/f16 passthrough promoted)."""
        raw = self.raw(name)
        dt = raw.dtype
        if dt == np.uint16:                      # bf16
            out = bf16_to_f32(raw)
        elif dt in (np.float32,):
            out = raw.astype(np.float32)
        elif dt == np.float16:
            out = raw.astype(np.float32)
        elif dt in (np.int32, np.int16, np.uint16, np.uint8, np.int8):
            out = raw
        else:
            out = raw.astype(np.float32)
        return np.asarray(out)

    # ----------------------------------------------------------- layer loads
    def layer_names(self, i):
        """Non-expert tensor names of layer i (experts live in expert_index)."""
        return sorted(self.layers.get(i, {}))

    def layer_type(self, i):
        names = self.layers.get(i, {})
        attn = "dsa" if any(".self_attn.kv_a_proj" in n or ".self_attn.q_a_proj" in n
                            for n in names) else "kda"
        moe = "+moe" if (i, 0) in self.expert_index else ""
        if i < FIRST_DENSE and not moe:
            return attn + "+dense"
        return attn + moe

    def load_layer(self, i, experts=None):
        """Non-expert weights of layer i as {name: f32}.

        Expert matrices are NOT materialized here (a MoE layer stores 288 x 3
        of them; resident is only what routing selects). Touch experts with
        loader.expert(i, e) (LRU) or prefetch with experts=[...].
        """
        out = {}
        for name in self.layer_names(i):
            out[name] = self.load_tensor(name)
        for e in (experts or []):
            self.expert(i, e)
        return out

    def load_router(self, i):
        """(gate W f32 [288, 4096], e_score_correction_bias f32 [288])."""
        w = self.load_tensor(f"model.language_model.layers.{i}.mlp.gate.weight")
        bias = self.load_tensor(
            f"model.language_model.layers.{i}.mlp.gate.e_score_correction_bias")
        return w.astype(np.float32), bias.astype(np.float32)

    def load_shared(self, i):
        """Shared expert as (wg, wu, wd) f32 [in, out] (moe_ref convention)."""
        pre = f"model.language_model.layers.{i}.mlp.shared_experts"
        g = self.load_tensor(f"{pre}.gate_proj.weight")
        u = self.load_tensor(f"{pre}.up_proj.weight")
        d = self.load_tensor(f"{pre}.down_proj.weight")
        return g.T.astype(np.float64), u.T.astype(np.float64), d.T.astype(np.float64)

    # ------------------------------------------------------- experts + LRU
    def complete_expert(self, layer, expert):
        """True if all 12 EXL3 tensors of (layer, expert) are in real shards."""
        need = {p: set(EXPERT_SUFFIXES) for p in EXPERT_PROJS}
        got = self.expert_index.get((layer, expert), {})
        return all(set(got.get(p, {})) == need[p] and p in got for p in EXPERT_PROJS) \
            and all(got[p][s].shard in self.shards
                    for p in EXPERT_PROJS for s in EXPERT_SUFFIXES)

    def expert(self, layer, expert, proj=None):
        """Dequantized expert weights {proj: f16 [in, out]}, LRU-cached.

        proj=None -> all three matrices; proj='down_proj' -> just that one.
        """
        key = (layer, expert)
        hit = key in self._lru
        if hit:
            self.stats["expert_hits"] += 1
            self._lru.move_to_end(key)
        else:
            self.stats["expert_misses"] += 1
        if hit and (proj is not None or len(self._lru[key]) == len(EXPERT_PROJS)):
            return self._lru[key] if proj is None else {proj: self._lru[key][proj]}
        if not self.complete_expert(layer, expert):
            raise KeyError(f"expert {expert} of layer {layer} not fully present "
                           f"in real shards")
        want = EXPERT_PROJS if proj is None else (proj,)
        out = dict(self._lru.get(key, {}))
        for p in want:
            if p in out:
                continue
            t = self.expert_index[key][p]
            t0 = time.perf_counter()
            trellis = self._check(t["trellis"].shard).raw(t["trellis"])
            suh = self._check(t["suh"].shard).raw(t["suh"])
            svh = self._check(t["svh"].shard).raw(t["svh"])
            w = dequant_exl3_fast(trellis, suh, svh)     # f16 [in, out], BLAS path
            self.stats["dequant_seconds"] += time.perf_counter() - t0
            self.stats["dequants"] += 1
            out[p] = w
        if key in self._lru:
            self._lru.move_to_end(key)
        self._lru[key] = out
        while len(self._lru) > self.max_experts:
            self._lru.popitem(last=False)
            self.stats["expert_evictions"] += 1
        return out if proj is None else {proj: out[proj]}

    def lru_bytes(self):
        """Resident bytes held by the expert LRU (fp16 dequantized weights)."""
        return sum(w.nbytes for v in self._lru.values() for w in v.values())

    # ---------------------------------------------------------- byte counts
    @staticmethod
    def _stored(info: TensorInfo):
        return info.nbytes

    @staticmethod
    def _effective_f32(info: TensorInfo):
        numel = int(np.prod(info.shape))
        if info.dtype == "BF16":
            return numel * 4                    # bf16 -> f32 resident
        if info.dtype == "F32":
            return numel * 4
        return info.nbytes                      # exl3 / ints: counted at storage

    def layer_bytes(self, i):
        """Byte accounting for layer i (experts counted from their index).

        stored          : bytes on disk (bf16 2 B/w, exl3 0.5 B/w + scales)
        effective_f32   : bytes if every non-expert tensor were f32 resident
        expert_stored   : EXL3 bytes of the 288 routed experts
        expert_f16_all  : bytes if ALL 288 experts were dequantized (f16)
        expert_f16_topk : bytes for the 8 routed-in experts of one token (f16)
        resident_no_lru : non-expert f32 + LRU footprint actually held now
        """
        infos = list(self.layers.get(i, {}).values())
        stored = sum(self._stored(x) for x in infos)
        eff = sum(self._effective_f32(x) for x in infos)
        ex_infos = [x for key, projs in self.expert_index.items() if key[0] == i
                    for p in projs.values() for x in p.values()]
        ex_stored = sum(self._stored(x) for x in ex_infos)
        # dequantized size from the trellis geometry: [in/16, out/16, 64] i16
        ex_f16 = 0
        for key, projs in self.expert_index.items():
            if key[0] != i:
                continue
            for p, sfx in projs.items():
                if "trellis" not in sfx:        # shard mid-arrival
                    continue
                sh = sfx["trellis"].shape
                ex_f16 += sh[0] * sh[1] * 16 * 16 * 2   # [in, out] f16
        ex_f16_topk = ex_f16 // N_EXPERTS * 8
        resident = eff + self.lru_bytes()
        return {"layer": i, "type": self.layer_type(i), "n_tensors": len(infos),
                "stored": stored, "effective_f32": eff,
                "expert_stored": ex_stored, "expert_f16_all": ex_f16,
                "expert_f16_topk": ex_f16_topk, "resident_no_lru": resident}

    def model_bytes(self):
        """Whole-checkpoint totals from the discovered (real) shards."""
        per_layer = [self.layer_bytes(i) for i in sorted(self.layers)]
        tot_stored = sum(l["stored"] + l["expert_stored"] for l in per_layer) \
            + sum(self._stored(x) for x in self.other.values())
        return {
            "layers": per_layer,
            "stored_total": tot_stored,
            "body_f32_total": sum(l["effective_f32"] for l in per_layer)
                              + sum(self._effective_f32(x) for x in self.other.values()),
            "expert_stored_total": sum(l["expert_stored"] for l in per_layer),
            "shards_ok": len(self.shards),
            "shards_bad": len(self.bad_shards),
            "complete": self.tree_status()["complete_marker"],
        }


def format_gib(n):
    return f"{n / (1 << 30):8.2f} GiB"


if __name__ == "__main__":
    ck = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/apex/rick/vybforge/glm53-checkpoint"
    ld = GLM53Loader(ck, max_experts=16)
    st = ld.tree_status()
    print(f"tree: complete={st['complete_marker']} shards_ok={st['shards_ok']} "
          f"indexed={st['indexed_shards']} bad={st['shards_bad']}")
    for i in (0, 2, 3):
        if i in ld.layers:
            b = ld.layer_bytes(i)
            print(f"layer {i} ({b['type']}): {b['n_tensors']} tensors, "
                  f"stored {format_gib(b['stored'])}, "
                  f"experts stored {format_gib(b['expert_stored'])}, "
                  f"all-288 f16 would be {format_gib(b['expert_f16_all'])}, "
                  f"top-8 f16 {format_gib(b['expert_f16_topk'])}")
    mb = ld.model_bytes()
    print(f"discovered so far: stored {format_gib(mb['stored_total'])}, "
          f"body f32 {format_gib(mb['body_f32_total'])}, "
          f"experts {format_gib(mb['expert_stored_total'])}")
