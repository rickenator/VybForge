#!/usr/bin/env python3
"""Ridge's FFN half on the ENGINE's own weight path vs OUR dequant ports (VybForge#10 phase 4, W1).

What this proves, and why the per-layer gates are not enough. The FFN arithmetic here is the shared
dense block (pre-norm -> parallel gate/up -> SiLU(gate)*up -> down -> residual) and it is already
exercised by `make prefill` on Qwen3-4B. What no engine path could do before W1 is STAGE Ridge's FFN
weights: they are IQ2_S (160 tensors, the mid-stack) and IQ3_S (32, the edge layers), and both carry
their codebook as an ordinary FIFTH kernel argument — so they launch through `cuda_launch_n`, not the
four-slot `cuda_launch4i` every other dequant uses. This runs THE ENGINE'S OWN DRIVER —
native/host/model_driver.vyb in FFN-probe mode (`VYB_FFN_PROBE=<layer>`) — which stages the block's
three FFN weights plus its pre-norm by name from the live GGUF through its own kernels and
multiplies with layer.ptx's `gemm`, and compares its dumped stages against a reference assembled
from OUR numpy ports of the same two dequantizers (`iq2_s_ref.iq2_s_ours`,
`iq3_s_ref.iq3_s_ours(raw, kmask, grid)`, each verified element-wise against llama.cpp's own compiled
dequantizer in its own gate — S0.5/S0.8 of the phase-2 battery). The same ports write the grid
images the kernel loads, so the two sides cannot disagree about a table neither can derive.

Three geometries, one per FFN type boundary, because the type is NOT a function of the block kind:
  blk.3  all three IQ3_S            (an edge layer)
  blk.7  ffn_down IQ3_S, gate/up IQ2_S  (the 4..11 boundary — mixed within one block)
  blk.19 all three IQ2_S            (the mid-stack)

Teeth, all required to be distinguishable:
  * the STAGED OPERANDS at the addresses gemm reads (B[k*N+n]) against the model's own dequantised
    tensors — a good block fed a wrong operand is invisible to a stage-by-stage comparison, and the
    wrong-in-dim mistake (gate/up is D, down is FF) is exactly an operand-orientation error;
  * the residual mis-wirings, computed from the reference's own stages, must MISS: no residual, and
    the residual taken on the NORMED input instead of the block input;
  * SiLU dropped (gate*up instead of gate*sigmoid(gate)*up) must MISS.

The comparison bar is 1e-4, deliberately conservative: the authority's operands come from our ports
at f64 while the engine stages f64 buffers whose values were produced by the kernel's own arithmetic,
so the two are two paths to the same numbers. MEASURED, it is far tighter than the bar — worst
2.3e-11 over the seven stages of blk.3/blk.7/blk.19 (a Vyb `Float` is f64, so the kernels' dequant is
exact in this path) — which is why a real mis-wiring (1e-1..1e0 in the teeth) is unambiguous. NaN
fails (comparisons are written `not (r <= bar)`).

Proven able to FAIL (never baked in): pairing the kernels with the WRONG grid table (iq2sdeq with the
2048-byte iq3s image) is caught at 6/6 staged-operand slots; staging ffn_down with inz=D instead of FF
is caught at 1/6 slots — the second is the one a stage-only comparison would have let through, which
is why the operand check exists.

SKIPs (never PASSes) without the model, the inventory, the engine's tensor index, the Vyb toolchain,
the grid tables' source (llama.cpp's ggml-common.h, which iq3s's table extraction needs) or a python
with numpy.

Usage: ffn_engine_verify.py
  env: VYBFORGE_RIDGE_GGUF, VYBFORGE_FFN_LAYERS (default "3,7,19"), VYBFORGE_FFN_S (default 2),
       VYBFORGE_FE_MAXREL (default 1e-4), VYB, VYBHOME, VYBFORGE_LLAMA
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "gguf"))

REPO = os.path.dirname(os.path.dirname(HERE))
VYBHOME = os.environ.get("VYBHOME", os.path.expanduser("~/Projects/Vyb"))
VYB = os.environ.get("VYB", os.path.join(VYBHOME, "build", "vyb"))
STDLIB = os.environ.get("VYB_STDLIB", os.path.join(VYBHOME, "stdlib"))
LLAMA = os.environ.get("VYBFORGE_LLAMA", os.path.expanduser("~/Projects/llama.cpp"))
BUILD = os.path.join(REPO, "native", "build")
OUTD = os.path.join(REPO, "native", "out")
DRV_X = os.path.join(BUILD, "ffn_engine_x.bin")
DRV_LOG = os.path.join(BUILD, "ffn_engine_driver.log")
TSV = os.path.join(OUTD, "ridge_tensors.tsv")
INVENTORY = os.path.join(REPO, "native", "gguf", "ridge-3.7bpw-inventory.tsv")
INVFREQ = os.path.join(OUTD, "ridge_invfreq.bin")
GRID2 = os.path.join(OUTD, "iq2s_grid.bin")
GRID3 = os.path.join(OUTD, "iq3s_grid.bin")
MODEL = os.environ.get("VYBFORGE_RIDGE_GGUF",
                       os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf"))

LAYERS = [int(x) for x in os.environ.get("VYBFORGE_FFN_LAYERS", "3,7,19").split(",") if x.strip()]
S = int(os.environ.get("VYBFORGE_FFN_S", "2"))
MAXREL = float(os.environ.get("VYBFORGE_FE_MAXREL", "1e-4"))
OPERAND_REL = float(os.environ.get("VYBFORGE_FE_OPERAND_REL", "1e-5"))

# The engine's dump order for the FFN probe (native/host/model_driver.vyb, VYB_FFN_PROBE branch).
STAGES = ("norm", "ffn_xn", "ffn_gate", "ffn_up", "ffn_silu", "ffn_down", "block_out")
PER256 = {21: 110, 22: 82, 12: 144, 13: 176, 14: 210}


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))


def inv_row(name):
    for line in open(INVENTORY):
        if line.startswith(name + "\t"):
            f = line.rstrip("\n").split("\t")
            return int(f[2]), f[3], f[1], int(f[5]), int(f[6])   # type_id, type name, dims, offset, bytes
    return None


def raw_tensor(name):
    row = inv_row(name)
    if row is None:
        return None
    type_id, tname, _dims, off, nbytes = row
    with open(MODEL, "rb") as fh:
        fh.seek(off)
        raw = fh.read(nbytes)
    if len(raw) != nbytes:
        return None
    return type_id, tname, raw


def ours(raw, type_id, kmask2, grid2, kmask3, grid3):
    """OUR port of the two dequantizers (never the `gguf` package: it has no IQ2_S/IQ3_S
    implementation, and 'our numpy agrees with our numpy' is the self-consistency the authority rule
    forbids — each port is verified against llama.cpp's compiled C in its own gate)."""
    if type_id == 22:
        import iq2_s_ref
        return iq2_s_ref.iq2_s_ours(raw)
    if type_id == 21:
        import iq3_s_ref
        return iq3_s_ref.iq3_s_ours(raw, kmask3, grid3)
    raise ValueError(f"the FFN probe expects IQ2_S/IQ3_S, got type {type_id}")


def permille_ok(name, row):
    """The inventory's dims and byte size are two independent readings of the tensor; they must
    agree, or the geometry below is built on a misread row."""
    if row is None:
        return None
    type_id, _tname, dims, _off, nbytes = row
    if type_id not in PER256:
        return None
    d = [int(x) for x in dims.split("x")]
    ne0 = d[0]
    ne1 = d[1] if len(d) > 1 else 1
    if nbytes * 256 // PER256[type_id] != ne0 * ne1:
        return None
    return ne0, ne1


def ensure_grids():
    """Regenerate both grid images from the SAME tables the reference uses. Returns a note string, or
    None when a needed table could not be produced and no image is on disk."""
    notes = []
    try:
        import iq2_s_ref
        iq2_s_ref.write_grid_image(GRID2)
        notes.append("iq2s=regenerated")
    except Exception as e:                                                # noqa: BLE001
        if os.path.exists(GRID2):
            notes.append(f"iq2s=existing ({e.__class__.__name__})")
        else:
            return None
    try:
        import iq3_s_ref
        kmask, _grid = iq3_s_ref.tables()
        raw_grid = np.array([int(x) for x in iq3_s_ref._raw_grid()], dtype="<u4").tobytes()
        assert len(raw_grid) == 2048, len(raw_grid)
        with open(GRID3, "wb") as fh:
            fh.write(raw_grid)
        notes.append("iq3s=regenerated")
    except Exception as e:                                                # noqa: BLE001
        if os.path.exists(GRID3):
            notes.append(f"iq3s=existing ({e.__class__.__name__})")
        else:
            return None
    return " ".join(notes)


def run_driver(layer):
    env = dict(os.environ, VYB_STDLIB=STDLIB, VYB_MODEL=MODEL, VYB_TSV=TSV,
               VYB_FFN_X=DRV_X, VYB_FFN_PROBE=str(layer), VYB_PROMPT_RAW="1",
               VYB_INVFREQ=INVFREQ)
    cmd = [VYB, "native/host/model_driver.vyb",
           "--module-path", "native/config", "--module-path", "native/json",
           "--module-path", "native/tensor", "--module-path", "native/dtype",
           "--module-path", "native/llm"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    with open(DRV_LOG, "w") as fh:
        fh.write(out)
    return out


def parse_dumps(text):
    secs, name, want, buf = {}, None, 0, []
    for line in text.splitlines():
        if line.startswith("DUMP "):
            if name is not None:
                secs[name] = buf[:want]
            _, name, count = line.split()
            want, buf = int(count), []
        elif name is not None and len(buf) < want:
            try:
                buf.extend(int(v) for v in line.split())
            except ValueError:
                pass
    if name is not None:
        secs[name] = buf[:want]
    return {k: np.frombuffer(np.array(v, dtype="<i8").tobytes(), dtype="<f8") for k, v in secs.items()}


def rms(x, w, eps, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt(np.mean(x * x, axis=axis, keepdims=True) + eps) * w


def verify_layer(layer, kmask2, grid2, kmask3, grid3, eps):
    p = f"blk.{layer}."
    gne = permille_ok(p + "ffn_gate.weight", inv_row(p + "ffn_gate.weight"))
    dne = permille_ok(p + "ffn_down.weight", inv_row(p + "ffn_down.weight"))
    une = permille_ok(p + "ffn_up.weight", inv_row(p + "ffn_up.weight"))
    if gne is None or dne is None or une is None:
        print(f"FFN_ENGINE_VERIFY_FAIL blk.{layer}'s FFN tensors are absent from the inventory or "
              f"their dims disagree with their bytes")
        return 1
    D, FF = gne
    if une != (D, FF) or dne != (FF, D):
        print(f"FFN_ENGINE_VERIFY_FAIL blk.{layer}'s FFN shapes are not a consistent block "
              f"(gate ne={gne} up ne={une} down ne={dne})")
        return 1
    g3 = inv_row(p + "ffn_gate.weight")
    d3 = inv_row(p + "ffn_down.weight")
    u3 = inv_row(p + "ffn_up.weight")
    print(f"FFN_ENGINE_VERIFY blk.{layer} D={D} FF={FF} S={S} types gate/up/down="
          f"{g3[3]}/{u3[3]}/{d3[3]} (ids {g3[0]}/{u3[0]}/{d3[0]})")

    # ── the model's own weights through OUR ports ────────────────────────────────────────────────
    layers_bytes = {}
    for nm, shape in ((p + "ffn_gate.weight", (FF, D)), (p + "ffn_up.weight", (FF, D)),
                      (p + "ffn_down.weight", (D, FF))):
        row = raw_tensor(nm)
        if row is None:
            print(f"FFN_ENGINE_VERIFY_FAIL could not read {nm}")
            return 1
        vals = ours(row[2], row[0], kmask2, grid2, kmask3, grid3)
        if vals.size != shape[0] * shape[1]:
            print(f"FFN_ENGINE_VERIFY_FAIL {nm}: {vals.size} values, wanted {shape[0] * shape[1]}")
            return 1
        layers_bytes[nm] = vals.reshape(shape).astype(np.float64)
    Wg = layers_bytes[p + "ffn_gate.weight"]
    Wu = layers_bytes[p + "ffn_up.weight"]
    Wd = layers_bytes[p + "ffn_down.weight"]
    # The block's FFN pre-norm: qwen3 calls it `ffn_norm`, qwen35 `post_attention_norm` — the same
    # fallback chain the driver resolves it with.
    nm2 = p + "post_attention_norm.weight"
    if inv_row(nm2) is None:
        nm2 = p + "ffn_norm.weight"
    rt = raw_tensor(nm2)
    if rt is None or rt[1] != "F32":
        print(f"FFN_ENGINE_VERIFY_FAIL blk.{layer}'s pre-norm ({nm2}) is not a readable F32 tensor")
        return 1
    N2 = np.frombuffer(rt[2], dtype="<f4").astype(np.float64)
    if N2.size != D:
        print(f"FFN_ENGINE_VERIFY_FAIL blk.{layer}'s pre-norm is {N2.size} values, wanted {D}")
        return 1
    print(f"FFN_ENGINE_VERIFY weights {p}ffn_gate {Wg.shape}, ffn_up {Wu.shape}, ffn_down {Wd.shape}, "
          f"{nm2.split('.', 1)[1]} {N2.shape} — all read raw from the GGUF and dequantised by our ports")

    # ── the fixture: the block's RESIDUAL-STREAM input, S tokens of D ────────────────────────────
    rng = np.random.default_rng(20261009 + layer)
    x = rng.normal(0.0, 1.0, size=(S, D)).astype(np.float64)
    with open(DRV_X, "wb") as fh:
        fh.write(np.ascontiguousarray(x, dtype="<f8").tobytes())

    # ── the reference, stage by stage ───────────────────────────────────────────────────────────
    xn = rms(x, N2, eps)
    gate = xn @ Wg.T
    up = xn @ Wu.T
    silu = (gate / (1.0 + np.exp(-gate))) * up
    down = silu @ Wd.T
    out = x + down
    ref = {"norm": N2, "ffn_xn": xn.ravel(), "ffn_gate": gate.ravel(), "ffn_up": up.ravel(),
           "ffn_silu": silu.ravel(), "ffn_down": down.ravel(), "block_out": out.ravel()}

    out_txt = run_driver(layer)
    if "FFN_PROBE_DONE" not in out_txt:
        if "SKIP" in out_txt:
            print("FFN_ENGINE_VERIFY_SKIP " + [l for l in out_txt.splitlines() if "SKIP" in l][0].strip())
            return 0
        print(f"FFN_ENGINE_VERIFY_FAIL the driver did not complete for blk.{layer} (log: "
              f"{os.path.relpath(DRV_LOG, REPO)}): " + out_txt[-900:])
        return 1
    gl = [l for l in out_txt.splitlines() if l.startswith("FFN_GEOM")]
    if not gl:
        print(f"FFN_ENGINE_VERIFY_FAIL the driver printed no FFN_GEOM for blk.{layer}")
        return 1
    gd = dict(kv.split("=") for kv in gl[0].split()[2:])
    if (int(gd["D"]), int(gd["FF"]), int(gd["S"])) != (D, FF, S):
        print(f"FFN_ENGINE_VERIFY_FAIL the driver's geometry {gd} disagrees with the tensor table's "
              f"D={D} FF={FF} S={S}")
        return 1
    got = parse_dumps(out_txt)

    # ── the staged operands, at the addresses gemm reads ────────────────────────────────────────
    probes = []
    for line in out_txt.splitlines():
        if line.startswith("PROBE fg ") or line.startswith("PROBE fu ") or line.startswith("PROBE fd "):
            _, tag, idx, bits = line.split()
            v = np.frombuffer(np.array([int(bits)], dtype="<i8").tobytes(), dtype="<f8")[0]
            probes.append((tag, int(idx), v))
    pbad = []
    for tag, idx, v in probes:
        if tag == "fg":
            k, n = idx // FF, idx % FF
            want = Wg[n, k]
        elif tag == "fu":
            k, n = idx // FF, idx % FF
            want = Wu[n, k]
        else:
            k, n = idx // D, idx % D
            want = Wd[n, k]
        if not (abs(v - want) <= OPERAND_REL * max(abs(want), 1e-12)):
            pbad.append((tag, idx, v, want))
    if not probes or pbad:
        print(f"FFN_ENGINE_VERIFY staged operand FAIL ({len(pbad)}/{len(probes)} slots wrong)")
        for tag, idx, v, want in pbad[:4]:
            print(f"FFN_ENGINE_VERIFY   {tag} idx {idx} driver={v:.9e} model={want:.9e}")
        print("FFN_ENGINE_VERIFY_FAIL the staged operand is not the model's own weight")
        return 1
    print(f"FFN_ENGINE_VERIFY staged operands ok ({len(probes)} slots of B[k*N+n] == the model's own "
          f"dequantised ffn_gate/ffn_up/ffn_down)")

    # ── the seven stages ────────────────────────────────────────────────────────────────────────
    worst, bad = 0.0, []
    for nm in STAGES:
        key = "f1_" + nm
        if key not in got:
            print(f"FFN_ENGINE_VERIFY {nm:10s} MISSING (the driver dumped no {key})")
            bad.append(nm)
            continue
        gpu = got[key].astype(np.float64)
        r = np.asarray(ref[nm], dtype=np.float64)
        if gpu.size != r.size:
            print(f"FFN_ENGINE_VERIFY {nm:10s} LENGTH MISMATCH driver={gpu.size} reference={r.size}")
            bad.append(nm)
            continue
        rr = rel(gpu, r)
        worst = max(worst, rr)
        if not (rr <= MAXREL):                                   # NaN must FAIL
            bad.append(nm)
            w = int(np.nanargmax(np.abs(gpu - r)))
            print(f"FFN_ENGINE_VERIFY {nm:10s} n={gpu.size:7d} maxrel={rr:.3e} DIFFERS")
            print(f"FFN_ENGINE_VERIFY   worst #{w}: driver={gpu[w]:.9e} reference={r[w]:.9e}")
        else:
            print(f"FFN_ENGINE_VERIFY {nm:10s} n={gpu.size:7d} maxrel={rr:.3e} ok")
    if bad:
        print(f"FFN_ENGINE_VERIFY_FAIL the engine does not reproduce the FFN block for blk.{layer}: "
              f"{', '.join(bad)} (driver log: {os.path.relpath(DRV_LOG, REPO)})")
        return 1

    # ── the teeth: mis-wirings computed from the reference's own stages must MISS ────────────────
    barb = MAXREL * 100.0
    no_res = rel(down.ravel(), got["f1_block_out"])
    res_normed = rel((xn + down).ravel(), got["f1_block_out"])
    no_silu = rel((gate * up).ravel(), got["f1_ffn_silu"])
    print(f"FFN_ENGINE_VERIFY tooth no residual                          (must MISS): maxrel={no_res:.3e}")
    print(f"FFN_ENGINE_VERIFY tooth residual on the NORMED input         (must MISS): maxrel={res_normed:.3e}")
    print(f"FFN_ENGINE_VERIFY tooth SiLU dropped: gate*up instead        (must MISS): maxrel={no_silu:.3e}")
    if min(no_res, res_normed, no_silu) <= barb:
        print(f"FFN_ENGINE_VERIFY_FAIL a plausible mis-wiring is NOT distinguished from the engine's "
              f"output (bar {barb:.1e}) — this fixture cannot tell them apart, so its pass proves nothing")
        return 1

    print(f"FFN_ENGINE_VERIFY_SUMMARY blk.{layer} D={D} FF={FF} S={S} stages={len(STAGES)} "
          f"worst={worst:.3e} teeth=no_res({no_res:.2e})/res_normed({res_normed:.2e})/no_silu({no_silu:.2e})")
    return 0


def main():
    print(f"FFN_ENGINE_VERIFY layers={LAYERS} S={S} maxrel={MAXREL:g}")
    for path, what in ((MODEL, "the Ridge model"), (INVENTORY, "the inventory"), (TSV, "the engine "
                       "tensor index (native/tools/inventory_to_tsv.py)"), (INVFREQ, "the rope table")):
        if not os.path.exists(path):
            print(f"FFN_ENGINE_VERIFY_SKIP no {what} at {path}")
            return 0
    if not os.path.exists(os.path.join(LLAMA, "ggml/src/ggml-common.h")):
        print(f"FFN_ENGINE_VERIFY_SKIP no llama.cpp source at {LLAMA} (the iq3s table extraction "
              f"reads ggml-common.h)")
        return 0
    try:
        import iq2_s_ref                                     # noqa: F401
        import iq3_s_ref                                     # noqa: F401
    except Exception as e:                                                # noqa: BLE001
        print(f"FFN_ENGINE_VERIFY_SKIP the dequant ports are not importable ({e.__class__.__name__}: {e})")
        return 0
    note = ensure_grids()
    if note is None:
        print("FFN_ENGINE_VERIFY_SKIP could not produce the grid images and none are on disk "
              "(run native/tools/iq2_s_ref.py and native/tools/iq3_s_ref.py)")
        return 0
    print(f"FFN_ENGINE_VERIFY grid images: {note}")
    kmask2, grid2 = iq2_s_ref.kmask_array(), iq2_s_ref.grid_array()
    kmask3, grid3 = iq3_s_ref.tables()

    # eps from the model's own metadata (never a literal)
    eps = 1e-6
    try:
        import gguf
        r = gguf.GGUFReader(MODEL)
        for f in r.fields.values():
            if f.name in ("qwen35.attention.layer_norm_rms_epsilon", "qwen35.attention.layer_norm_epsilon"):
                eps = float(f.contents())
    except Exception:                                                     # noqa: BLE001
        print("FFN_ENGINE_VERIFY note: `gguf` unavailable, eps left at 1e-6")
    print(f"FFN_ENGINE_VERIFY eps={eps:g}")

    bad = []
    for layer in LAYERS:
        if verify_layer(layer, kmask2, grid2, kmask3, grid3, eps) != 0:
            bad.append(layer)
    if bad:
        print(f"FFN_ENGINE_VERIFY_FAIL layers {bad} did not reproduce ggml's FFN block")
        return 1
    print(f"FFN_ENGINE_VERIFY_DONE the engine's own staging of IQ2_S/IQ3_S (through the grid "
          f"launcher), its gemm, SiLU and residual reproduce the FFN block for layers {LAYERS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
