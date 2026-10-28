# native/exl3 — P1: EXL3 4bpw dequant for the GLM-5.3 routed experts

Part of `doc/ROADMAP-GLM53-ON-VYB.md` (phase P1). Target: the routed experts of
GLM-5.3-Flash are the ONLY EXL3-quantized tensors (`quantization_config.scope =
glm53_routed_experts_only`, bits=4, codebook mcg, head_bits 16); everything else
is bf16 safetensors.

## Status

- `EXL3_REF_VERIFY: OK` — numpy reference (`exl3_ref.dequant_exl3`) matches the
  exllamav3 oracle (`ext.reconstruct_had_slice`, the exact kernel TensorFold
  serves GLM-5.3-Flash with) on REAL expert weights: max abs diff 1.2e-4 on
  values with std 0.0117 (pure fp16-butterfly rounding from the oracle's column
  transform; the rotated-basis compare in `bisect2.py` is bit-exact, 0.0).
- Remaining for P1: the Vyb port (`native/kernels/deq_exl3.vyb` + driver),
  gated against this reference (`EXL3_VYB_MATCH`).

## Format (pinned from exllamav3 source + oracle probes)

Per weight matrix (fixture: `layers.3.mlp.experts.0.down_proj`, [in 2048, out 4096]):

  trellis  int16 [in/16, out/16, 64]   64 uint16 = 1024 bits = 256 nibbles / 16x16 tile
  suh      fp16   [in]                 per-row (k) scale+sign
  svh      fp16   [out]                per-col (n) scale+sign
  mcg      int32  [1]                  stores 0xCBAC1FED — the mcg magic constant itself

Reconstruction: `W = diag(suh) (H/8) What (H/8) diag(svh)` per 128x128 block
(H = natural-order Sylvester Hadamard-128, 1/8 = 1/sqrt(128) = r_scale).

Value decode — each 16-bit window (codebook.cuh `decode_3inst`, cb=1):
  prod = (w16 * 0xCBAC1FED) mod 2^32
  y    = (prod & 0x8fff8fff) ^ 0x3b603b60      (lop3 0x6a = (a&b)^c)
  val  = fp16(y & 0xffff) + fp16(y >> 16)      (fp16 add)
Bit-exact vs the oracle (all-zero trellis -> 1.84375 both sides).

Window->slot placement: NOT fully derivable by hand — the 128 b-windows
(j%8 in 0..3, read from the later word) follow the sliding-window model, but the
128 s-windows (j%8 in 4..7, extracted through the funnel-shifted `s`) do not.
The verified composition lives in `tile_slot_composition.json`
(slot (r,c) -> [(nibble, bitpos) x4]), pinned by `build_map.py` +
`finalize_map.py` against the oracle with random-fille probes (3 seeds, all 256
slots exact). `nibble_slot_map.npy` / `s_slot_composition.json` are intermediate
artifacts of that derivation.

## Oracle setup (godzilla)

  uv venv ~/venvs/exl3 --python 3.12
  uv pip install --python ~/venvs/exl3/bin/python exllamav3 numpy
  # torch 2.14.1+cu130 comes with it; JIT-builds exllamav3_ext on first import

Gotcha 1: unset PYTHONPATH (the Hermes env leaks a py3.14 numpy into the venv).
Gotcha 2: torch's JIT compiles some of exllamav3's `libtorch/*.cpp` with g++;
their `util.cuh` uses `__halves2half2` in host-visible constructors and fails.
Patch the venv copy of `exllamav3_ext/util.cuh`: wrap the three multi-arg
constructors (half4, bfloat164, half8) in `#ifdef __CUDACC__` with empty host
fallbacks — no .cpp file actually calls them (verified by grep).
Gotcha 3: the build takes ~10 min; run it via `systemd-run --user` so nothing
reaps it (see `~/.cache/exl3-build/import.log`).

## Files

  exl3_ref.py           reference dequant + mcg decode + Hadamard (verification-only Python)
  verify_ref.py         EXL3_REF_VERIFY gate: reference vs oracle on the real fixture
  build_map.py          empirical slot-composition extraction (256 single-nibble probes)
  finalize_map.py       merge b/s compositions, verify on random probes, save the map
  probe_swindows.py     the probe run that exposed the s-window structure
  bisect2.py            rotated-basis bit-exactness check (0.0 diff)
  probe_hadamard.py     Hadamard-ordering hypothesis test
  tile_slot_composition.json   THE verified map (committed)
  fixture/              real tensor bytes from lefty:/srv/models/hf (NOT committed)

Regenerate the fixture: `ssh lefty python3 /tmp/extract_exl3_fixture.py
$SNAP /tmp/exl3_fixture` (script in the Hermes scratch dir) and scp the files
into `fixture/`. Weights are never committed (MODELS.md rule).
