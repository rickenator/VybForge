# native/mel — the voice loop's audio front-end (P1, done and gated)

Turns 16 kHz mono audio into the 80 x 3000 log-mel "image" Whisper consumes. This
is the only piece between the microphone (`native/audio`) and speech recognition,
and it is Vyb-native: the FFT, the mel filterbank, the framing and the
normalisation are all computed here.

Architecture and phase plan: `../../doc/VOICE-LOOP.md`.

## Files

| File | Role |
|---|---|
| `fft.vyb` | radix-2 complex FFT + Bluestein's algorithm for arbitrary N (the 400-point window) |
| `mel.vyb` | Slaney mel filterbank, reflect-padded framing, Hann, mel projection, log/floor/scale |
| `wav.vyb` | 16-bit PCM WAV reader (the gates and the loop's replay path need a file input) |
| `fft_probe.vyb`, `mel_probe.vyb` | the two gates |
| `oracle_mel.py` | writes the reference artifacts into `gold/` (transformers' own feature extractor) |
| `check_fft.py`, `check_mel.py` | the checkers (run by `build.sh`) |
| `build.sh` | `fft` / `mel` / `gold` / `all` |

## Run

```sh
./build.sh            # both gates
./build.sh gold       # regenerate gold/ from the oracle (needs ~/miniconda/bin/python)
```

## Gates and their evidence

`fft` — 400-point real DFT of two vectors (a random one and a real windowed frame
from the fixture) against `numpy.fft.rfft` of the same quantised input:

    random: bins=201 max|diff|=1.216e-06 rel=8.1e-08   over tol: 0
    frame:  bins=201 max|diff|=1.466e-06 rel=4.0e-08   over tol: 0

`mel` — the 80x201 Slaney filterbank and the full 80x3000 log-mel against
`transformers.WhisperFeatureExtractor` (which uses librosa's filter algorithm and
torch.stft) on jfk.wav:

    filters: 16080 values  max|diff|=9.997e-13   over 1e-9: 0     (i.e. exact)
    mel:     240000 values max|diff|=1.336e-05 rel=9.1e-06 over 2e-05: 0
             ref range [-0.5387, 1.4613]   vyb range [-0.5387, 1.4613]

The residual 1.3e-5 is float32 reference rounding (torch computes the STFT in
float32; Vyb computes it in float64), not a disagreement about the algorithm.

## Reference semantics implemented (get these wrong and STT degrades silently)

* audio is zero-padded/truncated to exactly 480000 samples (30 s) *before* framing
* reflect padding by n_fft/2 = 200 with torch's semantics (the edge sample is NOT
  repeated), so frame *t* starts at `t*160`
* periodic Hann (`0.5 - 0.5*cos(2*pi*i/400)`), not the symmetric window
* power spectrum |X|^2 over bins 0..200, then the 80 Slaney filters
* `log10(clamp(x, 1e-10))`, then floor at `global_max - 8`, then `(x + 4) / 4`
* exactly 3000 frames: the reference's `stft[..., :-1]` drops the frame that the
  30 s padding would otherwise add, so we never compute it

## Performance (measured on godzilla)

* Vyb CPU arithmetic is ~107 MFLOP/s (128x128x128 GEMM benchmark, `-O2`): fine for
  this front-end, **far too slow for an encoder** — Whisper-tiny's encoder is
  ~28 GFLOP, i.e. ~4 minutes on the host CPU. Neural work goes to the GPU kernels.
* The full 3000-frame mel takes ~3-4 s today, dominated by per-frame overhead.
  Two known wins, not yet taken: (a) a 30 s input is mostly silence — frames that
  no longer touch real audio are exactly the floor constant, so only
  `(n_audio + 400)/160` frames need computing (~7x for a 4 s utterance);
  (b) the mel projection skips zero filter weights but still walks all 201 bins
  per band per frame.

## Verification channel (why the probes print integers)

Vyb's `Float.to_string()` keeps only 6 significant digits, so it cannot carry a
1e-5 comparison. Everything is therefore printed as `(value * 10^k) as Int` and
compared in Python against float32 reference files. The gold artifacts are raw
`.f32` + a `.shape` sidecar because Vyb has no float container reader (and no
float bit-cast) — the comparison runs oracle-side for that reason.
