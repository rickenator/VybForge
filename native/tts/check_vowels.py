"""Gate L4's synthesis core: does the audio have the spectrum it was told to have?

Reads the probe's SPEC lines (what the Vyb side was asked to synthesise) and
MEASURES the written WAVs with numpy. Nothing here reads Vyb code or Vyb's
arithmetic -- the artifact under test is the sample stream.

WHAT IS COMPARED, and why peaks rather than the whole curve: any envelope estimate
smooths the spectrum, so its valleys come out shallower than the analytic response
by however much the smoothing costs -- 5 dB here, monotonically worse with
frequency, until the measurement floor takes over about 70 dB down. A curve
difference therefore measures the estimator, not the synthesiser. That was tried and
it failed on all six vowels; the fix was a better measurement, not a wider
tolerance. Peaks do not have that problem: they sit where the filters put them in
both curves, so the top three of each are matched and compared.

The predicted curve is the cascade response RE-DERIVED in numpy from the spec
(including the bandwidths the model exports), so resonator interaction appears on
both sides of the comparison.
"""
import sys, wave
import numpy as np

out_path = sys.argv[1]
lines = [l.rstrip("\n") for l in open(out_path)]

POS_TOL_HZ = 120.0     # or 8% of the peak's own frequency, whichever is larger
POS_TOL_FRAC = 0.08
LVL_TOL_DB = 3.0       # relative peak level
F0_TOL = 0.08
LIFTER_S = 0.0042
GRID = np.arange(200.0, 5000.0, 10.0)

specs, names, files = {}, [], {}
pending = None
i = 0
while i < len(lines):
    if lines[i] == "SPEC":
        v = [int(x) for x in lines[i + 1:i + 9]]
        pending = {"f1": v[0] / 100.0, "f2": v[1] / 100.0, "f3": v[2] / 100.0,
                   "f0": v[3] / 100.0, "ms": v[4],
                   "bw1": v[5] / 100.0, "bw2": v[6] / 100.0, "bw3": v[7] / 100.0}
        i += 9
        continue
    if lines[i] == "NAME":
        names.append(lines[i + 1])
        specs[lines[i + 1]] = pending
        i += 2
        continue
    if lines[i] == "FILE":
        files[names[-1]] = lines[i + 1]
        i += 2
        continue
    i += 1

def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected mono 16-bit, got {w.getnchannels()}ch "
                             f"{w.getsampwidth() * 8}bit")
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    return sr, data / 32768.0

def lpc_env_db(x, sr, p=12, grid=None):
    """All-pole (LPC) spectral envelope of the segment, via Levinson-Durbin.

    This is the estimator that replaced cepstral liftering. Liftering smooths the
    log spectrum, which means the LOG OF THE NOISE FLOOR gets smoothed in too: the
    floor is broadband in the log domain, so the resulting envelope is filled in and
    the weak peaks read high (measured F2 of /i/ came out 10 dB above the predicted
    response, with the error growing as the floor got closer). LPC fits the
    autocorrelation instead -- a least-squares fit in the time domain -- so it does
    not see the floor the same way, and it is a genuinely different estimator from
    the resonator arithmetic under test rather than a restatement of it.

    No pre-emphasis: standard formant analysis applies 1-0.97z^-1 to flatten the
    -6 dB/octave tilt of a real glottal source, and this source is a flat impulse
    train with no such tilt to remove.
    """
    xw = (x - x.mean()) * np.hamming(len(x))
    p = min(p, len(xw) - 1)
    full = np.correlate(xw, xw, mode="full")[len(xw) - 1:]
    r = full[:p + 1] / len(xw)
    # Levinson-Durbin, in the convention where the model is x[n] = -sum a_j x[n-j]:
    #   k = -(r[i] + sum_{j<i} a_j r[i-j]) / E ;  a_i = k ;  a_j += k a_{i-j} ;  E *= (1-k^2)
    # The first attempt had the sign inverted inside the dot product and the update
    # written as a difference: it converged to a nonsense envelope whose peaks were
    # at 2820 and 4510 Hz for /a/, i.e. no formant structure at all.
    a = np.zeros(p + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, p + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1:0:-1])) / (e + 1e-30)
        a_new = a.copy()
        a_new[i] = k
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a = a_new
        e *= (1.0 - k * k)
    g = np.sqrt(max(e, 1e-30))
    w = 2 * np.pi * grid / sr
    A = np.ones(len(grid), dtype=complex)
    for j in range(1, p + 1):
        A += a[j] * np.exp(-1j * w * j)
    return 20 * np.log10(g / (np.abs(A) + 1e-30))

def envelope(x, sr, n=8192, lifter_s=LIFTER_S):
    """Spectral envelope: cepstral liftering.

    The lifter cut sits between two constraints: ABOVE ~1/(formant spacing) so two
    formants 270 Hz apart stay separate peaks (/o/ at 570/840), and BELOW the pitch
    quefrency 1/f0 = 8.3 ms at 120 Hz so the harmonics are removed. At 2 ms the
    lifter merged /a/'s F1 and F2 (360 Hz apart) into a single peak at 873 Hz.
    """
    win = np.hanning(len(x))
    logmag = np.log(np.abs(np.fft.rfft(x * win, n=n)) + 1e-12)
    cep = np.fft.irfft(logmag, n=n)
    keep = int(lifter_s * sr)
    cep[keep:n - keep] = 0.0
    return np.fft.rfft(cep, n=n).real, np.fft.rfftfreq(n, 1.0 / sr)

def predicted_env_db(sp, sr, freqs, n=8192):
    """The cascade response, RE-DERIVED here, then taken through the SAME lifter.

    H(w) = A / (1 - B e^-jw - C e^-2jw) per formant, multiplied over formants, from
    the Klatt relationship rather than read out of the Vyb source:
    C = -exp(-2 pi BW/fs), B = 2 exp(-pi BW/fs) cos(2 pi F/fs), A = 1 - B - C.

    Liftering the prediction as well as the measurement is the whole point: the
    estimator flattens peak LEVELS toward each other (valleys cannot be measured
    through a smoother), so comparing a smoothed curve against an analytic one
    charges the synthesiser for the measurement. Positions agreed to a few Hz either
    way; levels were off by 3-6 dB until both sides went through the same pipeline.
    """
    H = np.ones(len(freqs), dtype=complex)
    for F, BW in ((sp["f1"], sp["bw1"]), (sp["f2"], sp["bw2"]), (sp["f3"], sp["bw3"])):
        c = -np.exp(-2 * np.pi * BW / sr)
        b = 2 * np.exp(-np.pi * BW / sr) * np.cos(2 * np.pi * F / sr)
        a = 1 - b - c
        w = 2 * np.pi * freqs / sr
        H *= a / (1 - b * np.exp(-1j * w) - c * np.exp(-2j * w))
    logmag = np.log(np.abs(H) + 1e-12)
    cep = np.fft.irfft(logmag, n=n)
    keep = int(LIFTER_S * sr)
    cep[keep:n - keep] = 0.0
    return 20 * np.log10(np.exp(np.fft.rfft(cep, n=n).real))

def est_f0(x, sr, lo=60.0, hi=400.0):
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= (ac[0] + 1e-12)
    lo_l, hi_l = int(sr / hi), int(sr / lo)
    seg = ac[lo_l:hi_l]
    return float(sr) / (lo_l + int(np.argmax(seg))) if len(seg) else 0.0

def top_peaks(curve, grid, k=3, min_sep_hz=120.0):
    """The k strongest local maxima at least min_sep_hz apart, as (freq, db)."""
    idx = [j for j in range(1, len(curve) - 1)
           if curve[j] >= curve[j - 1] and curve[j] >= curve[j + 1]]
    idx.sort(key=lambda j: -curve[j])
    out = []
    for j in idx:
        f = float(grid[j])
        if all(abs(f - g) >= min_sep_hz for (g, _) in out):
            out.append((f, float(curve[j])))
        if len(out) == k:
            break
    return out

bad = 0
for name in names:
    sp = specs[name]
    sr, x = read_wav(files[name])
    seg = x[len(x) // 2:len(x) // 2 + 2048]
    # measured: LPC envelope of the audio. predicted: the cascade response with the
    # same liftering the measurement used to need (harmless now, kept so both sides
    # are still passed through one pipeline).
    meas_db = lpc_env_db(seg, sr, 12, GRID)
    freqs = np.fft.rfftfreq(8192, 1.0 / sr)
    pred_db = predicted_env_db(sp, sr, freqs)
    p = np.interp(GRID, freqs, pred_db)
    m = meas_db
    m_rel, p_rel = m - m.max(), p - p.max()
    f0 = est_f0(x, sr)

    pm, pp = top_peaks(m_rel, GRID), top_peaks(p_rel, GRID)
    f0ok = abs(f0 - sp["f0"]) <= F0_TOL * sp["f0"]
    pairs = []
    ok = f0ok and len(pm) >= 3 and len(pp) >= 3
    if len(pm) >= 3 and len(pp) >= 3:
        for (fp, lp) in pp[:3]:
            d_f, fm, lm = min((abs(fm - fp), fm, lm) for (fm, lm) in pm)
            good = d_f <= max(POS_TOL_HZ, POS_TOL_FRAC * fp) and abs(lm - lp) <= LVL_TOL_DB
            ok = ok and good
            pairs.append((fp, fm, good))
    if not ok:
        bad += 1
    info = "  ".join(f"{fp:5.0f}->{fm:5.0f}{'' if good else ' BAD'}" for (fp, fm, good) in pairs)
    print(f"  [{'ok  ' if ok else 'FAIL'}] /{name}/ peaks (pred->meas) {info}"
          f"   f0 {sp['f0']:.0f}->{f0:.1f}{'' if f0ok else ' BAD'}")

if bad:
    print(f"VOWEL_GATE: FAIL ({bad}/{len(names)})")
    sys.exit(1)
print(f"VOWEL_GATE: OK  ({len(names)} vowels: top-3 envelope peaks within "
      f"max({POS_TOL_HZ:.0f} Hz, {POS_TOL_FRAC:.0%}) and {LVL_TOL_DB:.0f} dB of the "
      f"cascade response re-derived in numpy; f0 within {F0_TOL:.0%})")
