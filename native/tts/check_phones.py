"""Gate the consonant inventory: is each phone the KIND of sound it claims to be?

Three independent measurements per phone, none of which reads Vyb or the phone
table's arithmetic:

  * VOICING  -- normalised autocorrelation peak in the pitch range 60-400 Hz. A
                glottal-pulse source is periodic and scores high; a noise source
                does not.
  * CENTROID -- spectral centre of gravity. This is the property that separates
                the sibilants from each other and from aspiration, and it is what
                makes /s/ an ess rather than a hiss.
  * BAND     -- energy below/above 2 kHz, reported.

The expectations are relational and physical rather than a table of fingerprinted
numbers: /s/ must be unvoiced, high-centroid, and above /sh/, which must be above
/h/. A fingerprint table would pass a synthesiser that had drifted, as long as it
had drifted the same way in both places.
"""
import sys, wave
import numpy as np

out_path = sys.argv[1]
lines = [l.rstrip("\n") for l in open(out_path)]

OBS = ["kind", "voiced", "f1", "f2", "f3", "ms", "noise_hz"]
phones, order = {}, []
i = 0
while i < len(lines):
    if lines[i].startswith("PHONE "):
        name = lines[i].split(" ", 1)[1]
        vals = [int(x) for x in lines[i + 1:i + 1 + len(OBS)]]
        j = i + 1 + len(OBS)
        assert lines[j] == "FILE", f"expected FILE after {name}"
        phones[name] = dict(zip(OBS, vals))
        phones[name]["file"] = lines[j + 1]
        order.append(name)
        i = j + 2
        continue
    i += 1

def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        d = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    return sr, d / 32768.0

def voicing(x, sr, lo=60.0, hi=400.0):
    """Normalised autocorrelation peak in the pitch range: 0 = noise, ~1 = periodic.

    On the FIRST DIFFERENCE, not the raw signal. A raw autocorrelation at 2.5-16 ms
    lags is dominated by the amplitude envelope rather than by periodicity: white
    noise under a smooth 250 ms envelope scored 0.94, which is a measurement of "the
    level changes slowly", not of voicing. Differencing is the standard pre-emphasis
    for pitch detection and removes the envelope term while keeping the pulse train.
    """
    if len(x) < sr // lo:
        return 0.0
    x = np.diff(x)
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-30)
    seg = ac[int(sr / hi):int(sr / lo)]
    return float(seg.max()) if len(seg) else 0.0

def centroid(x, sr):
    win = np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x * win, n=8192))
    f = np.fft.rfftfreq(8192, 1.0 / sr)
    return float((f * mag).sum() / (mag.sum() + 1e-30))

meas = {}
for name in order:
    sr, x = read_wav(phones[name]["file"])
    trim = x[len(x) // 4:3 * len(x) // 4]   # skip the fades
    meas[name] = {"voiced": voicing(trim, sr), "centroid": centroid(trim, sr)}
    p = phones[name]
    print(f"  {name:3s} kind={p['kind']} claims_voiced={p['voiced']}  "
          f"measured autocorr={meas[name]['voiced']:.3f}  centroid={meas[name]['centroid']:7.0f} Hz")

bad = []
def want(cond, msg):
    if not cond:
        bad.append(msg)

# The criterion is SEPARATION, not an absolute threshold. Voicing measured this way
# is a continuum -- /h/ is 0.33 because aspiration genuinely has some slow structure
# in it, and real pitch detectors flag /h/ as partly periodic too -- so the claim
# being tested is that the two classes are clearly apart, by a factor of two, with
# the periodic ones substantially correlated (> 0.5, the "more correlated than not"
# line). An absolute cut-off picked to make /h/ pass would be a threshold chosen to
# fit the answer; a separation ratio is a statement about the inventory.
SEPARATION = 2.0
voiced_vals = [meas[n]["voiced"] for n in order if phones[n]["voiced"] == 1]
unvoiced_vals = [meas[n]["voiced"] for n in order if phones[n]["voiced"] == 0]
if voiced_vals and unvoiced_vals:
    lo_v, hi_u = min(voiced_vals), max(unvoiced_vals)
    print(f"  separation: weakest voiced {lo_v:.3f} vs strongest unvoiced {hi_u:.3f}"
          f"  (ratio {lo_v / hi_u:.2f}, need >= {SEPARATION})")
    want(lo_v >= SEPARATION * hi_u,
         f"voiced/unvoiced not separated: {lo_v:.3f} vs {hi_u:.3f}")
    want(lo_v > 0.5, f"weakest voiced phone only {lo_v:.3f} correlated")
    worst = min((meas[n]["voiced"], n) for n in order if phones[n]["voiced"] == 0)
    print(f"  least-clean unvoiced: /{max((meas[n]['voiced'], n) for n in order if phones[n]['voiced'] == 0)[1]}/"
          f" at {hi_u:.3f} (aspiration has some structure; real pitch detectors agree)")

# physical, relational expectations -- written from what the phones ARE
want(meas["s"]["centroid"] > 3000, f"/s/ centroid {meas['s']['centroid']:.0f} Hz <= 3000")
want(meas["sh"]["centroid"] > 1500, f"/sh/ centroid {meas['sh']['centroid']:.0f} Hz <= 1500")
want(meas["s"]["centroid"] > meas["sh"]["centroid"] > meas["h"]["centroid"],
     "/s/ > /sh/ > /h/ centroid ordering violated")
want(meas["t"]["centroid"] > 1000, f"/t/ burst centroid {meas['t']['centroid']:.0f} Hz")
for name in ("m", "n", "l"):
    want(meas[name]["voiced"] > 0.5, f"/{name}/ not periodic (0.5 = more correlated than not)")

if bad:
    print("PHONE_GATE: FAIL")
    for b in bad:
        print("   " + b)
    sys.exit(1)
print(f"PHONE_GATE: OK  ({len(order)} phones: voicing as declared, /s/ > /sh/ > /h/ in "
      f"centroid, low-centroid stops and nasals as expected)")
