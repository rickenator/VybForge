#!/usr/bin/env python3
"""oracle_mel.py — the gold reference for the Vyb-native mel front-end.

Not part of the runtime: this exists only to produce reference artifacts and to
check the Vyb implementation against them (the VybForge rule — Python verifies,
never runs).

    ~/llm-venv/bin/python oracle_mel.py            # writes gold/ under this dir

Writes:
  gold/pcm.f32        the 16 kHz mono samples the mel was computed from
  gold/mel.npy        mel spectrogram, shape (80, T)  — WhisperFeatureExtractor
  gold/mel_filters.npy the 80x201 Slaney mel filterbank transformers used
  gold/stft_power.npy |STFT|^2 of the first frames, for the FFT to be checked alone
  gold/ref_text.txt   reference transcription of the fixture (whole file)
  gold/pcm_head.f32   first 1200 samples (a small, fast debug case)
"""
import json
import os

import wave

import numpy as np
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
from transformers.audio_utils import mel_filter_bank

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold")
FIXTURE = os.path.join(HERE, "..", "..", "fixtures", "audio", "jfk.wav")
MODEL = "openai/whisper-tiny.en"

os.makedirs(GOLD, exist_ok=True)

# stdlib `wave` reader: the fixture is a plain PCM WAV, so no soundfile/librosa
# dependency is needed on the oracle side either.
with wave.open(FIXTURE, "rb") as w:
    sr = w.getframerate()
    nch = w.getnchannels()
    width = w.getsampwidth()
    raw = w.readframes(w.getnframes())
assert sr == 16000, f"fixture must be 16 kHz, got {sr}"
assert width == 2, f"expected 16-bit PCM, got {width * 8}-bit"
audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
if nch > 1:
    audio = audio.reshape(-1, nch).mean(axis=1)
audio = np.ascontiguousarray(audio, dtype=np.float32)
print("audio:", audio.shape, "sr", sr, "peak", float(np.abs(audio).max()))

proc = AutoProcessor.from_pretrained(MODEL)
fe = proc.feature_extractor
print("feature extractor:", json.dumps(
    {k: getattr(fe, k) for k in ("n_fft", "hop_length", "chunk_length", "n_samples", "nb_max_frames", "feature_size", "sampling_rate")}))

# --- the filters, computed the same way transformers does it -----------------
filters = mel_filter_bank(
    num_frequency_bins=1 + fe.n_fft // 2,
    num_mel_filters=fe.feature_size,
    min_frequency=0.0,
    max_frequency=fe.sampling_rate // 2,
    sampling_rate=fe.sampling_rate,
    norm="slaney",
    mel_scale="slaney",
)
print("filters:", filters.shape, "sum per bin range", filters.sum(1).min(), filters.sum(1).max())

# --- the mel, exactly as the feature extractor produces it -------------------
feats = fe(audio, sampling_rate=sr, return_tensors="np")["input_features"]   # (1, 80, 3000)
mel = feats[0]
print("mel:", mel.shape, mel.dtype, "min", mel.min(), "max", mel.max())

np.save(os.path.join(GOLD, "mel.npy"), mel)
np.save(os.path.join(GOLD, "mel_filters.npy"), filters)

# Raw little-endian float32 binaries + shape sidecars: the Vyb side has no .npy
# reader, and a flat f32 file needs nothing but io::read_at.
def dump_f32(name, arr):
    a = np.ascontiguousarray(arr, dtype="<f4")
    a.tofile(os.path.join(GOLD, name + ".f32"))
    with open(os.path.join(GOLD, name + ".shape"), "w") as f:
        f.write(" ".join(str(d) for d in a.shape) + "\n")

dump_f32("mel", mel)
dump_f32("mel_filters", filters)
audio.tofile(os.path.join(GOLD, "pcm.f32"))
audio[:1200].tofile(os.path.join(GOLD, "pcm_head.f32"))

# --- STFT power of the first ~2.56 s, so the FFT stage can be checked alone ---
seg = torch.from_numpy(audio[:40960].copy())
win = torch.hann_window(fe.n_fft, periodic=True)
stft = torch.stft(seg, n_fft=fe.n_fft, hop_length=fe.hop_length, window=win,
                  center=True, pad_mode="reflect", return_complex=True, normalized=False)
power = (stft.abs() ** 2).numpy()
print("stft power:", power.shape)
np.save(os.path.join(GOLD, "stft_power.npy"), power)
dump_f32("stft_power", power)

# A single 400-point frame's complex spectrum and a random input vector: the
# smallest possible target for the Vyb FFT, checked before anything else depends
# on it. Same window/padding as the real front-end (reflect, center).
# Offset 40000, not 0: the clip opens with digital silence, and the first frame
# of a reflect-padded silent region is exactly zero — useless as an FFT target.
frame = np.zeros(400, dtype=np.float32)
seg = np.pad(audio, 200, mode="reflect")[40000:40400]
win = np.hanning(401)[:400].astype(np.float32)     # periodic Hann == torch.hann_window
frame[:] = seg * win
spec = np.fft.rfft(frame.astype(np.float64), n=400)
dump_f32("frame_input", frame)
dump_f32("frame_spec_re", spec.real.astype(np.float32))
dump_f32("frame_spec_im", spec.imag.astype(np.float32))
rng = np.random.default_rng(1234)
rnd = rng.standard_normal(400).astype(np.float32)
dump_f32("fft_random_input", rnd)
rs = np.fft.rfft(rnd.astype(np.float64), n=400)
dump_f32("fft_random_re", rs.real.astype(np.float32))
dump_f32("fft_random_im", rs.imag.astype(np.float32))
print("frame ok: len", frame.shape, "spec", spec.shape)


# The FFT unit test input has to reach Vyb as 16-bit PCM (Vyb cannot read a
# float file), so the vector is quantised here and the gold spectrum is computed
# from the QUANTISED values. Whatever the quantisation costs, the comparison is
# still exact about the FFT itself.
def q16(x, peak):
    y = np.clip(x / peak, -1.0, 1.0)
    return (y * 32767.0).round().astype("<i2")

def write_wav16(name, samples_i16, rate=16000, channels=1):
    import struct
    data = samples_i16.astype("<i2").tobytes()
    with wave.open(os.path.join(GOLD, name), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)

MAX32 = np.float32(np.abs(frame).max())
qi = q16(frame, MAX32)
write_wav16("frame_input.wav", qi)
q = qi.astype(np.float64) / 32768.0
spec_q = np.fft.rfft(q, n=400)
dump_f32("frame_spec_q_re", spec_q.real.astype(np.float32))
dump_f32("frame_spec_q_im", spec_q.imag.astype(np.float32))
print("frame quantised peak", float(MAX32), "max |x| after quant",
      float(np.abs(q).max()))

MAXR = np.float32(np.abs(rnd).max())
qr = q16(rnd, MAXR)
write_wav16("fft_random_input.wav", qr)
qr_f = qr.astype(np.float64) / 32768.0
spec_r = np.fft.rfft(qr_f, n=400)
dump_f32("fft_random_q_re", spec_r.real.astype(np.float32))
dump_f32("fft_random_q_im", spec_r.imag.astype(np.float32))
print("fft random quantised peak", float(MAXR))

# --- reference transcription, for the STT gate later -------------------------
model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()
with torch.no_grad():
    ids = model.generate(feats.squeeze(0) if False else torch.from_numpy(feats), max_new_tokens=64)
text = proc.batch_decode(ids, skip_special_tokens=True)[0]
print("ref text:", text)
with open(os.path.join(GOLD, "ref_text.txt"), "w") as f:
    f.write(text + "\n")

# --- and the decoder's first few tokens, the greedy-decode target -----------
with torch.no_grad():
    dec = model.generate(torch.from_numpy(feats), max_new_tokens=8, return_dict_in_generate=True,
                         output_scores=False)
np.save(os.path.join(GOLD, "ref_token_ids.npy"), dec.sequences.numpy())
print("ref token ids:", dec.sequences.numpy()[0][:12].tolist())
print("done ->", GOLD)
