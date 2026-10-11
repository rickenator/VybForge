# native/tts — L4, the voice loop's synthesis side

Status: **the synthesis core is done and gated.** A phoneme's formant targets go
in; a WAV comes out whose spectrum is the cascade response those targets imply.

    ./native/tts/build.sh          # renders the vowel table, measures it, gates it

## What is here

| file | what it is |
|---|---|
| `synth.vyb` | Klatt cascade formant synthesiser + the vowel table + WAV packing |
| `host/synth_probe.vyb` | renders the table to `out/*.wav` and prints the SPEC it used |
| `check_vowels.py` | measures the WAVs (LPC envelope) against the response re-derived in numpy |

## The closed loop, and where it currently stands

`./native/tts/round_trip.sh hello` synthesises a word and transcribes it with the
Vyb Whisper. It does not pass:

    sent:     hello
    heard:    (bell dings)

and after formant transitions, and a bug fix:

    sent:     hello
    heard:    ♪♪

The progression is the useful part. Stationary phones -> "(bell dings)"; adding
transitions exposed a REAL BUG (the pitch period came from the first phone in the
sequence, so a word starting with an unvoiced phone divided by zero and emitted no
glottal pulses at all -- "hello" rendered its vowels as SILENCE and Whisper heard
"[BLANK_AUDIO]"); with that fixed the vowels sound and Whisper calls it MUSIC.

Whisper classifies the output as a non-speech SOUND, now with pitch in it. Four
rounds of the obvious fixes did not change that category:

    glottal tilt (2-pole) + per-class source amplitudes  -> "(singing)"
    + amplitude levels actually applied                  -> "(singing)"
    + pitch declination (1.12 -> 0.86 across the phrase)  -> "(singing)"

THE PATTERN IS THE FINDING. What changed each time was the *label*, never the
category: sound, then music, then singing. Whisper is a model of natural speech and
has no acoustic model of a three-formant sounds-almost-right signal; "(singing)" and
"(music)" are what it does with pitched non-speech, and Whisper never emits those
labels for the jfk fixture, which is real speech. So this instrument cannot
distinguish "my synthesiser is bad" from "my synthesiser is not what Whisper models",
and every further tweak measures Whisper rather than the synthesiser.

The honest next steps, in order:
  1. change the ORACLE before changing the synthesiser further -- a vowel-only
     stimulus (steady /a/, 500 ms) is the cheap decisive test: if Whisper labels a
     single steady vowel as music too, the instrument is the problem, not the
     sequence. A phone recogniser (or an ear) is the right judge for synthetic speech.
  2. then, if ASR-recognisable speech is the requirement rather than "Vyb can speak",
     this is the point to stop growing the formant synthesiser and port a neural
     vocoder (VITS/HiFiGAN), which is the same scale of port as Whisper and is what
     the formant core was always going to be replaced BY if intelligibility to ASR
     is the bar. The formant core would remain the thing that makes the loop speak
     with no checkpoint at all. That is a decision for Rick, not for me. That is an honest and useful
result, and the diagnosis is not mysterious: the synthesiser renders each phone as
a STATIONARY filter and concatenates them. Consonant-vowel formant TRANSITIONS are
the main carrier of intelligibility in speech -- the /e/ into /l/ into /o/ movement
is most of what makes "hello" that word -- and there are none here. The spectrum is
right (the vowel gate proves the filters do what they are told); the DYNAMICS are
absent.

The formant track now moves: over the first 60 ms of each phone F1/F2/F3/BW
interpolate from the previous phone's targets, and the resonator coefficients are
recomputed per sample. What is still missing is everything AROUND the segment:

  * an amplitude envelope -- every phone is peak-normalised to the same level and
    butts against the next, so the output is a sequence of equal-loudness bursts.
    A syllable has a rise and a fall, and ASR expects that modulation;
  * a real glottal PULSE rather than an impulse train (an impulse train is
    harmonically rich and buzzy, which is much of why this reads as music);
  * prosody: pitch declination, stress, and a falling contour at the end;
  * source amplitude per phone class -- stops and fricatives are much quieter than
    vowels in speech, and here they are not. This script exists as a measurement
rather than a gate until it passes, because a gate that is permanently red for work
not yet done hides the failures that matter.

## What is deliberately NOT here yet

Text. This is the acoustic core: it takes formant targets, not letters. Text ->
phonemes (a pronunciation dictionary plus letter-to-sound rules), the consonant
table, durations and prosody, and the connect-to-L1 playback path all sit on top of
this, and none of them are started. What the gate establishes is the narrow, honest
claim that the DSP does what it is told -- which is the part a listener cannot
debug by ear.

## Why formant synthesis first, and not a neural vocoder

A VITS/HiFiGAN vocoder is the same scale of port as Whisper: weeks of kernels, and
its output would be the same WAV. The formant core is what a neural path would
replace, it needs no checkpoint, and its correctness is measurable. The vocal tract
model is Klatt's, written from the literature:

    C = -exp(-2*pi*BW/fs)
    B =  2*exp(-pi*BW/fs)*cos(2*pi*F/fs)
    A =  1 - B - C                                  (unit DC gain)
    y[n] = A*x[n] + B*y[n-1] + C*y[n-2]              one two-pole filter per formant

Vowel targets are Peterson & Barney adult-male averages, f0 is 120 Hz, and
bandwidths (90/110/170 Hz) are exported from the module because the gate re-derives
the predicted response from them -- the checker must not restate a number the model
owns.

## The measurement, and three attempts it took to get right

The gate asks one question: does the audio have the spectrum it was told to have?
Answering it took three estimators, and the failures were all in the measurement
rather than the synthesiser. That distinction is the whole reason the file is
written this way:

1. **Raw FFT peaks** -- cannot work on a voiced sound at all: energy exists only at
   multiples of f0, so "the largest bin near F1" can only ever be a harmonic. /u/
   has F1 at 300 Hz and f0 at 120 Hz, so 240 or 360 was the only answer available.
2. **Cepstral liftering** -- fixes (1) but adds its own bias. Smoothing the log
   spectrum smooths the log of the NOISE FLOOR in with it, so weak peaks read high
   (measured F2 of /i/ came out 10 dB above the predicted response) and the error
   grows as the floor gets closer. Curve-vs-curve comparison was tried too and it
   measures the estimator, not the synthesiser.
3. **LPC (all-pole, Levinson-Durbin)** -- a least-squares fit to the
   autocorrelation, so the floor is not smoothed into the answer, and it is a
   genuinely different estimator from the resonator arithmetic under test. It
   agrees with the re-derived cascade response to within a few Hz at every formant
   of every vowel, and within 3 dB on relative peak levels.

Tolerances are stated before the numbers are seen: 120 Hz or 8% on a formant
position, 3 dB on relative peak level, 8% on f0.

## The prediction goes through the same pipeline

The analytic response is passed through the same lifter the measurement uses, so
both sides are treated identically. Peak positions matched to a few Hz either way;
levels were off by 3-6 dB until that was fixed.
