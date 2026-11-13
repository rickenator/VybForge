"""Gate L3: the dispatcher's replies, against expectations computed HERE.

The point of computing the expected time and date in Python rather than reading
them back is that the check must not be able to agree with a wrong answer. The
epoch and the UTC offset come from the driver (the offset IS what tz.vyb
computes); the calendar conversion and the formatting are redone independently
with zoneinfo + strftime semantics.

  check_dispatch.py <driver-output> <tzfile> <utc-offset-secs>
"""
import sys
from datetime import datetime, timedelta, timezone

out_path, tzfile, off = sys.argv[1], sys.argv[2], int(sys.argv[3])

lines = [l.rstrip("\n") for l in open(out_path)]
utts = []
now = None
for i, l in enumerate(lines):
    if l == "NOW":
        now = int(lines[i + 1])
    if l == "UTT":
        utts.append({
            "text": lines[i + 1],
            "reply": lines[i + 3],
            "quit": lines[i + 5] == "1",
        })
if now is None or not utts:
    print("DISPATCH_GATE: FAIL (driver emitted nothing to check)")
    sys.exit(1)

# Independent: apply the driver's offset to the driver's epoch, then convert.
local = datetime.fromtimestamp(now, tz=timezone.utc) + timedelta(seconds=off)
h12 = local.hour % 12 or 12
exp_time = f"It is {h12}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}."
exp_date = ("Today is " + local.strftime("%A, %B ") + str(local.day) + ", " +
            str(local.year) + ".")

HELP = ("I can tell you the time or the date, and I can transcribe a recording. "
        "Say goodbye to stop.")
FALLBACK = "I don't know how to do that yet. Say help to hear what I can do."

# Expectations are keyed on the NORMALISED utterance, so the same case can be
# written several ways in the fixture.
CASES = {
    "hello": ("Hello. I'm listening.", False),
    "": ("I didn't catch that.", False),
    "what time is it": (exp_time, False),
    "can you tell me the time?": (exp_time, False),
    "what is the date": (exp_date, False),
    "what day is it": (exp_date, False),
    "help": (HELP, False),
    "what can you do": (HELP, False),
    "transcribe my recording": ("Give me a WAV file and I can transcribe it locally "
                                "with the Vyb Whisper model.", False),
    "make me a sandwich": (FALLBACK, False),
    "goodbye": ("Goodbye.", True),
}

def norm(s):
    return " ".join(s.lower().split())

bad = 0
for u in utts:
    key = norm(u["text"])
    if key not in CASES:
        print(f"  no expectation for {u['text']!r} (add it to CASES)")
        bad += 1
        continue
    want_text, want_quit = CASES[key]
    ok = (u["reply"] == want_text) and (u["quit"] == want_quit)
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {u['text']!r} -> {u['reply']!r}"
          + ("" if ok else f"   want {want_text!r} quit={want_quit}"))
    if not ok:
        bad += 1

if bad:
    print(f"DISPATCH_GATE: FAIL ({bad}/{len(utts)})")
    sys.exit(1)
print(f"DISPATCH_GATE: OK  ({len(utts)} utterances, time/date recomputed independently"
      f" from epoch {now} + offset {off:+d}s)")
