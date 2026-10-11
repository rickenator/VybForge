"""Gate the greedy decode: the ids Vyb produced, token for token, against the
reference that fixtures/whisper/oracle_stt.py captured from the real model.

Reads the driver's "IDS n=" block out of its stdout and the gold ids file, and
exits non-zero on any difference -- including a length difference, because a
shorter or longer answer is not the same answer.
"""
import sys

out_path, gold_path = sys.argv[1], sys.argv[2]
lines = [l.strip() for l in open(out_path)]

got, want = [], []
for i, l in enumerate(lines):
    if l == "IDS n=":
        n = int(lines[i + 1])
        got = [int(x) for x in lines[i + 2:i + 2 + n]]
        break
want = [int(x) for x in open(gold_path).read().split()]

print(f"  ids: vyb {len(got)} vs reference {len(want)}")
if got == want:
    print(f"IDS_GATE: OK  ({' '.join(str(x) for x in got[:6])} ...)")
    sys.exit(0)
for i in range(max(len(got), len(want))):
    g = got[i] if i < len(got) else None
    w = want[i] if i < len(want) else None
    if g != w:
        print(f"IDS_GATE: FAIL at {i}: vyb={g} ref={w}")
        break
sys.exit(1)
