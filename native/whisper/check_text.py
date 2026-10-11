"""Gate the transcription text against the reference.

The reference is what transformers' batch_decode produced from the same ids --
so this checks the DETOKENIZER (byte-level BPE: ids -> token glyphs -> bytes ->
UTF-8) and not just the model. Compared character for character, including the
leading space and the punctuation.
"""
import sys

out_path, ref_path = sys.argv[1], sys.argv[2]
lines = [l.rstrip("\n") for l in open(out_path)]
got = None
for i, l in enumerate(lines):
    if l == "TEXT":
        got = lines[i + 1] if i + 1 < len(lines) else ""
        break
ref = open(ref_path).read().rstrip("\n")

if got is None:
    print("TEXT_GATE: FAIL (driver emitted no TEXT block)")
    sys.exit(1)
if got == ref:
    print(f"TEXT_GATE: OK  ({len(got)} chars)")
    sys.exit(0)
print("TEXT_GATE: FAIL")
print(f"  vyb: {got!r}")
print(f"  ref: {ref!r}")
sys.exit(1)
