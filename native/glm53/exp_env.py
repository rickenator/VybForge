"""Emit shell exports for the token-0 expert gate (ids + renormed weights) from the gold."""
import json

S40 = 1099511627776.0
d = json.load(open("native/out/gold_l3_moe.json"))
ids = ",".join(str(int(e)) for e in d["route"][0])
w = ",".join(str(int(round(float(x) * S40))) for x in d["rw"][0])
print(f'export VYBFORGE_EXPERT_IDS="{ids}"')
print(f'export VYBFORGE_EXPERT_W="{w}"')
