#!/usr/bin/env python3
"""Compute step-1 greedy argmax using decode_ref.py's EXACT functions
(imported from a functions-only copy _dr_fns.py). Settles the 12001-vs-73305
disagreement by removing any chance of my ad-hoc re-typing a different forward."""
import os, importlib.util, numpy as np
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("drf", os.path.join(repo, "native/gguf/_dr_fns.py"))
drf = importlib.util.module_from_spec(spec); spec.loader.exec_module(drf)
x = np.stack([drf.emb_row(0), drf.emb_row(1), drf.emb_row(31784)])
for L in range(36):
    x = drf.forward_layer(x, drf.layer_weights(L))
lg = drf.logits_for(x)[-1]
top = np.argsort(-lg)[:10]
print("step1 (decode_ref fns, fresh) top10 token:logit")
for t in top: print(f"  {t}: {lg[t]:.6f}")
print("GPU pick 73305:", lg[73305], " ref-file 12001:", lg[12001])
