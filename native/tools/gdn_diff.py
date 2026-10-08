#!/usr/bin/env python3
"""One self-contained run: generate inputs, run the harness (with the op's own dump), run the port,
and diff head 0 line by line. Nothing is re-read from disk that was not written microseconds earlier,
because the last ad-hoc comparison read a regenerated file and reported nonsense.

Build the instrumented library first:
    cd ~/Projects/llama.cpp && git apply <repo>/native/tools/ggml_gdn_debug.patch && cmake --build build --target ggml-cpu -j

Then: GGML_GDN_DEBUG=1 python native/tools/gdn_diff.py
"""
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gdn_ref import BUILD, BIN, H_K, H_V, S, build_authority, inputs, numpy_rule  # noqa: E402


def main():
    if not build_authority():
        print("GDN_DIFF_FAIL harness did not build")
        return 1
    q, k, v, g, be, st = inputs(1, 1)

    in_path = os.path.join(BUILD, "diff_in.bin")
    out_path = os.path.join(BUILD, "diff_out.bin")
    st_path = os.path.join(BUILD, "diff_state.bin")
    with open(in_path, "wb") as fh:
        for a in (q, k, v, g, be, st):
            fh.write(a.astype("<f4").tobytes())

    env = dict(os.environ, GGML_GDN_DEBUG="1")
    r = subprocess.run([BIN, in_path, out_path, st_path, str(S), str(H_K), str(H_V), "1", "1", "1"],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print("GDN_DIFF_FAIL harness: " + (r.stdout + r.stderr)[:300])
        return 1

    opv = {}
    for line in r.stderr.splitlines():
        if not line.startswith("GDN_VAL"):
            continue
        if "decay=" in line:
            m = re.search(r"decay=(\S+) beta=(\S+)", line)
            opv["decay"], opv["beta"] = float(m.group(1)), float(m.group(2))
        elif " i=" in line:
            m = re.search(r"i=(\d+) k=(\S+) q=(\S+) v=(\S+) delta=(\S+) s_out=(\S+) attn=(\S+)", line)
            if m:
                opv[int(m.group(1))] = [float(m.group(x)) for x in (2, 3, 4, 5, 6, 7)]
    if "decay" not in opv:
        print("GDN_DIFF_FAIL no GDN_VAL output — is the instrumented library built and loaded?")
        return 1

    ref_out = np.fromfile(out_path, dtype="<f4").reshape(S, H_V, 1, 1).astype(np.float64)
    ref_st = np.fromfile(st_path, dtype="<f4").reshape(S, S, H_V, 1).astype(np.float64)
    my_out, my_st = numpy_rule(q, k, v, g, be, st, bcast="mod")

    # port head 0, token 0
    kd = k[:, 0, 0, 0].astype(np.float64)
    qd = q[:, 0, 0, 0].astype(np.float64)
    vd = v[:, 0, 0, 0].astype(np.float64)
    m = st[:, :, 0, 0].astype(np.float64).copy() * np.exp(float(g[0, 0, 0, 0]))
    delta = (vd - m @ kd) * float(be[0, 0, 0, 0])
    m = m + np.outer(delta, kd)
    attn = (m @ qd) * (1.0 / np.sqrt(S))

    print(f"GDN_DIFF decay op={opv['decay']:.9g} port={np.exp(float(g[0,0,0,0])):.9g}")
    print(f"GDN_DIFF beta  op={opv['beta']:.9g} port={float(be[0,0,0,0]):.9g}")
    for i in range(4):
        ok, oq, ov, od, os_, oa = opv[i]
        pk, pq, pv = kd[i], qd[i], vd[i]
        pd, ps, pa = delta[i], m.ravel()[i], attn[i]
        flag = "MATCH" if abs(pk - ok) < 1e-7 * max(abs(ok), 1) else "DIFF"
        print(f"GDN_DIFF i={i} {flag} k op={ok:.9g} port={pk:.9g} | q op={oq:.9g} port={pq:.9g} "
              f"| v op={ov:.9g} port={pv:.9g}")
        print(f"GDN_DIFF      delta op={od:.9g} port={pd:.9g} | s_out op={os_:.9g} port={ps:.9g} "
              f"| attn op={oa:.9g} port={pa:.9g}")

    # whole-tensor verdict too
    ro = float(np.max(np.abs(my_out - ref_out)) / max(np.max(np.abs(ref_out)), 1e-30))
    rs = float(np.max(np.abs(my_st - ref_st)) / max(np.max(np.abs(ref_st)), 1e-30))
    print(f"GDN_DIFF whole-tensor maxrel out={ro:.3e} state={rs:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
