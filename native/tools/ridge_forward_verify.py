#!/usr/bin/env python3
"""A whole-model Ridge forward vs the captured oracle (VybForge#10 phase 4, W2/W3).

The acceptance test, and why it is not one of the probe gates. Every probe gate in this repo compares
the engine against a reference written here, so a shared misreading agrees with itself and stays green
(P4.5/P4.9/P4.10 all do this deliberately — they isolate ONE block). A whole 64-block forward has never
run, and per-layer agreement cannot see an error BETWEEN layers, so the only honest acceptance is a
comparison against an implementation that shares no code with ours: llama.cpp on the same GGUF, via the
fixtures `native/tools/ridge_oracle_capture.py` froze (llama's own /tokenize ids, its per-position top1
with the top-2 margins, and the per-position final hidden).

What is compared:
  * PER-POSITION TOP1 — our `prefill_top1_vyb.txt` (the argmax after each prompt position) against the
    fixture's `pos_top1` entries, under the fixture's own rule: EQUALITY where the oracle's top-2 margin
    clears `margin_bar`, MEMBERSHIP of {top1, top2} below it. A flat position is a coin flip that fp
    noise moves, and the 1-token prefix of any prompt is flat.
  * FINAL HIDDEN — our last-block hidden is PRE-output_norm (the driver writes it before the head), the
    oracle's `/embeddings` output is POST-output_norm, so `output_norm.weight` is applied here first.
    Comparing across that stage mismatch reads as ~0.7 cosine on a CORRECT forward, which is why the
    un-normalized cosine is printed as a negative beside the real number.

Provenance is checked BEFORE anything is compared: the fixture pins the llama-server version and a GGUF
id (sha256 of the first MiB), and a mismatch is a FAIL that says to recapture — a stale fixture that
still "matches" is worse than no fixture.

SKIPs (never PASSes) without the model, the inventory, the engine's tensor index, the tokenizer dir,
the Vyb toolchain or the fixtures. Expensive: it runs the FULL 64-block model once per prompt, so it is
not a step to run casually.

Usage: ridge_forward_verify.py [fixture-name ...]
  env: VYBFORGE_RIDGE_GGUF, VYBFORGE_RIDGE_FIXTURE (default the_capital_of_france_is),
       VYBFORGE_RF_COSMIN (default 0.99), VYBFORGE_RIDGE_TOK_DIR, VYB / VYBHOME, VYBFORGE_ORACLE_BIN,
       VYBFORGE_RF_SKIP_RUN=1 + VYBFORGE_RF_LOG (default native/out/ridge_run.log) to CHECK an
       already-completed forward instead of paying for another hour — the log must carry a
       PROMPT_SRC that matches the fixture, or the check refuses rather than compare another run's
       numbers.
"""
import glob
import hashlib
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(HERE))
VYBHOME = os.environ.get("VYBHOME", os.path.expanduser("~/Projects/Vyb"))
VYB = os.environ.get("VYB", os.path.join(VYBHOME, "build", "vyb"))
STDLIB = os.environ.get("VYB_STDLIB", os.path.join(VYBHOME, "stdlib"))
BUILD = os.path.join(REPO, "native", "build")
OUTD = os.path.join(REPO, "native", "out")
MODEL = os.environ.get("VYBFORGE_RIDGE_GGUF",
                       os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf"))
INVENTORY = os.path.join(REPO, "native", "gguf", "ridge-3.7bpw-inventory.tsv")
TSV = os.path.join(OUTD, "ridge_tensors.tsv")
INVFREQ = os.path.join(OUTD, "ridge_invfreq.bin")
TOKDIR = os.environ.get("VYBFORGE_RIDGE_TOK_DIR", os.path.join(REPO, "artifacts/ridge-tokenizer"))
ORACLE_BIN = os.environ.get("VYBFORGE_ORACLE_BIN",
                            os.path.expanduser("~/Projects/llama.cpp/build/bin/llama-server"))
FIXDIR = os.path.join(REPO, "native", "legit", "fixtures", "llama_ridge")
DRV_LOG = os.path.join(OUTD, "ridge_forward_driver.log")
TOPF = os.path.join(OUTD, "prefill_top1_vyb.txt")
HIDF = os.path.join(OUTD, "prefill_hidden_vyb.txt")
COSMIN = float(os.environ.get("VYBFORGE_RF_COSMIN", "0.99"))
WORSTREL = float(os.environ.get("VYBFORGE_RF_WORSTREL", "0.10"))


def field(path, key):
    for line in open(path):
        if line.startswith(key + " "):
            return line[len(key) + 1:].rstrip("\n")
    return None


def fields(path, key):
    out = []
    for line in open(path):
        if line.startswith(key + " "):
            out.append(line.split()[1:])
    return out


def inv_row(name):
    for line in open(INVENTORY):
        if line.startswith(name + "\t"):
            f = line.rstrip("\n").split("\t")
            return int(f[2]), f[3], f[1], int(f[5]), int(f[6])
    return None


def gguf_id(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read(1 << 20)).hexdigest()[:12]


def oracle_version():
    try:
        out = subprocess.run([ORACLE_BIN, "--version"], capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip().splitlines()[0].strip()
    except Exception as e:                                                # noqa: BLE001
        return "unknown (%s)" % e.__class__.__name__


def read_floats(path):
    vals = []
    for line in open(path):
        line = line.strip()
        if line:
            vals.append(float(line))
    return np.array(vals, dtype=np.float64)


def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps) * w


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_driver(prompt_text):
    env = dict(os.environ, VYB_STDLIB=STDLIB, VYB_MODEL=MODEL, VYB_TSV=TSV,
               VYB_INVFREQ=INVFREQ, VYB_LLM_DIR=TOKDIR, VYB_PROMPT=prompt_text)
    cmd = [VYB, "native/host/model_driver.vyb",
           "--module-path", "native/config", "--module-path", "native/json",
           "--module-path", "native/tensor", "--module-path", "native/dtype",
           "--module-path", "native/llm"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    with open(DRV_LOG, "w") as fh:
        fh.write(out)
    return out


def existing_run(prompt_text):
    """The driver output of an ALREADY-COMPLETED run, for when the forward has just been run by hand.

    A full Ridge forward is ~an hour, so paying for it twice to check it once is waste. But native/out
    is reused by every run, so the log must PROVE it is this prompt's: `PROMPT_SRC` has to match the
    fixture's text exactly, or the numbers beside it belong to a different prompt and the comparison
    would be silent nonsense (the shared-output-file trap, in its usual form).
    """
    path = os.environ.get("VYBFORGE_RF_LOG", os.path.join(OUTD, "ridge_run.log"))
    if not os.path.exists(path):
        print("RIDGE_FORWARD_VERIFY_FAIL no run log at %s to reuse (unset VYBFORGE_RF_SKIP_RUN to run "
              "the model)" % path)
        return None
    out = open(path, encoding="utf-8", errors="replace").read()
    if "MODEL_PREFILL_DONE" not in out:
        print("RIDGE_FORWARD_VERIFY_FAIL the run in %s did not finish (no MODEL_PREFILL_DONE)"
              % os.path.relpath(path, REPO))
        return None
    src = None
    for line in out.splitlines():
        i = line.find("PROMPT_SRC=<")
        if i >= 0:
            j = line.find(">", i)
            src = line[i + len("PROMPT_SRC=<"):j] if j > i else None
            break
    if src != prompt_text:
        print("RIDGE_FORWARD_VERIFY_FAIL %s is a run of a DIFFERENT prompt (%r) — refusing to compare "
              "the fixture's oracle against another run's output"
              % (os.path.relpath(path, REPO), src))
        return None
    print("RIDGE_FORWARD_VERIFY reusing the completed run in %s (PROMPT_SRC matches the fixture)"
          % os.path.relpath(path, REPO))
    return out


SKIP_RUN = os.environ.get("VYBFORGE_RF_SKIP_RUN", "") != ""


def verify_fixture(fx):
    name = os.path.basename(fx)
    print("RIDGE_FORWARD_VERIFY fixture=%s" % name)

    # ── provenance FIRST: a mismatch means the oracle moved, not that we are wrong ────────────────
    want_ver, want_gid = field(fx, "llama_version"), field(fx, "gguf_id")
    live_ver, live_gid = oracle_version(), gguf_id(MODEL)
    if want_ver != live_ver or want_gid != live_gid:
        print("RIDGE_FORWARD_VERIFY_FAIL the oracle moved: llama-server %r vs fixture %r, gguf %s vs %s "
              "— recapture required (native/tools/ridge_oracle_capture.py)"
              % (live_ver, want_ver, live_gid, want_gid))
        return 1
    print("RIDGE_FORWARD_VERIFY provenance ok (%s, gguf %s)" % (live_ver, live_gid))

    text = field(fx, "prompt_text")
    want_ids = [int(x) for x in (field(fx, "prompt_ids") or "").split()]
    bar = float(field(fx, "margin_bar") or "0.5")
    pos = fields(fx, "pos_top1")          # k top1 lp1 top2 lp2 exact|membership
    hid_name = field(fx, "hidden_file")
    hs = [int(x) for x in (field(fx, "hidden_shape") or "0 0").split()]
    H = np.fromfile(os.path.join(FIXDIR, hid_name), dtype="<f8").reshape(hs[0], hs[1]) if hid_name else None
    if not text or not want_ids or not pos or H is None:
        print("RIDGE_FORWARD_VERIFY_FAIL %s is missing prompt_ids/pos_top1/hidden" % name)
        return 1
    S, D = hs
    print("RIDGE_FORWARD_VERIFY prompt=<%s> S=%d D=%d positions=%d margin_bar=%.2f"
          % (text, S, D, len(pos), bar))

    # ── the encoding must be the oracle's (P4.11 checks this directly; here it is the run's input) ─
    # A gate may deliberately run a SHORTER prompt, and then the same fixture covers the first runS
    # positions: a causal model's token-k hidden cannot depend on the tokens after it, which is
    # exactly the property that catches a cross-token state bug. The ids check below is what makes
    # that sound (the fixture is keyed by ids).
    run_text = os.environ.get("VYBFORGE_RF_PROMPT") or text
    out = existing_run(run_text) if SKIP_RUN else run_driver(run_text)
    if out is None:
        return 1
    if "MODEL_PREFILL_DONE" not in out:
        if "SKIP" in out:
            print("RIDGE_FORWARD_VERIFY_SKIP " + [l for l in out.splitlines() if "SKIP" in l][0].strip())
            return 0
        print("RIDGE_FORWARD_VERIFY_FAIL the driver did not complete (log: "
              "%s): %s" % (os.path.relpath(DRV_LOG, REPO), out[-1200:]))
        return 1
    for line in out.splitlines():
        if line.startswith(("PROMPT_IDS=", "HEAD_CFG", "HEAD_TABLE", "EMBED_ROWS", "LMHEAD_MODE",
                            "LMHEAD_TOP1_DONE", "PREFILL_HIDDEN_DONE")):
            print("RIDGE_FORWARD_VERIFY driver: " + line)
    got_n = [l for l in out.splitlines() if l.startswith("PROMPT_N=")]
    got_i = [l for l in out.splitlines() if l.startswith("PROMPT_IDS=")]
    if got_i:
        run_ids = [int(x) for x in got_i[0].split("=", 1)[1].split()]
        # The run may be the fixture's prompt or any PREFIX of it — decided on the IDS, not on the
        # text and not on the token count. A text-prefix guess ("The cap") can tokenize to the same
        # COUNT with different ids, which would silently compare the wrong positions.
        if not run_ids or run_ids != want_ids[:len(run_ids)]:
            print("RIDGE_FORWARD_VERIFY_FAIL the run embedded ids %s, which are not a prefix of the "
                  "fixture's %s" % (run_ids[:8], want_ids[:8]))
            return 1
    else:
        # logs from before PROMPT_IDS was printed: fall back to the exact-length rule
        if not got_n or int(got_n[0].split("=")[1]) != len(want_ids):
            print("RIDGE_FORWARD_VERIFY_FAIL the driver embedded %s tokens, the fixture's prompt has "
                  "%d — the prompt stream is not the oracle's"
                  % (got_n[0] if got_n else "?", len(want_ids)))
            return 1
        run_ids = want_ids
    runS = len(run_ids)
    print("RIDGE_FORWARD_VERIFY the run covers %d of the fixture's %d positions (prefix rule on ids)"
          % (runS, len(want_ids)))

    # ── per-position top1, under the fixture's own equality/membership rule ───────────────────────
    ours = [int(v) for v in read_floats(TOPF).astype(np.int64)]
    n_ok, n_membership, n_bad, checked = 0, 0, 0, 0
    details = []
    for row in pos:
        k, t1, lp1, t2, lp2, rule = int(row[0]), int(row[1]), float(row[2]), int(row[3]), float(row[4]), row[5]
        if k > runS or k > len(ours):
            break
        got = ours[k - 1]
        margin = lp1 - lp2
        if rule == "exact":
            checked += 1
            if got == t1:
                n_ok += 1
            else:
                n_bad += 1
                details.append("pos %d: ours=%d oracle=%d (margin %.3f, exact)" % (k, got, t1, margin))
        else:
            checked += 1
            if got in (t1, t2):
                n_membership += 1
            else:
                n_bad += 1
                details.append("pos %d: ours=%d oracle={%d,%d} (margin %.3f, membership)" % (k, got, t1, t2, margin))
    print("RIDGE_FORWARD_VERIFY per-position top1: %d/%d agree (%d exact, %d membership)"
          % (n_ok + n_membership, checked, n_ok, n_membership))
    for d in details[:6]:
        print("RIDGE_FORWARD_VERIFY   " + d)
    if n_bad or checked == 0:
        print("RIDGE_FORWARD_VERIFY_FAIL per-position top1 disagrees with the oracle at %d of %d "
              "positions" % (n_bad, checked))
        return 1

    # ── the final hidden: ours is PRE-output_norm, the oracle's is POST-output_norm ───────────────
    row = inv_row("output_norm.weight")
    if row is None or row[1] != "F32":
        print("RIDGE_FORWARD_VERIFY_FAIL output_norm.weight is not a readable F32 tensor")
        return 1
    with open(MODEL, "rb") as fh:
        fh.seek(row[3])
        w = np.frombuffer(fh.read(row[4]), dtype="<f4").astype(np.float64)
    if w.size != D:
        print("RIDGE_FORWARD_VERIFY_FAIL output_norm is %d values, wanted %d" % (w.size, D))
        return 1
    eps = 1e-6
    try:
        import gguf
        r = gguf.GGUFReader(MODEL)
        for f in r.fields.values():
            if f.name in ("qwen35.attention.layer_norm_rms_epsilon", "qwen35.attention.layer_norm_epsilon"):
                eps = float(f.contents())
    except Exception:                                                     # noqa: BLE001
        print("RIDGE_FORWARD_VERIFY note: gguf unavailable, eps left at 1e-6")
    raw = read_floats(HIDF)
    if raw.size != runS * D:
        print("RIDGE_FORWARD_VERIFY_FAIL our hidden is %d values, wanted %d" % (raw.size, runS * D))
        return 1
    pre = raw.reshape(runS, D)
    post = rmsnorm(pre, w, eps)
    H = H[:runS]                      # the oracle's rows for the positions this run actually covers
    c_post, c_pre = cos(post.ravel(), H.ravel()), cos(pre.ravel(), H.ravel())
    rel = float(np.max(np.abs(post - H)) / max(float(np.max(np.abs(H))), 1e-30))
    print("RIDGE_FORWARD_VERIFY hidden cos(post-output_norm)=%.6f maxrel=%.3e" % (c_post, rel))
    # The stage tooth must use a STAGE-SENSITIVE measure. Cosine is SCALE-INVARIANT, so applying or
    # omitting output_norm barely moves it (measured: cos 0.9804 without the norm against 0.99961
    # with it) — the old `c_pre <= COSMIN - 0.1` bar therefore fails on a CORRECT forward, and
    # tuning it down would accept a fixture that cannot tell the stages apart. The quantity that
    # actually distinguishes them is the per-row NORM: our pre-norm rows are ~490 where the oracle's
    # are ~141, and POST-output_norm they agree to 0.1% (141.02 vs 140.95). So: the post-norm norms
    # must match, and the pre-norm norms must NOT.
    nrm_post = np.linalg.norm(post, axis=1)
    nrm_orac = np.linalg.norm(H, axis=1)
    nrm_pre = np.linalg.norm(pre, axis=1)
    ratio_post = float(np.max(nrm_post / nrm_orac))
    ratio_pre = float(np.min(nrm_pre / nrm_orac))
    print("RIDGE_FORWARD_VERIFY tooth stage norms: post ours/oracle max=%.4f (must be ~1), "
          "pre ours/oracle min=%.2f (must be >> 1)" % (ratio_post, ratio_pre))
    if not (c_post >= COSMIN):                       # NaN fails: not (x >= bar) is True for NaN
        print("RIDGE_FORWARD_VERIFY_FAIL the final hidden does not match the oracle (cos %.6f < %.4f)"
              % (c_post, COSMIN))
        return 1
    if not (ratio_post <= 1.05 and ratio_pre >= 2.0):
        print("RIDGE_FORWARD_VERIFY_FAIL the stage-mismatch negative is not distinguished "
              "(post-norm norm ratio %.4f, pre-norm norm ratio %.2f) — this fixture cannot tell "
              "post-norm from pre-norm, so its pass proves nothing" % (ratio_post, ratio_pre))
        return 1
    if not (rel <= WORSTREL):
        print("RIDGE_FORWARD_VERIFY_FAIL the hidden's maxrel %.3e exceeds %.3e" % (rel, WORSTREL))
        return 1
    print("RIDGE_FORWARD_VERIFY_SUMMARY fixture=%s S=%d positions=%d hidden_cos=%.6f hidden_maxrel=%.3e "
          "negatives=pre_norm_cos(%.4f)" % (name, S, checked, c_post, rel, c_pre))
    return 0


def field_of(text, key):
    for line in text.splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1]
    return None


def main():
    want = sys.argv[1:]
    if not want:
        only = os.environ.get("VYBFORGE_RIDGE_FIXTURE", "the_capital_of_france_is")
        want = [only] if only else []
    fxs = []
    for w in want:
        p = os.path.join(FIXDIR, w if w.endswith(".fix") else w + ".fix")
        if not os.path.exists(p):
            avail = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(FIXDIR, "*.fix")))]
            print("RIDGE_FORWARD_VERIFY_FAIL no fixture %s (available: %s)" % (p, ", ".join(avail)))
            return 1
        fxs.append(p)
    for path, what in ((MODEL, "the Ridge model"), (INVENTORY, "the inventory"),
                       (TSV, "the engine tensor index"), (INVFREQ, "the rope table"),
                       (os.path.join(TOKDIR, "vocab.json"), "the tokenizer dir")):
        if not os.path.exists(path):
            print("RIDGE_FORWARD_VERIFY_SKIP no %s at %s" % (what, path))
            return 0
    if not os.access(VYB, os.X_OK):
        print("RIDGE_FORWARD_VERIFY_SKIP no Vyb toolchain at %s" % VYB)
        return 0

    bad = []
    for fx in fxs:
        if verify_fixture(fx) != 0:
            bad.append(os.path.basename(fx))
    if bad:
        print("RIDGE_FORWARD_VERIFY_FAIL %s did not match the oracle" % ", ".join(bad))
        return 1
    print("RIDGE_FORWARD_VERIFY_DONE a whole 64-block Ridge forward reproduces the oracle's per-position "
          "top1 and its final hidden for %d prompt(s)" % len(fxs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
