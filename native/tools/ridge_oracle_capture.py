#!/usr/bin/env python3
"""Capture the INDEPENDENT decode oracle for Qwen3.8-27B "Ridge" (VybForge#10 phase 4, W2/W3).

Why this exists. `make prefill`'s gold and every probe gate compare the engine against a numpy
reference written in this repo, and both implement the same reading of the model — so a shared
misreading agrees with itself and the gate stays green while the model is wrong (that is how #11's
token salad hid for weeks; see the note in the skill). The acceptance test for a whole-model Ridge
forward is therefore a comparison against an implementation that shares no code with ours:
llama.cpp on the SAME GGUF. This script captures that oracle ONCE and freezes it, with provenance,
so a later divergence can be attributed rather than argued about.

What it captures, for one prompt:

  * the oracle's own TOKENIZER output for the prompt (`/tokenize`) — Ridge's ids are NOT Qwen3-4B's
    (measured: "The capital of France is" is [760,6511,314,9338,369] here, [785,6722,315,9625,374]
    on Qwen3-4B), so any comparison that reuses the dense ids is comparing nonsense;
  * PER-POSITION top1 — llama's greedy next token after each prefix length 1..S, with the top-2
    logprobs at each, so the margin is on the record before anything is gated on it;
  * the greedy CONTINUATION from the full prompt (the stream a decode gate compares token-for-token);
  * the FINAL HIDDEN per position (`/embeddings` with `--pooling none`), the numeric oracle;
  * provenance: the llama-server version/commit and a GGUF id (sha256 of the first MiB), because the
    oracle legitimately moves when the build or the quant file changes.

The endpoint is llama-server, not llama-cli (which dumps core on this box), launched CPU-only
(`-ngl 0`) so it can serve WHILE a GPU driver holds the 3090 — that is the whole point of the
configuration, and it is also why the numbers are f32 CPU arithmetic rather than f16 GPU arithmetic.

Two llama.cpp details this script exists to remember:
  * `--embeddings` must be passed explicitly or `/v1/embeddings` answers 501;
  * with `--pooling none` the OAI-compatible `/v1/embeddings` refuses ("Pooling type 'none' is not
    OAI compatible"), so the NATIVE `/embeddings` endpoint with `{"content": [ids]}` is the one that
    returns one vector per token. `/v1/embeddings` is not usable for per-position hidden.

Run from the repo root (any python with numpy; it needs no llama_cpp):
    python3 native/tools/ridge_oracle_capture.py [--prompt "..."] [--gen 8] [--name capital]

Env: VYBFORGE_RIDGE_GGUF, VYBFORGE_ORACLE_BIN, VYBFORGE_ORACLE_PORT (default 18091),
     VYBFORGE_ORACLE_OUT (default native/legit/fixtures/llama_ridge).
Exit codes: 0 ok, 1 capture failed, 2 prerequisites missing (no GGUF / no llama-server).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GGUF = os.environ.get("VYBFORGE_RIDGE_GGUF",
                      os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf"))
ORACLE_BIN = os.environ.get("VYBFORGE_ORACLE_BIN",
                            os.path.expanduser("~/Projects/llama.cpp/build/bin/llama-server"))
PORT = int(os.environ.get("VYBFORGE_ORACLE_PORT", "18091"))
OUTDIR = os.environ.get("VYBFORGE_ORACLE_OUT", os.path.join(ROOT, "native/legit/fixtures/llama_ridge"))
CTX = int(os.environ.get("VYBFORGE_ORACLE_CTX", "512"))
# The oracle must be reproducible, so its flags live here and nowhere else. CPU-only on purpose
# (see the docstring); `--no-warmup` keeps the log short.
FLAGS = ["-ngl", "0", "-c", str(CTX), "--host", "127.0.0.1", "--port", str(PORT),
         "--pooling", "none", "--embd-normalize", "-1", "--no-warmup", "--embeddings"]


def post(path, payload, timeout=600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def healthy():
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=3) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:                                                   # noqa: BLE001
        return False


def wait_healthy(deadline_s=300):
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        if healthy():
            return True
        time.sleep(2)
    return False


def server_version():
    try:
        out = subprocess.run([ORACLE_BIN, "--version"], capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip().splitlines()[0].strip()
    except Exception as e:                                              # noqa: BLE001
        return "unknown (%s)" % e.__class__.__name__


def gguf_id(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read(1 << 20)).hexdigest()[:12]


def one_pick(ids, n_probs=5):
    """Greedy next token after `ids`, plus the top-2 ids AND logprobs so the margin is recorded.

    `add_special:false` because our driver embeds no BOS; a BOS the driver never sees changes the
    answer. `cache_prompt:false` so each prefix is a clean forward (the point is the per-position
    numbers, not throughput).
    """
    d = post("/completion", {"prompt": [int(i) for i in ids], "n_predict": 1, "temperature": 0.0,
                             "top_k": 1, "n_probs": n_probs, "add_special": False,
                             "cache_prompt": False, "return_tokens": True})
    toks = d.get("tokens") or []
    cp = d.get("completion_probabilities") or []
    if not toks or not cp:
        raise RuntimeError("no tokens/probabilities in the oracle reply: %s" % str(d)[:200])
    tl = cp[0].get("top_logprobs") or []
    t1 = int(tl[0]["id"]) if len(tl) > 0 else int(cp[0]["id"])
    t2 = int(tl[1]["id"]) if len(tl) > 1 else -1
    p1 = float(tl[0]["logprob"]) if len(tl) > 0 else float(cp[0]["logprob"])
    p2 = float(tl[1]["logprob"]) if len(tl) > 1 else float("-inf")
    return int(toks[0]), t1, p1, t2, p2


def stream(ids, n, n_probs=5):
    d = post("/completion", {"prompt": [int(i) for i in ids], "n_predict": n, "temperature": 0.0,
                             "top_k": 1, "n_probs": n_probs, "add_special": False,
                             "cache_prompt": False, "return_tokens": True})
    toks = [int(t) for t in (d.get("tokens") or [])]
    steps = []
    for c in (d.get("completion_probabilities") or []):
        tl = c.get("top_logprobs") or []
        steps.append((int(c["id"]),
                      float(tl[0]["logprob"]) if len(tl) > 0 else float(c["logprob"]),
                      float(tl[1]["logprob"]) if len(tl) > 1 else float("-inf"),
                      int(tl[1]["id"]) if len(tl) > 1 else -1))
    return toks, steps


def hidden(ids):
    d = post("/embeddings", {"content": [int(i) for i in ids]})
    e = d[0]["embedding"]
    a = np.array(e, dtype=np.float64)
    if a.ndim != 2:
        raise RuntimeError("the oracle returned a pooled vector, not per-token hidden (shape %r) — "
                           "the server must run with --pooling none" % (a.shape,))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=8, help="tokens of greedy continuation to record")
    ap.add_argument("--name", default=None, help="fixture name (default: slug of the prompt)")
    args = ap.parse_args()
    if args.name is None:
        # One underscore per RUN of non-alphanumerics: a prompt of numbers otherwise becomes
        # "1__2__3__4__5__6__7" (a doubled separator per comma) and a long prompt silently truncates
        # to a name nobody would guess.
        slug, prev_sep = [], False
        for c in args.prompt.lower():
            if c.isalnum():
                slug.append(c)
                prev_sep = False
            elif not prev_sep:
                slug.append("_")
                prev_sep = True
        name = "".join(slug).strip("_")[:48].strip("_") or "prompt"
    else:
        name = args.name

    if not os.path.exists(GGUF):
        print("RIDGE_ORACLE_CAPTURE_FAIL no Ridge GGUF at %s" % GGUF)
        return 2
    if not (os.path.exists(ORACLE_BIN) and os.access(ORACLE_BIN, os.X_OK)):
        print("RIDGE_ORACLE_CAPTURE_FAIL no llama-server at %s" % ORACLE_BIN)
        return 2
    os.makedirs(OUTDIR, exist_ok=True)
    ver = server_version()
    gid = gguf_id(GGUF)
    print("RIDGE_ORACLE_CAPTURE oracle=%s" % ver)
    print("RIDGE_ORACLE_CAPTURE gguf_id=%s prompt=<%s>" % (gid, args.prompt))

    spawned = None
    if not healthy():
        # The log has to go somewhere that survives the run; native/out is gitignored build output.
        logdir = os.path.join(ROOT, "native/out/ridge_oracle")
        os.makedirs(logdir, exist_ok=True)
        logpath = os.path.join(logdir, "capture_server.log")
        logf = open(logpath, "w")
        spawned = subprocess.Popen([ORACLE_BIN, "-m", GGUF] + FLAGS, stdout=logf, stderr=logf)
        print("RIDGE_ORACLE_CAPTURE started %s (log: %s)" % (ORACLE_BIN, logpath))
        if not wait_healthy():
            spawned.terminate()
            print("RIDGE_ORACLE_CAPTURE_FAIL the oracle never became healthy (see %s)" % logpath)
            return 1
    else:
        print("RIDGE_ORACLE_CAPTURE reusing the server already on port %d" % PORT)

    try:
        tok = post("/tokenize", {"content": args.prompt})
        ids = [int(t) for t in tok["tokens"]]
        print("RIDGE_ORACLE_CAPTURE prompt_ids=%s" % ids)

        # per-position top1: the model's greedy pick AFTER each prefix length 1..S
        periop = []
        for k in range(1, len(ids) + 1):
            t, t1, p1, t2, p2 = one_pick(ids[:k])
            periop.append((k, t, t1, p1, t2, p2))
            print("RIDGE_ORACLE_CAPTURE   after %d tokens -> %d (logprob %.4f, runner-up %d %.4f, "
                  "margin %.4f)" % (k, t, p1, t2, p2, p1 - p2))

        gen_ids, gen_steps = stream(ids, args.gen)
        print("RIDGE_ORACLE_CAPTURE greedy %d: %s" % (len(gen_ids), gen_ids))

        H = hidden(ids)
        print("RIDGE_ORACLE_CAPTURE hidden shape %s norms %s" % (H.shape, np.round(np.linalg.norm(H, axis=1), 2)))

        # Determinism: an oracle that moves between runs cannot be a gate. temp 0 makes the picks
        # repeatable in principle; this MEASURES it rather than assuming it (the same rule the
        # attention authority's gate applies to itself). Re-issue the per-position picks and the
        # embeddings call and require bit-identical answers.
        for k in range(1, len(ids) + 1):
            again = one_pick(ids[:k])
            want = periop[k - 1]
            if again[0] != want[1] or again[1] != want[2] or abs(again[2] - want[3]) > 1e-6 \
               or abs(again[4] - want[5]) > 1e-6:
                print("RIDGE_ORACLE_CAPTURE_FAIL the oracle is NON-DETERMINISTIC at prefix %d: %r vs %r"
                      % (k, again, want))
                return 1
        H2 = hidden(ids)
        if not np.array_equal(H, H2):
            print("RIDGE_ORACLE_CAPTURE_FAIL the oracle's hidden is NON-DETERMINISTIC (max abs diff "
                  "%.3e)" % float(np.max(np.abs(H - H2))))
            return 1
        print("RIDGE_ORACLE_CAPTURE deterministic: %d picks and the hidden repeat exactly"
              % len(periop))

        # The first generated token must BE the per-position pick at k=S, or the two captures
        # disagree with each other and neither is a trustworthy oracle.
        if gen_ids and periop and gen_ids[0] != periop[-1][1]:
            print("RIDGE_ORACLE_CAPTURE_FAIL the per-position pick after %d tokens (%d) is not the "
                  "first greedy token (%d)" % (len(ids), periop[-1][1], gen_ids[0]))
            return 1

        BAR = 0.5      # nats; the margin bar a gate uses to decide equality vs membership
        margins = [p1 - p2 for _, _, _, p1, _, p2 in periop] + [s[1] - s[2] for s in gen_steps]
        n_flat = sum(1 for m in margins if m < BAR)
        mode = "exact" if n_flat == 0 else "membership"
        hid_name = name + ".hidden.f64"
        H.astype("<f8").tofile(os.path.join(OUTDIR, hid_name))

        lines = []
        lines.append("# Ridge decode oracle — captured by native/tools/ridge_oracle_capture.py.")
        lines.append("# The oracle is llama.cpp on the SAME GGUF, CPU-only (so it can serve beside a GPU")
        lines.append("# driver). A provenance mismatch below means the oracle moved: recapture, do not")
        lines.append("# compare against a stale fixture. `hidden_stage` says WHICH hidden this is.")
        lines.append("#")
        lines.append("# How to use it, and why the bar is not a fudge: greedy agreement is only meaningful")
        lines.append("# where the decision is decisive. Every recorded pick carries its top-2 margin (nats)")
        lines.append("# and the runner-up's id, so a gate checks EQUALITY where margin >= margin_bar and")
        lines.append("# MEMBERSHIP of {top1, top2} where it is below — a flat position is a coin flip that fp")
        lines.append("# noise can move, and a gate that demands equality there flakes on a correct forward.")
        lines.append("# Measured on this prompt: the 1-token prefix is inherently thin (after one token the")
        lines.append("# distribution is nearly flat), which is exactly the case the membership rule exists for.")
        lines.append("oracle llama-server")
        lines.append("llama_version %s" % ver)
        lines.append("llama_binary %s" % ORACLE_BIN)
        lines.append("llama_flags %s" % " ".join(FLAGS))
        lines.append("gguf %s" % os.path.basename(GGUF))
        lines.append("gguf_id %s" % gid)
        lines.append("prompt_text %s" % args.prompt)
        lines.append("prompt_ids %s" % " ".join(str(i) for i in ids))
        lines.append("margin_bar %.4f" % BAR)
        lines.append("mode %s" % mode)
        lines.append("mode_note %d of %d recorded picks are below the margin bar"
                      % (n_flat, len(margins)))
        lines.append("hidden_file %s" % hid_name)
        lines.append("hidden_stage post_output_norm")
        lines.append("hidden_shape %d %d" % (H.shape[0], H.shape[1]))
        # pos_top1 <k> <top1> <logprob1> <top2> <logprob2>
        for k, t, t1, p1, t2, p2 in periop:
            gate = "exact" if (p1 - p2) >= BAR else "membership"
            lines.append("pos_top1 %d %d %.6f %d %.6f %s" % (k, t1, p1, t2, p2, gate))
        lines.append("gen_ids %s" % " ".join(str(i) for i in gen_ids))
        # gen_step <token> <logprob1> <runner_up id> <logprob2>
        for t, p1, p2, t2 in gen_steps:
            lines.append("gen_step %d %.6f %d %.6f" % (t, p1, t2, p2))
        p = os.path.join(OUTDIR, name + ".fix")
        with open(p, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("RIDGE_ORACLE_CAPTURE_WROTE %s" % os.path.relpath(p, ROOT))
        print("RIDGE_ORACLE_CAPTURE_WROTE %s" % os.path.relpath(os.path.join(OUTDIR, hid_name), ROOT))
        print("RIDGE_ORACLE_CAPTURE mode=%s min_margin=%.4f flat=%d/%d"
              % (mode, min(margins), n_flat, len(margins)))
        print("RIDGE_ORACLE_CAPTURE_DONE")
        return 0
    finally:
        if spawned is not None:
            spawned.terminate()
            try:
                spawned.wait(timeout=30)
            except Exception:                                           # noqa: BLE001
                spawned.kill()


if __name__ == "__main__":
    sys.exit(main())
