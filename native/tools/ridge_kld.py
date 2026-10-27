# usage: kld.py base.kld TAG  (env passes through to vyb: e.g. VYB_RGMAP=1)
import sys, os, struct, subprocess, numpy as np
base, tag = sys.argv[1], sys.argv[2]
f = open(base, "rb"); assert f.read(8) == b"_logits_"
n_ctx, = struct.unpack("i", f.read(4)); n_vocab, n_chunk = struct.unpack("ii", f.read(8))
toks = np.frombuffer(f.read(4 * n_ctx * n_chunk), dtype=np.int32)
nv = 2 * ((n_vocab + 1) // 2) + 4
first = n_ctx // 2; nt = n_ctx - 1 - first
repo = os.path.expanduser("~/Projects/VybForge-ursela-27b")
kl = []; top = []; nll = []; argeq = 0
for c in range(n_chunk):
    t = toks[c * n_ctx:(c + 1) * n_ctx]
    q = np.frombuffer(f.read(2 * nv * nt), dtype=np.uint16).reshape(nt, nv)
    dump = f"/tmp/kld_{tag}_{c}.bin"
    env = dict(os.environ, VYB_STDLIB=os.path.expanduser("~/Projects/Vyb/stdlib"), VYB_PFB="2", VYB_GEN=str(nt),
               VYB_PROMPT_IDS=" ".join(map(str, t[:first + 1])), VYB_FORCE_IDS=" ".join(map(str, t[first + 1:n_ctx - 1])), VYB_LOGDUMP=dump)
    env.pop("PYTHONPATH", None)
    r = subprocess.run([os.path.expanduser("~/Projects/Vyb/build/vyb"), "native/host/ridge_decode.vyb"] + sum([["--module-path", "native/" + m] for m in ["config", "json", "tensor", "dtype", "llm"]], []),
                       cwd=repo, env=env, capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if "IQ_GEMV" in ln or "ERR" in ln or "TOK_S" in ln: print(ln)
    L = np.fromfile(dump, dtype=np.float64).reshape(-1, n_vocab)[:nt].astype(np.float64); pass
    assert L.shape[0] == nt, L.shape
    for i in range(nt):
        sc, mn = np.frombuffer(q[i, :4].tobytes(), dtype=np.float32)
        lpb = mn + sc * q[i, 4:4 + n_vocab].astype(np.float64)   # same decode + same sum as llama.cpp kl_divergence
        x = L[i] - L[i].max(); lpv = x - np.log(np.exp(x).sum())
        m = lpb > -16.0
        kl.append(float((np.exp(lpb[m]) * (lpb[m] - lpv[m])).sum()))
        top.append(int(np.argmax(lpb) == np.argmax(lpv)))
        nll.append(-lpv[t[first + 1 + i]])
kl = np.array(kl)
print(f"KLD_{tag} n={len(kl)} mean={kl.mean():.5f} median={np.median(kl):.5f} p99={np.percentile(kl,99):.4f} max={kl.max():.4f} top1={100*np.mean(top):.2f}% ppl={np.exp(np.mean(nll)):.4f}")
