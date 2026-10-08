#!/usr/bin/env python3
"""Capture the independent decode oracle for the llama.cpp gate (VybForge#22).

Writes frozen fixtures under native/legit/fixtures/llama_decode/. Each pins a prompt's ids,
llama.cpp's greedy continuation, and the provenance (llama.cpp version + GGUF id) so a later
divergence can be attributed rather than argued about.

Three cases, because greedy streams are only stable where the decision is decisive:
  weather      exact match, chosen for comfortable margins
  cap_tf       teacher-forced: seed = prompt + llama's own prefix, one token expected
  cap_tie      a documented near-tie: membership, not equality

Two routes are used deliberately, and neither is interchangeable:
  - token IDS come from the low-level loop (reset/eval/sample/eval) — logprobs returns strings
  - PROBABILITIES come from create_completion(logprobs=..) — llm.scores came back unfilled here

Regenerate from the repo root:
  env -u PYTHONPATH .venv/bin/python native/tools/llama_decode_capture.py
"""
import hashlib, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
MODEL = os.environ.get("VYBFORGE_QWEN3_GGUF", os.path.expanduser("~/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"))
OUT = os.path.join(ROOT, "native/legit/fixtures/llama_decode")

try:
    import llama_cpp
    from llama_cpp import Llama
except ImportError:
    sys.exit("llama_cpp not importable — run this with the repo's .venv python")

llm = Llama(model_path=MODEL, n_ctx=512, logits_all=True, verbose=False, seed=0)
with open(MODEL, "rb") as fh:
    gguf_id = hashlib.sha256(fh.read(1 << 20)).hexdigest()[:12]      # first MiB identifies the build
ver = getattr(llama_cpp, "__version__", "unknown")
os.makedirs(OUT, exist_ok=True)
print(f"llama_cpp {ver}  gguf(first MiB) {gguf_id}")


def greedy_ids(seed_ids, n):
    llm.reset(); llm.eval(seed_ids)
    out = []
    for _ in range(n):
        t = llm.sample(temp=0.0)
        out.append(int(t)); llm.eval([t])
    return out


def ids_line(ids):
    return " ".join(str(int(i)) for i in ids)


def text(ids):
    return llm.detokenize([int(i) for i in ids]).decode("utf-8", "replace")


def write(name, header, fields):
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        fh.write("# llama.cpp greedy oracle fixture (VybForge#22)\n")
        fh.write("# regenerate: env -u PYTHONPATH .venv/bin/python native/tools/llama_decode_capture.py\n")
        fh.write(f"# {header}\n")
        for k, v in fields:
            fh.write(f"{k} {v}\n")
    print(f"  wrote {os.path.relpath(p, ROOT)}")


P_WEATHER = "The weather today is sunny and"
P_CAPITAL = "The capital of France is"

# ---- case 1: clean exact match ----------------------------------------------------------
i1 = llm.tokenize(P_WEATHER.encode())
g1 = greedy_ids(i1, 8)
write("decode_weather.fix",
      "exact match; chosen for comfortable margins so it cannot flake on a near-tie",
      [("mode", "exact"),
       ("prompt_text", P_WEATHER),
       ("prompt_ids", ids_line(i1)),
       ("expect_ids", ids_line(g1)),
       ("expect_text", text(g1).strip()),
       ("llama_version", ver), ("gguf_id", gguf_id)])

# ---- case 2: teacher-forced one step past the near-tie -----------------------------------
i2 = llm.tokenize(P_CAPITAL.encode())
g2 = greedy_ids(i2, 3)
seed = i2 + g2
g_tf = greedy_ids(seed, 1)
write("decode_capital_teacherforced.fix",
      "teacher-forced: seed = prompt + llama's own first three tokens, one token expected",
      [("mode", "exact"),
       ("prompt_text", text(seed)),
       ("prompt_ids", ids_line(seed)),
       ("expect_ids", ids_line(g_tf)),
       ("expect_text", text(g_tf).strip()),
       ("llama_version", ver), ("gguf_id", gguf_id)])

# ---- case 3: the near-tie itself, as membership ------------------------------------------
# Two llama.cpp routes are used here on purpose: the low-level greedy loop (which the fixture
# ids come from) and create_completion's logprobs. They disagree about the argmax at this step,
# which is the whole reason this case asserts membership instead of equality.
ll = greedy_ids(i2, 2)                      # low-level route: step 0, step 1
done = llm.create_completion(P_CAPITAL, max_tokens=2, temperature=0.0, logprobs=5, echo=False)
lp = done["choices"][0]["logprobs"]
step1 = sorted(lp["top_logprobs"][1].items(), key=lambda kv: -kv[1])
(a_s, a_lp), (b_s, b_lp) = step1[0], step1[1]
lvl_ids = sorted({int(t) for t in llm.tokenize(a_s.encode())} | {int(t) for t in llm.tokenize(b_s.encode())})
allowed = sorted(set(lvl_ids) | {ll[1]})
agree = "agree" if ll[1] in lvl_ids else "DISAGREE"
write("decode_capital_neartie.fix",
      (f"documented near-tie at step 1, and the two llama.cpp routes {agree} there: the low-level "
       f"greedy loop takes {text([ll[1]])!r} while create_completion ranks {a_s!r} (lp {a_lp:.4f}) "
       f"over {b_s!r} (lp {b_lp:.4f}), a {abs(a_lp - b_lp):.4f} logprob gap. Either side is accepted."),
      [("mode", "membership"),
       ("prompt_text", P_CAPITAL),
       ("prompt_ids", ids_line(i2)),
       ("check_step", "1"),
       ("allowed_ids", ids_line(allowed)),
       ("low_level_ids", ids_line(ll)),
       ("logprob_rank_ids", ids_line(lvl_ids)),
       ("min_gap", f"{abs(a_lp - b_lp):.4f}"),
       ("note", "both routes' argmax are accepted; this step is not a stable decision"),
       ("llama_version", ver), ("gguf_id", gguf_id)])

print("done")
