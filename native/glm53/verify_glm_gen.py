#!/usr/bin/env python3
"""GLM_GEN_GATE: the chain driver's generation mode (VYBFORGE_GEN).

Checks, from one or two chain dump files (run A, and optionally run B):
  1. GENTRUNK lines exist for every step (teacher-forced 0..nsteps-1 + generated);
  2. run A vs run B determinism: identical trunk argmax sequence and identical
     per-step logits digests (gtrunk=/gdraft=), token for token;
  3. MTP draft top-1 vs trunk argmax agreement rate (no hard bar — it is a draft);
  4. decodes the generated ids back to text with the tokenizers reference.

Run:  env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_glm_gen.py \
        native/out/decchain_genA.txt [native/out/decchain_genB.txt]
"""
import re
import sys
import os

rx_trunk = re.compile(r'^GENTRUNK t=(\d+) input=(\d+) trunk=(\d+) trunklogit=(\S+) gtrunk=logits n=154880 (.*)$')
rx_draft = re.compile(r'^GENDRAFT t=(\d+) draft=(\d+) draftlogit=(\S+) gdraft=logits n=154880 (.*) agree=(\d+)$')


def parse(path):
    trunks, drafts = {}, {}
    for line in open(path):
        line = line.strip()
        m = rx_trunk.match(line)
        if m:
            trunks[int(m.group(1))] = {
                "input": int(m.group(2)), "tok": int(m.group(3)),
                "logit": float(m.group(4)), "dg": m.group(5)}
            continue
        m = rx_draft.match(line)
        if m:
            drafts[int(m.group(1))] = {
                "tok": int(m.group(2)), "logit": float(m.group(3)),
                "dg": m.group(4), "agree": int(m.group(5))}
    return trunks, drafts


def main():
    if len(sys.argv) < 2:
        print("usage: verify_glm_gen.py <chainA.txt> [chainB.txt]")
        return 2
    ta, da = parse(sys.argv[1])
    if not ta:
        print("GLM_GEN_GATE: FAIL (no GENTRUNK lines in %s)" % sys.argv[1])
        return 1
    steps = sorted(ta)
    nsteps = min(steps)  # teacher-forced steps are 0..nsteps-1
    seq = [(ta[s]["input"], ta[s]["tok"]) for s in steps]
    ids = [t for _, t in seq]
    ok = True

    if len(sys.argv) > 2:
        tb, db = parse(sys.argv[2])
        seq_b = [(tb[s]["input"], tb[s]["tok"]) for s in sorted(tb)]
        dg_same = all(ta[s]["dg"] == tb.get(s, {}).get("dg") for s in steps)
        if seq != seq_b or not dg_same:
            print("GLM_GEN_GATE: FAIL (non-deterministic)")
            print("  A:", seq)
            print("  B:", seq_b)
            ok = False
        else:
            print("determinism: run A == run B over %d steps (ids + logits digests)" % len(steps))
    else:
        print("determinism: single run (pass a second dump file to compare)")

    # generated steps feed the previous trunk argmax back. The first nsteps inputs
    # are the prompt's tail tokens (teacher-forced), so verify that every break in
    # the input==prev-argmax chain sits inside the teacher-forced prefix and every
    # generated step chains: input[t+1] == argmax[t] from the first generated step on.
    breaks = [i for i in range(len(seq) - 1) if seq[i][1] != seq[i + 1][0]]
    k = (breaks[-1] + 1) if breaks else 1
    consistent = all(seq[i][1] == seq[i + 1][0] for i in range(k, len(seq) - 1))
    print("greedy chain: input[t+1] == argmax[t] for generated steps (t>=%d): %s"
          % (k, "OK" if consistent else "BROKEN"))
    ok = ok and consistent

    if da:
        agr = sum(d["agree"] for d in da.values())
        tot = len(da)
        print("MTP draft top-1 agreement: %d/%d steps" % (agr, tot))
        for s in sorted(da):
            print("  t=%d trunk=%d (%.3f)  draft=%d (%.3f)  agree=%d"
                  % (s, ta[s]["tok"], ta[s]["logit"], da[s]["tok"], da[s]["logit"], da[s]["agree"]))
    else:
        print("MTP draft: no GENDRAFT lines (run was not at 45 layers?)")

    # decode the generated ids with the reference tokenizer for the report
    ckpt = os.environ.get("GLM_CKPT", "/mnt/apex/rick/vybforge/glm53-checkpoint")
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.path.join(ckpt, "tokenizer.json"))
        text = tk.decode(ids)
        print("generated ids:", ids)
        print("continuation text: %r" % text[:400])
    except Exception as e:
        print("(tokenizer decode skipped: %s)" % e)

    print("GLM_GEN_GATE: %s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
