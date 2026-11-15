#!/usr/bin/env python3
# GLM_TOKENIZER_ROUNDTRIP gate: run the Vyb-native tokenizer over the GLM corpus
# (test_glm_tokenizer.vyb) and compare ids against the `tokenizers` reference
# loaded from the real GLM-5.3 checkpoint, plus check the Vyb encode->decode
# round trip per prompt (reported by the driver as "RT ok" / "RT X").
#
# The gate is exactness on the ASCII corpus (prompts 0..23). Prompts 24..26 are
# non-ASCII divergence probes: the Vyb pre-tokenizer implements the GPT-2
# regex's ASCII behaviour only (no Unicode \p{L}/\p{N} classes), so CJK/emoji
# pre-token segmentation can differ from the `tokenizers` reference. Their ids
# are reported per prompt (documented divergence, same policy as the Qwen3
# harness); the Vyb round trip still holds for all of them.
import subprocess, re, os, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CKPT = os.environ.get("GLM_CKPT", "/mnt/apex/rick/vybforge/glm53-checkpoint")

# Must mirror test_glm_tokenizer.vyb's p0..p26 exactly (index = prompt id).
CORPUS = [
    "I want a disposable container rootfs for CI. What should we decide first?",
    "Add a package to a QEMU x86_64 boot test.",
    "Can VybOS already build a kernel and boot it in QEMU?",
    "Name this an ARM64 appliance image host.",
    "Apply this configuration to the current machine now.",
    "Enable a service on a rescue image with a serial console.",
    "Make this target an ARM64 image instead of x86_64.",
    "I no longer need interactive console access on a QEMU x86_64 boot test.",
    "I want an x86_64 development VM. What should we decide first?",
    "Add a package to an ARM64 appliance image.",
    "Name this a rescue image with a serial console host.",
    "Enable a service on a QEMU x86_64 boot test.",
    "I no longer need interactive console access on an ARM64 appliance image.",
    "I want a reproducible server image. What should we decide first?",
    "Add a package to a rescue image with a serial console.",
    "Name this a QEMU x86_64 boot test host.",
    "Enable a service on an ARM64 appliance image.",
    "I no longer need interactive console access on a rescue image with a serial console.",
    "The kernel version is 6.12.41, don't you know? It's 2026-11-13.",
    "  leading spaces and   multiple   gaps\there\nnewline ends: ",
    "id=12345 count=007 x9y8z7 3.14159 100%",
    "Hello, world! (parenthetical) [bracketed] {braced} <angled> a/b\\c|d&e*f",
    "line one\nline two\n\nline four\ttabbed\tend ",
    "What's the system's initramfs? It'll be built; we've decided.",
    "Resume: caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u2014 em-dash, ellipsis\u2026 done.",
    "\u4e2d\u6587\u6d4b\u8bd5\uff1a\u5185\u6838\u914d\u7f6e\u4e0e\u5f15\u5bfc\u53c2\u6570\u3002",
    "emoji test: ok \U0001f44d\U0001f3fd done",
]
N_ASCII = 24

try:
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(CKPT, "tokenizer.json"))
except Exception as e:
    print("GLM_TOKENIZER_ROUNDTRIP: SKIP (tokenizers lib unavailable:", e, ")")
    sys.exit(0)

env = dict(os.environ,
           VYBFORGE_TOKDIR="artifacts/glm53-tokenizer",
           VYBFORGE_DIGMAX="3",
           VYBFORGE_TOKIGNORE="1")
out = subprocess.run(
    [os.environ.get("VYB", os.path.expanduser("~/Projects/Vyb/build/vyb")),
     "native/tokenizer/test_glm_tokenizer.vyb", "--module-path", "native/tokenizer"],
    capture_output=True, text=True, cwd=ROOT, env=env)
if out.returncode != 0:
    print("GLM_TOKENIZER_ROUNDTRIP: FAIL (driver rc=%d)" % out.returncode)
    print(out.stderr[-3000:])
    sys.exit(1)

n = ok = rt_ok = 0
fail = False
for i, line in enumerate(out.stdout.splitlines()):
    m = re.match(r"^(\d+): \[([\d,]*)\] RT (ok|X)$", line)
    if not m:
        continue
    idx = int(m.group(1))
    ids = [int(x) for x in m.group(2).split(",") if x != ""]
    text = CORPUS[idx]
    ref = tk.encode(text, add_special_tokens=False).ids
    n += 1
    if m.group(3) == "ok":
        rt_ok += 1
    else:
        print(f"X  #{idx} round-trip broken: {text[:60]!r}")
        fail = True
    if ids == ref:
        ok += 1
    else:
        print(f"D  #{idx} (non-ASCII divergence)" if idx >= N_ASCII
              else f"X  #{idx} ASCII mismatch")
        print(f"   text={text[:60]!r}")
        print(f"   vyb={ids[:24]}{'...' if len(ids) > 24 else ''}")
        print(f"   ref={ref[:24]}{'...' if len(ref) > 24 else ''}")
        if idx < N_ASCII:
            fail = True

ascii_ok = sum(1 for idx in range(N_ASCII))
print(f"GLM_TOKENIZER_ROUNDTRIP: {'OK' if not fail else 'FAIL'}  "
      f"encode exact {ok}/{n} (gate = the {N_ASCII} ASCII prompts: "
      f"{'all exact' if not fail else 'MISMATCH'}), "
      f"round-trip {rt_ok}/{n}; non-ASCII probes documented above")
sys.exit(0 if not fail else 1)
