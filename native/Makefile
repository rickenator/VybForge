# VybForge native GPU substrate — builds kernels to sm_86 PTX and runs the
# Vyb-native host drivers. No Python in the runtime path; verification is done
# inside Vyb (a numpy reference verifies the standalone single-layer forward).
# Run `make -f native/Makefile verify` from the repo root.
# The Vyb checkout. VYBHOME is the canonical knob (VybForge#15); Vyb's build is
# expected to publish it in $VYBHOME/SOURCEME_VYB (rickenator/Vyb#424). VYBDIR stays
# as the make-level name the existing targets already use.
VYBHOME ?= $(HOME)/Projects/Vyb
VYBDIR  ?= $(VYBHOME)
VYB     ?= $(VYBDIR)/build/vyb
ROOT    := $(shell git -C "$(dir $(abspath $(lastword $(MAKEFILE_LIST))))/.." rev-parse --show-toplevel 2>/dev/null || echo "$(abspath $(dir $(abspath $(lastword $(MAKEFILE_LIST))))/..)")
KSRC    := $(ROOT)/native/kernels/%.vyb
KPTX    := $(ROOT)/native/build/%.ptx
MODPATH := --module-path $(ROOT)/native/kernels
# S0.2 substrate modules: the model's own config (config + its json reader), the tensor core,
# and the dtype table. All three are pure (no CUDA), so a driver declaring its own extern "C"
# can import them (Vyb#442 keeps native/tensor/tensor.vyb itself off this list).
S02MODS := --module-path native/config --module-path native/json --module-path native/tensor --module-path native/dtype

KERNELS := gemm rmsnorm math_probe layer deq q4k q6k backprop train dtcvt gdn
PTX     := $(addprefix $(ROOT)/native/build/, $(addsuffix .ptx, $(KERNELS)))

.PHONY: all verify gemm rmsnorm mathprobe layer gguf gguf-real q4k stack tensor decode gdecode clean

all: verify

# Compile a kernel source to PTX via kernel mode (with vmath in scope).
# A kernel is an artifact of the compiler that emitted it, so the rule depends on a stamp of
# WHICH toolchain last built it. An mtime prerequisite alone cannot tell that a .ptx came from
# a different compiler -- and that is exactly the case when a gate is run against a second
# checkout, where the kernel would otherwise be reused and the gate would certify the wrong
# toolchain.
TOOLSTAMP := $(ROOT)/native/build/.toolchain
.PHONY: FORCE
FORCE:
$(TOOLSTAMP): FORCE
	@mkdir -p $(ROOT)/native/build
	@printf '%s %s %s\n' "$(VYB)" "$$(stat -c '%s %Y' '$(VYB)' 2>/dev/null || echo missing)" \
	    "$$(sha256sum '$(VYB)' 2>/dev/null | cut -c1-16)" > $@.tmp
	@cmp -s $@.tmp $@ || mv -f $@.tmp $@
	@rm -f $@.tmp

$(KPTX): $(KSRC) $(TOOLSTAMP)
	@mkdir -p $(ROOT)/native/build $(ROOT)/native/out
	cd $(ROOT)/native/build && $(VYB) --kernel ../kernels/$*.vyb --ptx $*.ptx $(MODPATH)

gemm: $(ROOT)/native/build/gemm.ptx
	cd $(ROOT) && $(VYB) native/host/gemm_driver.vyb

rmsnorm: $(ROOT)/native/build/rmsnorm.ptx
	cd $(ROOT) && $(VYB) native/host/rmsnorm_driver.vyb

mathprobe: $(ROOT)/native/build/math_probe.ptx
	cd $(ROOT) && $(VYB) native/host/math_probe_driver.vyb

# The GDN recurrent layer's kernels (l2 norm, delta-rule step, gated RMS epilogue). The input fixture
# is written by the verifier, which also does the comparison — so the gate is the entry point:
#   ./native/legit/run_gdn_kernel_gate.sh
gdn: $(ROOT)/native/build/gdn.ptx
	cd $(ROOT) && env -u PYTHONPATH $(ROOT)/.venv/bin/python native/tools/gdn_kernel_verify.py

# The whole recurrent block wired on the GPU vs ggml's own graph, stage by stage. Needs the rmsnorm
# kernel too (the driver imports attn_norm from it) and libggml for the authority:
#   ./native/legit/run_gdn_layer_gate.sh
gdn-layer: $(ROOT)/native/build/gdn.ptx $(ROOT)/native/build/rmsnorm.ptx
	cd $(ROOT) && env -u PYTHONPATH $(ROOT)/.venv/bin/python native/tools/gdn_layer_kernel_verify.py

layer: $(ROOT)/native/build/layer.ptx
	cd $(ROOT) && $(VYB) native/host/layer_driver.vyb
	python3 native/tools/verify_layer.py

# The synthetic GGUF fixture, written by the Vyb port (P1.7). It is git-ignored
# and generated on demand, so `verify` depends on it explicitly rather than
# assuming a previous `make gguf` left it behind. Gate (byte parity against the
# Python original's output): native/legit/run_gguf_gate.sh
$(ROOT)/native/gguf/test.gguf: $(ROOT)/native/gguf/mk_fixture.vyb
	@mkdir -p $(ROOT)/native/gguf
	cd $(ROOT) && $(VYB) native/gguf/mk_fixture.vyb

gguf: $(ROOT)/native/gguf/test.gguf
	cd $(ROOT) && $(VYB) native/gguf/parse_gguf.vyb
	python3 $(ROOT)/native/gguf/verify_gguf.py

# Phase-1 swap-in: real Qwen3-4B GGUF metadata reader (uses stdlib io::read_at #207)
gguf-real:
	cd $(ROOT) && $(VYB) native/gguf/read_real_meta.vyb
	cd $(ROOT) && python3 native/gguf/verify_gguf_meta.py

# Phase-2 swap-in: q4_K dequant kernel on a real model weight block
q4k: $(ROOT)/native/build/q4k.ptx
	cd $(ROOT) && $(VYB) native/host/q4k_driver.vyb
	cd $(ROOT) && python3 native/gguf/q4k_verify.py

# Phase-3 swap-in (issue #5 storage boundary): stream real Q4_K weights via
# read_at + bulk q4k dequant to f32, packed stays canonical, memory reported
q4k-load: $(ROOT)/native/build/q4k.ptx
	cd $(ROOT) && $(VYB) native/host/q4k_load_driver.vyb
	cd $(ROOT) && python3 native/gguf/attnq_slice_verify.py

# Phase-3 step 2: q6_K dequant (attn_v/ffn_down/token_embd bodies)
q6k: $(ROOT)/native/build/q6k.ptx
	cd $(ROOT) && $(VYB) native/host/q6k_load_driver.vyb
	cd $(ROOT) && python3 native/gguf/q6k_verify.py

stack: $(ROOT)/native/build/layer.ptx
	cd $(ROOT) && $(VYB) native/host/stack_driver.vyb
	cd $(ROOT) && python3 native/tools/verify_stack.py

tensor: $(ROOT)/native/build/gemm.ptx $(ROOT)/native/build/rmsnorm.ptx
	cd $(ROOT) && $(VYB) native/tensor/test_tensor.vyb --module-path native/tensor --module-path $(VYBDIR)/bindings/cuda

gdecode:
	cd $(ROOT) && $(VYB) native/gdecode/test_contract.vyb --module-path native/gdecode
	cd $(ROOT) && python3 native/gdecode/verify_contract.py

gdecode-pipeline:
	cd $(ROOT) && ./native/gdecode/run_pipeline.sh

# Schema + corpus agreement guard in the runtime language (Vyb port of the former Python verifier)
schemacheck:
	cd $(ROOT) && $(VYB) tests/test_schema.vyb --module-path native/json

# T1 ground truth: tokenize real corpus responses -> token ids + CE labels with the
# Vyb Qwen3 tokenizer, cross-checked against transformers (reference only).
#
# RESOLVED (Vyb#432, fixed by 1a9484b9 / merged 74220c3d): the whole-module import
# splice now gives colliding declarations a per-module identity, so
# `import tokenizer::{encode}` + `import json_parse` coexist as written — the
# consumer-side one-liner (`import json_parse::{parse}`) was NOT needed and is not
# applied. First green run 2026-10-05 against Vyb 7dd83764:
# `ENCODE_CORPUS_VERIFY: OK  (4/4 exact match, ids+lables derived)`.
# The oracle needs the repo's own `.venv` to be the numpy/transformers the process
# actually imports: an ambient PYTHONPATH pointing at another environment's
# site-packages makes it resolve a foreign numpy and the script SKIPs (reads as a
# missing dependency when the real cause is the leak) -- invoke it with a clean
# PYTHONPATH, e.g. `env -u PYTHONPATH make -f native/Makefile encode-corpus`.
encode-corpus:
	cd $(ROOT) && $(VYB) native/train/encode_corpus.vyb --module-path native/tokenizer --module-path native/json
	cd $(ROOT) && .venv/bin/python native/train/verify_encode_corpus.py

# T2 training-forward activation cache: run the forward + dump cached activations; the
# reference computes the same layer-0 tensors; verify_train_fwd.py gates max-rel < 2e-5.
fwd-cache: $(ROOT)/native/build/layer.ptx
	cd $(ROOT) && .venv/bin/python native/train/train_fwd_ref.py
	cd $(ROOT) && $(VYB) native/train/train_fwd.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_train_fwd.py

# T3.1 backward kernels (dSiLU, dresid): synthetic inputs -> analytic vs kernel grads.
bwd-probe: $(ROOT)/native/build/backprop.ptx
	cd $(ROOT) && .venv/bin/python native/train/bwd_ref.py
	cd $(ROOT) && $(VYB) native/train/bwd_probe.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_bwd.py

# T3.2 dGEMM + LoRA grads (dU/dV/ds/dx via dgemm_at/dgemm_bt) vs analytic reference.
bwd-lora: $(ROOT)/native/build/backprop.ptx
	cd $(ROOT) && .venv/bin/python native/train/bwd2_ref.py
	cd $(ROOT) && $(VYB) native/train/bwd2_probe.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_bwd2.py

# T3.3 dRMSNorm grads (dx = dy.w.inv - x.inv^3.T/D ; dw row-sum) vs analytic reference.
bwd-norm: $(ROOT)/native/build/backprop.ptx
	cd $(ROOT) && .venv/bin/python native/train/bwd3_ref.py
	cd $(ROOT) && $(VYB) native/train/bwd3_probe.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_bwd3.py

# T3.4 dRoPE adjoint (inverse split-half rotation) vs analytic reference.
bwd-rope: $(ROOT)/native/build/backprop.ptx
	cd $(ROOT) && .venv/bin/python native/train/bwd4_ref.py
	cd $(ROOT) && $(VYB) native/train/bwd4_probe.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_bwd4.py

# T3.5 dAttention (softmax backward, atomics for cross-query dK/dV) vs analytic reference.
bwd-attn: $(ROOT)/native/build/backprop.ptx
	cd $(ROOT) && .venv/bin/python native/train/bwd5_ref.py
	cd $(ROOT) && $(VYB) native/train/bwd5_probe.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_bwd5.py

# T4 LoRA apply (gemmf + addscaled: h=x@W+alpha_r*(x@U@V)) + one AdamW step vs analytic.
t4-lora: $(ROOT)/native/build/backprop.ptx
	cd $(ROOT) && .venv/bin/python native/train/t4_ref.py
	cd $(ROOT) && $(VYB) native/train/t4_probe.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_t4.py

# M2b real blk.0 forward with LoRA on all 7 projections + activation cache, vs numpy.
train-layer: $(ROOT)/native/build/train.ptx
	cd $(ROOT) && .venv/bin/python native/train/train_layer_ref.py
	cd $(ROOT) && $(VYB) native/train/train_layer.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_train_layer.py

# M2c real blk.0 backward chain -> LoRA grads (7 projections) + dx, vs numpy oracle.
train-layer-back: $(ROOT)/native/build/train.ptx
	cd $(ROOT) && .venv/bin/python native/train/train_layer_back_ref.py
	cd $(ROOT) && $(VYB) native/train/train_layer_back.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_train_layer_back.py

# M2d teacher-forced real blk.0 LoRA TRAINING LOOP: forward + MSE + verified backward +
# AdamW on U/V (7 pairs), overfit one fixed target; per-step loss vs numpy oracle + descent.
train-layer-train: $(ROOT)/native/build/train.ptx
	cd $(ROOT) && .venv/bin/python native/train/train_layer_ref.py
	cd $(ROOT) && .venv/bin/python native/train/train_layer_train_ref.py
	cd $(ROOT) && $(VYB) native/train/train_layer_train.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_train_layer_train.py

# M2e-1 FULL 36-layer LoRA forward + per-layer activation cache, vs numpy (train_full_ref).
train-full-forward: $(ROOT)/native/build/train.ptx
	cd $(ROOT) && .venv/bin/python native/train/train_full_ref.py
	cd $(ROOT) && $(VYB) native/train/train_full.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_train_full.py

# M2e-2 FULL 36-layer LoRA backward -> per-projection grads + chained dx, vs numpy oracle.
train-full-back: $(ROOT)/native/build/train.ptx
	cd $(ROOT) && .venv/bin/python native/train/train_full_ref.py
	cd $(ROOT) && .venv/bin/python native/train/train_full_back_ref.py
	cd $(ROOT) && $(VYB) native/train/train_full_back.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_train_full_back.py

# M2e-3 FULL 36-layer LoRA TRAINING LOOP on GPU, per-step loss vs numpy oracle.
# (Slow: ~40 min/step, weight streaming is per-8-byte H2D. Run with a long timeout / in bg.)
train-full-loop: $(ROOT)/native/build/train.ptx
	cd $(ROOT) && .venv/bin/python native/train/train_full_loop_ref.py
	cd $(ROOT) && $(VYB) native/train/train_full_loop.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_train_full_loop.py

# M2e-3 PACKED-WEIGHT-CACHE variant: weights dequantized device-side ONCE into PKD, forward/backward
# index slabs instead of streaming GGUF per-8-byte -> ~3 min/step (was ~40). ROCKSOLVED: step 2+ were
# re-seeding forward with the clobbered XA scratch (input aliased by layer swaps); fixed with a
# pristine XIN buffer refilled into XA each step. GPU now matches the FD-verified oracle per-step and
# DESCENDS. Gate = verify_train_full_loop.py (per-step loss + L0/17/35 adapter/moment parity).
train-full-loop-pkd: $(ROOT)/native/build/train.ptx
	cd $(ROOT) && .venv/bin/python native/train/train_full_loop_ref.py
	cd $(ROOT) && $(VYB) native/train/train_full_loop_pkd.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_train_full_loop.py

# Fast reduced-mode repro for the packed full-loop (VybForge#12). Override layers/steps
# via make vars M2E3_MAXL / M2E3_NSTP (default 4 layers / 2 steps => minutes, not 18h).
# e.g.  make -f native/Makefile train-full-loop-pkd-fast M2E3_MAXL=2 M2E3_NSTP=1
M2E3_MAXL ?= 4
M2E3_NSTP ?= 2
train-full-loop-pkd-fast: $(ROOT)/native/build/train.ptx
	cd $(ROOT) && env M2E3_MAXL=$(M2E3_MAXL) M2E3_NSTP=$(M2E3_NSTP) .venv/bin/python native/train/train_full_loop_ref.py
	cd $(ROOT) && env M2E3_MAXL=$(M2E3_MAXL) M2E3_NSTP=$(M2E3_NSTP) $(VYB) native/train/train_full_loop_pkd.vyb
	cd $(ROOT) && env M2E3_MAXL=$(M2E3_MAXL) M2E3_NSTP=$(M2E3_NSTP) .venv/bin/python native/train/verify_train_full_loop.py

# M2e-3-CE: the PRODUCTION corpus objective on GPU — teacher-forced next-token CE over a real
# record-0 slice (S=4): embed via token_embd, 36-layer LoRA forward, output_norm, TIED lm_head
# (h @ token_embd.T), softmax CE + head-back -> the 36-layer backward -> AdamW on all U/V.
# Finite head gradient -> real descent (CE 15.49->0.697). Gate = verify_ce.py (per-step loss
# vs the numpy oracle + L0 Uq final parity).
train-ce-loop: $(ROOT)/native/build/train.ptx
	cd $(ROOT) && .venv/bin/python native/train/train_corpus_ce_ref.py
	cd $(ROOT) && $(VYB) native/train/train_full_ce.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_ce.py

# Frozen-context KV training ORACLE (numpy): caches the 9-token goal context per-layer roped-K/V
# once; per-token response forward == full-sequence; masked CE on response only; FROZEN-CONTEXT
# backward (context-row q/k/v attention grads zeroed, context LoRA frozen); AdamW on response LoRA.
# Writes the GPU target (kvresp_train_loss_ref.txt + step-1 dU_q/dV_q refs). Must descend.
kvresp-train-oracle:
	cd $(ROOT) && .venv/bin/python native/train/kvresp_train_ref.py

# FROZEN-CONTEXT KV TRAINING on GPU (kvresp_train.vyb): reuses the validated batched S=93
# forward+backward+AdamW engine (train_full_ce.vyb) with the context-region (rows 0..NCctx-1)
# attention q/k/v grads ZEROED after dattn -- context K/V frozen, response-only LoRA updates.
# Gate = KVRESP_TRAIN_VERIFY: step-1 dU_q/dV_q (L0/17/35) match the numpy oracle to ~e-6,
# per-step CE loss matches kvresp_train_loss_ref.txt (15.957->14.095) and descends.
kvresp-train: $(ROOT)/native/build/train.ptx
	cd $(ROOT) && .venv/bin/python native/train/kvresp_train_ref.py
	cd $(ROOT) && $(VYB) native/train/kvresp_train.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_kvresp_train.py

# TRUE per-token KV trainer (kvresp_train_kv.vyb): context built once per-token, response forward
# per-token attending the combined roped-k/v cache, unchanged S=93 frozen backward + AdamW.
# Same oracle gate. (nv only: kvresp_train_kv shares output files with kvresp_train.)
kvresp-train-kv: $(ROOT)/native/build/train.ptx
	cd $(ROOT) && .venv/bin/python native/train/kvresp_train_ref.py
	cd $(ROOT) && $(VYB) native/train/kvresp_train_kv.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_kvresp_train.py

decode: $(ROOT)/native/build/layer.ptx
	cd $(ROOT) && VYB_STDLIB=$(VYBDIR)/stdlib $(VYB) native/host/decode_driver.vyb --module-path native/llm
	cd $(ROOT) && python3 native/tools/verify_decode.py

# Real Qwen3-4B greedy decode: token stream == numpy decode_ref.py (decode_ids_ref.txt)
decode-real: $(ROOT)/native/build/layer.ptx $(ROOT)/native/build/q4k.ptx $(ROOT)/native/build/q6k.ptx $(ROOT)/native/build/qwen3.ptx
	cd $(ROOT) && .venv/bin/python native/gguf/decode_ref.py
	cd $(ROOT) && VYB_STDLIB=$(VYBDIR)/stdlib $(VYB) native/host/decode_driver.vyb --module-path native/llm
	cd $(ROOT) && .venv/bin/python native/tools/verify_decode_real.py

# KV-cache decode: builds the seed forward once (caching per-layer roped-K/V), then
# each gen step computes ONLY the new position. Must reproduce the same token stream
# as decode-real (verified against the same numpy reference). The reference is
# regenerated here -- this target used to rely on whatever decode_ref.py last wrote,
# which is the stale-gold trap that hid the chat-prompt breakage.
decode-kv: $(ROOT)/native/build/layer.ptx $(ROOT)/native/build/q4k.ptx $(ROOT)/native/build/q6k.ptx $(ROOT)/native/build/qwen3.ptx
	cd $(ROOT) && .venv/bin/python native/gguf/decode_ref.py
	cd $(ROOT) && $(VYB) native/host/kv_driver.vyb
	cd $(ROOT) && .venv/bin/python native/tools/verify_decode_real.py

# kvctx: the per-layer roped-K/V context cache vs the numpy reference (kvctx_ref.py).
# Both sides regenerated here, so a stale .bin cannot be what the driver is measured against.
kvctx: $(ROOT)/native/build/train.ptx
	cd $(ROOT) && .venv/bin/python native/train/kvctx_ref.py
	cd $(ROOT) && $(VYB) native/train/kvctx.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_kvctx.py

# kvrespfwd: the KV-cache response forward (S=93) vs the authoritative numpy full forward
# (kvresp_ref.py, ~30-60 min: it is a 93-token CPU forward).
kvrespfwd: $(ROOT)/native/build/train.ptx
	cd $(ROOT) && .venv/bin/python native/train/kvresp_ref.py
	cd $(ROOT) && $(VYB) native/train/kvrespfwd.vyb
	cd $(ROOT) && .venv/bin/python native/train/verify_kvrespfwd.py

# b1: LoRA-applied autoregressive decode of the goal context (m2e_l*.bin seed
# adapters) -> native/out/loradec_tokens.txt. The interviewer-runtime first pass.
loradec: $(ROOT)/native/build/train.ptx $(ROOT)/native/build/qwen3.ptx
	cd $(ROOT) && $(VYB) native/host/loradec_driver.vyb

# #14: tuned decode -> text -> post-filter/repair + schema-coerce -> schema-valid
# agent-response contract (writes native/out/contract_loradec.json). Runs decode
# in JSON mode (VYB_JSON_MODE=1) so it opens with the '{"' contract opener; the
# coerce is tolerant of truncated JSON (decode token budget often cuts mid-object).
# All three stages are Vyb; the independent jsonschema verification of the result
# lives in native/legit/run_coerce_gate.sh (verification-only, never in this path).
loradec-contract: $(ROOT)/native/build/train.ptx $(ROOT)/native/build/qwen3.ptx
	cd $(ROOT) && VYB_JSON_MODE=1 $(VYB) native/host/loradec_driver.vyb
	cd $(ROOT) && $(VYB) native/gdecode/loradec_emit.vyb --module-path native/tokenizer
	cd $(ROOT) && $(VYB) native/gdecode/coerce_contract.vyb --module-path native/json

# build all kernels then run every verified driver
verify: $(PTX) $(ROOT)/native/gguf/test.gguf
	@echo "== GEMM =="
	cd $(ROOT) && $(VYB) native/host/gemm_driver.vyb
	@echo "== RMSNORM =="
	cd $(ROOT) && $(VYB) native/host/rmsnorm_driver.vyb
	@echo "== VMATH =="
	cd $(ROOT) && $(VYB) native/host/math_probe_driver.vyb
	@echo "== ONE-LAYER FORWARD (numpy reference) =="
	cd $(ROOT) && $(VYB) native/host/layer_driver.vyb
	cd $(ROOT) && python3 native/tools/verify_layer.py
	@echo "== MULTI-LAYER STACK (4x, numpy reference) =="
	cd $(ROOT) && $(VYB) native/host/stack_driver.vyb
	cd $(ROOT) && python3 native/tools/verify_stack.py
	@echo "== TENSOR:: WRAPPER (gemm + rmsnorm via module + cuLaunchKernel) =="
	cd $(ROOT) && $(VYB) native/tensor/test_tensor.vyb --module-path native/tensor --module-path $(VYBDIR)/bindings/cuda
	@echo "== GREEDY AUTOREGRESSIVE DECODE (numpy reference) =="
	cd $(ROOT) && VYB_STDLIB=$(VYBDIR)/stdlib $(VYB) native/host/decode_driver.vyb --module-path native/llm
	cd $(ROOT) && python3 native/tools/verify_decode.py
	@echo "== STOCHASTIC SAMPLER (temp + top_k + top_p + LCG) =="
	cd $(ROOT) && $(VYB) native/sampler/test_sampler.vyb --module-path native/sampler
	cd $(ROOT) && python3 native/sampler/verify_sampler.py
	@echo "== GGUF READER + Q4_0 DEQUANT ON-GPU =="
	cd $(ROOT) && $(VYB) native/gguf/parse_gguf.vyb
	cd $(ROOT) && python3 native/gguf/verify_gguf.py
	cd $(ROOT) && $(VYB) native/gguf/dequant_gguf.vyb
	@echo "== JSON PARSER =="
	cd $(ROOT) && $(VYB) native/json/test_json.vyb --module-path native/json
	@echo "== BPE TOKENIZER (vs transformers reference) =="
	cd $(ROOT) && $(VYB) native/tokenizer/test_tokenizer.vyb --module-path native/tokenizer
	cd $(ROOT) && (.venv/bin/python native/tokenizer/verify_tokenizer.py || echo "TOKENIZER_VERIFY: SKIP (venv)")
	@echo "== G-DECODE: agent-response CONTRACT EMISSION (jsonschema) =="
	cd $(ROOT) && $(VYB) native/gdecode/test_contract.vyb --module-path native/gdecode
	cd $(ROOT) && python3 native/gdecode/verify_contract.py

clean:
	rm -rf $(ROOT)/native/build $(ROOT)/native/out

# Phase-3 swap-in step 4: real Qwen3-4B blk.0 layer-0 forward vs numpy reference
# (gate). Builds the layer/kernels + qwen3 kernel, runs the staged forward.
layer0: $(ROOT)/native/build/layer.ptx $(ROOT)/native/build/q4k.ptx $(ROOT)/native/build/q6k.ptx $(ROOT)/native/build/qwen3.ptx
	cd $(ROOT) && .venv/bin/python native/gguf/layer0_ref.py
	cd $(ROOT) && $(VYB) native/host/layer0_driver.vyb
	cd $(ROOT) && .venv/bin/python native/tools/verify_layer0.py

# wcheck: dequant attn_v (L0/L4/L35) and ffn_down (L4/L35) on the GPU and compare the
# first 6 f64 values of each buffer against numpy's read_weight for the same tensors.
# Both sides are regenerated here; the comparison is verify_wcheck.py, not eyeballing.
wcheck: $(ROOT)/native/build/q4k.ptx $(ROOT)/native/build/q6k.ptx
	cd $(ROOT) && .venv/bin/python native/gguf/wcheck_ref.py > native/out/wcheck_ref.log 2>&1
	cd $(ROOT) && $(VYB) native/host/wcheck_driver.vyb | tee native/out/wcheck_driver.log
	cd $(ROOT) && .venv/bin/python native/tools/verify_wcheck.py

# Full-36-layer Qwen3-4B prefill + tied lm_head vs numpy reference (prefill gate).
# model_driver now composes llm:: (stdlib/vllm CPU decode) as the closed-loop
# real-model chat head, so it needs the llm module path + the stdlib vllm engine.
# VYB_PROMPT_RAW=1 selects the synthetic 2-token prefix [0,1]: this gate is numeric
# (hidden + per-position top1 vs prefill_ref.py, which is built for exactly that
# prefix), so it stays tokenizer-free on purpose. The driver's decoded text is NOT
# a result here -- with a 2-character punctuation input there is no language being
# requested; read PREFILL_HIDDEN_MATCH / PREFILL_TOP1_MATCH. For an English
# response see chat-real / chat-prompt below, which run a real prompt.
prefill: $(ROOT)/native/build/layer.ptx $(ROOT)/native/build/q4k.ptx $(ROOT)/native/build/q6k.ptx $(ROOT)/native/build/qwen3.ptx
	cd $(ROOT) && .venv/bin/python native/gguf/prefill_ref.py
	cd $(ROOT) && VYB_PROMPT_RAW=1 VYB_STDLIB=$(VYBDIR)/stdlib $(VYB) native/host/model_driver.vyb $(S02MODS) --module-path native/llm
	cd $(ROOT) && .venv/bin/python native/tools/verify_prefill.py

# Real-model llm.chat closed loop: full 36-layer forward -> tied lm_head argmax ->
# stdlib/vllm CPU decode to text, on the RTX 3090. Requires the real GGUF + the
# real BPE tokenizer dir (VYB_LLM_DIR, default artifacts/vybos-configurator-lora).
# Runs a REAL prompt (CHAT_PROMPT), so LLM_CHAT_RESPONSE is a genuine English
# continuation. The numpy gold is regenerated here (encode probe -> prompt_ids ->
# prompt_ref.py) and the driver's per-position top1 must match it token-for-token:
# a committed gold file must never be the thing that is trusted, because a stale
# gold and a buggy forward can agree and leave the gate green.
chat-real: $(ROOT)/native/build/layer.ptx $(ROOT)/native/build/q4k.ptx $(ROOT)/native/build/q6k.ptx $(ROOT)/native/build/qwen3.ptx
	cd $(ROOT) && VYB_STDLIB=$(VYBDIR)/stdlib VYB_PROMPT="$(CHAT_PROMPT)" $(VYB) native/llm/llm_encode_probe.vyb --module-path native/llm | tee native/out/chat_prompt_encode.log
	cd $(ROOT) && .venv/bin/python native/tools/emit_prompt_ids.py "$(CHAT_PROMPT)"
	cd $(ROOT) && .venv/bin/python native/gguf/prompt_ref.py
	cd $(ROOT) && VYB_PROMPT="$(CHAT_PROMPT)" VYB_STDLIB=$(VYBDIR)/stdlib $(VYB) native/host/model_driver.vyb $(S02MODS) --module-path native/llm 2>&1 | tee native/out/chat_real.log
	cd $(ROOT) && .venv/bin/python native/tools/verify_chat_real.py

# Checkpoint B — arbitrary-prompt llm.chat: GPU token_embd gather embeds a real
# prompt's ids (from the stdlib/vllm CPU tokenizer), full 36-layer forward, then
# per-position lm_head argmax must equal the numpy gold token-for-token.
# The gold is regenerated HERE on every run (encode probe -> prompt_ids ->
# prompt_ref.py), because the prompt_top1_ref.txt sitting in native/out/ was dated
# 2026-09-11 -- before the 09-13 orientation fix landed in the numpy reference --
# and did not even correspond to CHAT_PROMPT (its ids were for another prompt
# entirely). Together with the quoting bug below, that means this gate had never
# compared anything. Regenerating in-target is what makes that impossible.
CHAT_PROMPT ?= The capital of France is
chat-prompt: $(ROOT)/native/build/layer.ptx $(ROOT)/native/build/q4k.ptx $(ROOT)/native/build/q6k.ptx $(ROOT)/native/build/qwen3.ptx
	cd $(ROOT) && VYB_STDLIB=$(VYBDIR)/stdlib VYB_PROMPT="$(CHAT_PROMPT)" $(VYB) native/llm/llm_encode_probe.vyb --module-path native/llm | tee native/out/chat_prompt_encode.log
	cd $(ROOT) && .venv/bin/python native/tools/emit_prompt_ids.py "$(CHAT_PROMPT)"
	cd $(ROOT) && .venv/bin/python native/gguf/prompt_ref.py
	cd $(ROOT) && VYB_PROMPT="$(CHAT_PROMPT)" VYB_STDLIB=$(VYBDIR)/stdlib $(VYB) native/host/model_driver.vyb $(S02MODS) --module-path native/llm 2>&1 | tee native/out/chat_prompt.log
	cd $(ROOT) && .venv/bin/python native/tools/verify_chat_prompt.py

# Manual alias kept for a quick gold-only refresh (the gate targets above already
# regenerate it, so this is no longer needed to make chat-prompt/chat-real pass).
chat-prompt-ref:
	cd $(ROOT) && VYB_STDLIB=$(VYBDIR)/stdlib VYB_PROMPT="$(CHAT_PROMPT)" $(VYB) native/llm/llm_encode_probe.vyb --module-path native/llm | tee native/out/chat_prompt_encode.log
	cd $(ROOT) && .venv/bin/python native/tools/emit_prompt_ids.py "$(CHAT_PROMPT)"
	cd $(ROOT) && .venv/bin/python native/gguf/prompt_ref.py

# Checkpoint C — fluent autoregressive llm.chat over an arbitrary prompt.
# decode_driver (VYB_PROMPT_IDS prompt mode) seeds the recompute-full-prefix
# greedy gen loop from the prompt ids and writes the full generated stream; the
# numpy autoregressive gold (chatgen_ref.py) is compared token-for-token, and the
# generated continuation is decoded to a String via the stdlib/vllm CPU head.
CHAT_GEN ?= 3
chat-gen: $(ROOT)/native/build/layer.ptx $(ROOT)/native/build/q4k.ptx $(ROOT)/native/build/q6k.ptx $(ROOT)/native/build/qwen3.ptx
	cd $(ROOT) && .venv/bin/python native/gguf/chatgen_ref.py
	cd $(ROOT) && VYB_STDLIB=$(VYBDIR)/stdlib VYB_PROMPT_IDS=native/out/prompt_ids.txt VYB_GEN=$(CHAT_GEN) $(VYB) native/host/decode_driver.vyb --module-path native/llm 2>&1 | tee native/out/chat_gen.log
	cd $(ROOT) && .venv/bin/python native/tools/verify_chatgen.py

# Resident throughput: decode_driver's model (packed weights, embedding, norms)
# loads ONCE onto the GPU, then NRUNS follow-up interactions each run the proven
# recompute-full-prefix gen loop (GEN tokens). Per-run wall time is printed and
# averaged -> tokens/sec once the model is resident. No re-load per interaction.
CHAT_GEN ?= 3
RESIDENT_RUNS ?= 3
resident-bench: $(ROOT)/native/build/layer.ptx $(ROOT)/native/build/q4k.ptx $(ROOT)/native/build/q6k.ptx $(ROOT)/native/build/qwen3.ptx
	cd $(ROOT) && VYB_STDLIB=$(VYBDIR)/stdlib VYB_PROMPT_IDS=native/out/prompt_ids.txt VYB_GEN=$(CHAT_GEN) VYB_RUNS=$(RESIDENT_RUNS) $(VYB) native/host/resident_driver.vyb --module-path native/llm 2>&1 | tee native/out/resident_bench.log
	cd $(ROOT) && .venv/bin/python native/tools/verify_resident_bench.py

# ── Ridge GGUF tensor inventory (VybForge#10, phase 3 groundwork) ────────────────
# The inventory is a FUNCTION OF THE FILE, so regenerate it here rather than editing
# the TSV by hand; the coverage section names the quant types this engine does not
# implement yet, which is the phase-3 work list. Needs the reference GGUF:
#   hf download empero-ai/Qwen3.8-27B-Ridge-GGUF Qwen3.8-27B-Ridge-3.7bpw.gguf \
#     --local-dir $(HOME)/Models/qwen38-27b-ridge
# Override RIDGE_GGUF for a different file (e.g. the vision tower, which has its own
# mmproj-*.gguf and inventories cleanly through the same tool).
RIDGE_GGUF ?= $(HOME)/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf
ridge-inventory:
	cd $(ROOT) && .venv/bin/python native/gguf/ridge_inventory.py "$(RIDGE_GGUF)" \
	  --tsv native/gguf/ridge-3.7bpw-inventory.tsv --meta native/out/ridge_meta.txt

# ── S0.2e capability / layer descriptor gate (VybForge#10 phase 2) ───────────────
# model_caps.vyb answers "what does this model REQUIRE of the engine" (layer kinds and their
# state, the quant types present, vision/MTP/tokenizer) and refuses with a NAMED reason when
# this build cannot run it. The gate checks the dense model, the Ridge target, the vision tower
# and a truncated file. The Ridge/mmproj cases SKIP when those files are not on disk.
caps:
	./native/legit/run_caps_gate.sh

# ── Q8_0 dequant (VybForge#10 phase 3, first Ridge kernel) ──────────────────────
# The Gated-DeltaNet state path of Qwen3.8-27B Ridge (ssm_alpha/ssm_beta, 96 tensors).
# Checks the GPU kernel against native/tools/q8_0_ref.py on real tensors, and that reference
# against the INDEPENDENT python gguf dequantizer — so the comparison is not self-referential.
# Needs the 12 GiB Ridge GGUF; reports SKIP without it.
q8_0: $(ROOT)/native/build/q8_0.ptx
	./native/legit/run_q8_0_gate.sh

# ── IQ2_S dequant (VybForge#10 phase 3, the largest Ridge type) ─────────────────
# 160 tensors / 4.25 GiB of the mid-stack FFN. The authority is llama.cpp's own dequant, copied
# verbatim out of the local checkout and compiled as-is (the gguf package has no IQ2_S), and the
# grid table rides in the packed buffer because the shared launcher takes four arguments.
iq2_s: $(ROOT)/native/build/iq2s.ptx
	./native/legit/run_iq2_s_gate.sh

# ── Q5_K dequant (VybForge#10 phase 3, full-attention weight type) ──────────────
# 51 tensors / 0.80 GiB: the full-attention layers' q/k/v. Same authority pattern as iq2_s —
# llama.cpp's own dequantize_row_q5_K + get_scale_min_k4, extracted verbatim and compiled.
q5k: $(ROOT)/native/build/q5k.ptx
	./native/legit/run_q5k_gate.sh

# ── IQ3_S dequant (VybForge#10 phase 3, edge-layer FFN type) ────────────────────
# 32 tensors / 1.14 GiB: blk.0-3 and blk.60-63's ffn_gate/up/down. Grid-based like IQ2_S (512
# entries), with the table as an ordinary fifth argument.
iq3_s: $(ROOT)/native/build/iq3s.ptx
	./native/legit/run_iq3_s_gate.sh

# ── BF16 dequant (VybForge#10 phase 3, the vision tower's weight type) ──────────
# 110 tensors / 0.85 GiB in the mmproj. Not a block quant — 2 bytes per element, and the kernel
# uses the language's own ld_bf16. Checked against the independent python gguf package.
bf16: $(ROOT)/native/build/bf16.ptx
	./native/legit/run_bf16_gate.sh


