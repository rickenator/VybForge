# VybForge native GPU substrate — builds kernels to sm_86 PTX and runs the
# Vyb-native host drivers. No Python in the runtime path; verification is done
# inside Vyb (a numpy reference verifies the standalone single-layer forward).
# Run `make -f native/Makefile verify` from the repo root.
VYBDIR ?= /home/rick/Projects/Vyb
VYB     ?= $(VYBDIR)/build/vyb
ROOT    := $(shell git -C "$(dir $(abspath $(lastword $(MAKEFILE_LIST))))/.." rev-parse --show-toplevel 2>/dev/null || echo "$(abspath $(dir $(abspath $(lastword $(MAKEFILE_LIST))))/..)")
KSRC    := $(ROOT)/native/kernels/%.vyb
KPTX    := $(ROOT)/native/build/%.ptx
MODPATH := --module-path $(ROOT)/native/kernels

KERNELS := gemm rmsnorm math_probe layer deq q4k q6k backprop train
PTX     := $(addprefix $(ROOT)/native/build/, $(addsuffix .ptx, $(KERNELS)))

.PHONY: all verify gemm rmsnorm mathprobe layer gguf gguf-real q4k stack tensor decode gdecode clean

all: verify

# Compile a kernel source to PTX via kernel mode (with vmath in scope)
$(KPTX): $(KSRC)
	@mkdir -p $(ROOT)/native/build $(ROOT)/native/out
	cd $(ROOT)/native/build && $(VYB) --kernel ../kernels/$*.vyb --ptx $*.ptx $(MODPATH)

gemm: $(ROOT)/native/build/gemm.ptx
	cd $(ROOT) && $(VYB) native/host/gemm_driver.vyb

rmsnorm: $(ROOT)/native/build/rmsnorm.ptx
	cd $(ROOT) && $(VYB) native/host/rmsnorm_driver.vyb

mathprobe: $(ROOT)/native/build/math_probe.ptx
	cd $(ROOT) && $(VYB) native/host/math_probe_driver.vyb

layer: $(ROOT)/native/build/layer.ptx
	cd $(ROOT) && $(VYB) native/host/layer_driver.vyb
	python3 native/tools/verify_layer.py

gguf:
	mkdir -p $(ROOT)/native/gguf
	python3 $(ROOT)/native/gguf/mk_fixture.py
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

# Vyb port of tests/test_schema.py (schema + corpus agreement guard) in the runtime language
schemacheck:
	cd $(ROOT) && $(VYB) tests/test_schema.vyb --module-path native/json

# T1 ground truth: tokenize real corpus responses -> token ids + CE labels with the
# Vyb Qwen3 tokenizer, cross-checked against transformers (reference only).
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
	cd $(ROOT) && $(VYB) native/host/decode_driver.vyb
	cd $(ROOT) && python3 native/tools/verify_decode.py

# Real Qwen3-4B greedy decode: token stream == numpy decode_ref.py (decode_ids_ref.txt)
decode-real: $(ROOT)/native/build/layer.ptx $(ROOT)/native/build/q4k.ptx $(ROOT)/native/build/q6k.ptx $(ROOT)/native/build/qwen3.ptx
	cd $(ROOT) && .venv/bin/python native/gguf/decode_ref.py
	cd $(ROOT) && $(VYB) native/host/decode_driver.vyb
	cd $(ROOT) && .venv/bin/python native/tools/verify_decode_real.py

# KV-cache decode: builds the seed forward once (caching per-layer roped-K/V), then
# each gen step computes ONLY the new position. Must reproduce the same token stream
# as decode-real (verified against the same numpy reference).
decode-kv: $(ROOT)/native/build/layer.ptx $(ROOT)/native/build/q4k.ptx $(ROOT)/native/build/q6k.ptx $(ROOT)/native/build/qwen3.ptx
	cd $(ROOT) && $(VYB) native/host/kv_driver.vyb
	cd $(ROOT) && .venv/bin/python native/tools/verify_decode_real.py

# b1: LoRA-applied autoregressive decode of the goal context (m2e_l*.bin seed
# adapters) -> native/out/loradec_tokens.txt. The interviewer-runtime first pass.
loradec: $(ROOT)/native/build/train.ptx $(ROOT)/native/build/qwen3.ptx
	cd $(ROOT) && $(VYB) native/host/loradec_driver.vyb

# build all kernels then run every verified driver
verify: $(PTX)
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
	cd $(ROOT) && $(VYB) native/host/decode_driver.vyb
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

# Full-36-layer Qwen3-4B prefill + tied lm_head vs numpy reference (prefill gate)
prefill: $(ROOT)/native/build/layer.ptx $(ROOT)/native/build/q4k.ptx $(ROOT)/native/build/q6k.ptx $(ROOT)/native/build/qwen3.ptx
	cd $(ROOT) && .venv/bin/python native/gguf/prefill_ref.py
	cd $(ROOT) && $(VYB) native/host/model_driver.vyb
	cd $(ROOT) && .venv/bin/python native/tools/verify_prefill.py
