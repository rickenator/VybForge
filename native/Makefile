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

KERNELS := gemm rmsnorm math_probe layer deq
PTX     := $(addprefix $(ROOT)/native/build/, $(addsuffix .ptx, $(KERNELS)))

.PHONY: all verify gemm rmsnorm mathprobe layer gguf stack tensor decode gdecode clean

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

decode: $(ROOT)/native/build/layer.ptx
	cd $(ROOT) && $(VYB) native/host/decode_driver.vyb
	cd $(ROOT) && python3 native/tools/verify_decode.py

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
