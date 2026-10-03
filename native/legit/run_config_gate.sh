#!/usr/bin/env bash
# run_config_gate.sh — S0.2a gate: the model config contract vs independent authorities.
#
# What it proves. native/config/model_config.vyb is the single place a model's dimensions come
# from. This gate checks every field it reports against something that is NOT that code:
#
#   Qwen3-4B GGUF (the model the engine runs today)
#     authority 1  llama.cpp's own gguf-dump — its metadata section AND its tensor table, so
#                  the dims the reader derives from the tensor index (vocab, ffn, nq, nkv,
#                  layers, tied) are checked by a second, independent reading of that index
#     authority 2  the python `gguf` package, when importable (reported SKIPPED otherwise)
#     authority 3  the rope_theta -> invfreq table the kernels actually consume
#
#   SpikingBrain-7B config.json (a SECOND real model: different architecture, different field
#   names — norm_eps for rms_norm_eps, no head_dim, an explicit tie_word_embeddings, GLA/SWA)
#     authority 1  transformers.AutoConfig (SKIPPED when the architecture is not loadable)
#     authority 2  a plain json.load cross-check
#
# Discrimination — a gate that cannot fail is decoration, so inputs are chosen to break the
# reader on purpose: a config.json with hidden_size mutated (the reader must report the mutated
# value, proving it reads the file rather than remembering Qwen3), one with
# num_key_value_heads removed (the documented fallback must fire AND be labelled as a
# default), and a truncated GGUF (must report a failed check, not a plausible config).
#
# An absent model is reported `skipped`, never PASS.
#
# Knobs: MC_PY=<python with numpy[,transformers]>  VYBFORGE_MC_GGUF / _HF / _INVFREQ / _OUTDIR
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

fail=0
step() { printf '%-46s %s\n' "$1" "$2"; }
cd "$root"

work="${VYBFORGE_MC_OUTDIR:-$root/native/out/config}"
mkdir -p "$work"
mods=(--module-path native/config --module-path native/json)
ref=native/config/mc_ref.py

echo "S0.2a config gate — $(date '+%F %T')"
echo "toolchain: $VYB  ($(git -C "$(dirname "$(dirname "$VYB")")" log --oneline -1 2>/dev/null || echo 'not a git checkout'))"
echo

# ---- reference interpreter: needs numpy; transformers is used when present ----
py=""
for cand in "${MC_PY:-}" "$root/.venv/bin/python" "$HOME/Projects/VybAIConf/.venv/bin/python" python3; do
  if [ -n "$cand" ] && "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then step "reference interpreter" "FAIL (no python with numpy)"; exit 1; fi
step "reference interpreter" "$py"
if "$py" -c 'import transformers' >/dev/null 2>&1; then
  step "transformers available" "yes ($("$py" -c 'import transformers;print(transformers.__version__)' 2>/dev/null))"
else
  step "transformers available" "no (HF authority will be reported SKIPPED)"
fi

# ---- model discovery: env override first, then a small candidate list ----
gguf="${VYBFORGE_MC_GGUF:-}"
if [ -z "$gguf" ]; then
  for c in "$HOME/Models/qwen3/Qwen3-4B-Q4_K_M.gguf" "$root/native/out/qwen3_4b.gguf"; do
    [ -f "$c" ] && { gguf="$c"; break; }
  done
fi
hf="${VYBFORGE_MC_HF:-}"
if [ -z "$hf" ]; then
  for c in "$HOME/Models/spikingbrain-v1-7b-base/config.json"; do
    [ -f "$c" ] && { hf="$c"; break; }
  done
fi
invfreq="${VYBFORGE_MC_INVFREQ:-$root/native/out/layer0_invfreq.bin}"

probe() {  # probe <path> <outfile> ; echoes the probe's exit status
  env VYBFORGE_CONFIG="$1" "$VYB" native/config/mc_probe.vyb "${mods[@]}" >"$2" 2>&1
  return $?
}
mc() { grep -o "^MC_$1=.*" "$2" | head -1 | cut -d= -f2- ; }

# ── the GGUF the engine actually runs ────────────────────────────────────────────────
if [ -n "$gguf" ]; then
  probe "$gguf" "$work/qwen3.probe.txt"
  prc=$?
  n=$(grep -c '^MC_DONE' "$work/qwen3.probe.txt")
  if [ "$n" != "1" ]; then
    step "GGUF probe" "FAIL (probe did not complete, rc=$prc)"
    tail -5 "$work/qwen3.probe.txt" | sed 's/^/      /'; fail=1
  else
    step "GGUF probe" "$(mc KIND "$work/qwen3.probe.txt") arch=$(mc ARCH "$work/qwen3.probe.txt") check=$(mc CHECK "$work/qwen3.probe.txt")"
    chk=$(mc CHECK "$work/qwen3.probe.txt")
    if [ "$chk" != "0" ]; then
      step "GGUF config valid" "FAIL (mc_check=$chk)"; fail=1
    else
      step "GGUF config valid" "all required dims present"
    fi

    # authority 1: llama.cpp's own reader
    "$py" "$ref" authority-gguf "$gguf" "$work/qwen3.auth.json" >"$work/qwen3.auth.log" 2>&1
    if grep -q "MCREF_AUTHORITY=" "$work/qwen3.auth.log"; then
      step "authority: llama.cpp gguf-dump" "$(grep -o 'MCREF_AUTHORITY=.*' "$work/qwen3.auth.log" | head -1 | cut -d= -f2-)"
      out="$("$py" "$ref" check "$work/qwen3.probe.txt" "$work/qwen3.auth.json" 2>&1)"
      if echo "$out" | grep -q "MCGATE_OK"; then
        step "GGUF vs authority" "$(echo "$out" | grep -o 'MCGATE_FIELDS_OK [0-9]*' | head -1)"
        echo "$out" | grep -E 'MCGATE_AUTHORITY_SKIPPED' | sed 's/^/      /'
      else
        step "GGUF vs authority" "FAIL"
        # Show a marker mismatch when there is one, and the raw tail otherwise — a crash in the
        # authority must never look like a silent failure.
        if echo "$out" | grep -qE 'MCGATE_(MISMATCH|FAIL)'; then
          echo "$out" | grep -E 'MCGATE_(MISMATCH|FAIL)' | head -8 | sed 's/^/      /'
        else
          echo "$out" | tail -6 | sed 's/^/      /'
        fi
        fail=1
      fi
    else
      step "authority: llama.cpp gguf-dump" "SKIPPED ($(grep -o 'MCREF_AUTHORITY=.*' "$work/qwen3.auth.log" | head -1)) — not a PASS"
    fi

    # authority 3: the invfreq table the kernels consume must follow from the config's theta
    out="$("$py" "$ref" invfreq "$work/qwen3.probe.txt" "$invfreq" 2>&1)"
    if echo "$out" | grep -q "MCGATE_OK"; then
      if echo "$out" | grep -q "MCGATE_INVFREQ_OK"; then
        step "rope_theta vs kernel invfreq" "$(echo "$out" | grep -o 'MCGATE_INVFREQ_OK.*' | head -1)"
      else
        step "rope_theta vs kernel invfreq" "SKIPPED ($(echo "$out" | grep -o 'MCGATE_AUTHORITY_SKIPPED.*' | head -1))"
      fi
    else
      step "rope_theta vs kernel invfreq" "FAIL"; echo "$out" | grep MCGATE_FAIL | sed 's/^/      /'; fail=1
    fi

    # discrimination: a truncated GGUF must NOT yield a plausible config
    head -c 4096 "$gguf" > "$work/truncated.gguf"
    probe "$work/truncated.gguf" "$work/truncated.probe.txt"
    trc=$?
    tdone=$(grep -c '^MC_DONE' "$work/truncated.probe.txt")
    tchk="$([ "$tdone" = "1" ] && mc CHECK "$work/truncated.probe.txt" || echo "no-verdict")"
    tlayers="$([ "$tdone" = "1" ] && mc LAYERS "$work/truncated.probe.txt" || echo "?")"
    if [ "$tdone" = "1" ] && [ "$tchk" != "0" ]; then
      step "discrimination: truncated GGUF" "correctly refused (check=$tchk, layers=$tlayers, notes=$(mc NOTES "$work/truncated.probe.txt"))"
    else
      step "discrimination: truncated GGUF" "FAIL (reported check=$tchk layers=$tlayers — a broken file must not look valid)"
      tail -3 "$work/truncated.probe.txt" | sed 's/^/      /'; fail=1
    fi
  fi
else
  step "GGUF model" "skipped (no Qwen3 GGUF found — not a PASS)"
fi

# ── a SECOND real model: different arch, different field names ───────────────────────
if [ -n "$hf" ]; then
  probe "$hf" "$work/hf.probe.txt"
  if grep -q '^MC_DONE' "$work/hf.probe.txt"; then
    step "HF probe ($(basename "$(dirname "$hf")"))" "arch=$(mc ARCH "$work/hf.probe.txt") layers=$(mc LAYERS "$work/hf.probe.txt") hidden=$(mc HIDDEN "$work/hf.probe.txt") head_dim=$(mc HEAD_DIM "$work/hf.probe.txt") eps=$(mc RMS_EPS "$work/hf.probe.txt")"
    "$py" "$ref" authority-hf "$hf" "$work/hf.auth.json" >"$work/hf.auth.log" 2>&1
    out="$("$py" "$ref" check "$work/hf.probe.txt" "$work/hf.auth.json" 2>&1)"
    if echo "$out" | grep -q "MCGATE_OK"; then
      step "HF vs authority" "$(echo "$out" | grep -o 'MCGATE_FIELDS_OK [0-9]*' | head -1) (derived: $(mc DERIVED "$work/hf.probe.txt"))"
      echo "$out" | grep -E 'MCGATE_AUTHORITY_SKIPPED' | sed 's/^/      /'
    else
      step "HF vs authority" "FAIL"
      echo "$out" | grep -E 'MCGATE_(MISMATCH|FAIL)' | head -8 | sed 's/^/      /'; fail=1
    fi

    # discrimination: mutate a value -> the reader must report the mutation
    "$py" "$ref" mutate "$hf" "$work/mut_hidden.json" set hidden_size 4096 >/dev/null 2>&1
    probe "$work/mut_hidden.json" "$work/mut_hidden.probe.txt"
    got=$(mc HIDDEN "$work/mut_hidden.probe.txt")
    if [ "$got" = "4096" ]; then
      step "discrimination: mutated hidden_size" "correctly reported 4096 (file edited to 4096)"
    else
      step "discrimination: mutated hidden_size" "FAIL (reported hidden=$got for a file edited to 4096)"
      fail=1
    fi
    # discrimination: drop a key -> the documented fallback must fire AND be labelled
    "$py" "$ref" mutate "$hf" "$work/no_kvh.json" del num_key_value_heads >/dev/null 2>&1
    probe "$work/no_kvh.json" "$work/no_kvh.probe.txt"
    gkv=$(mc KV_HEADS "$work/no_kvh.probe.txt"); ghd=$(mc HEADS "$work/no_kvh.probe.txt")
    gdef=$(mc DEFAULTED "$work/no_kvh.probe.txt")
    if [ "$gkv" = "$ghd" ] && echo "$gdef" | grep -q "kv_heads"; then
      step "discrimination: no num_key_value_heads" "kv_heads=$gkv (=heads) and labelled defaulted"
    else
      step "discrimination: no num_key_value_heads" "FAIL (kv_heads=$gkv heads=$ghd defaulted='$gdef')"
      fail=1
    fi
  else
    step "HF probe" "FAIL (probe did not complete)"; tail -4 "$work/hf.probe.txt" | sed 's/^/      /'; fail=1
  fi
else
  step "second model (HF config.json)" "skipped (not found — not a PASS)"
fi

echo
if [ $fail -eq 0 ]; then echo "S0.2a CONFIG GATE: PASS"; else echo "S0.2a CONFIG GATE: FAIL"; fi
exit $fail
