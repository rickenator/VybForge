#!/usr/bin/env bash
# P4.4 layer-kind dispatch gate — VybForge#10 phase 4, unit 9c step 1.
#
# What it proves. The engine's per-layer loop has to decide, per block, between the conventional
# attention branch and the recurrent (Gated DeltaNet) branch, and that decision must come from the
# model's OWN tensor table rather than from the architecture name or a hardcoded block list. The
# rule now lives in one place — `model_caps::mc_layer_kind`, which the capability descriptor counts
# with — and this gate checks it three ways, none of which is our own summary agreeing with itself:
#
#   1. LK_SELFTEST — the rule's own name table, run by the probe before it touches a file. It
#      contains the traps that a prefix or `.contains("attn_q")` test classifies wrongly
#      (`attn_qkv.weight`, `attn_gate.weight` = RECURRENT; `attn_q_norm.weight` = not the marker).
#      A rule that fails any of them would run the wrong branch on 48 of Ridge's 64 text blocks.
#   2. the real interleave on both models — Ridge's 64 text blocks must be exactly 48 recurrent +
#      16 attention, with the attention blocks at N % 4 == 3 and NO other block attention; the
#      dense Qwen3-4B must be 36 attention + 0 recurrent. The expected sets are written out here,
#      not read back from the probe.
#   3. an INDEPENDENT per-layer cross-check — native/gguf/ridge_inventory.py (Python, written
#      separately from the Vyb reader) lists the tensor NAMES. For every text block, exactly one of
#      `blk.N.attn_q.weight` / `blk.N.ssm_alpha.weight` must exist in that list, and it must be the
#      one the Vyb rule's kind names. Two implementations, one file, per-layer agreement.
#
# It also pins the two facts that a kind-only count gets wrong, and would get wrong silently:
#   * the Ridge plan has 65 classified blocks but the TEXT model is 64: block 64 is the MTP draft
#     head and carries `attn_q.weight` too, so 65-block navigation yields "17 attention" — the gate
#     asserts the draft block is reported and is EXCLUDED from the text count, and that the text
#     count agrees with the capability descriptor (16 + 48), which a naive comparison would fail.
#   * `mc_layer_kinds` is three-valued: "" (unreadable table) and "-" (readable table, no text
#     blocks) must produce DIFFERENT LK_ERROR lines. The mmproj tower (real GGUF, no blk.N layers)
#     and a truncated model must be distinguishable; a gate that let them share a message could not
#     tell a broken download from a clip model.
#
# SKIPs (never PASSes) without a Vyb toolchain. The models are 12 GiB downloads: a checkout without
# them SKIPs those cases, but a gate that proves nothing is a FAIL.
#
# NOTE: the JIT prints main()'s return value and always exits 0 (doc/PYTHON-CLEANUP.md), so this
# gate reads LK_ lines out of the output. It never trusts `$?` for the verdict.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

fail=0
skipped=0
proven=0
step() { printf '%-46s %s\n' "$1" "$2"; }

echo "P4.4 layer-kind gate — $(date '+%F %T')"
if ! . "$root/vybenv.sh" >/dev/null 2>&1 || [ ! -x "$VYB" ]; then
  step "Vyb toolchain" "SKIP (no compiler)"
  echo; echo "P4.4 LAYER-KIND GATE: SKIP (no toolchain — nothing to run)"; exit 0
fi
echo "toolchain: $VYB"
echo

work="${VYBFORGE_LK_OUTDIR:-$root/native/out/layerkind}"
mkdir -p "$work"
mods=(--module-path native/config --module-path native/json)

probe() {  # probe <gguf-or-empty> <outfile>
  local src="$1" out="$2"
  if [ -z "$src" ]; then env -u VYBFORGE_MODEL "$VYB" native/config/layerkind_probe.vyb "${mods[@]}" >"$out" 2>&1
  else env VYBFORGE_MODEL="$src" "$VYB" native/config/layerkind_probe.vyb "${mods[@]}" >"$out" 2>&1; fi
}
lv() { grep -o "^LK_$1=.*" "$2" | head -1 | cut -d= -f2- ; }
# NOTE one argument: the file. A helper whose parameter is `$2` of a one-argument call is an
# unbound variable under `set -u`, and the empty command substitution it produces reads as "no
# message" instead of as an error — which is how the tower case first reported FAIL.
lmsg() { grep -o "^LK_ERROR=.*" "$1" | head -1 | cut -d= -f2- ; }
done_ok() { [ "$(grep -c '^LK_DONE' "$1")" = "1" ]; }

# an independent parser for the per-layer name cross-check (stdlib only; no venv needed)
py=""
for cand in "${LK_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import struct' >/dev/null 2>&1; then py="$cand"; break; fi
done

# ── 1. the rule's own table (the traps) ────────────────────────────────────────────────
st="$work/selftest.txt"
probe "" "$st"
if grep -q '^LK_SELFTEST=PASS' "$st"; then
  step "layer-kind rule (name traps)" "PASS (ssm_alpha=2, attn_q=1, attn_qkv/attn_gate/attn_q_norm/ssm_norm=0)"
  proven=$((proven + 1))
else
  step "layer-kind rule (name traps)" "FAIL"
  sed 's/^/      /' "$st" | head -8
  fail=1
fi

# The expected attention set for a `full_attention_interval` of 4: N % 4 == 3.
attn_expect() {  # <text_layers> -> the attention block indices, one per line
  local tl="$1" n
  for ((n = 0; n < tl; n++)); do [ $((n % 4)) = 3 ] && echo "$n"; done
}

plan_attn() {  # <outfile> <limit> -> attention block indices below <limit>, sorted
  awk -v lim="$2" '$1=="LK_LAYER" && $3==1 && $2<lim { print $2 }' "$1" | sort -n
}
plan_gdn() {   # <outfile> <limit>
  awk -v lim="$2" '$1=="LK_LAYER" && $3==2 && $2<lim { print $2 }' "$1" | sort -n
}
plan_blocks() { awk '$1=="LK_LAYER" { print $2 }' "$1" | sort -n; }

# Independent per-layer cross-check against the Python parser's tensor NAMES.
lk_indep_check() {  # <label> <outfile> <tsv> <text_layers>
  local label="$1" out="$2" tsv="$3" tl="$4" bad="" n kind want ap gp
  local names="$work/names.$$.txt"
  awk -F'\t' '$1 ~ /^blk\.[0-9]+\.(attn_q|ssm_alpha)\.weight$/ { print $1 }' "$tsv" >"$names"
  for ((n = 0; n < tl; n++)); do
    kind="$(awk -v n="$n" '$1=="LK_LAYER" && $2==n { print $3 }' "$out")"
    ap=0; gp=0
    grep -qx "blk.$n.attn_q.weight" "$names" && ap=1
    grep -qx "blk.$n.ssm_alpha.weight" "$names" && gp=1
    want=-1
    if [ "$ap" = 1 ] && [ "$gp" = 0 ]; then want=1; fi
    if [ "$ap" = 0 ] && [ "$gp" = 1 ]; then want=2; fi
    if [ "$want" = -1 ]; then bad="$bad blk$n(markers=$ap/$gp)"; continue; fi
    [ "$kind" = "$want" ] || bad="$bad blk$n(kind=$kind want=$want)"
  done
  rm -f "$names"
  if [ -n "$bad" ]; then
    step "$label independent per-layer check" "FAIL:$bad"
    fail=1
  else
    step "$label independent per-layer check" "OK ($tl text blocks, Vyb kinds == python tensor names)"
    proven=$((proven + 1))
  fi
}

# ── 2. the #10 target: the hybrid interleave ───────────────────────────────────────────
ridge="${VYBFORGE_RIDGE_GGUF:-$HOME/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf}"
if [ ! -f "$ridge" ]; then
  step "Ridge (hybrid 48 recurrent + 16 attention)" "SKIP (not on disk)"
  skipped=$((skipped + 1))
else
  out="$work/ridge.txt"
  probe "$ridge" "$out"
  if ! done_ok "$out"; then
    step "Ridge (hybrid 48 recurrent + 16 attention)" "FAIL (probe did not complete)"
    tail -5 "$out" | sed 's/^/      /'
    fail=1
  else
    bad=""
    [ "$(lv TEXT_LAYERS "$out")" = "64" ] || bad="$bad text=$(lv TEXT_LAYERS "$out")"
    [ "$(lv TEXT_ATTN "$out")" = "16" ] || bad="$bad text_attn=$(lv TEXT_ATTN "$out")"
    [ "$(lv TEXT_GDN "$out")" = "48" ] || bad="$bad text_gdn=$(lv TEXT_GDN "$out")"
    [ "$(lv BLOCKS "$out")" = "65" ] || bad="$bad blocks=$(lv BLOCKS "$out")"
    [ "$(lv DRAFT_BLOCKS "$out")" = "64" ] || bad="$bad draft=$(lv DRAFT_BLOCKS "$out")"
    # the plan must cover every block 0..64 exactly once: no gap, no duplicate
    if [ "$(plan_blocks "$out" | sort -nu | tr '\n' ' ')" != "$(seq 0 64 | tr '\n' ' ')" ]; then
      bad="$bad plan-not-0..64"
    fi
    # the attention blocks must be EXACTLY N % 4 == 3, in the text range
    if ! diff <(plan_attn "$out" 64) <(attn_expect 64) >/dev/null; then
      bad="$bad attention-set!=N%4==3"
    fi
    # and the recurrent set is the complement (48 of them) — asserted separately so a plan that
    # called EVERYTHING recurrent cannot pass the two counts above by symmetry. Compared against
    # an ascending generator, NOT `comm`: comm needs lexicographic input and `seq 0 63` is not
    # lexicographically sorted, so it errors out and a `diff` on its output would agree for the
    # wrong reason.
    if ! diff <(plan_gdn "$out" 64) <(awk 'BEGIN{for(n=0;n<64;n++) if(n%4!=3) print n}') >/dev/null; then
      bad="$bad recurrent-set!=complement"
    fi
    # the draft head is IN the plan but OUT of the text count — the distinction a kind-only count
    # collapses (it would read 17 attention layers for a 16-layer attention model)
    [ "$(lv ATTN_LAYERS "$out")" = "17" ] || bad="$bad attn_layers=$(lv ATTN_LAYERS "$out")"
    if [ -n "$bad" ]; then
      step "Ridge (hybrid 48 recurrent + 16 attention)" "FAIL:$bad"
      fail=1
    else
      step "Ridge (hybrid 48 recurrent + 16 attention)" "OK (64 text = 16 attn(N%4==3) + 48 gdn, draft blk 64 excluded)"
      proven=$((proven + 1))
    fi
    # descriptor agreement: the caps walk uses the same rule over the same file
    caps="$work/ridge_caps.txt"
    env VYBFORGE_MODEL="$ridge" "$VYB" native/config/caps_probe.vyb "${mods[@]}" >"$caps" 2>&1
    cv() { grep -o "^CAPS_$1=.*" "$caps" | head -1 | cut -d= -f2- ; }
    if [ "$(cv TEXT_LAYERS)" = "$(lv TEXT_LAYERS "$out")" ] && \
       [ "$(cv ATTN_LAYERS)" = "$(lv TEXT_ATTN "$out")" ] && \
       [ "$(cv GDN_LAYERS)" = "$(lv TEXT_GDN "$out")" ]; then
      step "Ridge descriptor agreement" "OK (caps 16+48 text == plan's text kinds)"
      proven=$((proven + 1))
    else
      step "Ridge descriptor agreement" "FAIL (caps $(cv ATTN_LAYERS)+$(cv GDN_LAYERS) vs plan $(lv TEXT_ATTN)+$(lv TEXT_GDN))"
      fail=1
    fi
    if [ -n "$py" ]; then
      tsv="$work/ridge.tsv"
      if env -u PYTHONPATH "$py" native/gguf/ridge_inventory.py "$ridge" --tsv "$tsv" >/dev/null 2>&1; then
        lk_indep_check "Ridge" "$out" "$tsv" 64
      else
        step "Ridge independent per-layer check" "SKIP (independent parser failed to run)"
        skipped=$((skipped + 1))
      fi
    else
      step "Ridge independent per-layer check" "SKIP (no python to cross-check with)"
      skipped=$((skipped + 1))
    fi
  fi
fi

# ── 3. the dense model the engine runs today ───────────────────────────────────────────
gguf="${VYBFORGE_QWEN3_GGUF:-$HOME/Models/qwen3/Qwen3-4B-Q4_K_M.gguf}"
if [ ! -f "$gguf" ]; then
  step "dense Qwen3-4B (36 attention, 0 recurrent)" "SKIP (not on disk)"
  skipped=$((skipped + 1))
else
  out="$work/qwen3.txt"
  probe "$gguf" "$out"
  if ! done_ok "$out"; then
    step "dense Qwen3-4B (36 attention, 0 recurrent)" "FAIL (probe did not complete)"
    tail -5 "$out" | sed 's/^/      /'
    fail=1
  else
    bad=""
    [ "$(lv TEXT_LAYERS "$out")" = "36" ] || bad="$bad text=$(lv TEXT_LAYERS "$out")"
    [ "$(lv TEXT_ATTN "$out")" = "36" ] || bad="$bad attn=$(lv TEXT_ATTN "$out")"
    [ "$(lv TEXT_GDN "$out")" = "0" ] || bad="$bad gdn=$(lv TEXT_GDN "$out")"
    [ "$(lv DRAFT_BLOCKS "$out")" = "" ] || bad="$bad draft=$(lv DRAFT_BLOCKS "$out")"
    if ! diff <(plan_attn "$out" 36) <(seq 0 35) >/dev/null; then bad="$bad attention-set"; fi
    if [ -n "$bad" ]; then
      step "dense Qwen3-4B (36 attention, 0 recurrent)" "FAIL:$bad"
      fail=1
    else
      step "dense Qwen3-4B (36 attention, 0 recurrent)" "OK (all 36 layers attention, no draft head)"
      proven=$((proven + 1))
    fi
    if [ -n "$py" ]; then
      tsv="$work/qwen3.tsv"
      if env -u PYTHONPATH "$py" native/gguf/ridge_inventory.py "$gguf" --tsv "$tsv" >/dev/null 2>&1; then
        lk_indep_check "Qwen3-4B" "$out" "$tsv" 36
      else
        step "Qwen3-4B independent per-layer check" "SKIP (independent parser failed to run)"
        skipped=$((skipped + 1))
      fi
    fi
  fi
fi

# ── 4. the two no-plan cases must stay DISTINGUISHABLE ─────────────────────────────────
mmproj="${VYBFORGE_MMPROJ_GGUF:-$HOME/Models/qwen38-27b-ridge/mmproj-Qwen3.8-27B-BF16.gguf}"
msg_tower=""
if [ ! -f "$mmproj" ]; then
  step "vision tower (no text blocks)" "SKIP (not on disk)"
  skipped=$((skipped + 1))
else
  out="$work/mmproj.txt"
  probe "$mmproj" "$out"
  msg_tower="$(lmsg "$out")"
  if ! grep -q '^LK_SELFTEST=PASS' "$out"; then
    step "vision tower (no text blocks)" "FAIL (rule selftest did not pass on this run)"
    fail=1
  elif [ -z "$msg_tower" ] || grep -q '^LK_LAYER' "$out"; then
    # a tower must not have text layers invented for it, and must say why it has none
    step "vision tower (no text blocks)" "FAIL (invented blk.N layers or refused without a reason)"
    fail=1
  elif grep -q '^LK_DONE' "$out"; then
    step "vision tower (no text blocks)" "FAIL (printed a plan for a file with no text blocks)"
    fail=1
  else
    step "vision tower (no text blocks)" "OK (readable table, no layer block: '$msg_tower')"
    proven=$((proven + 1))
  fi
fi

msg_trunc=""
if [ -f "$gguf" ]; then
  trunc="$work/truncated.gguf"
  head -c 200000 "$gguf" >"$trunc"
  if ! cmp -s <(head -c 200000 "$gguf") "$trunc"; then
    step "truncated file (unreadable table)" "FAIL (fixture is not a prefix of $(basename "$gguf"))"
    fail=1
  else
    out="$work/trunc.txt"
    probe "$trunc" "$out"
    msg_trunc="$(lmsg "$out")"
    if [ -z "$msg_trunc" ] || grep -q '^LK_LAYER' "$out" || grep -q '^LK_DONE' "$out"; then
      step "truncated file (unreadable table)" "FAIL (a metadata-truncated file produced a plan)"
      fail=1
    else
      step "truncated file (unreadable table)" "OK ('$msg_trunc')"
      proven=$((proven + 1))
    fi
  fi
else
  step "truncated file (unreadable table)" "SKIP (no dense model to truncate)"
  skipped=$((skipped + 1))
fi

# the point of the two cases above: the SAME empty answer must not mean two different things
if [ -n "$msg_tower" ] && [ -n "$msg_trunc" ]; then
  if [ "$msg_tower" = "$msg_trunc" ]; then
    step "no-plan reasons are distinct" "FAIL (tower and truncated file share a message)"
    fail=1
  else
    step "no-plan reasons are distinct" "OK (tower vs truncated file: different reasons)"
    proven=$((proven + 1))
  fi
fi

echo
if [ "$proven" = "0" ]; then
  echo "P4.4 LAYER-KIND GATE: FAIL (nothing was proven; skipped=$skipped)"; exit 1
fi
if [ "$fail" = "0" ]; then
  echo "P4.4 LAYER-KIND GATE: PASS ($proven cases, $skipped skipped)"
else
  echo "P4.4 LAYER-KIND GATE: FAIL"
fi
exit $fail
