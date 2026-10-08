#!/usr/bin/env bash
# S0.2e capability gate — VybForge#10 phase 2.
#
# What it proves. native/config/model_caps.vyb decides whether this build can RUN a model and
# refuses with a NAMED reason when it cannot. That decision is checked against four real inputs,
# none of which is a fixture written to agree with the code:
#
#   Qwen3-4B GGUF (what the engine runs today)
#       -> SUPPORTED, dense, every layer an attention layer, and the type counts are
#          cross-checked against an INDEPENDENT parser (native/gguf/ridge_inventory.py, Python,
#          written separately from the Vyb reader) — two implementations must agree.
#   Qwen3.8-27B Ridge (the #10 target)
#       -> UNSUPPORTED for exactly TWO STRUCTURAL reasons — the Gated-DeltaNet layer kind and the
#          MTP head — with every text quant type implemented (Q8_0, Q4_K, Q5_K, Q6_K, IQ2_S,
#          IQ3_S). The check asserts each of those types is ABSENT from the refusal list. It also
#          pins the hybrid facts the phase-2 descriptor exists to express: 64 text blocks = 16
#          attention + 48 recurrent, interval 4, ssm state_size 128, one nextn layer.
#   mmproj-BF16.gguf (the vision tower)
#       -> UNSUPPORTED for ONE reason: vision itself. Its weight type (BF16) is implemented, so the
#          check asserts BF16 is ABSENT from the refusal and that the tower is still filed as a
#          vision-encoder layout, NOT as a broken text model.
#   a truncated file
#       -> UNSUPPORTED with a read reason and no crash. A metadata-intact, table-less file must
#          never read as SUPPORTED — that is the worst output this module could produce.
#
# A model that is not on disk is reported SKIP (the Ridge file is a 12 GiB download; a checkout
# without it still runs this gate honestly). SKIP does not fail the gate, but a gate that proves
# NOTHING does: if every case skipped, that is a FAIL.
#
# NOTE: the JIT prints main()'s return value and always exits 0 (doc/PYTHON-CLEANUP.md), so this
# gate reads CAPS_* lines out of the output. It never trusts `$?` for the verdict.

set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

fail=0
skipped=0
proven=0
step() { printf '%-46s %s\n' "$1" "$2"; }
cd "$root"

work="${VYBFORGE_CAPS_OUTDIR:-$root/native/out/caps}"
mkdir -p "$work"
mods=(--module-path native/config --module-path native/json)

echo "S0.2e capability gate — $(date '+%F %T')"
echo "toolchain: $VYB"
echo

probe() {  # probe <gguf> <outfile>
  env VYBFORGE_MODEL="$1" "$VYB" native/config/caps_probe.vyb "${mods[@]}" >"$2" 2>&1
}
cv() { grep -o "^CAPS_$1=.*" "$2" | head -1 | cut -d= -f2- ; }
done_ok() { [ "$(grep -c '^CAPS_DONE' "$1")" = "1" ]; }

# independent parser, for the type-count cross-check
py=""
for cand in "${CAPS_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && "$cand" -c 'import sys' >/dev/null 2>&1; then py="$cand"; break; fi
done

py_type_pairs() {  # <gguf> -> sorted NAME:count lines, from the Python inventory
  "$py" native/gguf/ridge_inventory.py "$1" --tsv "$work/xcheck.tsv" 2>/dev/null | awk '
    /per type/ { f = 1; next }
    /^---/     { f = 0 }
    f && NF >= 2 { print $1 ":" $2 }' | sort
}
vyb_type_pairs() { tr ',' '\n' <<<"$1" | sed '/^$/d' | sort; }

# Compare the Vyb reader's per-type counts against the Python reader's. Both parse the same
# tensor table; a disagreement means one of them is wrong, and that is exactly what a
# self-consistent fixture could never show.
# NOTE every variable here is `local`, and the path argument is named `src`: an earlier version
# assigned `gguf=$3` without `local`, which silently clobbered the caller's model path and made
# the later truncation case read the WRONG file. The cross-checks were still correct, which is
# how a bug like that survives a green run.
type_crosscheck() {  # <label> <outfile> <gguf>
  local label="$1" out="$2" src="$3" vt pt
  if [ -z "$py" ]; then
    step "$label type cross-check" "SKIP (no python to cross-check with)"
    return 0
  fi
  vt="$(vyb_type_pairs "$(cv TYPES "$out")")"
  pt="$(py_type_pairs "$src")"
  if [ "$vt" = "$pt" ]; then
    step "$label type cross-check" "OK ($(wc -l <<<"$vt" | tr -d ' ') types, Vyb == python)"
  else
    step "$label type cross-check" "FAIL"
    diff <(echo "$vt") <(echo "$pt") | sed 's/^/      /' | head -8
    fail=1
  fi
}

# ── 1. the dense model the engine runs ─────────────────────────────────────────────────
gguf="${VYBFORGE_QWEN3_GGUF:-}"
if [ -z "$gguf" ] || [ ! -f "$gguf" ]; then
  gguf=""
  for c in "$HOME/Models/qwen3/Qwen3-4B-Q4_K_M.gguf" "$root/native/out/qwen3_4b.gguf"; do
    [ -f "$c" ] && { gguf="$c"; break; }
  done
fi
if [ -z "$gguf" ]; then
  step "dense model (Qwen3-4B)" "SKIP (not found)"
  skipped=$((skipped + 1))
else
  out="$work/dense.txt"
  probe "$gguf" "$out"
  if ! done_ok "$out"; then
    step "dense model (Qwen3-4B)" "FAIL (probe did not complete)"
    tail -5 "$out" | sed 's/^/      /'
    fail=1
  else
    arch="$(cv ARCH "$out")"; layout="$(cv LAYOUT "$out")"
    nl="$(cv N_LAYERS "$out")"; al="$(cv ATTN_LAYERS "$out")"
    gl="$(cv GDN_LAYERS "$out")"; ml="$(cv MTP_LAYERS "$out")"
    vv="$(cv VISION "$out")"; uns="$(cv UNSUPPORTED "$out")"
    bad=""
    [ "$(cv VERDICT "$out")" = "SUPPORTED" ] || bad="$bad verdict"
    [ "$arch" = "qwen3" ] || bad="$bad arch=$arch"
    [ "$layout" = "dense" ] || bad="$bad layout=$layout"
    # attn == layers AND layers > 0, so "0 == 0" cannot pass this
    [ "$al" = "$nl" ] || bad="$bad attn=$al/$nl"
    [ "$nl" -gt 0 ] 2>/dev/null || bad="$bad n_layers=$nl"
    [ "$gl" = "0" ] || bad="$bad gdn=$gl"
    [ "$ml" = "0" ] || bad="$bad mtp=$ml"
    [ "$vv" = "0" ] || bad="$bad vision=$vv"
    [ -z "$uns" ] || bad="$bad unsupported=[$uns]"
    if [ -n "$bad" ]; then
      step "dense model (Qwen3-4B)" "FAIL:$bad"
      fail=1
    else
      step "dense model (Qwen3-4B)" "SUPPORTED (dense, $al/$nl attention layers, no GDN/MTP/vision)"
      proven=$((proven + 1))
    fi
    type_crosscheck "dense model (Qwen3-4B)" "$out" "$gguf"
  fi
fi

# ── 2. the #10 target: hybrid + new quant types + MTP ─────────────────────────────────
ridge="${VYBFORGE_RIDGE_GGUF:-$HOME/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf}"
if [ ! -f "$ridge" ]; then
  step "Ridge text model (#10 target)" "SKIP (not on disk)"
  skipped=$((skipped + 1))
else
  out="$work/ridge.txt"
  probe "$ridge" "$out"
  if ! done_ok "$out"; then
    step "Ridge text model (#10 target)" "FAIL (probe did not complete)"
    tail -5 "$out" | sed 's/^/      /'
    fail=1
  else
    uns="$(cv UNSUPPORTED "$out")"
    bad=""
    [ "$(cv VERDICT "$out")" = "UNSUPPORTED" ] || bad="$bad verdict=$(cv VERDICT "$out")"
    [ "$(cv ARCH "$out")" = "qwen35" ] || bad="$bad arch=$(cv ARCH "$out")"
    [ "$(cv LAYOUT "$out")" = "hybrid-recurrent" ] || bad="$bad layout=$(cv LAYOUT "$out")"
    # the hybrid facts phase 2 exists to express
    [ "$(cv N_LAYERS "$out")" = "65" ] || bad="$bad n_layers=$(cv N_LAYERS "$out")"
    [ "$(cv TEXT_LAYERS "$out")" = "64" ] || bad="$bad text=$(cv TEXT_LAYERS "$out")"
    [ "$(cv ATTN_LAYERS "$out")" = "16" ] || bad="$bad attn=$(cv ATTN_LAYERS "$out")"
    [ "$(cv GDN_LAYERS "$out")" = "48" ] || bad="$bad gdn=$(cv GDN_LAYERS "$out")"
    [ "$(cv MTP_LAYERS "$out")" = "1" ] || bad="$bad mtp=$(cv MTP_LAYERS "$out")"
    [ "$(cv ATTN_INTERVAL "$out")" = "4" ] || bad="$bad interval=$(cv ATTN_INTERVAL "$out")"
    [ "$(cv SSM_STATE_SIZE "$out")" = "128" ] || bad="$bad ssm_state=$(cv SSM_STATE_SIZE "$out")"
    [ "$(cv NEXTN_TENSORS "$out")" = "4" ] || bad="$bad nextn_tensors=$(cv NEXTN_TENSORS "$out")"
    # the refusal list must be EXACTLY these two, each named, and no others: the text model's quant
    # types are all implemented now, so what remains is structural. Every type that LEFT the list
    # (Q8_0, Q5_K, IQ2_S, IQ3_S) is asserted ABSENT, so a capability that silently stopped being
    # reported cannot read as a passing gate.
    nreasons="$(tr ',' '\n' <<<"$uns" | sed '/^$/d' | wc -l | tr -d ' ')"
    [ "$nreasons" = "2" ] || bad="$bad reasons=$nreasons"
    for want in \
      "UNSUPPORTED_LAYER_KIND gated-deltanet layers=48" \
      "UNSUPPORTED_CAPABILITY mtp nextn_predict_layers=1"; do
      grep -qF "$want" <<<"$uns" || bad="$bad missing[$want]"
    done
    grep -qF "Q8_0" <<<"$uns" && bad="$bad Q8_0-still-refused"
    grep -qF "IQ2_S" <<<"$uns" && bad="$bad IQ2_S-still-refused"
    grep -qF "Q5_K" <<<"$uns" && bad="$bad Q5_K-still-refused"
    grep -qF "IQ3_S" <<<"$uns" && bad="$bad IQ3_S-still-refused"
    if [ -n "$bad" ]; then
      step "Ridge text model (#10 target)" "FAIL:$bad"
      echo "      unsupported=[$uns]" | sed 's/^/      /'
      fail=1
    else
      step "Ridge text model (#10 target)" "UNSUPPORTED with 2 named reasons (GDN 48 layers, MTP; all five text quant types implemented)"
      proven=$((proven + 1))
    fi
    type_crosscheck "Ridge text model" "$out" "$ridge"
  fi
fi

# ── 3. the vision tower ───────────────────────────────────────────────────────────────
mmproj="${VYBFORGE_MMPROJ_GGUF:-$HOME/Models/qwen38-27b-ridge/mmproj-Qwen3.8-27B-BF16.gguf}"
if [ ! -f "$mmproj" ]; then
  step "vision tower (mmproj)" "SKIP (not on disk)"
  skipped=$((skipped + 1))
else
  out="$work/mmproj.txt"
  probe "$mmproj" "$out"
  if ! done_ok "$out"; then
    step "vision tower (mmproj)" "FAIL (probe did not complete)"
    tail -5 "$out" | sed 's/^/      /'
    fail=1
  else
    uns="$(cv UNSUPPORTED "$out")"
    bad=""
    [ "$(cv VERDICT "$out")" = "UNSUPPORTED" ] || bad="$bad verdict=$(cv VERDICT "$out")"
    [ "$(cv ARCH "$out")" = "clip" ] || bad="$bad arch=$(cv ARCH "$out")"
    [ "$(cv LAYOUT "$out")" = "vision-encoder" ] || bad="$bad layout=$(cv LAYOUT "$out")"
    [ "$(cv VISION "$out")" = "1" ] || bad="$bad vision=$(cv VISION "$out")"
    # The tower's only remaining refusal is vision itself: its weight type (BF16) is implemented,
    # so the type reason must be ABSENT. One reason, and it must be the capability one.
    grepf="$(grep -oF 'BF16' <<<"$uns")"
    [ -z "$grepf" ] || bad="$bad BF16-still-refused"
    nreasons="$(tr ',' '\n' <<<"$uns" | sed '/^$/d' | wc -l | tr -d ' ')"
    [ "$nreasons" = "1" ] || bad="$bad reasons=$nreasons"
    grep -qF "UNSUPPORTED_CAPABILITY vision" <<<"$uns" || bad="$bad no-vision-reason"
    if [ -n "$bad" ]; then
      step "vision tower (mmproj)" "FAIL:$bad"
      fail=1
    else
      step "vision tower (mmproj)" "UNSUPPORTED (clip layout, vision only — BF16 type implemented)"
      proven=$((proven + 1))
    fi
    type_crosscheck "vision tower" "$out" "$mmproj"
  fi
fi

# ── 4. a malformed file must not read as supported ────────────────────────────────────
# A prefix cut inside the metadata/table is a file we cannot describe; it must be REFUSED with a
# read reason, never reported as a model. (A prefix cut inside the DATA region is a different
# case and is deliberately not tested here: the capability profile is a property of the metadata
# and the tensor table, so such a file still describes its model correctly — the descriptor is
# not a completeness check on the weights, and pretending otherwise would be a false gate.)
if [ -n "$gguf" ]; then
  trunc="$work/truncated.gguf"
  head -c 200000 "$gguf" >"$trunc"
  # Provenance: the fixture must really be a prefix of the model under test. Without this, a
  # step-4 that read some other file would still look green — which is exactly what happened
  # while the cross-check helper was clobbering the model path.
  if ! cmp -s <(head -c 200000 "$gguf") "$trunc"; then
    step "truncated file" "FAIL (fixture is not a prefix of $(basename "$gguf"))"
    fail=1
  else
    out="$work/trunc.txt"
    probe "$trunc" "$out"
    if ! done_ok "$out"; then
      step "truncated file" "FAIL (crashed instead of degrading)"
      fail=1
    else
      uns="$(cv UNSUPPORTED "$out")"
      if [ "$(cv VERDICT "$out")" != "UNSUPPORTED" ]; then
        step "truncated file" "FAIL (read as $(cv VERDICT "$out"))"
        fail=1
      else
        case "$uns" in
          UNSUPPORTED_READ*)
            step "truncated file" "UNSUPPORTED ($uns)"
            proven=$((proven + 1)) ;;
          "")
            step "truncated file" "FAIL (refused without a reason)"
            fail=1 ;;
          *)
            step "truncated file" "FAIL (reason is not a read failure: $uns)"
            fail=1 ;;
        esac
      fi
    fi
  fi
else
  step "truncated file" "SKIP (no dense model to truncate)"
  skipped=$((skipped + 1))
fi

echo
if [ "$proven" = "0" ]; then
  echo "S0.2e CAPABILITY GATE: FAIL (nothing was proven; skipped=$skipped)"
  exit 1
fi
if [ "$fail" = "0" ]; then
  echo "S0.2e CAPABILITY GATE: PASS ($proven cases, $skipped skipped)"
else
  echo "S0.2e CAPABILITY GATE: FAIL"
fi
exit $fail
