#!/usr/bin/env bash
# run_safetensors_gate.sh — S0.3 gate for native/torchload/safetensors.vyb (the native safetensors
# reader) on a REAL container file.
#
# Three independent sides must agree:
#   1. the Vyb reader (native/torchload/st_probe.vyb via safetensors.vyb),
#   2. native/torchload/st_ref.py — its own stdlib parse of the same container (not a wrapper around
#      the Vyb code, so a reader bug cannot be mirrored by shared implementation),
#   3. native/torchload/st_libcheck.py — the upstream `safetensors` library's own view (count, key
#      order, dtype, shape; lazy Slice API so bf16 is described rather than materialized).
# The listings from (1) and (2) must be byte-identical; (3) must agree with the listing.
#
# Beyond parity the gate asserts structural invariants so a *truncated* repeatable-bug listing
# cannot pass: the claimed TENSORS count must equal the emitted rows, and per tensor
# numel * dtype-size must equal the data length (0 mismatches). Every tensor name/shape/dtype is
# checked; tensor BYTES are checked by sha256 under the sampling rule documented in st_probe.vyb
# (all tensors on files <= 300 MB, which is the default input here; a deterministic sample plus a
# bounded 1 MiB window on larger shards).
#
# Usage: ./native/legit/run_safetensors_gate.sh [file.safetensors]
#        default: artifacts/vybos-configurator-lora/adapter_model.safetensors (504 BF16 tensors)
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

cd "$root"
file="${1:-artifacts/vybos-configurator-lora/adapter_model.safetensors}"
export VYBFORGE_ST_FILE="$file"
vyblist="native/out/st_listing_vyb.txt"
reflist="native/out/st_listing_ref.txt"
work="${TMPDIR:-/tmp}/st-gate.$$"
mkdir -p "$work"
fail=0
step() { printf '%-52s %s\n' "$1" "$2"; }

if [ ! -s "$file" ]; then
  step "input file" "MISSING ($file)"
  echo; echo "SAFETENSORS GATE: FAIL"; exit 1
fi
step "input" "$(basename "$file") $(wc -c < "$file") B sha256=$(sha256sum "$file" | cut -c1-12)"

# 1. Vyb reader
if VYBFORGE_ST_OUT="$vyblist" "$VYB" --module-path native/json native/torchload/st_probe.vyb >"$work/vyb.log" 2>&1; then
  step "vyb reader" "$(grep -o 'ST_PROBE_OK.*' "$work/vyb.log" | tail -1)"
else
  step "vyb reader" "FAIL"; tail -6 "$work/vyb.log" | sed 's/^/      /'; fail=1
fi

# 2. independent stdlib reference
if VYBFORGE_ST_OUT="$reflist" .venv/bin/python native/torchload/st_ref.py >"$work/ref.log" 2>&1; then
  step "reference (independent stdlib parse)" "$(tail -1 "$work/ref.log")"
else
  step "reference (independent stdlib parse)" "FAIL"; tail -6 "$work/ref.log" | sed 's/^/      /'; fail=1
fi

# 3. byte-exact listing parity
if [ -s "$vyblist" ] && [ -s "$reflist" ] && cmp -s "$vyblist" "$reflist"; then
  step "listing parity (byte-exact)" "$(wc -l < "$vyblist") lines"
else
  step "listing parity (byte-exact)" "FAIL"
  diff "$reflist" "$vyblist" 2>/dev/null | head -12 | sed 's/^/      /'
  fail=1
fi

# 4. structural invariants
n="$(awk '/^TENSORS /{print $2}' "$vyblist" 2>/dev/null)"
rows="$(grep -c '|' "$vyblist" 2>/dev/null)"
if [ "${n:-x}" = "${rows:-y}" ]; then
  step "count self-check" "TENSORS $n == $rows rows"
else
  step "count self-check" "FAIL TENSORS=${n:-missing} rows=${rows:-0}"; fail=1
fi
mm="$(awk '/^MISMATCH /{print $2}' "$vyblist" 2>/dev/null)"
if [ "${mm:-x}" = "0" ]; then
  step "layout check (numel*dtype-size == bytes)" "ok, 0 mismatches"
else
  step "layout check (numel*dtype-size == bytes)" "FAIL (${mm:-missing} mismatches)"; fail=1
fi

# 5. upstream library third opinion
.venv/bin/python native/torchload/st_libcheck.py "$vyblist" >"$work/lib.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  step "safetensors library cross-check" "$(tail -1 "$work/lib.log")"
elif [ $rc -eq 2 ]; then
  step "safetensors library cross-check" "skipped ($(tail -1 "$work/lib.log"))"
else
  step "safetensors library cross-check" "FAIL"; tail -6 "$work/lib.log" | sed 's/^/      /'; fail=1
fi

rm -rf "$work"
echo
if [ $fail -eq 0 ]; then echo "SAFETENSORS GATE: PASS"; else echo "SAFETENSORS GATE: FAIL"; fi
exit $fail
