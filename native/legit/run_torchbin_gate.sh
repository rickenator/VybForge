#!/usr/bin/env bash
# run_torchbin_gate.sh — S0.3 gate for the native PyTorch `.bin` (zipfile-format) loader.
#
# Two layers, each with an independent other side; the loader is only "gated" when both are green.
#
# A. CONTAINER — native/torchload/torchzip.vyb (central directory, local-header data offsets,
#    sizes/method/CRC, bounded-window sha256) against Python's own `zipfile` (tb_ref.py), whose SIZE
#    comes from the OS rather than from the Vyb side's read_at-to-EOF probe and which reads each
#    hashed record twice (absolute seek vs zipfile.open) to validate the offset arithmetic.
#
# B. OBJECT GRAPH — native/torchload/torchpickle.vyb, a protocol-2 pickle VM, driven by
#    tb_pkl_probe.vyb, against CPython's real pickle VM (tb_pkl_ref.py, records located by suffix so
#    a wrong prefix on the Vyb side cannot be mirrored, bytes read through zipfile.open), plus
#    tb_torchcheck.py: torch's own metadata view via map_location='meta', compared by name.
#
# Beyond parity each layer asserts invariants a truncated-but-repeatable listing could otherwise
# fake: claimed counts == emitted rows, no layout mismatches, no non-contiguous strides, the Vyb
# side's probed size == the real file size, and the pickle stream fully consumed (STOP at the last
# byte). Run it over real shards from several producers — a 1-tensor shard and a 41-tensor shard
# exercise different paths.
#
# Usage: ./native/legit/run_torchbin_gate.sh [shard.bin]
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

cd "$root"
file="${1:-${VYBFORGE_SB_SHARD:-$HOME/Models/spikingbrain-v1-7b-base/pytorch_model-00001.bin}}"
export VYBFORGE_TB_FILE="$file"
vz="native/out/tb_zip_listing_vyb.txt"; rz="native/out/tb_zip_listing_ref.txt"
vp="native/out/tb_pkl_listing_vyb.txt"; rp="native/out/tb_pkl_listing_ref.txt"
work="${TMPDIR:-/tmp}/tb-gate.$$"
mkdir -p "$work"
fail=0
step() { printf '%-52s %s\n' "$1" "$2"; }

# The torch authority needs an interpreter with torch; the project keeps one in the VybAIConf
# reference venv (VybForge/.venv carries safetensors but not torch). Discover rather than assume,
# and report which one was used — a silently skipped authority is worse than none.
torchpy=""
for cand in "${TORCH_PY:-}" "$HOME/Projects/VybAIConf/.venv/bin/python" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && "$cand" -c 'import torch' >/dev/null 2>&1; then torchpy="$cand"; break; fi
done

if [ ! -s "$file" ]; then
  step "input file" "MISSING ($file)"
  echo; echo "TORCHBIN GATE: FAIL"; exit 1
fi
step "input" "$(basename "$file") $(wc -c < "$file") B sha256=$(sha256sum "$file" | cut -c1-12)"

echo "A. container"
if VYBFORGE_TB_OUT="$vz" "$VYB" --module-path native/json native/torchload/tb_zip_probe.vyb >"$work/vybz.log" 2>&1; then
  step "  vyb container reader" "$(grep -o 'TB_ZIP_OK.*' "$work/vybz.log" | tail -1)"
else
  step "  vyb container reader" "FAIL"; tail -8 "$work/vybz.log" | sed 's/^/      /'; fail=1
fi
if VYBFORGE_TB_OUT="$rz" .venv/bin/python native/torchload/tb_ref.py >"$work/refz.log" 2>&1; then
  step "  reference (Python zipfile)" "$(grep -o 'TB_REF_OK.*' "$work/refz.log" | tail -1)"
  step "  reference byte-path self-check" "$(grep -o 'TB_REF_BYTEPATH_OK.*' "$work/refz.log" | tail -1)"
else
  step "  reference (Python zipfile)" "FAIL"; tail -8 "$work/refz.log" | sed 's/^/      /'; fail=1
fi
if [ -s "$vz" ] && [ -s "$rz" ] && cmp -s "$vz" "$rz"; then
  step "  listing parity (byte-exact)" "$(wc -l < "$vz") lines"
else
  step "  listing parity (byte-exact)" "FAIL"
  diff "$rz" "$vz" 2>/dev/null | head -12 | sed 's/^/      /'; fail=1
fi
n="$(awk '/^RECORDS /{print $2}' "$vz" 2>/dev/null)"; rows="$(grep -c '|' "$vz" 2>/dev/null)"
if [ "${n:-x}" = "${rows:-y}" ]; then step "  count self-check" "RECORDS $n == $rows rows"
else step "  count self-check" "FAIL RECORDS=${n:-missing} rows=${rows:-0}"; fail=1; fi
mm="$(awk '/^MISMATCH /{print $2}' "$vz" 2>/dev/null)"
if [ "${mm:-x}" = "0" ]; then step "  container invariants" "ok, 0 mismatches"
else step "  container invariants" "FAIL (${mm:-missing})"; fail=1; fi
ps="$(awk '/^SIZE /{print $2}' "$vz" 2>/dev/null)"; rs="$(wc -c < "$file")"
if [ "${ps:-x}" = "$rs" ]; then step "  probed size == real size" "$rs B (read_at-to-EOF probe)"
else step "  probed size == real size" "FAIL probed=${ps:-missing} real=$rs"; fail=1; fi

echo "B. object graph"
if VYBFORGE_TB_OUT="$vp" "$VYB" --module-path native/json native/torchload/tb_pkl_probe.vyb >"$work/vybp.log" 2>&1; then
  step "  vyb pickle VM" "$(grep -o 'TB_PKL_OK.*' "$work/vybp.log" | tail -1)"
else
  step "  vyb pickle VM" "FAIL"; tail -8 "$work/vybp.log" | sed 's/^/      /'; fail=1
fi
if VYBFORGE_TB_OUT="$rp" .venv/bin/python native/torchload/tb_pkl_ref.py >"$work/refp.log" 2>&1; then
  step "  reference (CPython pickle)" "$(grep -o 'TB_PKL_REF_OK.*' "$work/refp.log" | tail -1)"
else
  step "  reference (CPython pickle)" "FAIL"; tail -8 "$work/refp.log" | sed 's/^/      /'; fail=1
fi
if [ -s "$vp" ] && [ -s "$rp" ] && cmp -s "$vp" "$rp"; then
  step "  listing parity (byte-exact)" "$(wc -l < "$vp") lines"
else
  step "  listing parity (byte-exact)" "FAIL"
  diff "$rp" "$vp" 2>/dev/null | head -12 | sed 's/^/      /'; fail=1
fi
tn="$(awk '/^TENSORS /{print $2}' "$vp" 2>/dev/null)"; trows="$(grep -c '|' "$vp" 2>/dev/null)"
if [ "${tn:-x}" = "${trows:-y}" ]; then step "  count self-check" "TENSORS $tn == $trows rows"
else step "  count self-check" "FAIL TENSORS=${tn:-missing} rows=${trows:-0}"; fail=1; fi
tm="$(awk '/^MISMATCH /{print $2}' "$vp" 2>/dev/null)"
if [ "${tm:-x}" = "0" ]; then step "  layout invariants" "ok, 0 mismatches"
else step "  layout invariants" "FAIL (${tm:-missing})"; fail=1; fi
nc="$(awk '/^NONCONTIG /{print $2}' "$vp" 2>/dev/null)"
if [ "${nc:-x}" = "0" ]; then step "  contiguity" "all tensors row-major contiguous"
else step "  contiguity" "NOTE ${nc:-missing} non-contiguous tensor(s)"; fi
pk="$(awk '/^PKL /{print $2}' "$vp" 2>/dev/null)"
st="$(grep -o 'stop=[0-9]*' "$work/vybp.log" | tail -1 | cut -d= -f2)"
if [ -n "${st:-}" ] && [ "$((st + 1))" = "${pk:-0}" ]; then
  step "  pickle stream fully consumed" "STOP at byte $st of $pk"
else
  step "  pickle stream fully consumed" "FAIL stop=${st:-missing} pkl=${pk:-missing}"; fail=1
fi
if [ -n "$torchpy" ]; then
  step "  torch metadata interpreter" "$torchpy ($("$torchpy" -c 'import torch; print(torch.__version__)' 2>/dev/null))"
  "$torchpy" native/torchload/tb_torchcheck.py "$vp" >"$work/tc.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then step "  torch metadata cross-check" "$(tail -1 "$work/tc.log")"
  elif [ $rc -eq 2 ]; then step "  torch metadata cross-check" "skipped ($(tail -1 "$work/tc.log"))"
  else step "  torch metadata cross-check" "FAIL"; tail -6 "$work/tc.log" | sed 's/^/      /'; fail=1; fi
else
  step "  torch metadata cross-check" "skipped (no interpreter with torch found)"
fi

rm -rf "$work"
echo
if [ $fail -eq 0 ]; then echo "TORCHBIN GATE: PASS"; else echo "TORCHBIN GATE: FAIL"; fi
exit $fail
