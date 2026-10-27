#!/usr/bin/env bash
# build_ridge_kernels.sh: compile every PTX module the 27B Ridge decoder (native/host/ridge_decode.vyb)
# and the 4B resident gate load, into native/build/. Kernels that use more or less than the default
# 32 KB static shared buffer carry their size here (--kernel-shared, see the generator headers).
#   VYB_BIN (default ~/Projects/Vyb/build/vyb), VYB_STDLIB (default <vyb>/../../stdlib)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../build" 2>/dev/null || { mkdir -p "$(dirname "$(readlink -f "$0")")/../build"; cd "$(dirname "$(readlink -f "$0")")/../build"; }
VYB_BIN=${VYB_BIN:-$HOME/Projects/Vyb/build/vyb}
export VYB_STDLIB=${VYB_STDLIB:-$(dirname "$(dirname "$(readlink -f "$VYB_BIN")")")/stdlib}
# name[:shared-bytes]
K="attn35 gdn layer q4k q5k q6k qwen3 iq2s iq3s q8_0 rope rattn rconv32 rdelta:8192 rdelta2:12288
   rdelta2w:40960 rdelta2x:33040 rdeltag rgemv rgemv2 rgemv4 rgemv5:8192 rgemv6:8192 rgemv7:8192 rgemv8:8192 rpf:20480 rpf32
   rpf_gemm:27648 rpf_gemm128:45056 rpf_gemm128h:45056 rq4k rq4kq rq5kq rq6k rq6k8 rq6kr2 rq80s"
fail=0
for e in $K; do
  n=${e%%:*}; s=""; [[ $e == *:* ]] && s="--kernel-shared ${e##*:}"
  if "$VYB_BIN" --kernel ../kernels/$n.vyb --ptx $n.ptx $s > /tmp/brk_$n.log 2>&1; then echo "ok   $n"; else echo "FAIL $n (see /tmp/brk_$n.log)"; fail=1; fi
done
exit $fail
