#!/usr/bin/env bash
# vybenv.sh — resolve the Vyb toolchain in one place (VybForge#15).
#
# Source it; do not execute it.
#
#   . "$root/vybenv.sh"
#
# Exports:
#
#   VYBHOME     the Vyb checkout root — the canonical knob
#   VYB         the compiler/JIT binary       ($VYBHOME/build/vyb)
#   VYB_STDLIB  the standard library          ($VYBHOME/stdlib)
#   VYBOSHOME   the VybOS checkout root       (sibling repo; the repair and ledger
#   VYBOS       kept as VYBOSHOME's alias      gates need it)
#
# Resolution order for VYBHOME — first hit wins:
#   1. $VYBHOME from the environment. If $VYBHOME/SOURCEME_VYB exists it is sourced
#      first and its values are taken as given; Vyb's own build is expected to write
#      that file (rickenator/Vyb#424), so this is the forward-compatible path.
#   2. Derived from an explicit $VYB_BIN or $VYB that points at the binary.
#   3. $HOME/Projects/Vyb, if it looks like a Vyb checkout.
#
# Nothing here is a config file and nothing is written to your shell: export VYBHOME
# once (e.g. from $VYBHOME/SOURCEME_VYB) instead of the project guessing a path.
# Set VYBENV_QUIET=1 to suppress the notice printed when VYBHOME had to be inferred.

_vybenv_home_from_bin() {
  ( cd "$(dirname "$1")/.." 2>/dev/null && pwd )
}

if [ -n "${VYBHOME:-}" ] && [ -f "${VYBHOME}/SOURCEME_VYB" ]; then
  # shellcheck disable=SC1090
  . "${VYBHOME}/SOURCEME_VYB"
fi

_vybenv_inferred=""
if [ -z "${VYBHOME:-}" ]; then
  if [ -n "${VYB_BIN:-}" ]; then
    VYBHOME="$(_vybenv_home_from_bin "$VYB_BIN")"
  elif [ -n "${VYB:-}" ] && [ ! -d "${VYB}" ]; then
    VYBHOME="$(_vybenv_home_from_bin "$VYB")"
  elif [ -d "${HOME}/Projects/Vyb/build" ]; then
    VYBHOME="${HOME}/Projects/Vyb"
  fi
  _vybenv_inferred=1
fi

VYB="${VYB:-${VYBHOME:-}/build/vyb}"
VYB_STDLIB="${VYB_STDLIB:-${VYBHOME:-}/stdlib}"

if [ -z "${VYBHOME:-}" ]; then
  VYBHOME="$(cd "$(dirname "${VYB}")/.." 2>/dev/null && pwd)"
  VYB_STDLIB="${VYB_STDLIB:-${VYBHOME}/stdlib}"
fi

if [ ! -x "${VYB}" ]; then
  echo "vybenv.sh: the Vyb toolchain was not found." >&2
  echo "  set VYBHOME to your Vyb checkout — Vyb's build is expected to write" >&2
  echo "  \$VYBHOME/SOURCEME_VYB (rickenator/Vyb#424), after which:" >&2
  echo "      export VYBHOME=\$HOME/Projects/Vyb   # in your shell" >&2
  echo "      . \"\$VYBHOME/SOURCEME_VYB\"" >&2
  echo "  looked for VYBHOME='${VYBHOME:-<unset>}' and binary '${VYB}'" >&2
  return 1
fi

if [ -n "${_vybenv_inferred}" ] && [ "${VYBENV_QUIET:-0}" != "1" ]; then
  echo "vybenv: VYBHOME was not set; inferred ${VYBHOME} (export VYBHOME to be explicit)" >&2
fi
_vybenv_inferred=""

export VYBHOME VYB VYB_STDLIB

# VybOS is a sibling checkout; only the repair and ledger gates use it, so an
# unresolvable VybOS is left empty and those gates say what they need.
if [ -z "${VYBOSHOME:-}" ]; then
  if [ -n "${VYBOS:-}" ]; then
    VYBOSHOME="$(cd "${VYBOS}" 2>/dev/null && pwd)"
  elif [ -d "${HOME}/Projects/VybOS" ]; then
    VYBOSHOME="${HOME}/Projects/VybOS"
  fi
fi
if [ -n "${VYBOSHOME:-}" ] && [ -z "${VYBOS:-}" ]; then
  VYBOS="${VYBOSHOME}"
fi
export VYBOSHOME VYBOS
