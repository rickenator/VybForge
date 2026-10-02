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
# Precedence follows Vyb's own SOURCEME_VYB (rickenator/Vyb#424, PR #426):
#
#   1. $VYBHOME/SOURCEME_VYB, sourced when VYBHOME already names a checkout — Vyb
#      ships it at the toplevel, and it resolves the home itself and exports the pair.
#   2. $VYBHOME — canonical. VYB and VYB_STDLIB are derived from it, and a stale
#      exported VYB does NOT win: mixing one checkout's binary with another's stdlib
#      is the failure this rule exists to prevent.
#   3. An explicit $VYB / $VYB_BIN hint — used only to DERIVE the home (…/build/vyb,
#      …/build, or the checkout root) when VYBHOME is unset.
#   4. $HOME/Projects/Vyb, if that looks like a Vyb checkout.
#
# Nothing here is a config file and nothing is written to your shell: export VYBHOME
# once (or source your checkout's SOURCEME_VYB) instead of the project guessing.
# Set VYBENV_QUIET=1 to suppress the notice printed when VYBHOME had to be inferred.

if [ -n "${VYBHOME:-}" ] && [ -f "${VYBHOME}/SOURCEME_VYB" ]; then
  # shellcheck disable=SC1090
  . "${VYBHOME}/SOURCEME_VYB"
fi

_vybenv_inferred=""
if [ -z "${VYBHOME:-}" ]; then
  _vybenv_inferred=1
  _vybenv_hint="${VYB:-${VYB_BIN:-}}"
  case "${_vybenv_hint}" in
    "")
      if [ -d "${HOME}/Projects/Vyb/build" ]; then
        VYBHOME="${HOME}/Projects/Vyb"
      fi
      ;;
    */build/vyb)
      VYBHOME="$(cd "$(dirname "${_vybenv_hint}")/.." 2>/dev/null && pwd)"
      ;;
    */build)
      VYBHOME="$(cd "${_vybenv_hint}/.." 2>/dev/null && pwd)"
      ;;
    *)
      if [ -d "${_vybenv_hint}/build" ]; then
        VYBHOME="$(cd "${_vybenv_hint}" 2>/dev/null && pwd)"
      elif [ -d "${_vybenv_hint}/../stdlib" ]; then
        VYBHOME="$(cd "${_vybenv_hint}/.." 2>/dev/null && pwd)"
      fi
      ;;
  esac
  unset _vybenv_hint
fi

if [ -z "${VYBHOME:-}" ]; then
  echo "vybenv.sh: the Vyb toolchain was not found." >&2
  echo "  point VYBHOME at your Vyb checkout and source its environment file" >&2
  echo "  (rickenator/Vyb#424 ships SOURCEME_VYB at the checkout toplevel):" >&2
  echo "      export VYBHOME=\$HOME/Projects/Vyb" >&2
  echo "      . \"\$VYBHOME/SOURCEME_VYB\"" >&2
  echo "  no \$VYBHOME, no \$VYB/\$VYB_BIN hint, and no \$HOME/Projects/Vyb checkout." >&2
  return 1
fi

# VYBHOME is canonical: derive the pair from it rather than trusting an inherited
# VYB, exactly as SOURCEME_VYB does.
VYB="${VYBHOME}/build/vyb"
VYB_STDLIB="${VYBHOME}/stdlib"

if [ ! -x "${VYB}" ]; then
  echo "vybenv.sh: VYBHOME='${VYBHOME}' has no built '${VYB}'." >&2
  echo "  build it there, or point VYBHOME at a checkout that is built." >&2
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
