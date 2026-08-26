#!/usr/bin/env bash
set -euo pipefail

echo "Vyb local mode is a CPU fallback. For a useful interview, use GPU-backed Ollama or ./run.sh with a hosted backend; see README.md." >&2
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$root/bin/vyb-configurator"
