#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vyb_bin="${VYB_BIN:-/home/rick/Projects/Vyb/build/vyb}"

mkdir -p "$root/bin" "$root/data"
"$vyb_bin" "$root/training/generate_dataset.vyb" --build "$root/bin/vybos-training-data" -O2
# Native Vyb prints main()'s integer return after stdout. Keep only JSONL rows.
"$root/bin/vybos-training-data" | sed -n '/^{/p' >"$root/data/vybos-configurator-all.jsonl"
python3 - "$root/data/vybos-configurator-all.jsonl" "$root/data/vybos-configurator-train.jsonl" "$root/data/vybos-configurator-eval.jsonl" <<'PY'
import json
import sys

source, train, evaluation = map(__import__('pathlib').Path, sys.argv[1:])
rows = [json.loads(line) for line in source.read_text().splitlines() if line]
for path, split in ((train, 'train'), (evaluation, 'eval')):
    selected = [row for row in rows if row['metadata']['split'] == split]
    path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in selected))
    print(f'{path.name}: {len(selected)} valid records')
PY
