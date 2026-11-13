# Loader performance probes

Standalone Vyb programs that measure the loader path against the real checkpoint. Run
them with the module paths the loader needs:

    VYB_STDLIB=~/Projects/Vyb/stdlib ~/Projects/Vyb/build/vyb \
      --module-path native/json --module-path native/torchload \
      --module-path native/glm53 --module-path ~/Projects/Vyb/stdlib/collections \
      native/glm53/perf/st_open_all_shards.vyb

- `st_open_all_shards.vyb` — parses all 83 safetensors headers and prints the total. With
  `LIVE=150000` it first builds 150k live Strings, which is how the Vyb string-registry
  live-count drift showed up: 1.4 s empty heap vs 127 s with a live heap before the fix
  (`Vyb 3347863c`), 1.09 s after.
- `st_open_in_heap.vyb` — the same measurement for a few shards, with `HASH=1` to also
  hold a 150k-entry HashMap, for isolating heap-state effects.
