Title: bug: stdlib/vllm build_vocab_from maps EVERY token to id 0 when vocab.json is pretty-printed (silent wrong map)

Filed as https://github.com/rickenator/Vyb/issues/487 — this file is the in-repo copy of that report's
body, so the finding survives without the tracker and the next session can compare the two.

## Observed

`build_vocab_from` in `stdlib/vllm/mod.vyb` returns a map whose every value is 0 when the
`vocab.json` it reads is pretty-printed (a space after the `:`). Nothing errors and the key count is
correct, so the map looks well-formed.

Minimal reproducer (no model, no tokenizer, two fixtures differing ONLY in whitespace):

    pretty/vocab.json    {\n    "!": 1,\n    ",": 11\n}\n
    compact/vocab.json   {"!":1,",":11}

    TOK_DIR=<dir>/pretty  $VYB native/probes/vllm_vocab_whitespace/probe.vyb
    TOK_DIR=<dir>/compact $VYB native/probes/vllm_vocab_whitespace/probe.vyb

    pretty    KEYS=2  ZERO_VALUED=2  LOOKUP_BANG=0   LOOKUP_COMMA=0
    compact   KEYS=2  ZERO_VALUED=0  LOOKUP_BANG=1   LOOKUP_COMMA=11

The probe reads `m.keys`/`m.vals` directly, so `ZERO_VALUED` is a count over the inserted entries —
the walk inserts both keys and gives both the value 0.

## Mechanism

`read_int(s, i)` starts at `i`, accumulates digits, and breaks on the first character that is not
`0`-`9`. Its caller advances from the key to the `:` and passes the position right after it — which in
a pretty-printed file is a SPACE. `read_int` therefore breaks before reading any digit and returns
`v = 0` with `next` still on the space; the outer walk then scans forward to the next `"`, so it
reaches the end of the file having inserted every key with the value 0.

## Impact

Real model repos ship pretty-printed vocab.json — measured on `Qwen/Qwen3.8-27B`'s (6,722,759 bytes).
Consumed through the real tokenizer path, the failure is a right-length, all-zero id stream:

    PROMPT_IDS=0,0,0,0,0        # "The capital of France is"
    PROMPT_N=5

Since a tokenizer's obvious sanity check is the token COUNT, the defect survives review and shows up
much later as a wrong model. Compacting the same file (5,234,494 bytes) gives the correct ids
(`760,6511,314,9338,369`), which is the confirmation that the formatting, not the parse target, is the
trigger.

## Build

Vyb at `09fa22b8` (== origin/main when this was written); reproduced on Linux x86_64 with the repo's
`build/vyb`, no flags. The failing construct needs nothing but a `vocab.json` with a space after a
colon.

## Note on the fix

Recorded as an observation, not a request: whitespace does not terminate a JSON number, so either
`read_int` should skip whitespace before the digits, or `build_vocab_from` should refuse a file it
cannot parse instead of returning a map of zeros. The silent-zero map is the part that costs time.

A downstream consumer (VybForge) currently compacts the file and asserts a known token maps to a
non-zero id so the pathology fails loudly on its side; that is a labelled consumer-side guard, not a
fix, and it goes away once the reader handles whitespace.
