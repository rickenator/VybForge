# Probe: `build_vocab_from` maps every token to id 0 on a pretty-printed vocab.json

Minimal reproducer for a defect in the **Vyb compiler's** stdlib (`stdlib/vllm/mod.vyb`), found while
pointing VybForge's tokenizer path at a real model's `vocab.json` (Qwen3.8-27B "Ridge"). Upstream
tree checked: `rickenator/Vyb` at `09fa22b8` (== origin/main at the time of writing), defect present.
Tracker search for it (`vocab`, `tokenizer`, `read_int`, `vllm`) found no existing report.

**Status: the report is drafted in `ISSUE.md` and NOT yet filed** on `rickenator/Vyb` — filing needs a
go-ahead. When it is filed, put the issue URL here so the next session can re-check it against the
current upstream instead of re-deriving the finding.

## What it shows

Two `vocab.json` fixtures that differ **only in whitespace**:

    pretty/vocab.json    {\n    "!": 1,\n    ",": 11\n}\n     (the shape the official Qwen repo ships)
    compact/vocab.json   {"!":1,",":11}                        (a control)

Run:

    . ./vybenv.sh
    TOK_DIR=native/probes/vllm_vocab_whitespace/pretty  $VYB native/probes/vllm_vocab_whitespace/probe.vyb
    TOK_DIR=native/probes/vllm_vocab_whitespace/compact $VYB native/probes/vllm_vocab_whitespace/probe.vyb

Observed:

    pretty    KEYS=2  ZERO_VALUED=2  LOOKUP_BANG=0   LOOKUP_COMMA=0
    compact   KEYS=2  ZERO_VALUED=0  LOOKUP_BANG=1   LOOKUP_COMMA=11   (correct)

The walk inserts **every** key — so the map's size is right and nothing errors — while **every value is
0**. That is the whole defect: a wrong map that looks well-formed.

## Mechanism

`build_vocab_from` reads the key, advances to the `:`, and calls `read_int`. `read_int`'s loop breaks on
the FIRST non-digit, so the space in `": 1"` ends the number before it starts: it returns `v = 0` and
`next` pointing at the space. The outer walk then scans forward to the next `"`, so it reaches the end
of the file having recorded every key with the value 0. No short read, no error, no size mismatch.

## Impact, as measured downstream

Pointing VybForge's real tokenizer path (`llm_encode` over stdlib/vllm) at the actual
`Qwen/Qwen3.8-27B` `vocab.json` (6,722,759 bytes, pretty-printed) returned the right token COUNT with
every id zero:

    PROMPT_IDS=0,0,0,0,0        # "The capital of France is"
    PROMPT_N=5

Compacting the same file (5,234,494 bytes) made it return the oracle's ids exactly
(`760,6511,314,9338,369`). So the failure is invisible to anything that checks the number of tokens,
which is the natural sanity check a consumer writes.

## Fix direction (the maintainer's call — recorded only as the observation)

Whitespace after `:` (and generally between tokens) does not terminate a JSON number, so either
`read_int` should skip leading whitespace, or `build_vocab_from` should refuse a file its reader cannot
parse rather than returning a map of zeros. Either would turn this into a loud failure.

## Downstream workaround (labelled, temporary)

VybForge compacts the file before use and asserts a known token maps to a non-zero id
(`native/tools/ridge_encoder_check.py`, gate P4.11), so the pathology fails loudly on our side while the
upstream fix is pending. That code is a consumer-side guard, not a fix — it is deleted when the reader
handles whitespace.
