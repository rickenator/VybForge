# Handoff → vyb-lang: add a bounded/offset file read to the Vyb stdlib

**For:** the Vyb compiler/stdlib implementation agent (`rickenator/Vyb`).
**From:** VybForge (`rickenator/VybForge`). Owner-approved stdlib change.
**Blocked by (use-case):** VybForge's Vyb-native GGUF loader must read the
header + tensor-index prefix of a **2.5 GB** Qwen3 GGUF. Vyb io today can only
`read_all` a whole file into memory, which is impossible here — a single 2.5 GB
`String` is far past the runtime string-registry/memory limits. We need a
stateless **bounded, offset file read** so the loader can pull just the first
few hundred KB (metadata + tensor index) and, later, tensors by offset.

## Requested API (`stdlib/io/mod.vyb`, alongside the existing `File`)

```vyb
# Read up to `n` bytes at absolute byte `off` of the open file.
# Absent (`String?` empty) only if the read fails; returns fewer than n bytes at
#   EOF (caller can detect truncation by length).
share(all) read_at(f<File>, off<Int>, n<Int>)<String?>
```
A one-line stopgap alias is also fine if `read_at` is more work, but offsets are
needed for tensor loading, so prefer the full `read_at(fd, off, n)`.

## Implementation sites (mirror the existing `vyb_io_read_all_opt` / socket `vyb_net_recv_opt` patterns)

1. **`src/vre/semantic.cpp`** (~line 2165, the io allowlist): add
   `|| name == "vyb_io_read_at"`.
2. **`src/vre/llvm/cgen_expr.cpp`** (~line 3104, `rtName` map): add
   `else if (fname == "vyb_io_read_at") rtName = "__vyb_file_read_at";`
   Then add an args-codegen block mirroring `vyb_net_recv_opt`
   (src line ~5371): `(fd, off, maxlen)` → three i64 args → call
   `__vyb_file_read_at(fd, off, maxlen, <slot>)`, returning the `String?`
   `{ {ptr,len}, has }` struct (same shape as `read_all_opt`).
3. **`src/main.cpp`** (~lines 109–110 and the `runtimeSymbols` registration
   ~2306): forward-declare + export
   `int64_t __vyb_file_read_at(int64_t fd, int64_t off, int64_t maxlen, vyb_file_str* out)`
   and implement it with `pread(fd, out->ptr, (size_t)maxlen, (off_t)off)`
   (glibc `pread` — thread-safe, no `lseek` state). Return 0 on success,
   nonzero on failure; set `out->len` to the bytes actually read.
4. **`stdlib/io/mod.vyb`** (near `read_all`, ~line 123): add the `read_at`
   wrapper forwarding to `vyb_io_read_at`.

Look at how `__vyb_file_read_all`/`__vyb_io_read_all_opt` build the returned
`vyb_file_str` for exact ABI consistency (String = `{char*, i64 len}`).

## Acceptance (a Vyb test in `rickenator/Vyb` + one here)

- `io.read_at(f, 0, 24)` on the real GGUF returns the 16-byte magic+version
  prefix of the file (and any leading bytes).
- Reading at a later offset returns exactly the bytes there; near-EOF returns a
  short read (len < n) without error; a full `read_all` still works unchanged.
- Negative/oversized `off` returns absent (error), no crash.
- Existing suite stays green.

## Cross-repo

- Motivating consumer `rickenator/VybForge`: after this lands, its GGUF reader
  (`native/gguf/`) will switch to `read_at` and stop using the temporary
  `freedom { exec_output("head -c …", …) }` workaround it is using meanwhile.
