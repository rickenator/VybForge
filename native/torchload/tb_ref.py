#!/usr/bin/env python3
"""Independent reference for the container layer of the torchbin gate.

Enumerates a PyTorch zipfile-format `.bin` shard with Python's own `zipfile` and emits the same
canonical listing as native/torchload/tb_zip_probe.vyb. Deliberately independent where it can be:

  * the file SIZE comes from the OS (`os.path.getsize`), while the Vyb side finds it by probing
    read_at to EOF — so every gate run validates that probe;
  * the bytes of each hashed record are read TWICE, once by absolute seek and once through
    `zipfile.open(name).read()` (Python's own decompression path), and the two hashes must agree
    (reported as TB_REF_BYTEPATH_OK) — that validates the Vyb-side absolute data offset without
    sharing any code with it;
  * sizes/method/CRC come from the central directory Python parses.

EOCD location note (the thing that cost a gate run): acceptance is on the EOCD's own comment
length (eocd + 22 + comment_len == file size), NOT on cd_off + cd_size == eocd. torch writes a
ZIP64 end-of-central-directory record plus its locator between the directory and the EOCD, so the
directory ends 76 bytes before the EOCD on every real shard. The 32-bit directory values stay
authoritative unless they are sentinels, in which case the ZIP64 record holds the real ones.

The only shared algorithm is the local-header data-offset arithmetic (30 + fnlen + extralen), which
is the format's documented layout.

SAMPLING RULE (identical in tb_zip_probe.vyb — change both together): a record's sha column is the
sha256 of its first `hlen` bytes, hlen = on-disk length if the FILE is <= 300,000,000 bytes, else
min(on-disk length, 1,048,576); files over the threshold hash only indices
{0, 1, n-2, n-1, argmin(bytes), argmax(bytes)}, deduped ascending, unhashed records print "-".
"""
import hashlib
import os
import struct
import sys
import zipfile

BIG = 300_000_000
WIN = 1_048_576
EOCD_SIG = b"PK\x05\x06"
Z64_LOC_SIG = b"PK\x06\x07"
Z64_EOCD_SIG = b"PK\x06\x06"
SENTINEL = 0xFFFFFFFF
META_SUFFIXES = ("/data.pkl", "/byteorder", "/version", "/.data/serialization_id")


def locate_directory(f, size):
    """-> (eocd, tail, tail_start, ent, cd_off, cd_size, zip64) or (-1, ...) on failure."""
    tail_start = max(0, size - 65557)
    f.seek(tail_start)
    tail = f.read(size - tail_start)
    j = len(tail) - 22
    while j >= 0:
        if tail[j:j + 4] == EOCD_SIG:
            cmlen = struct.unpack("<H", tail[j + 20:j + 22])[0]
            if tail_start + j + 22 + cmlen == size:
                ent, cds, cdo = struct.unpack("<HII", tail[j + 10:j + 20])
                zip64 = j >= 20 and tail[j - 20:j - 16] == Z64_LOC_SIG
                if ent == 0xFFFF or cdo == SENTINEL or cds == SENTINEL:
                    zip64 = True
                if zip64 and j >= 20 and tail[j - 20:j - 16] == Z64_LOC_SIG:
                    zoff = struct.unpack("<Q", tail[j - 12:j - 4])[0]
                    f.seek(zoff)
                    z = f.read(56)
                    if z[:4] == Z64_EOCD_SIG:
                        ent = struct.unpack("<Q", z[32:40])[0]
                        cds = struct.unpack("<Q", z[40:48])[0]
                        cdo = struct.unpack("<Q", z[48:56])[0]
                return tail_start + j, tail, tail_start, ent, cdo, cds, zip64
        j -= 1
    return -1, tail, tail_start, 0, 0, 0, False


def main():
    path = os.environ.get("VYBFORGE_TB_FILE", os.path.expanduser("~/Models/spikingbrain-v1-7b-base/pytorch_model-00001.bin"))
    out = os.environ.get("VYBFORGE_TB_OUT", "native/out/tb_zip_listing_ref.txt")
    size = os.path.getsize(path)

    with open(path, "rb") as f:
        eocd, tail, _, ent, cd_off, cd_size, zip64 = locate_directory(f, size)
        if eocd < 0:
            print(f"TB_REF_FAIL no EOCD path={path}")
            return 1
        if ent != len(zipfile.ZipFile(path).infolist()):
            print(f"TB_REF_FAIL EOCD entry count {ent} != central directory {len(zipfile.ZipFile(path).infolist())}")
            return 1
        zf = zipfile.ZipFile(path)
        infos = zf.infolist()
        recs = []
        for info in infos:
            f.seek(info.header_offset)
            lh = f.read(30)
            if lh[:4] != b"PK\x03\x04":
                print(f"TB_REF_FAIL bad local header for {info.filename}")
                return 1
            fnlen, extralen = struct.unpack("<HH", lh[26:30])
            doff = info.header_offset + 30 + fnlen + extralen
            recs.append((info.filename, info.compress_type, info.compress_size, info.file_size,
                         info.CRC, info.header_offset, doff))

        big = size > BIG
        n = len(recs)
        picked = set()
        if big:
            sizes = [r[2] for r in recs]
            imin = min(range(n), key=lambda i: sizes[i])
            imax = max(range(n), key=lambda i: sizes[i])
            for i in (0, 1, n - 2, n - 1, imin, imax):
                if 0 <= i < n:
                    picked.add(i)

        lines = [f"RECORDS {n}", f"SIZE {size}",
                 f"PREFIX {(recs[0][0].split('/')[0] if recs else '')}",
                 f"CD {cd_off} {cd_size}", f"CDGAP {eocd - (cd_off + cd_size)}",
                 f"EOCD {eocd}", f"ZIP64 {'true' if zip64 else 'false'}",
                 f"BIG {'true' if big else 'false'}"]
        bad = hc = datarecs = 0
        bypath_checked = 0
        for i, (name, method, clen, rawlen, crc, lh_off, doff) in enumerate(recs):
            if doff + clen > size:
                bad += 1
            if clen > 0 and doff <= 0:
                bad += 1
            if clen == SENTINEL or rawlen == SENTINEL or lh_off == SENTINEL:
                bad += 1
            if not name.endswith(META_SUFFIXES):
                datarecs += 1
            go = (not big) or (i in picked)
            hl = min(clen, WIN) if big else clen
            sha = "-"
            if go:
                f.seek(doff)
                raw = f.read(hl)
                sha = hashlib.sha256(raw).hexdigest() if len(raw) == hl else "SHORT"
                if sha != "SHORT":
                    hc += 1
                    alt = zf.open(name).read(hl)
                    if hashlib.sha256(alt).hexdigest() != sha:
                        print(f"TB_REF_FAIL byte paths disagree for {name}")
                        return 1
                    bypath_checked += 1
            lines.append(f"{name}|{method}|{clen}|{rawlen}|{crc:08x}|{doff}|{sha}")
        lines += [f"DATARECS {datarecs}", f"HASHED {hc}", f"MISMATCH {bad}"]

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"TB_REF_OK records={n} datarecs={datarecs} size={size} hashed={hc} mismatch={bad} out={out}")
    print(f"TB_REF_BYTEPATH_OK {bypath_checked} records hashed identically by raw seek and by zipfile")
    return 1 if bad > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
