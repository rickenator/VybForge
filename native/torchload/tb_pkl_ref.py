#!/usr/bin/env python3
"""Independent reference for the object-graph layer of the torchbin gate.

Reads a PyTorch `.bin` shard with Python's own `zipfile` + `pickle` (with `find_class` and
`persistent_load` overridden to record what torch's loader would resolve) and emits the same
canonical listing as native/torchload/tb_pkl_probe.vyb.

Independence, deliberately:
  * the object graph comes from CPython's real pickle VM, not a re-implementation of it;
  * the storage record is located by SUFFIX over `zf.namelist()` ("/data/<key>"), not by rebuilding
    the "<prefix>/data/<key>" name the Vyb side constructs — so a wrong prefix on the Vyb side shows
    up as differing hashes instead of being mirrored;
  * tensor bytes are read through `zf.open(record)` (Python's own decompression path), skipping to
    the window rather than seeking inside the file, so the Vyb side's absolute-offset arithmetic is
    validated rather than shared.

SAMPLING RULE (identical in tb_pkl_probe.vyb — change both together): file <= 300,000,000 bytes:
every tensor hashed over its whole length. Larger: a tensor is hashed if its byte length <=
1,048,576 (in full) or its index is in {0, 1, n-2, n-1, argmin(bytes), argmax(bytes)} (first
1,048,576 bytes); others print "-".
"""
import hashlib
import os
import pickle
import sys
import zipfile

BIG = 300_000_000
WIN = 1_048_576

DTYPE_OF = {
    "torch.BFloat16Storage": "BF16",
    "torch.HalfStorage": "F16",
    "torch.FloatStorage": "F32",
    "torch.DoubleStorage": "F64",
    "torch.ByteStorage": "U8",
    "torch.CharStorage": "I8",
    "torch.ShortStorage": "I16",
    "torch.IntStorage": "I32",
    "torch.LongStorage": "I64",
    "torch.BoolStorage": "BOOL",
    "torch.UntypedStorage": "U8",
    "torch.Float8_e4m3fnStorage": "F8_E4M3",
    "torch.Float8_e5m2Storage": "F8_E5M2",
}
DTSIZE = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4,
          "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
          "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


class StorageMark:
    def __init__(self, name):
        self.name = name


def make_class(mod, name):
    full = f"{mod}.{name}"
    if "Storage" in name:
        return StorageMark(full)
    if name == "_rebuild_tensor_v2":
        def rebuild(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None):
            return {"kind": "tensor", "storage": storage, "soff": int(storage_offset),
                    "size": tuple(int(x) for x in size), "stride": tuple(int(x) for x in stride)}
        return rebuild
    if name == "OrderedDict":
        return dict
    raise KeyError(f"unsupported pickle global {full}")


class Recorder(pickle.Unpickler):
    def find_class(self, m, n):
        return make_class(m, n)

    def persistent_load(self, pid):
        # ('storage', <StorageClass>, key, location, numel)
        st, key, loc, numel = pid[1], pid[2], pid[3], pid[4]
        return {"kind": "storage", "dtype": DTYPE_OF.get(st.name, ""), "key": key,
                "loc": loc, "numel": int(numel)}


def main():
    path = os.environ.get("VYBFORGE_TB_FILE", "/home/rick/Models/spikingbrain-v1-7b-base/pytorch_model-00001.bin")
    out = os.environ.get("VYBFORGE_TB_OUT", "native/out/tb_pkl_listing_ref.txt")
    size = os.path.getsize(path)
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pklname = [n for n in names if n.endswith("/data.pkl")]
    if not pklname:
        print(f"TB_PKL_REF_FAIL no data.pkl in {path}")
        return 1
    pklname = pklname[0]
    pkl_bytes = zf.getinfo(pklname).file_size

    with zf.open(pklname) as fh:
        graph = Recorder(fh).load()
    if not isinstance(graph, dict):
        print(f"TB_PKL_REF_FAIL top-level object is {type(graph)}")
        return 1

    entries = [(name, t) for name, t in graph.items()]
    nten = len(entries)
    big = size > BIG

    def nbytes_of(t):
        dts = t["storage"]["dtype"]
        n = 1
        for d in t["size"]:
            n *= d
        return n * DTSIZE.get(dts, 0)

    picked = set()
    if big:
        sizes = [nbytes_of(t) for _, t in entries]
        imin = min(range(nten), key=lambda i: sizes[i])
        imax = max(range(nten), key=lambda i: sizes[i])
        for i in (0, 1, nten - 2, nten - 1, imin, imax):
            if 0 <= i < nten:
                picked.add(i)

    lines = [f"TENSORS {nten}", f"PKL {pkl_bytes}", f"BIG {'true' if big else 'false'}"]
    bad = hc = noncontig = 0
    for i, (name, t) in enumerate(entries):
        st = t["storage"]
        dts = st["dtype"]
        dims = list(t["size"])
        sds = list(t["stride"])
        soff = t["soff"]
        snumel = st["numel"]
        numel = 1
        for d in dims:
            numel *= d
        dsz = DTSIZE.get(dts, 0)
        nbytes = numel * dsz

        contig = True
        acc = 1
        for w in range(len(dims) - 1, -1, -1):
            if sds[w] != acc:
                contig = False
            acc *= dims[w]
        if not contig:
            noncontig += 1
        if len(dims) != len(sds):
            bad += 1

        rec = [n for n in names if n.endswith("/data/" + str(st["key"]))]
        if not rec:
            bad += 1
            recname = None
        else:
            recname = rec[0]
            if zf.getinfo(recname).file_size != snumel * dsz:
                bad += 1
            if soff + numel > snumel:
                bad += 1

        go = True
        hl = nbytes
        if big:
            go = (i in picked) or (nbytes <= WIN)
            if nbytes > WIN:
                hl = WIN
        sha = "-"
        if go:
            if nbytes <= 0:
                sha = "ZERO"
            elif recname is None or dsz <= 0:
                sha = "ERR"
            else:
                with zf.open(recname) as fh:
                    skip = soff * dsz
                    while skip > 0:
                        chunk = fh.read(min(skip, 1 << 20))
                        if not chunk:
                            break
                        skip -= len(chunk)
                    data = fh.read(hl)
                sha = hashlib.sha256(data).hexdigest() if len(data) == hl else "ERR"
                if sha != "ERR":
                    hc += 1
        lines.append("|".join([
            name, dts, ",".join(str(d) for d in dims), str(numel), ",".join(str(s) for s in sds),
            str(soff), str(st["key"]), str(snumel), str(nbytes), "true" if contig else "false", sha]))
    lines += [f"HASHED {hc}", f"MISMATCH {bad}", f"NONCONTIG {noncontig}"]

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"TB_PKL_REF_OK tensors={nten} pkl={pkl_bytes} hashed={hc} mismatch={bad} noncontig={noncontig} out={out}")
    return 1 if bad > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
