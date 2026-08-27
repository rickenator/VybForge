#!/usr/bin/env python3
"""Deterministic VybOS desired-state pipeline driver (first pass).

Plumbing only -- the validation/merge/render logic lives in the UTF Vyb program
tools/apply.vyb (compile Vyb-native with the stable toolchain). This driver:
  1. reads the baseline from config/default-state.json,
  2. feeds confirmed patches (JSONL) to tools/apply.vyb over stdin,
  3. splits the applier output into issue list / SPEC / rendered PROGRAM,
  4. writes out/spec.json (the machine-contract) and out/system.vyb, and
  5. compiles + runs the rendered program to PROVE it reproduces spec.json.

Usage: apply_interview.py [patches.jsonl]   (default out/patches.jsonl)
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
VYB = os.environ.get("VYB_BIN", str(pathlib.Path.home() / "Projects/Vyb-vybos/build/vyb"))


def main() -> int:
    patches_file = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "out/patches.jsonl"))
    if not patches_file.is_file():
        print(f"no patches file: {patches_file}", file=sys.stderr)
        return 2
    baseline = json.loads((ROOT / "config/default-state.json").read_text())
    outdir = ROOT / "out"
    outdir.mkdir(exist_ok=True)

    # run the Vyb applier with baseline scalar fields injected and patches on stdin
    env = dict(os.environ, APP_SYSTEM=baseline["system"], APP_HOSTNAME=baseline["hostname"])
    apply = ROOT / "tools/apply.vyb"
    run = subprocess.run([VYB, str(apply)], input=patches_file.read_text(), text=True,
                         capture_output=True, env=env)
    if run.returncode != 0:
        sys.stderr.write(run.stdout)
        sys.stderr.write(run.stderr)
        return 1

    sections = {}
    cur = None
    for line in run.stdout.splitlines():
        if line.startswith("ISSUES::"):
            cur = "ISSUES"
        elif line.startswith("SPEC::"):
            cur = "SPEC"
        elif line.startswith("PROGRAM::"):
            cur = "PROGRAM"
        elif cur:
            sections.setdefault(cur, []).append(line)

    if "ISSUES" in sections:
        print("applier validation issues:")
        print("\n".join(sections["ISSUES"]))
        return 1

    spec = "\n".join(l for l in sections.get("SPEC", []) if l.strip()).strip()
    # drop the bare "0" that Vyb prints for main()'s return, and blank lines
    prog = "\n".join(l for l in sections.get("PROGRAM", []) if l.strip() and l.strip() != "0").strip()
    try:
        parsed = json.loads(spec)
    except json.JSONDecodeError as e:
        print(f"applier emitted invalid spec JSON: {e}", file=sys.stderr)
        return 1

    (outdir / "spec.json").write_text(spec + "\n")
    (outdir / "system.vyb").write_text(prog + "\n")

    # PROVE the rendered system.vyb reproduces the spec when compiled + run.
    exe = outdir / "system-vyb-bin"
    build = subprocess.run([VYB, str(outdir / "system.vyb"), "--build", str(exe), "-O2"],
                           capture_output=True, text=True)
    if build.returncode != 0:
        print("rendered system.vyb failed to compile:", file=sys.stderr)
        sys.stderr.write(build.stdout + build.stderr)
        return 1
    run2 = subprocess.run([str(exe)], capture_output=True, text=True)
    spec_line = next((l.strip() for l in run2.stdout.splitlines() if l.strip().startswith("{")), None)
    if spec_line != spec:
        print("VERIFY FAIL: rendered program output != merged spec", file=sys.stderr)
        print("  spec : " + spec, file=sys.stderr)
        print("  prog : " + str(spec_line), file=sys.stderr)
        return 1

    print(f"merged spec : {spec}")
    print(f"src         : {len(parsed['pkgs'])} pkgs, {len(parsed['services'])} services, "
          f"system={parsed['system']!r}, hostname={parsed['hostname']!r}")
    print(f"wrote {outdir / 'spec.json'} and {outdir / 'system.vyb'}; "
          f"rendered program compiled and reproduces spec ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
