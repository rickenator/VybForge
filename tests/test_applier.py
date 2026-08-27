import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VYB = os.environ.get("VYB_BIN", str(Path.home() / "Projects/Vyb-vybos/build/vyb"))


def has_toolchain() -> bool:
    return Path(VYB).is_file()


@unittest.skipUnless(has_toolchain(), "Vyb toolchain not present")
class ApplierPipelineTest(unittest.TestCase):
    """Runs the deterministic Vyb applier driver end-to-end on small patch sets."""

    def _run(self, patches: str) -> subprocess.CompletedProcess:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write(patches)
            name = fh.name
        try:
            return subprocess.run(
                ["/usr/bin/env", "python3", str(ROOT / "tools/apply_interview.py"), name],
                capture_output=True, text=True,
                env={**os.environ, "VYB_BIN": VYB},
            )
        finally:
            Path(name).unlink(missing_ok=True)

    def test_merge_add_replace_and_renders_reproducing_spec(self):
        proc = self._run(
            '{"path":"system","op":"replace","value":"aarch64-linux","reason":"r"}\n'
            '{"path":"hostname","op":"replace","value":"vyb-appliance","reason":"r"}\n'
            '{"path":"pkgs","op":"add","value":{"name":"openssh","version":"9.9p1","source":"upstream"},"reason":"r"}\n'
            '{"path":"services","op":"add","value":{"name":"nginx","command":"nginx -g daemon off;"},"reason":"r"}\n'
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        spec = json.loads((ROOT / "out/spec.json").read_text())
        self.assertEqual(spec["system"], "aarch64-linux")
        self.assertEqual(spec["hostname"], "vyb-appliance")
        self.assertEqual([p["name"] for p in spec["pkgs"]], ["openssh"])
        self.assertEqual([s["name"] for s in spec["services"]], ["nginx"])

    def test_rejects_invalid_patches(self):
        proc = self._run(
            '{"path":"pkgs","op":"add","value":{"name":"openssh","version":"1","source":"u"},"reason":"r"}\n'
            '{"path":"pkgs","op":"add","value":{"name":"openssh","version":"2","source":"u"},"reason":"dup"}\n'
            '{"path":"nope","op":"add","value":{},"reason":"bad"}\n'
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("duplicate package", proc.stdout)
        self.assertIn("unknown path", proc.stdout)


if __name__ == "__main__":
    unittest.main()
