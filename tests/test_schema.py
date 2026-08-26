import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MockConfigurationTest(unittest.TestCase):
    def test_mock_contains_the_required_system_domains(self):
        config = json.loads((ROOT / "config/mock-system.json").read_text())
        self.assertEqual(set(config), {"system", "kernel", "toolchain", "packages", "services", "users", "storage", "network", "boot"})
        self.assertEqual(config["kernel"]["package"], "linux")
        self.assertTrue(any(item["name"] == "gcc" for item in config["toolchain"]))
        self.assertTrue(any(item["selected"] for item in config["packages"]))

    def test_agent_response_schema_requires_review_gate(self):
        schema = json.loads((ROOT / "config/agent-response.schema.json").read_text())
        self.assertIn("requires_confirmation", schema["required"])
        self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
