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

    def test_proposed_change_items_match_the_corpus(self):
        """the response schema, generator, Vyb client, and training corpus must
        all describe proposed_changes items as {path, op, value, reason}."""
        schema = json.loads((ROOT / "config/agent-response.schema.json").read_text())
        item = schema["properties"]["proposed_changes"]["items"]
        self.assertEqual(set(item["required"]), {"path", "op", "value", "reason"})
        self.assertIn("op", item["properties"])
        seen = set()
        for path in sorted((ROOT / "data").glob("*.jsonl")):
            for line in path.read_text().splitlines():
                record = json.loads(line)
                answer = json.loads(record["messages"][-1]["content"])
                for change in answer.get("proposed_changes", []):
                    seen.add(frozenset(change))
                    self.assertEqual(set(change), {"path", "op", "value", "reason"})
                    self.assertIn(change["op"], {"add", "replace", "remove"})
                    self.assertIsInstance(change["path"], str)
                    self.assertIsInstance(change["reason"], str)
        self.assertTrue(seen, "expected at least one proposed_changes item in the corpus")




if __name__ == "__main__":
    unittest.main()
