import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("configurator", ROOT / "app/configurator.py")
configurator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(configurator)


class BackendPayloadTest(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads((ROOT / "config/agent-response.schema.json").read_text())

    def test_chat_schema_uses_chat_completions_shape(self):
        shape = configurator.structured_format(self.schema, "json_schema")
        self.assertEqual(shape["type"], "json_schema")
        self.assertIn("json_schema", shape)
        self.assertTrue(shape["json_schema"]["strict"])

    def test_responses_schema_uses_responses_shape(self):
        shape = configurator.responses_structured_format(self.schema, "json_schema")
        self.assertEqual(shape["type"], "json_schema")
        self.assertNotIn("json_schema", shape)
        self.assertEqual(shape["schema"], self.schema)

    def test_responses_output_fallback(self):
        self.assertEqual(configurator.response_text({"output": [{"content": [{"type": "output_text", "text": "{}"}]}]}), "{}")


if __name__ == "__main__":
    unittest.main()
