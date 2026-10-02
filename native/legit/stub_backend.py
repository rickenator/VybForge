#!/usr/bin/env python3
"""Record the exact request a driver sends, and answer with a canned contract.

Verification harness only (never part of the pipeline): lets the P1.4 gate
compare the Vyb configurator's request bodies byte-for-byte against the Python
driver's, per backend, and proves both parse the same reply shapes.

Usage: stub.py <record_dir> [--port N]
Writes <record_dir>/<n>-<backend>.path and .body for each request.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

RECORD = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8099
MODE = os.environ.get("STUB_MODE", "question")


def contract() -> str:
    """Canned configurator contract.

    MODE=question (default) keeps the P1.4 baseline contract stable;
    MODE=changes carries proposed_changes for the P1.5 applier gate.
    """
    if MODE == "changes":
        return json.dumps({
            "kind": "proposal",
            "message": "Apply the requested desired-state changes.",
            "missing_fields": [],
            "proposed_changes": [
                {"path": "hostname", "op": "replace", "value": "vyb-appliance-2",
                 "reason": "user asked for this hostname"},
                {"path": "pkgs", "op": "add",
                 "value": {"name": "curl", "version": "8.5.0", "source": "repo"},
                 "reason": "needed to fetch the payload"},
            ],
            "requires_confirmation": True,
        })
    return json.dumps({
        "kind": "question",
        "message": "Which hostname should the appliance use?",
        "missing_fields": ["hostname"],
        "proposed_changes": [],
        "requires_confirmation": False,
    })


CONTRACT = contract()
SEQ = {"n": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = SEQ["n"]
        SEQ["n"] += 1
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        with open(os.path.join(RECORD, f"{n}.path"), "w") as fh:
            fh.write(self.path)
        with open(os.path.join(RECORD, f"{n}.body"), "w") as fh:
            fh.write(body)
        with open(os.path.join(RECORD, f"{n}.auth"), "w") as fh:
            fh.write(self.headers.get("Authorization", ""))
        if self.path.endswith("/api/chat"):
            reply = {"message": {"content": CONTRACT}}
        elif self.path.endswith("/chat/completions"):
            reply = {"choices": [{"message": {"content": CONTRACT}}]}
        else:
            # Responses shape with a reasoning item first: the driver must skip
            # it and take the typed output_text item (configurator.py's rule).
            reply = {
                "output": [
                    {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "thinking..."}]},
                    {"type": "message", "content": [{"type": "output_text", "text": CONTRACT}]},
                ]
            }
        raw = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
