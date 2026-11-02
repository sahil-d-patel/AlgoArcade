# agent

The MCP server that exposes the harness to a language model, and the loop that drives a model through it for evaluation.

`server.py` serves the harness on stdio, one backend per process, with `get_state` and `press` on the fake backend, and `.mcp.json` at the repo root launches it. The `navigate_to` and `advance_dialogue` rungs, the level option and the screenshots follow in AGT-04 (Jayden).

Contract: `docs/harness-spec.md`, MCP tool surface. Nothing but protocol messages may reach standard output.
