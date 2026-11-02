# agent

The MCP server that exposes the harness to a language model, and the loop that drives a model through it for evaluation.

`server.py` serves the harness on stdio, one backend per process, with `get_state` on the fake backend. `press` and the project config follow in AGT-02 (Jayden), the `navigate_to` and `advance_dialogue` rungs in AGT-04.

Contract: `docs/harness-spec.md`, MCP tool surface. Nothing but protocol messages may reach standard output.
