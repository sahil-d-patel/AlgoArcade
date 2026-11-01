# agent

The MCP server that exposes the harness to a language model, and the loop that drives a model through it for evaluation.

`server.py` serves one game on stdio, chosen by `--game` (`fake` until ENV-04 and ENV-05 land), with `get_state` and `press`. Both return the state as compact JSON and structured content, and a 2x PNG of the screen when asked. `.mcp.json` at the repo root launches it.

`loop.py` runs one attempt: `python -m algoarcade.agent.loop --name pilot --max-actions 20 --max-dollars 1`. The model id and the four prices per million tokens are config values; the defaults are Haiku 5.5's card for prompts up to 100K tokens, and the loop does not switch to the long-prompt card. The key comes from `ANTHROPIC_API_KEY`. Logs go to `data/llm/<name>/`.

`navigate_to`, `advance_dialogue` and the level option follow in AGT-04 (Jayden).

Contract: `docs/harness-spec.md`, MCP tool surface, and `docs/technical-plan.md`, LLM arm. Nothing but protocol messages may reach standard output.
