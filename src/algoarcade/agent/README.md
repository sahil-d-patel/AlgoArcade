# agent

The MCP server that exposes the harness to a language model, and the loop that drives a model through it for evaluation.

`server.py` serves one game on stdio, chosen by `--game`, with `get_state` and `press`. Both return the state as compact JSON and structured content, and a 2x PNG of the screen when asked. `--game firered` needs `data/roms/firered.gba` and the mGBA binding; `--state NAME` loads `data/states/NAME.state` first and refuses a manifest state whose SHA-256 differs from the manifest's. `.mcp.json` at the repo root launches it.

`loop.py` runs one attempt on a segment and prints the outcome, the presses, the dollars, the mean tokens of each kind and dollars per press, and a projected cost for 300 requests. Every request resends the whole history, so the projection does not scale the per-press cost: it prices the first request as measured, then 299 more whose prompt grows by the run's mean growth per request, each on the short or long card by its size. With `ANTHROPIC_API_KEY` set, on macOS with the `DYLD_FALLBACK_LIBRARY_PATH` export from `docs/setup.md`:

```sh
uv run python -m algoarcade.agent.loop --name pilot --segment seg_01_house --max-actions 20 --max-dollars 1
uv run python -m algoarcade.agent.loop --name seg01 --segment seg_01_house --max-actions 300 --max-dollars 2
```

The first is the 20-action pilot, the second the full seg_01 attempt, graded by the manifest's goal for seg_01_house (`left_house`: the player is on the Pallet Town map). Logs go to `data/llm/<name>/`, which must not exist yet: `steps.jsonl` has one line per request, `attempt.json` the totals, the per-press means and the projection. `--segment fake` runs the fake game without a ROM.

The dollar cap is hard: before each request the loop adds an upper bound on that request's cost to the spend so far, and stops if the sum would pass the cap. The model id and the prices per million tokens are config values. The defaults are Haiku 5.5's: one card for prompts up to 100K tokens and a dearer one beyond, which the loop uses for a request whose prompt, or the bound on it, is past 100K.

`navigate_to`, `advance_dialogue` and the level option follow in AGT-04 (Jayden).

Contract: `docs/harness-spec.md`, MCP tool surface, and `docs/technical-plan.md`, LLM arm. Nothing but protocol messages may reach standard output.
