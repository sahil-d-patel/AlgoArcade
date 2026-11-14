# agent

The MCP server that exposes the harness to a language model, and the loop that drives a model through it for evaluation.

`server.py` serves one game on stdio, chosen by `--game`, at a tool level chosen by `--level` (default 1). At L0, `get_state` and `press` return only a 2x PNG of the screen. At L1 they return the state as compact JSON and structured content, and the PNG when asked. L2 adds `navigate_to` and `advance_dialogue` from `tools.py`. `--trace FILE` writes every press with the state after it, one JSON line each. `--game firered` needs `data/roms/firered.gba` and the mGBA binding; `--state NAME` loads `data/states/NAME.state` first and refuses a manifest state whose SHA-256 differs from the manifest's, or that would run on a ROM other than rev 1, the one the states were captured on. `.mcp.json` at the repo root launches it.

`loop.py` runs one attempt on a segment and prints the outcome, the presses, the dollars, the mean tokens of each kind and dollars per press, and a projected cost for a 300-action attempt. Every request resends the whole history, so the projection does not scale the per-press cost. It takes 300 divided by the run's mean presses per request, rounded up, as the number of requests. It prices the first as measured and each later one with a prompt grown by the run's mean growth per request, on the short or long card by its size. With `ANTHROPIC_API_KEY` set, on macOS with the `DYLD_FALLBACK_LIBRARY_PATH` export from `docs/setup.md`:

```sh
uv run python -m algoarcade.agent.loop --name pilot --segment seg_01_house --max-actions 20 --max-dollars 1
uv run python -m algoarcade.agent.loop --name seg01 --segment seg_01_house --max-actions 300 --max-dollars 2
uv run python -m algoarcade.agent.loop --name seg02-l2 --segment seg_02_pallet --level 2 --max-actions 300 --max-dollars 2
```

The first is the 20-action pilot, the second the full seg_01 attempt at L1, graded by the manifest's goal for seg_01_house (`left_house`: the player is on the Pallet Town map), the third seg_02 at L2. The segments are `seg_01_house`, `seg_02_pallet` and `seg_03_rival`, and `fake` runs the fake game without a ROM. Logs go to `data/llm/<name>/`, which must not exist yet: `steps.jsonl` has one line per request, `presses.jsonl` the server's trace, `attempt.json` the totals, the per-press means and the projection. The attempt is also run `<name>` under `data/runs/` in the logging schema (`parquet.py`), which must not exist either.

An action is a button press, whichever tool made it: the loop counts and grades from the trace, and lowers a tool's `max_presses` to the presses left.

The dollar cap is hard: before each request the loop adds an upper bound on that request's cost to the spend so far, and stops if the sum would pass the cap. The model id and the prices per million tokens are config values. The defaults are Haiku 5.5's: one card for prompts up to 100K tokens and a dearer one beyond, which the loop uses for a request whose prompt, or the bound on it, is past 100K.

`tools.py` holds the game a server owns and the L2 tools. `navigate_to` walks A* over the live collision grid of the current map, other characters blocked, and stops on arrival, a battle, a text box, a map change, no path or its press cap. `advance_dialogue` presses A until the box closes or a script menu waits, and returns the text. Each game's overworld reads are a `Field`: `games/fake.py` and `games/firered/field.py`.

Contract: `docs/harness-spec.md`, MCP tool surface, and `docs/technical-plan.md`, LLM arm. Nothing but protocol messages may reach standard output.
