# backends

One class per emulator, all behind the `GameBackend` protocol in `base.py`. This is the only package that knows which console is running.

- `fake.py`: a ROM-free toy console for tests and CI.
- `mgba.py`: FireRed on mGBA. Arrives with ENV-04 (Rishi).
- `pyboy.py`: Super Mario Land on PyBoy (ENV-03, Akshaj). Refuses any ROM but the pinned dump.

Contract: `docs/harness-spec.md`, GameBackend. A new backend adds one entry to the registry in `tests/conformance/test_backend.py` and must pass that suite.
