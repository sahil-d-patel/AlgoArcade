# backends

One class per emulator, all behind the `GameBackend` protocol in `base.py`. This is the only package that knows which console is running.

- `fake.py`: a ROM-free toy console for tests and CI.
- `mgba.py`: FireRed on mGBA, with direct EWRAM/IWRAM views and native save-state bytes (ENV-04).
- `pyboy.py`: Super Mario Land on PyBoy. Arrives with ENV-03 (Akshaj).

Contract: `docs/harness-spec.md`, GameBackend. A new backend adds one entry to the registry in `tests/conformance/test_backend.py` and must pass that suite.
