# logs

The logging schema, the Parquet writer and the fake logger. Parquet is the source of truth for every figure we publish.

Owner: Sahil (OPS-03). The schema freezes on November 8 under the tag `schema-v1`. After that, changing a field is a PM decision and means re-running training.

- `schema.py`: the four tables (`steps`, `visits`, `episodes`, `runs`) and `validate`, which rejects a missing column, an extra column, a wrong type or a null in a required column.
- `writer.py`: `RollingWriter`, one per (process, table). It validates each batch and rolls to a new file every 100,000 rows under `data/runs/run_id=<id>/<table>/`. Only closed files end in `.parquet`, so they can be read while the run goes on. Write batches of a few thousand rows: each write is one row group.
- `fake.py`: a seeded synthetic run in the same schema. `uv run python -m algoarcade.logs.fake --run-id fake0 --seed 0`.

A per-step row takes 8.9 bytes on disk on fake-logger data (pyarrow defaults, Snappy), so one 15M-step run at every 4th step is about 33 MB. Real rows compress worse; with random values in the clock and three reward columns it is 46 bytes, or 172 MB.

Contract: `docs/harness-spec.md`, Logging schema.
