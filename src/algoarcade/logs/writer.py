"""Rolling Parquet files for one table of one run. Contract: docs/harness-spec.md, Logging schema."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from algoarcade.logs.schema import TABLES, validate

# A guess: about 13 minutes of per-step rows at 500 steps/s with every 4th step logged.
ROWS_PER_FILE = 100_000


class RollingWriter:
    """Writes `<root>/run_id=<run_id>/<table>/<writer_id>-<n>.parquet`, starting a new file once
    one holds `rows_per_file` rows. An open Parquet file cannot be read, so the open file ends in
    `.inprogress` and is renamed when it closes: a glob on `*.parquet` finds only readable files.
    `writer_id` must be unique per process writing this table."""

    def __init__(
        self,
        run_id: str,
        table: str,
        writer_id: str,
        root: Path = Path("data/runs"),
        rows_per_file: int = ROWS_PER_FILE,
    ) -> None:
        self._table = table
        self._schema = TABLES[table]
        self._dir = root / f"run_id={run_id}" / table
        self._writer_id = writer_id
        self._rows_per_file = rows_per_file
        self._file_index = 0
        self._writer: pq.ParquetWriter | None = None
        self._path = Path()
        self._rows = 0

    def write(self, batch: pa.Table) -> None:
        validate(batch, self._table)
        if self._writer is None:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._path = self._dir / f"{self._writer_id}-{self._file_index:05d}.parquet"
            self._writer = pq.ParquetWriter(f"{self._path}.inprogress", self._schema)
            self._rows = 0
        self._writer.write_table(batch.select(self._schema.names).cast(self._schema))
        self._rows += batch.num_rows
        if self._rows >= self._rows_per_file:
            self.close()

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        Path(f"{self._path}.inprogress").rename(self._path)
        self._writer = None
        self._file_index += 1
