"""Rolling Parquet files for one table of one run. Contract: docs/harness-spec.md, Logging schema."""

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from algoarcade.logs.schema import TABLES, validate

# A guess: about 13 minutes of per-step rows at 500 steps/s with every 4th step logged.
ROWS_PER_FILE = 100_000


def run_dir(run_id: str, root: Path = Path("data/runs")) -> Path:
    return root / f"run_id={run_id}"


class RollingWriter:
    """Writes `<run_dir>/<table>/<writer_id>-<n>.parquet`, starting a new file once
    one holds `rows_per_file` rows. An open Parquet file cannot be read, so the open file is
    `_<writer_id>-<n>.parquet.inprogress` until it closes: a glob on `*.parquet` and a folder read
    with pyarrow both skip it. `writer_id` must be unique per process writing this table.

    A writer refuses a folder that already holds files from its `writer_id`, so a reused run id
    cannot mix or overwrite data. With `resume=True` it numbers on after the highest file."""

    def __init__(
        self,
        run: Path,
        table: str,
        writer_id: str,
        rows_per_file: int = ROWS_PER_FILE,
        resume: bool = False,
    ) -> None:
        self._table = table
        self._schema = TABLES[table]
        self._dir = run / table
        self._writer_id = writer_id
        self._rows_per_file = rows_per_file
        own = re.compile(rf"_?{re.escape(writer_id)}-(\d+)\.parquet(\.inprogress)?")
        taken = [int(m[1]) for p in self._dir.glob("*") if (m := own.fullmatch(p.name))]
        if taken and not resume:
            raise FileExistsError(
                f"{self._dir} already has files from writer {writer_id!r}; "
                "use a new run id, or resume=True to append"
            )
        self._file_index = max(taken, default=-1) + 1
        self._writer: pq.ParquetWriter | None = None
        self._path = Path()
        self._rows = 0

    def write(self, batch: pa.Table) -> None:
        validate(batch, self._table)
        if self._writer is None:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._path = self._dir / f"{self._writer_id}-{self._file_index:05d}.parquet"
            self._writer = pq.ParquetWriter(self._open_path, self._schema)
            self._rows = 0
        self._writer.write_table(batch.select(self._schema.names).cast(self._schema))
        self._rows += batch.num_rows
        if self._rows >= self._rows_per_file:
            self.close()

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._open_path.rename(self._path)
        self._writer = None
        self._file_index += 1

    @property
    def _open_path(self) -> Path:
        return self._path.with_name(f"_{self._path.name}.inprogress")
