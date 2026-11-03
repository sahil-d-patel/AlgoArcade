import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "fetch_mgba.py"
spec = importlib.util.spec_from_file_location("fetch_mgba", SCRIPT)
assert spec is not None
assert spec.loader is not None
fetch_mgba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_mgba)


@pytest.fixture
def archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        zip_file.writestr("mgba/__init__.py", "")
    return buffer.getvalue()


def test_refuses_an_archive_whose_sha256_differs(archive: bytes, tmp_path: Path) -> None:
    with pytest.raises(fetch_mgba.ChecksumError):
        fetch_mgba.install(archive, "0" * 64, tmp_path)
    assert not (tmp_path / "mgba").exists()


def test_installs_an_archive_whose_sha256_matches(archive: bytes, tmp_path: Path) -> None:
    fetch_mgba.install(archive, hashlib.sha256(archive).hexdigest(), tmp_path)
    assert (tmp_path / "mgba" / "__init__.py").is_file()


def test_second_run_does_not_download_again(
    archive: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads = []

    def fake_download(url: str) -> bytes:
        downloads.append(url)
        return archive

    monkeypatch.setattr(fetch_mgba, "download", fake_download)
    monkeypatch.setattr(fetch_mgba, "host_archive", lambda: "toy")
    monkeypatch.setitem(
        fetch_mgba.ARCHIVES, "toy", ("toy.zip", hashlib.sha256(archive).hexdigest())
    )
    monkeypatch.setattr(fetch_mgba.sysconfig, "get_path", lambda name: str(tmp_path))
    fetch_mgba.main()
    fetch_mgba.main()
    assert len(downloads) == 1
