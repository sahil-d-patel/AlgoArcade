"""Download the prebuilt libmgba-py binding for this OS and unpack it into the project environment.

Run with: uv run python scripts/fetch_mgba.py
Contract: docs/setup.md (The mGBA binding).
"""

import hashlib
import io
import platform
import sys
import sysconfig
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "0.2.0-2"
BASE_URL = f"https://github.com/hanzi/libmgba-py/releases/download/{RELEASE}"

# SHA-256 of each upstream archive, computed from the release downloads on 2026-10-25.
ARCHIVES = {
    "win64": (
        "libmgba-py_0.2.0_win64.zip",
        "857280d7d952c03dea17ee148266a0ebb13e07985e446a8b0041bda728cd4513",
    ),
    "macos-arm64": (
        "libmgba-py_0.2.0_macos-arm64.zip",
        "abd90c6e4fd98d2a0bffbda16d5b94fa6944654b61353d4592af7b4b4a3484b7",
    ),
    "ubuntu-lunar": (
        "libmgba-py_0.2.0_ubuntu-lunar.zip",
        "16f4426551ec3699f46ad450b89cda8841e51475526d4427b660883adf45fc10",
    ),
}

# Written inside the unpacked package so a second run can tell it is already installed.
MARKER = ".archive-sha256"


class ChecksumError(ValueError):
    pass


def host_archive() -> str:
    system, machine = sys.platform, platform.machine().lower()
    if system == "win32" and machine in {"amd64", "x86_64"}:
        return "win64"
    if system == "darwin" and machine == "arm64":
        return "macos-arm64"
    if system == "linux" and machine == "x86_64":
        return "ubuntu-lunar"
    raise SystemExit(f"No libmgba-py archive for {system} {machine}. See docs/setup.md.")


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def install(archive: bytes, expected_sha256: str, target: Path) -> None:
    actual = hashlib.sha256(archive).hexdigest()
    if actual != expected_sha256:
        raise ChecksumError(f"archive SHA-256 is {actual}, expected {expected_sha256}")
    with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
        zip_file.extractall(target)
    (target / "mgba" / MARKER).write_text(expected_sha256)


def main() -> None:
    name, sha256 = ARCHIVES[host_archive()]
    target = Path(sysconfig.get_path("platlib"))
    marker = target / "mgba" / MARKER
    if marker.is_file() and marker.read_text() == sha256:
        print(f"{name} is already installed in {target}")
        return
    print(f"Downloading {name}")
    install(download(f"{BASE_URL}/{name}"), sha256, target)
    print(f"Installed mgba into {target}")


if __name__ == "__main__":
    main()
