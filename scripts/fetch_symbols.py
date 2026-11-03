from pathlib import Path
from urllib.request import urlopen

COMMIT = "3b401eaf14f193946899eb7c056d5ac02a470126"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "pokefirered"


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name in ("pokefirered.sym", "pokefirered_rev1.sym"):
        url = f"https://raw.githubusercontent.com/pret/pokefirered/{COMMIT}/{name}"
        with urlopen(url, timeout=30) as response:
            content = response.read()
        if not any(
            line.endswith(" gPlayerPartyCount") for line in content.decode("utf-8").splitlines()
        ):
            raise ValueError(f"{url}: missing gPlayerPartyCount symbol")
        path = DESTINATION / name
        path.write_bytes(content)
        print(f"Saved {path.name}: {len(content)} bytes from {COMMIT}")


if __name__ == "__main__":
    main()
