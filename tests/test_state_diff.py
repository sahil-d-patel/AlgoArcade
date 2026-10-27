import pytest
from state_diff import EWRAM_BASE, EWRAM_SIZE, compare_states

from algoarcade.backends.fake import FakeBackend


class StateBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.state = b"original" + bytes(EWRAM_SIZE)

    def save_state(self) -> bytes:
        return self.state

    def load_state(self, state: bytes) -> None:
        if len(state) != EWRAM_SIZE + 8:
            raise ValueError("Invalid synthetic save state")
        self.state = state

    def read(self, address: int, size: int) -> bytes:
        assert (address, size) == (EWRAM_BASE, EWRAM_SIZE)
        return self.state[8:]


def test_prints_memory_addresses_not_save_file_offsets(capsys: pytest.CaptureFixture[str]) -> None:
    backend = StateBackend()
    original = backend.save_state()
    first = bytearray(EWRAM_SIZE)
    second = bytearray(EWRAM_SIZE)
    first[0], second[0] = 0x12, 0x34
    first[0x2552C], second[0x2552C] = 0x0C, 0x0D
    first[-1], second[-1] = 0xFE, 0xFF
    compare_states(backend, b"header-A" + first, b"header-B" + second)
    assert capsys.readouterr().out == (
        "0x02000000: 12 -> 34\n0x0202552c: 0c -> 0d\n0x0203ffff: fe -> ff\n"
    )
    assert backend.save_state() == original


def test_different_headers_with_identical_memory_print_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = StateBackend()
    compare_states(backend, b"header-A" + bytes(EWRAM_SIZE), b"header-B" + bytes(EWRAM_SIZE))
    assert capsys.readouterr().out == ""


def test_restores_original_state_after_failed_load() -> None:
    backend = StateBackend()
    original = backend.save_state()
    with pytest.raises(ValueError, match="Invalid synthetic save state"):
        compare_states(backend, b"header-A" + bytes(EWRAM_SIZE), b"invalid")
    assert backend.save_state() == original


def test_rejects_short_memory_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = StateBackend()
    original = backend.save_state()
    monkeypatch.setattr(backend, "read", lambda address, size: b"\x00")
    with pytest.raises(ValueError, match="full EWRAM"):
        compare_states(backend, original, original)
    assert backend.save_state() == original
