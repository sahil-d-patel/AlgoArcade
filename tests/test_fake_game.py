from algoarcade.backends.base import Button
from algoarcade.backends.fake import FakeBackend
from algoarcade.games.fake import FakeReader, FakeState


def play(*buttons: Button) -> FakeState:
    backend = FakeBackend()
    for button in buttons:
        backend.press([button])
    return FakeReader().read(backend)


def test_start_state() -> None:
    assert play() == FakeState(
        room=0,
        x=1,
        y=1,
        chest_opened=False,
        frame=0,
        story_flags=0,
        party_level_sum=5,
        badge_count=0,
        milestone_count=0,
    )


def test_wall_blocks() -> None:
    assert play(Button.UP).location == (0, 1, 1)
    assert play(Button.LEFT).location == (0, 1, 1)


def test_door_changes_room_both_ways() -> None:
    to_door = [Button.DOWN, Button.DOWN, *[Button.RIGHT] * 8]
    assert play(*to_door[:-1]).location == (0, 8, 3)
    assert play(*to_door).location == (1, 1, 3)
    assert play(*to_door, Button.LEFT).location == (0, 8, 3)


def test_chest_sets_flag_only_when_a_is_pressed_on_it() -> None:
    to_chest = [Button.RIGHT, Button.RIGHT]
    assert not play(Button.A).chest_opened
    assert not play(*to_chest).chest_opened
    assert play(*to_chest, Button.A).chest_opened
    assert play(*to_chest, Button.A, Button.DOWN).chest_opened


def test_each_press_advances_24_frames() -> None:
    assert play(Button.B, Button.B).frame == 48
