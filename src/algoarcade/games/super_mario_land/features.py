"""The observation vector AlgoArcadeEnv builds from a SuperMarioLandState, each value scaled into
[0, 1]. The bounds are guesses except world (1-4), level (1-3), the 400-second timer and Mario's
screen y (one byte); the env clips anything above."""

from collections.abc import Callable

from algoarcade.games.super_mario_land.reader import SuperMarioLandState

FEATURES: dict[str, Callable[[SuperMarioLandState], float]] = {
    "world": lambda s: s.world[0] / 4,
    "level": lambda s: s.world[1] / 3,
    "x_tile": lambda s: s.x_tile / 512,
    "mario_y": lambda s: s.mario_y / 255,
    "time_left": lambda s: s.time_left / 400,
    "lives_left": lambda s: s.lives_left / 10,
}
