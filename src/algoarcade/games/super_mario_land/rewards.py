"""Super Mario Land's reward table: the registry components that read only the location.

`explore` pays each new (world, level, x_tile, y_tile); its jumping exploit is in
docs/harness-spec.md, Reward registry. The penalties stay off, as on FireRed.
"""

from collections.abc import Callable

from algoarcade.games.super_mario_land.reader import SuperMarioLandState
from algoarcade.rewards.base import RewardComponent
from algoarcade.rewards.explore import Explore
from algoarcade.rewards.penalties import AntiLoop, Time
from algoarcade.rewards.registry import WEIGHTS

COMPONENTS: dict[str, Callable[[], RewardComponent[SuperMarioLandState]]] = {
    "explore": Explore,
    "anti_loop": AntiLoop,
    "time": Time,
}
DEFAULT_WEIGHTS = {"explore": WEIGHTS["explore"]}
