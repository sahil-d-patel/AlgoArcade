"""Terms on party health: heal pays for HP restored, blackout charges for a whiteout."""

from typing import Protocol, override

from algoarcade.games.base import GameState
from algoarcade.rewards.base import RewardComponent


class Party(GameState, Protocol):
    @property
    def party_count(self) -> int: ...

    @property
    def party_hp_frac(self) -> float:
        """Party HP over party max HP."""
        ...


def _valid(state: Party) -> bool:
    return 0.0 <= state.party_hp_frac <= 1.0


def blacked_out(prev: Party, curr: Party) -> bool:
    """The party went from all fainted to healed. In FireRed that is a whiteout: DoWhiteOut in
    src/overworld.c heals the party and warps in one call. Losing the Oak's Lab battle also heals
    a fainted party, without a warp (PalletTown_ProfessorOaksLab/scripts.inc), and counts too.
    The party_count checks keep an empty party, which reads 0.0, from looking fainted."""
    return (
        prev.party_count > 0
        and curr.party_count == prev.party_count
        and prev.party_hp_frac == 0.0
        and _valid(curr)
        and curr.party_hp_frac > 0.0
    )


class Heal(RewardComponent[Party]):
    """Pays the rise in party HP fraction. A whiteout heals for free, so it never pays; neither
    does a step where the party size changed, which moves the fraction without healing anyone."""

    name = "heal"

    @override
    def reset(self, state: Party) -> None:
        pass

    @override
    def score(self, prev: Party, curr: Party) -> float:
        if (
            not (_valid(prev) and _valid(curr))
            or curr.party_count != prev.party_count
            or blacked_out(prev, curr)
        ):
            return 0.0
        return max(0.0, curr.party_hp_frac - prev.party_hp_frac)


class Blackout(RewardComponent[Party]):
    """1.0 on the step the party is healed after fainting. Its weight is negative."""

    name = "blackout"

    @override
    def reset(self, state: Party) -> None:
        pass

    @override
    def score(self, prev: Party, curr: Party) -> float:
        return 1.0 if blacked_out(prev, curr) else 0.0
