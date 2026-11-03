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


class Heal(RewardComponent[Party]):
    """Pays the rise in party HP fraction from one trusted read to the next. A read is trusted
    once the next read agrees with it, so one bad read, high or low, neither pays nor moves the
    baseline; ENV-08 cannot vouch for HP, which sits outside the checksummed block. A rise from a
    fainted party is a whiteout heal (see Blackout) and pays nothing. A change in party size
    moves the fraction without healing anyone, so it resets the baseline without paying."""

    name = "heal"

    def __init__(self) -> None:
        self._trusted: tuple[int, float] | None = None

    @override
    def reset(self, state: Party) -> None:
        self._trusted = None

    @override
    def score(self, prev: Party, curr: Party) -> float:
        read = (curr.party_count, curr.party_hp_frac)
        if read != (prev.party_count, prev.party_hp_frac) or not _valid(curr):
            return 0.0
        trusted, self._trusted = self._trusted, read
        if trusted is None or trusted[0] != read[0] or trusted[1] == 0.0:
            return 0.0
        return max(0.0, read[1] - trusted[1])


class Blackout(RewardComponent[Party]):
    """1.0 on the step the party goes from all fainted to healed. Its weight is negative.

    In FireRed that is a whiteout: DoWhiteOut in src/overworld.c (pret/pokefirered at ce6406d8)
    heals the party and warps in one call. Losing the Oak's Lab battle also heals a fainted party,
    without a warp (data/maps/PalletTown_ProfessorOaksLab/scripts.inc), and counts too. The
    party_count checks keep an empty party, which reads 0.0, from looking fainted. One bad read of
    0.0 charges a false blackout; it never pays."""

    name = "blackout"

    @override
    def reset(self, state: Party) -> None:
        pass

    @override
    def score(self, prev: Party, curr: Party) -> float:
        fired = (
            prev.party_count > 0
            and curr.party_count == prev.party_count
            and prev.party_hp_frac == 0.0
            and _valid(curr)
            and curr.party_hp_frac > 0.0
        )
        return 1.0 if fired else 0.0
