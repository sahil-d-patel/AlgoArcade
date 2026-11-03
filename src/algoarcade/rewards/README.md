# rewards

Reward components and the registry that combines them. `base.py` holds the protocol: a name, a reset step and a scoring step over two game states.

`registry.py` builds components from a game's component table and a config of name to weight; FireRed's table is `COMPONENTS`. `RewardRegistry.step` returns the weighted reward and each raw value as `reward/<name>`. FireRed has ten: `explore`, four progress terms (`events`, `level`, `badge`, `milestone`) that share one class, `NewMaximum` in `progress.py`, `heal` and `blackout` in `party.py`, and the penalties `anti_loop`, `anti_spam` and `time` in `penalties.py`. `WEIGHTS` holds their starting weights and `DEFAULT_WEIGHTS` the five that are on by default. To add a component, see `.claude/skills/reward-component/SKILL.md`.

Contract: `docs/harness-spec.md`, Reward registry. Every component returns zero on a no-op step and on the reset step except the penalties `time` and `anti_spam`, which never pay and are off by default, and every pull request that adds one says how an agent would exploit it.
