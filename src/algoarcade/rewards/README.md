# rewards

Reward components and the registry that combines them. `base.py` holds the protocol: a name, a reset step and a scoring step over two game states.

`registry.py` builds components from a game's component table and a config of name to weight; FireRed's table is `COMPONENTS`. `RewardRegistry.step` returns the weighted reward and each raw value as `reward/<name>`. `RewardRegistry.reset` takes an archive start's visited set, which `explore` treats as already paid. FireRed has ten: `explore`, four progress terms (`events`, `level`, `badge`, `milestone`) that share one class, `NewMaximum` in `progress.py`, `heal` and `blackout` in `party.py`, and the penalties `anti_loop`, `anti_spam` and `time` in `penalties.py`. `WEIGHTS` holds their starting weights and `DEFAULT_WEIGHTS` the five that are on by default. To add a component, see `.claude/skills/reward-component/SKILL.md`.

Contract: `docs/harness-spec.md`, Reward registry. Every component returns zero on the reset step and on a no-op step, with two exceptions: a progress term pays on the step after a rise, when the second read confirms it, and the penalties `time` and `anti_spam`, which never pay and are off by default, can charge on either. Every pull request that adds a component says how an agent would exploit it.
