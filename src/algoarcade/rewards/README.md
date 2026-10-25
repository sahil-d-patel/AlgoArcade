# rewards

Reward components and the registry that combines them. `base.py` holds the protocol: a name, a reset step and a scoring step over two game states.

`registry.py` builds components from a config of name to weight. `RewardRegistry.step` returns the weighted reward and each raw value as `reward/<name>`. Its components are `explore` and four progress terms (`events`, `level`, `badge`, `milestone`) that share one class, `NewMaximum` in `progress.py`. To add a component, see `.claude/skills/reward-component/SKILL.md`.

Contract: `docs/harness-spec.md`, Reward registry. Every component returns zero on a no-op step and on the reset step, and every pull request that adds one says how an agent would exploit it.
