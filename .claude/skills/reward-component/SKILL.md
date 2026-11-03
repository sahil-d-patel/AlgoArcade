---
name: reward-component
description: The contract for reward terms in this repo and the tests each one must ship with. Use when adding, changing or tuning a reward component or its weight, working under src/algoarcade/rewards, or investigating an agent that is farming reward without making progress.
paths: src/algoarcade/rewards/**
---

# Adding or changing a reward component

An agent will find every way a reward can be earned without doing what you meant. Assume that, and build the guard with the component.

A component implements the `RewardComponent` protocol in `src/algoarcade/rewards/base.py`, which is generic over the state it reads: a `name`, a `reset(state)` called with the first state of an episode, and a `score(prev, curr)` that returns a float. It keeps its own per-episode memory (a visited set, a best-so-far) and clears it in `reset`. Its weight lives in config, never in the component.

To add one:

1. Write the component. A term that pays when an integer count reaches a new maximum needs no new class: `NewMaximum("name", lambda s: s.some_count, maximum=..., max_rise=...)`, from `rewards/progress.py`. Set `maximum` to the largest count the game allows and `max_rise` to the most it can rise in one step. Cite the source for both beside the table entry. A read past either bound pays nothing and does not become the best, and the best is set only by two consecutive reads that agree, never by the reset read alone. Test both: a misread followed by real gains (`test_a_misread_badge_byte_neither_pays_nor_silences_later_badges`), and a bad reset read followed by real gains (`test_a_bad_reset_read_neither_becomes_the_best_nor_freezes_the_term`). If it needs a field `Progress` lacks, add the field to `Progress`, to the state class of each game whose table includes the component, and to `FakeState` so the tests can drive it. A game whose table leaves the component out does not need the field.
2. Add it to the game's component table. FireRed's is `COMPONENTS` in `rewards/registry.py`. The registry builds only names in the table it is given and raises with the valid names for any other.
3. Add the tests below to `tests/test_registry.py`.

`RewardRegistry(COMPONENTS, weights).step(prev, curr)` returns the weighted sum and an info dict with each configured component's raw value as `reward/<name>`.

Every component ships with four things.

An isolation test: `score` returns exactly 0.0 on the reset step and on a step where nothing changed, such as a press into a wall. The parametrised `test_zero_on_reset_and_on_a_no_op_step` runs this for every name in `COMPONENTS`. A reward that fires on reset teaches the agent to die quickly.

An exploit test: a scripted policy that tries the obvious farm (walk between two tiles, mash a button in a menu, faint on purpose, level in the first grass forever) and must earn less than a bound you state in the test. The fake game cannot earn most counts, so these tests set the count with `poke` and play the farm through it.

A log column. The schema has one `reward_<name>` column per component (`docs/harness-spec.md`, Logging schema). After November 1 a new component adds a column, which is a PM decision.

An exploit note in the pull request: one or two sentences on how an agent would game this component and what stops it.

The registry tests also check, over random rollouts, that the reward equals the sum of weight times raw value and that no component goes negative. Add the new name to `WEIGHTS` in that file so the rollouts include it.

Prefer terms that pay for a new maximum (a new flag, a first visit this episode, a higher level sum) over terms that pay for a change. Use coordinates for novelty, never screen hashes or a sliding window. Validate any RAM field before paying on it. Penalties and the heal reward exist but are off by default; turning one on is experiment E5, not a tuning step.

When you tune a weight, change one weight, state the ratio you are aiming for (badge above everything, events above level), and watch the per-component chart and five minutes of video before trusting the return curve.

The contract is in `docs/harness-spec.md` under "Reward registry", and the reasoning in `docs/technical-plan.md` under "Reward design".
