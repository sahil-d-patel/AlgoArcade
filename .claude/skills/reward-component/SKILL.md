---
name: reward-component
description: The contract for reward terms in this repo and the tests each one must ship with. Use when adding, changing or tuning a reward component or its weight, working under src/algoarcade/rewards, or investigating an agent that is farming reward without making progress.
paths: src/algoarcade/rewards/**
---

# Adding or changing a reward component

An agent will find every way a reward can be earned without doing what you meant. Assume that, and build the guard with the component.

A component implements the `RewardComponent` protocol in `src/algoarcade/rewards/base.py`, which is generic over the state it reads: a `name`, a `reset(state)` called with the first state of an episode, and a `score(prev, curr)` that returns a float. It keeps its own per-episode memory (a visited set, a best-so-far) and clears it in `reset`. Its weight lives in config, never in the component.

FireRed's ten components, all in `COMPONENTS` in `rewards/registry.py`:

| Component | Class | Weight | Default |
|---|---|---|---|
| events, level, badge, milestone | `NewMaximum` (`progress.py`) | +1.0, +0.5, +10.0, +2.0 | on |
| explore | `Explore` (`explore.py`) | +0.02 | on |
| heal | `Heal` (`party.py`) | +0.3 | off |
| blackout | `Blackout` (`party.py`) | -5.0 | off |
| anti_loop, anti_spam, time | `AntiLoop`, `AntiSpam`, `Time` (`penalties.py`) | -0.01, -0.02, -0.001 | off |

`WEIGHTS` beside the table holds these starting weights; `DEFAULT_WEIGHTS` is the five that are on. The predicted exploit for each is in `docs/harness-spec.md`, Reward registry. A raw value is never negative: a penalty returns 1.0 when it fires and its weight carries the sign.

To add one:

1. Write the component. A term that pays when an integer count reaches a new maximum needs no new class: `NewMaximum("name", lambda s: s.some_count, maximum=..., max_rise=...)`, from `rewards/progress.py`. Set `maximum` to the largest count the game allows and `max_rise` to the most it can rise in one step. Cite the source for both beside the table entry. A read past either bound pays nothing and does not become the best, and the best is set only by two consecutive reads that agree, never by the reset read alone. Only the first step pays when it sets the best; setting it again after a rejected read pays nothing. A bad read that repeats on two consecutive steps can become the best and silence the term for the episode. Only reader validation prevents that, and `test_a_misread_repeated_on_two_steps_becomes_the_best` pins it down. Test both: a misread followed by real gains (`test_a_misread_badge_byte_neither_pays_nor_silences_later_badges`), and a bad reset read followed by real gains (`test_a_bad_reset_read_neither_becomes_the_best_nor_freezes_the_term`). If it needs a field no protocol has (`Progress`, `Party` in `party.py`, `Dialogue` in `penalties.py`), add it to a protocol, to `FireRedFields` in `registry.py` if it is a new protocol, to the state class of each game whose table includes the component, and to `FakeState` so the tests can drive it. A game whose table leaves the component out does not need the field.
2. Add it to the game's component table and its weight to `WEIGHTS`. FireRed's table is `COMPONENTS` in `rewards/registry.py`. The registry builds only names in the table it is given and raises with the valid names for any other.
3. Add the tests below: isolation and unit tests in `tests/test_registry.py` or a file named for the component's module, exploit tests in `tests/test_exploits.py`.

`RewardRegistry(COMPONENTS, weights).step(prev, curr)` returns the weighted sum and an info dict with each configured component's raw value as `reward/<name>`.

Every component ships with four things.

An isolation test: `score` returns exactly 0.0 on the reset step and on a step where nothing changed, such as a press into a wall. The parametrised `test_zero_on_reset_and_on_a_no_op_step` runs this for every name in `COMPONENTS` except `time`. A reward that fires on reset teaches the agent to die quickly. The only exceptions are the penalties `time` and `anti_spam`, which never pay and are off by default: `time` charges every step, and `anti_spam` charges an unchanged step with a text box open, including a reset into one. `test_time_and_anti_spam_charge_the_reset_step` pins both. A new exception has to be a penalty, and the spec has to name it.

An exploit test. `tests/test_exploits.py` has one scripted policy per known exploit: two-tile oscillation, menu spam, faint loop, instant reset and grass camping. Each feeds the registry a sequence of states and asserts the total reward stays under a bound written beside the assert. The rule for a new component: if one of the five already tries its obvious farm, add the component to that test's config and check the bound still holds; otherwise add a `test_exploit_<policy>` there with its own bound. Then remove the guard by hand, watch the test fail, and restore it; a bound that holds without the guard proves nothing. `uv run pytest -k exploit` must select only these policies, so no other test name contains "exploit".

A log column. The frozen schema has one `reward_<name>` column for each of the ten (`docs/harness-spec.md`, Logging schema). An eleventh component adds a column after the November 1 freeze, which is a PM decision.

An exploit note in the pull request: one or two sentences on how an agent would game this component and what stops it.

The registry tests also check, over random rollouts, that the reward equals the sum of weight times raw value and that no component goes negative. The rollouts use `WEIGHTS`, so a component in it is included; if it reads a new fake byte, poke that byte in `rollout` too.

Prefer terms that pay for a new maximum (a new flag, a first visit this episode, a higher level sum) over terms that pay for a change. Use coordinates for novelty, never screen hashes or a sliding window. Validate any RAM field before paying on it. Penalties and the heal reward exist but are off by default; turning one on is experiment E5, not a tuning step.

When you tune a weight, change one weight, state the ratio you are aiming for (badge above everything, events above level), and watch the per-component chart and five minutes of video before trusting the return curve.

The contract is in `docs/harness-spec.md` under "Reward registry", and the reasoning in `docs/technical-plan.md` under "Reward design".
