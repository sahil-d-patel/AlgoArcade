# runqueue

The experiment queue for the training box (OPS-04, Sahil). An experiment is a Python file naming a base config, its arms and its seeds. The queue expands it into runs and trains them one after another, unattended.

- `queue.py`: `submit`, `run` and `status`, run as `uv run python -m algoarcade.runqueue <command>`.
- `train.py`: what `run` starts for each attempt, `train()` on the run's saved config.

The commands, the experiment file and the nightly task are in `docs/setup.md`, Training box.

Contract: `docs/technical-plan.md`, Run queue. Nobody launches runs on the box by hand.
