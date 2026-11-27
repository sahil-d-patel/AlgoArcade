---
id: OPS-05
title: Evaluation tooling: video renderer and statistics module
team: pm
owner: Akshaj (Co-PM)
due: 2026-11-08
estimate: 7h
depends_on: TRN-05
---
## Context
Two tools every later result depends on. Footage: a reward exploit is easier to spot in a video than in a curve, and the weekly video review is how we catch one before it costs a multi-day run. Statistics: every comparison we publish reports the same numbers, computed the same way, and rliable is archived, so we write them over scipy. TRN-08 needs clips as evidence this week, VIZ-02, VIZ-04, VIZ-05, VIZ-06 and ENV-11 build on these.

## Deliverable
- One script: `uv run python -m algoarcade.viz.render --checkpoint <path> --seed 0 --out data/videos/<name>.mp4`.
- It loads a checkpoint and plays one seeded episode from the fixed start state with rendering on.
- It writes mp4 with imageio-ffmpeg (libx264, yuv420p, explicit fps). Frames get a nearest-neighbour integer upscale from 240x160 (GBA) or 160x144 (Game Boy).
- Overlays: one bar per reward component, the step count, and map_group, map_num, x, y and badges.
- `--grid` composes several episodes side by side in numpy.
- docs/video-review.md with the first weekly entry.
- `src/algoarcade/training/stats.py`, citing Agarwal et al. 2021 (arXiv 2108.13264): IQM as `scipy.stats.trim_mean(scores, 0.25)`; a percentile bootstrap interval over seeds, 95% by default, stratified by task when tasks are pooled; probability of improvement as Mann-Whitney U over n times m; a performance profile. Every estimate prints its seed count, for example `IQM 0.62 [0.48, 0.71] (n=5)`, and warns below four seeds.

## Acceptance criteria
- [ ] The command on a TRN-05 FireRed checkpoint writes an mp4. Read back with imageio, it has the episode's frame count and a width and height divisible by 16.
- [ ] The same command runs on a TRN-05 Super Mario Land checkpoint with no code change.
- [ ] Two runs with the same checkpoint and seed print the same episode return and final (map_group, map_num, x, y).
- [ ] A test on the fake backend renders 10 frames and asserts one bar per enabled reward component.
- [ ] `--grid` with four seeds writes one mp4 with four panels per frame. One frame is pasted in the pull request.
- [ ] docs/video-review.md has an entry for the week of Nov 8: run ids, at least five minutes watched, and what we saw. Each of the six exploit signatures is marked seen or not seen.
- [ ] `uv run pytest tests/test_stats.py` passes in CI, including: IQM of `[0, 1, 2, 3, 4, 5, 6, 100]` is 3.5; probability of improvement of `[1, 2]` over `[1, 0]` is 3.5/4 (a tie counts half); the interval of `[0.4] * 5` is exactly `(0.4, 0.4)`; the profile of `[0.1, 0.4, 0.6, 0.9]` at `[0.0, 0.5, 1.0]` is `[1.0, 0.5, 0.0]`.
- [ ] A 3-seed IQM call warns that it equals the mean, and every printed estimate contains `n=`.
- [ ] The `experiment` skill states the reporting rule: IQM, 95% interval, n, and five seeds for a headline.

## Notes
Contract: docs/harness-spec.md, Reward registry and Logging schema. The six exploit signatures: oscillation, menu spam, faint loop, reset farming, grass camping, warp flicker.

Gotchas: imageio rescales any frame whose width or height is not divisible by 16, and defaults to 10 fps. PyBoy updates its frame buffer in place, so copy every frame. A checkpoint needs the normalisation statistics TRN-05 saves beside it. Text overlays need Pillow. If it is not in `uv.lock`, ask a PM before adding it.

`trim_mean` cuts `int(0.25 * n)` values per side, which is zero below four seeds. `mannwhitneyu(x, y).statistic` is U for the first argument; the hand case catches a swapped order.

The module paths are proposals. Clips stay in `data/videos/`, which is git-ignored. Out of scope: batch rendering (VIZ-02), dashboard embedding.
