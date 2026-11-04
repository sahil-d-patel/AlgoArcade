---
id: VIZ-05
title: E3 and E5 analysed, reward decomposition and the hack catalogue
team: training
owner: Pri (Reward Engineer)
due: 2026-11-22
estimate: 6h
depends_on: TRN-10, OPS-05
labels: viz, experiment
---
## Context
E3 asks whether RND beats tile counts, and E5 asks whether penalties help or hurt on FireRed. A return curve hides which reward term the agent is chasing, so this item also charts what each component contributed and writes up every exploit we saw. VIZ-01 assembles the pages and DOC-05 writes them up.

## Deliverable
- Tables for E3 (none, count, RND; 5 seeds) and E5 (3 seeds, exploratory) from the OPS-05 statistics module.
- `src/algoarcade/viz/decomposition.py`: a stacked chart of mean reward per episode by component, per checkpoint window, for the baseline and each E5 arm.
- docs/hack-catalogue.md: each exploit with what the agent did, the component that paid for it, the fix, the run id and a clip.
- A dashboard page built from the catalogue, with clips from the OPS-05 renderer under `data/videos/hacks/`.

## Acceptance criteria
- [ ] The pull request has the E3 table (IQM, 95% interval, seed count, probability of improvement for count over none and RND over count) and the E5 table labelled exploratory.
- [ ] For every window of one real baseline run, the components sum to the logged total reward within 1e-6. The largest difference is pasted.
- [ ] `uv run pytest tests/test_decomposition.py` passes in CI on fake-logger data.
- [ ] The catalogue covers the six scripted exploits (oscillation, menu spam, faint loop, reset farming, grass camping, warp flicker), each marked observed (run id, clip) or not observed (test name).
- [ ] Every clip plays in the rendered page with Wi-Fi off, and `git ls-files '*.mp4'` prints nothing.
- [ ] The `reward-component` skill lists every exploit in the catalogue.

## Notes
Contract: docs/harness-spec.md (Reward registry, Logging schema) and docs/technical-plan.md (Reward design).

Plot raw values: a percentage-stacked chart breaks with mixed signs. If the sum check fails, the bug is upstream and gets its own issue. RND reward that never decays is usually an input bug.

Do not stage an exploit to fill the catalogue.

Fallback by Sunday: the W&B per-component charts, exported as images.

Out of scope: new components, new runs.
