# Agent Playbook (AGENTS.md)

**Machine-facing companion to [`CONTRIBUTING.md`](CONTRIBUTING.md).** Humans read CONTRIBUTING; AI coding agents read this. It is the strict, imperative version of how to help a member or PM work on this repo.

**Do only the work asked in the current task.** Follow the rules below while doing it. Don't expand scope on your own.

---

## Table of Contents

1. [What this repo is](#1-what-this-repo-is)
2. [How to work](#2-how-to-work)
3. [Plan before you touch code](#3-plan-before-you-touch-code)
4. [Work on the right branch](#4-work-on-the-right-branch)
5. [Editing files in this repo](#5-editing-files-in-this-repo)
6. [Working with data](#6-working-with-data)
7. [When to decide vs. ask](#7-when-to-decide-vs-ask)
8. [Hard prohibitions](#8-hard-prohibitions)
9. [Quick reference](#9-quick-reference)
10. [AlgoArcade: commands and invariants](#10-algoarcade-commands-and-invariants)
11. [Skills and agents](#11-skills-and-agents)

---

## 1. What this repo is

This is **AlgoArcade**, an **Aggie Data Science Club** project: a framework for training and comparing game-playing agents, proven on Pokémon FireRed. A team of PMs and members runs it out of this repo. The files an agent works with:

| File / folder | What the agent uses it for |
|---|---|
| [`README.md`](README.md) | One-page overview of the project. Read once for orientation. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | The human workflow. Honor it — don't re-explain it back to the user. |
| [`DELIVERABLES.md`](DELIVERABLES.md) | The living plan of what the project will ship. Update when a deliverable is added, split, or resequenced. |
| [`DATA.md`](DATA.md) | Documentation of data sources. Update whenever a new source is adopted. |
| [`docs/`](docs/) | [`technical-plan.md`](docs/technical-plan.md) (what and why), [`harness-spec.md`](docs/harness-spec.md) (the interface contract and logging schema), [`development-plan.md`](docs/development-plan.md) (who and when), [`decisions.md`](docs/decisions.md), [`setup.md`](docs/setup.md). Read the spec section for the area you are changing. |
| [`src/algoarcade/`](src/algoarcade/) | The package. Each subfolder's README names its owner and contract. |
| [`tests/`](tests/) | Unit tests and the backend conformance suite. |
| [`tools/`](tools/) | The file guard and the work-item loader. |
| [`data/`](data/) | Local datasets. **Git-ignored.** Never commit anything under it. |

If any of these files are missing or clearly out of date for what the user is doing, say so before proceeding.

---

## 2. How to work

- **Small, focused changes.** Do the smallest thing that satisfies the task. Don't add scope, abstraction, features, or ceremony that wasn't asked for.
- **Broad reads, narrow writes.** Read whatever you need to understand the repo. Only change what the task requires.
- **Match the repo's conventions.** Mirror existing style, naming, and structure. If a convention isn't established, propose one and let the user confirm.
- **Keep the user oriented.** Say what you're about to do, do it, then say what changed — briefly. Don't narrate the workflow at the user.
- **Follow instructions verbatim.** If the user gives an explicit direction that conflicts with a default here, follow the user.

---

## 3. Plan before you touch code

For any non-trivial task (more than a one-line edit):

1. Write a short working plan at `.planning/<task-name>.md`. This directory MUST be listed in `.gitignore` — these are private notes, not repo artifacts.
2. The plan MUST cover:
   - **Goal** — one paragraph.
   - **Scope in / Scope out** — bulleted.
   - **Assumptions** — anything you inferred.
   - **Files you expect to change**.
   - **Items to escalate** — anything opinion-shaped per [§7](#7-when-to-decide-vs-ask). Ask the user before proceeding on these.
   - **Acceptance criteria** — how you (and the user) will know it's done.
3. **Re-read the plan** in a fresh step before editing. That read pass catches contradictions the write pass missed.
4. If the plan changes mid-flight, update the plan file first, then continue.

Skip the plan file only for trivial edits (fix a typo, tweak one line). When in doubt, write it.

---

## 4. Work on the right branch

Features live on their own branches. The agent's rules for staying on the right one:

- **Never work directly on `main`.** If the user asks you to edit code on `main`, stop and ask which branch to use.
- **One task, one branch.** Work on the branch dedicated to the task's issue: `issue-<n>/<slug>`, cut from an up-to-date `main`, with a pull request opened against `main`. If the branch doesn't exist yet, create it and confirm the name with the user.
- **Match branch conventions.** Look at existing branches (`git branch -a`) and mirror the naming shape. If nothing's established, propose a scheme and get user approval before creating branches.
- **Small commits, meaningful messages.** Each commit should be one coherent step, with a message that says *why* — not just *what*.
- **Push your work** so the user can see it. Members push to their fork (`origin`) and open the PR against `sahil-d-patel/AlgoArcade` `main`; sync `main` from `upstream`. Don't sit on uncommitted or unpushed changes for long stretches.
- **Stay in your lane.** Don't pull in unrelated changes from another feature branch just because they're convenient. Keep the feature's history clean.
- **How work moves between branches is a human decision.** Don't merge branches, rebase across them, or delete them unless the user explicitly asks.

---

## 5. Editing files in this repo

- **Documentation files** (`README.md`, `CONTRIBUTING.md`, `DELIVERABLES.md`, `DATA.md`, `AGENTS.md`) — edit surgically. Preserve tone, structure, and cross-links. If you rename a section, update every reference to it.
- **`DELIVERABLES.md`** — update the table and timeline whenever a deliverable is added, dropped, split, or rescheduled. Note *why* in the row's description, not just *what*.
- **`DATA.md`** — add a new row to the source register for every dataset the team adopts. Fill in every column (Source, Origin/URL, Access method, License, Sensitivity, Notes). If any column is unknown, put `TBD` and flag it to the user.
- **Code files** — follow the repo's existing linters/formatters. Don't reformat unrelated code. Don't add dependencies, refactor, or introduce new patterns unless asked.
- **`.gitignore`** — confirm `data/` and `.planning/` remain ignored. Never remove these entries without explicit user approval.

---

## 6. Working with data

The `data/` folder is git-ignored and holds datasets the team pulls down locally. Rules for the agent:

- **Never commit anything under `data/`.** Not raw files, not cleaned files, not sample rows, not `.parquet`/`.csv` fragments. If the user asks you to commit data, refuse and explain why — this is a hard rule from [`DATA.md`](DATA.md).
- **Commit the pipeline, not the data.** The scripts/notebooks that fetch and process data must be reproducible and tracked in git. The outputs go under `data/` locally and stay there.
- **Document sources in [`DATA.md`](DATA.md)**, not in code comments. If you help the user adopt a new source, update the source register in the same change.
- **Sensitivity checks.** Before writing any pipeline that fetches data, fill in its License and Sensitivity columns in the [`DATA.md`](DATA.md) source register. If anything is unclear (license, PII, permission), stop and ask a PM.
- **Local layout.** Keep the layout in [`DATA.md`](DATA.md) (`roms/`, `states/`, `pokefirered/`, `runs/`, `videos/`, `llm/`). Don't invent new top-level folders inside `data/` without approval.

---

## 7. When to decide vs. ask

Default to **deciding**. Interrupt the user only when it's genuinely warranted. Before asking, run this test:

- **Is there one obviously-right answer?** → **Decide and do it.** Correctness fixes, typos, mechanical refactors, tests for existing behavior, following an explicit instruction, or picking the plainly-simpler of two options — those are yours to make. Say briefly what you decided, so it stays auditable.
- **Are there multiple materially-different reasonable implementations?** → **Ask** one focused "this way or that way?" question with a recommendation. Trivial or cosmetic differences don't count — pick the clean one and move on.
- **Is it one of the always-subjective categories below?** → **Ask.**
- **Would it add scope, abstraction, or infrastructure the task doesn't need?** → **Don't do it** — or do the minimal version and note the choice.

### Always escalate

- **Analysis / product direction** — what a deliverable does, what a model predicts, what a chart shows, what wording appears in a report.
- **Data-source decisions** — adopting a new source, changing an existing one, or anything that touches license/permission/sensitivity per [`DATA.md`](DATA.md).
- **Naming** — files, columns, symbols, endpoints, config keys, dataset names.
- **Dependency additions or version bumps** — new packages, major upgrades, library swaps.
- **Architectural tradeoffs** — pipeline structure, storage format, sync vs. async, notebook vs. script vs. module.
- **Anything else your judgment flags as opinion-shaped.** When in doubt, ask.

Batch questions when you have several. Never fan a single decision out into many small ones.

---

## 8. Hard prohibitions

The agent MUST NOT:

- Push directly to `main`, or make any change to `main` without explicit user approval.
- Force-push (`--force`, `--force-with-lease`) any branch without explicit user instruction for that specific push.
- Bypass hooks (`--no-verify`).
- Amend or rewrite commits that have already been pushed.
- Delete any branch or worktree without user approval.
- Commit anything under `data/`, or any secrets, tokens, credentials, or `.env` files. If one is discovered committed, stop and alert the user.
- Adopt a new data source, add a dependency, or make an architectural change without confirming per [§7](#7-when-to-decide-vs-ask).
- Remove `data/` or `.planning/` from `.gitignore`.
- Re-explain the human workflow ([`CONTRIBUTING.md`](CONTRIBUTING.md)) back to the user unprompted.

---

## 9. Quick reference

Before starting a task:
- [ ] Read the relevant repo files ([`README.md`](README.md), [`DELIVERABLES.md`](DELIVERABLES.md), [`DATA.md`](DATA.md)) enough to understand context.
- [ ] Wrote `.planning/<task-name>.md` with goal, scope, files, escalations, and acceptance criteria.
- [ ] Re-read the plan.
- [ ] Escalated any subjective items to the user and got answers.
- [ ] Confirmed which branch the work belongs on.

While working:
- [ ] On a feature branch, not `main`.
- [ ] Small, coherent commits with meaningful messages.
- [ ] No files committed under `data/`.
- [ ] No new dependencies, sources, or architectural changes made silently.
- [ ] Related docs updated ([`DELIVERABLES.md`](DELIVERABLES.md), [`DATA.md`](DATA.md)) in the same change when relevant.

When finishing:
- [ ] Acceptance criteria met.
- [ ] Pushed the branch.
- [ ] Summarized what changed, what was left out, and any items still awaiting a user decision.

---

## 10. AlgoArcade: commands and invariants

**Commands.** Run everything through `uv run`; never `pip install` into the environment.

- `uv sync` sets up. `uv run poe check` is the gate: format, lint, types, tests. Run it before you say anything works, and paste its real output.
- `uv run pytest tests/conformance -q` runs one folder; `uv run pytest -k <word>` runs matching tests; `uv run poe fmt` formats.

**Invariants.** Each of these fails silently if broken.

- Memory addresses come from the pokefirered symbol table, with the symbol name and commit recorded. Never from a wiki or from memory.
- Save block pointers are dereferenced on every read. Never cache the address they point to.
- A location is always `(map_group, map_num, x, y)`.
- Every reward component is named, logged on its own, returns zero on the reset step and on a no-op step (except the penalties time and anti_spam, which never pay and are off by default, and a progress term, which pays on the step after a rise when the second read confirms it), and ships with an isolation test and an exploit note.
- Gamma is an explicit config field and is never left at a library default. The baseline is 0.997.
- A step-budget end is truncation, not termination.
- A headline claim needs five seeds. Fewer is labeled exploratory.
- The logging schema is frozen after November 8. Changing a logged field after that is a PM decision.
- Nobody debugs on the training box. Debug locally with one environment and 1,000 steps.
- ROMs, save states, symbol files, map binaries and anything under `data/` never enter git.

**Write the smallest code that does the job.**

- Build what the task asks for. No extra features, options, or refactors of code you were not asked to touch.
- No comment that restates the code and no docstring that restates the signature. Comment only what the code cannot say.
- No error handling for cases that cannot happen. Validate at boundaries (files, ROMs, user input, API responses) and trust it inside.
- No helper for a single use and no abstraction with one implementation. The four protocols (GameBackend, GameState, StateReader, RewardComponent) are the exceptions.
- Delete debug prints, commented-out code and scratch files before you finish.

**Write plainly.** Docs, issues, commit messages and PR text follow `.claude/skills/plain-writing/SKILL.md`: lead with the fact, say it once, specifics over adjectives, claim only what you ran.

**Clean up before every PR.** Run `.claude/skills/pr-cleanup/SKILL.md` on the branch: delete comments that restate the code, code that only looks careful, debug leftovers and unrelated changes, and shorten the prose in the files you changed.

---

## 11. Skills and agents

Skills live at `.claude/skills/<name>/SKILL.md`. Claude Code loads them on its own. In any other tool, open the file when the situation matches: `plain-writing` (any prose a person will read), `pr-cleanup` (clean a branch before every PR), `deslop` (strip bloat from code), `ram-field` (a value read from game memory), `reward-component` (a reward term), `experiment` (define, launch or report a run), `start-issue` and `open-pr` (the issue workflow).

Agents live at `.claude/agents/<name>.md`: `ticket-implementer` (a ticket from plan to a committed branch), `pr-reviewer` (a full pull request review) and `rl-reviewer` (the RL and emulator checklist only). In any other tool, follow the body of the file as instructions. `.codex/agents/rl-reviewer.toml` is the Codex copy of `rl-reviewer`; change both together.
