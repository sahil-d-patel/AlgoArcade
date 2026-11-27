# Contributing Guide

This is the playbook for running an **Aggie Data Science Club** project on GitHub. It is **not** a general Git tutorial — it assumes you can already commit and push. What it explains is *how we work together in this repo*: who does what, how work flows from an idea to `main`, and which Git commands you reach for at each step.

Read your role's section first. Everyone should read [The Flow](#the-flow-issue--main) and [Staying in Sync](#staying-in-sync) — they apply to all of us.

> **Two roles, one repo.** **Project Managers (PMs)** are *managers*: they turn goals into issues, review pull requests, and are the only people who approve merges to `main`. **Members** are *developers*: they take an issue, build it on a branch, open a pull request against `main`, and respond to review. If you are an AI agent working in this repo, also read [`AGENTS.md`](AGENTS.md) — it is the strict machine-facing version of this same workflow.

> **Companion files.** [`README.md`](README.md) is the overview, [`DELIVERABLES.md`](DELIVERABLES.md) the plan, [`DATA.md`](DATA.md) the data rules, and [`docs/`](docs/) the design, schedule and setup.

---

## Table of Contents

1. [The mental model](#the-mental-model)
2. [The flow: issue → `main`](#the-flow-issue--main)
3. [Staying in sync](#staying-in-sync)
4. [For Members (developers)](#for-members-developers)
5. [For Project Managers (managers)](#for-project-managers-managers)
6. [Branch naming](#branch-naming)
7. [Worktrees (optional)](#worktrees-optional)
8. [Git commands in context](#git-commands-in-context)
9. [Checks before you open a PR](#checks-before-you-open-a-pr)
10. [Golden rules](#golden-rules)

---

## The mental model

Every change starts as a **GitHub Issue** (the *why*) and ends as a **merge into `main`** (the *done*). In between, the work lives on one branch per issue, cut from `main`:

```
main                          ← protected. Changes arrive only through reviewed pull requests.
 ├── issue-42/mgba-backend    ← one branch per work item, cut from the latest main
 └── issue-57/reward-registry ← another work item, built in parallel
```

- **`main`** is the source of truth. It should always be in a known-good state. Nobody pushes to it directly.
- **A work item branch** is created per issue, from the latest `main`. It holds that issue's work and opens a pull request **straight into `main`**.
- **Big issue?** Split it into several branches, each with its own small pull request into `main`. Small pull requests get reviewed; sprawling ones get skimmed.

Work flows one way: issue → branch → pull request → `main`. When `main` receives the pull request, the whole team is back on the same page.

---

## The flow: issue → `main`

This is the loop the whole project runs on.

1. **Open an Issue.** Describe the problem, the goal, and a checklist of acceptance criteria. No issue, no work. (Ours are loaded from [`tools/github/issues/`](tools/github/issues/).)
2. **Cut a branch** from the latest `main`, named for the issue.
3. **Build.** Small, focused commits. Push early so others can see progress.
4. **Open a Pull Request** from your branch into **`main`**. Put `Closes #42` in the description.
5. **Review.** CI runs the checks. A PM (the code owners) reviews; teammates can review too. Address the comments.
6. **Merge.** Once CI is green and the review is approved, a PM merges to `main`.
7. **Clean up.** Delete the merged branch (local + remote). **History is preserved by the merge commit** — deleting the branch label does not delete the commits.
8. **The Issue closes** automatically when the pull request merges.

---

## Staying in sync

**This is the most common source of pain, so do it reflexively.** GitHub is the source of truth, not your laptop. Before you start any work — and regularly while you work — pull the latest. Members work from a fork, so the team repository is `upstream`:

```sh
git switch main
git fetch upstream
git merge --ff-only upstream/main   # fast-forward your local main to match the team repository
git push origin main                # keep your fork's main current too
```

PMs work in the team repository directly, so for them it is `origin`: `git switch main && git pull origin main`. The rest of this guide writes `upstream`; PMs read it as `origin` and skip the push to a fork.

When you cut a new branch, cut it from an **up-to-date** `main`. If your branch has fallen behind `main` while you were building, bring it forward before opening/updating your PR (see [rebase vs. merge](#git-commands-in-context)).

If you skip this, you will build on stale code and hit avoidable merge conflicts. **When in doubt, `git fetch` and look before you leap:**

```sh
git fetch upstream            # update your view of the team's branches (changes nothing local)
git status                    # are you ahead/behind? clean/dirty?
```

---

## For Members (developers)

You own the *building*. Your job is to turn one issue into a clean, reviewed PR.

Members do not have write access to the repository. You work from your own fork and open pull requests from it into `main`. Only the PMs push branches to the repository itself.

**0. Once: fork and clone.** Fork https://github.com/sahil-d-patel/AlgoArcade on GitHub, then:
```sh
git clone https://github.com/<your-username>/AlgoArcade.git
cd AlgoArcade
git remote add upstream https://github.com/sahil-d-patel/AlgoArcade.git
```
`origin` is your fork; `upstream` is the team repository. (`gh repo fork sahil-d-patel/AlgoArcade --clone` does all three.)

**1. Pick up an issue.** Comment on it to claim it, and a PM assigns you, so no two people duplicate work. Read the acceptance criteria — that's your definition of done.

**2. Sync, then branch off `main`.**
```sh
git switch main
git fetch upstream
git merge --ff-only upstream/main
git push origin main          # keep your fork's main current too
git switch -c issue-42/mgba-backend
```

**3. Build in small commits.** Each commit should be one coherent step, with a message that says *why*, not just *what*.

**4. Push to your fork and open a PR into the team repository's `main`:**
```sh
git push -u origin issue-42/mgba-backend
gh pr create --repo sahil-d-patel/AlgoArcade --base main --fill
```
On the GitHub website, "Compare & pull request" does the same: base repository `sahil-d-patel/AlgoArcade`, base `main`. Leave "Allow edits by maintainers" checked.
Write a description a reviewer can act on: what changed, what you deliberately left out, and how you verified it.

**5. Respond to review.** Read the whole review before touching anything. Fix what's clearly right yourself; only ask the PM when there's a real judgment call (naming, scope, product behavior). Push follow-up commits and re-request review.

**6. After the merge,** delete your branch, sync `main` as in step 2, and move to the next issue.

**You are also a reviewer.** When a teammate requests review, read their PR and leave comments. Reviewing is half the job of being a developer here — it's how we catch bugs before `main` and how everyone learns the codebase.

---

## For Project Managers (managers)

You own the *coordination and the gate to `main`*. You mostly work in GitHub's UI, not the terminal.

**1. Turn goals into Issues.** Break the project (see [`DELIVERABLES.md`](DELIVERABLES.md)) into issues small enough for one person to finish in a few days. Give each a clear title, acceptance-criteria checklist, and labels. Assign each one when a member claims it with a comment.

**2. Maintain the board.** Keep issues triaged and prioritized so members always know what to pick up next. Link related issues; close duplicates.

**3. Review pull requests.** This is your core review responsibility:
   - Every PR targets `main`. The PMs are the code owners, so one of you reviews every PR; a member's review is welcome but does not count toward the required approval.
   - A first-time contributor's PR from a fork may wait for you to click **Approve and run** before CI starts.
   - **The merge is yours.** Confirm CI is green, the review is approved, and the issue's acceptance criteria are met.

   Use GitHub's review tools deliberately: **Comment** to ask questions, **Request changes** to block until fixed, **Approve** to sign off. Review the *diff*, not the whole repo.

**4. Gatekeep `main`.** Only you merge to `main`, and only after the PR is green and reviewed. Never enable auto-merge on a PR targeting `main`. Prefer **branch protection** on `main` (require a PR + at least one approval) so the rule is enforced, not just trusted.

**5. Keep the team in sync.** After a merge to `main`, remind members to sync `main` from `upstream` before continuing — anyone behind is building on outdated code.

**6. Own the subjective calls.** Members escalate product/UX, naming, dependencies, and architecture decisions to you. Decide, and record the decision in the issue or PR so it's auditable.

---

## Branch naming

Name every branch for its issue so the list is obvious at a glance:

```
issue-<n>/<slug>        # <n> is the GitHub issue number, <slug> says what the branch builds
```

**One hard rule:** inside an `issue-<n>/` namespace, every branch name is a **flat leaf** — no branch name may be a *prefix* of another. Git stores refs as files, so a branch named `issue-1/guide` makes it **impossible** to also create `issue-1/guide/body` (you'll get `cannot lock ref ... 'issue-1/guide' exists`). Use `issue-1/guide` and `issue-1/guide-body` instead.

---

## Worktrees (optional)

A **worktree** lets you check out multiple branches into **separate folders at the same time**, all backed by the one repository. Reach for one when you need to work on two issues at once, or review a teammate's branch without stashing your own work.

```sh
# create a new branch AND a folder for it, based off main
git worktree add -b issue-42/mgba-backend ../AlgoArcade.worktrees/mgba-backend main

# work in that folder like a normal clone (it needs its own `uv sync`)
cd ../AlgoArcade.worktrees/mgba-backend

# see all active worktrees
git worktree list

# when the branch is merged and the folder is no longer needed
git worktree remove ../AlgoArcade.worktrees/mgba-backend
```

---

## Git commands in context

Reach for these at the moments described — not as isolated trivia.

- **`git fetch upstream`** — *"What has changed on GitHub?"* Updates your remote-tracking view without changing your files. Safe anytime; do it before deciding whether you're behind.
- **`git merge --ff-only upstream/main`** (on `main`) — *"Bring my local `main` up to date."* Run before cutting a branch. It refuses to run if your `main` has commits of its own, which it never should.
- **`git push -u origin <branch>`** — *"Publish my branch"* to your fork (for PMs, to the team repository). The `-u` sets the tracking branch so later `git push`/`git pull` need no arguments.
- **`git rebase <base>`** — *"Replay my commits on top of the latest base."* Use it to bring your branch forward onto an updated `main` so your PR is a clean, linear diff. **Only rebase branches that are yours and not yet merged.** Rebasing shared history rewrites commits others may have — don't.
  ```sh
  git switch issue-42/mgba-backend
  git fetch upstream
  git rebase upstream/main
  ```
- **`git merge`** — *"Combine another branch into this one, preserving both histories."* This is how a reviewed PR lands on `main`. We keep merge commits so the small-commit trail stays auditable.
- **`git reset`** — *"Move my branch pointer / unstage things."* Use `git reset --soft HEAD~1` to undo the last *local* commit but keep the changes, or `git restore --staged <file>` to unstage. **Never `git reset --hard` on work you haven't backed up, and never rewrite commits you've already pushed to a shared branch.**
- **`--force` / `--force-with-lease`** — avoid. Only ever force-push a branch that is exclusively yours, and prefer `--force-with-lease` so you don't clobber someone else's push. If you feel you need to force-push a shared branch, stop and ask a PM.

> If a rebase or merge produces conflicts: don't panic. Open the conflicted files, resolve the `<<<<<<<`/`=======`/`>>>>>>>` markers, `git add` them, and continue (`git rebase --continue` or commit the merge). If it's a mess, `git rebase --abort` / `git merge --abort` puts you back where you started.

---

## Checks before you open a PR

Run the project check and make it pass:

```sh
uv run poe check
```

It runs the formatter, the linter, the type checker and the tests. CI runs the same thing on Linux, Windows and macOS, and the pre-commit hook (`uv run prek install`, once) runs most of it on every commit. Setup is in [`docs/setup.md`](docs/setup.md).

Before you open the PR, cut the diff down to the change itself: no comments that restate the code, no debug leftovers, no unrelated edits. The `pr-cleanup` skill does this pass.

Then fill in the pull request template. Paste the real output under "How it was verified". If you did not run something, say so. Two questions in the template are specific to this project:

- **Touches a reward component?** Say how an agent would exploit it.
- **Touches the logging schema after November 8?** Say which runs need re-running. A PM has to agree first.

---

## Golden rules

- **No work without an issue.** The issue is the *why*.
- **Never push to `main`.** It moves only through a reviewed PR merged by a PM.
- **Sync before you branch.** Cut from an up-to-date `main`; pull often.
- **Small slices, small PRs.** If a PR is sprawling, split it.
- **One issue, one branch, one PR into `main`.** Merged with a merge commit.
- **Review each other's work.** Everyone is a reviewer.
- **Don't rewrite shared history.** No force-push / hard-reset on branches others use.
- **Delete merged branches, keep the commits.** Clean tree, full history.
