# Contributing

This repository backs a research programme (see `docs/roadmap.md`).
The workflow described here applies both to the author and to anyone
opening a pull request in the future.

## Branch policy

- `main` is **append-only** and corresponds to work already public:
  a submitted arXiv preprint, an accepted paper, or a released
  software version. Nothing that has not yet been made public ships
  to `main`.
- **Every non-trivial change starts on a topic branch**, never on
  `main` directly. Naming conventions:
  - `wip/<package-stub>` — work in progress tied to a forthcoming
    paper listed in `docs/roadmap.md`. Kept local until the paper
    is public. Must not be pushed to `origin`.
  - `feat/<slug>` — contributor-driven improvements (bug fixes,
    refactors, tests) that do not depend on unpublished research.
    These may be pushed and opened as pull requests.
- Direct commits to `main` are reserved for minor, non-research
  hygiene (typo fixes, CI tweaks, metadata updates).

## Squash-merge policy

Merges into `main` use **squash-merge**, not rebase-merge or
regular merge commits:

```bash
git switch main
git merge --squash <topic-branch>
git commit -m "Add <subject>: <one-line context>"
git push
git branch -D <topic-branch>
```

The rationale:

- `main`'s history stays **one commit per logical contribution**
  (one paper, one feature, one round of paper revisions).
- Intermediate checkpoints, dead ends, and exploratory commits on
  the topic branch do not leak into the public history.
- Every `main` commit is self-contained and reviewable.
- Bisecting over the research programme's history lands on coherent
  states, not half-implementations.

The only exceptions to squash-merge:

1. A topic branch that is already a single clean commit — in which
   case fast-forward is equivalent to squash.
2. A release / tagging commit — handled with `git tag`, not merge.

## Commit message format

One-line subject under 72 characters, imperative mood
("Add", "Fix", "Update", not "Added" / "Fixes"). Body optional but
recommended for squash-merges so the collapsed context is not lost.
Paragraph-wrap at 72 columns.

Example:

```
Add Pareto n-scaling paper: submitted to Quantum

Multi-seed sweeps on MaxCut over 3-regular random graphs at
n = 6, 8, 10 qubits, replicating the Fig 2 frontier at each size.
Adds src/pqcgraphs/experiments/exp_b_pareto_scaling.py and the
corresponding multi-seed runner. Results JSONs under results/pareto_*.
```

No `Co-Authored-By` lines unless a human co-author is listed in the
paper's author list.

## Pre-push discipline

Before every `git push` to `origin`:

1. Confirm you are on `main` or a branch intended to be public
   (`feat/*`). Run `git branch --show-current`.
2. Refuse to push `wip/*` to `origin` — these are local-only. A
   pre-push hook is recommended; see `local/WORKFLOW.md` (local
   scratch area) for the template.
3. Run the test suite (`pytest tests/`) if the diff touches
   `src/` or `tests/`.
4. If the diff touches `docs/manuscript*`, recompile and verify the
   PDF page count is within target-journal limits.

## Tagging releases

Every paper-associated merge to `main` gets a tag:

```bash
git tag -a v<N>.<M>-<venue> -m "Paper <N> submitted / published at <venue>"
git push origin v<N>.<M>-<venue>
```

Example: `v1.0-quantum-arxiv`, `v2.0-joss`.

Tags are immutable anchors for arXiv Data-Availability links and for
citations of a specific framework version.

## Pull requests from external contributors

If you are opening a PR against this repository:

- Target: `main`.
- Start from `feat/<slug>` (no `wip/*`; that namespace is reserved
  for the author's unpublished research).
- Include tests for new functionality.
- Keep PRs focused — one logical change per PR so squash-merge
  produces a clean `main` history.
- By submitting a PR, you agree to license your contribution under
  MPL-2.0, matching the rest of the codebase.

## Questions

Research-direction questions should go to Ruben Dario Guerrero
<rudaguerman@gmail.com>. Code-level issues can be opened on the
GitHub tracker once the repository's issue feature is enabled.
