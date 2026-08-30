# Contributing to FabricPC

## Development setup

```bash
git clone https://github.com/trueagi-io/FabricPC.git
cd FabricPC
pip install -e ".[all,dev]"   # full contributor install; ".[dev,experiments]" reproduces CI
pre-commit install
pytest -q
```

`[dev]` brings pytest, hypothesis, black, ruff, mypy, and pre-commit. The pre-commit hooks run
`ruff check` (lint) and `black` (format); both versions are pinned so the hooks and a local run
agree. The narrower `.[dev,experiments]` install also reproduces CI: tests that need `[tfds]`
or `[viz]` skip via `pytest.importorskip` when those extras are absent.

## What CI checks on a pull request

- **Tests** (`.github/workflows/test.yml`): `pytest -q` on Python 3.11 and 3.13 against the newest
  JAX, plus a leg on Python 3.11 pinned to `jax==0.7.0`, the declared floor in `pyproject.toml`. A
  change that raises the real JAX minimum fails that leg; raise the declared floor in the same PR
  and say why.
- **Lint** (`.github/workflows/lint.yml`): the pre-commit hooks, run over all files. The
  installed local hook checks only the files in each commit; `pre-commit run --all-files`
  reproduces the CI check exactly.
- **Doc snippets**: code blocks in `docs/user_guides/` and `README.md` are parse-, import-, and
  signature-checked by `tests/test_doc_snippets.py`. If your change alters the signature of an API
  a guide demonstrates, update the guide; the test fails otherwise. Behavior changes that keep the
  signature pass the check, so verify affected guides by hand.

## Pull-request expectations

- Develop on a branch named `username/your_feature_name`; rebase on `main` before opening the PR.
- New behavior comes with tests and docstrings. Demos must match their baseline results (the
  `Results:` block in each demo's module docstring, e.g. `examples/mnist_demo.py`), or the PR
  explains the divergence.
- Every number, benchmark, and `file:line` reference in a PR description or design document must
  be real and reproducible. State the command that produced a measurement.
- Link the issue the PR resolves. Trivial fixes (typos, broken links) need no issue.

## Design-first workflow

Every non-trivial change starts with a design document. The document is committed to
`docs/dev_plans_archive/` before the PR is marked ready for review, and the PR description
links it. One branch and one PR carry both the design and the implementation.

1. Claim the issue by commenting on it, so effort is not duplicated.
2. Write the design document, either in `docs/dev_plans/` (in-flight designs; the directory is
   emptied before merge) or directly in `docs/dev_plans_archive/`. The document must sit in the
   archive before the PR is ready, so starting there skips the move.
3. Review the design critically yourself. If the issue is labeled `design`, open the PR as a
   draft containing only the design document and get a maintainer review before implementing.
4. Implement on the same branch.
5. Make sure the document sits in `docs/dev_plans_archive/`, the record of completed designs,
   and reflects the final implementation; then mark the PR ready for review.

A design document must:

- State the problem and the intended outcome, grounded to specific files and lines.
- List the alternative approaches considered and rejected, each with its pros and cons, alongside
  the chosen approach.
- Include a migration inventory when the change touches shared code: every call site that must
  move, and the tests that pin current behavior before the move.

Completed designs in `docs/dev_plans_archive/` show the expected shape.

## Migration, not fallbacks

When a change fixes or extends shared code, migrate every existing caller in the same change. Do
not add dual-mode flags, compatibility shims, or `legacy_*` parameters: they persist as dead code
and leave the improved path exercised only by new callers. If a genuine compatibility need exists
(external users, staged rollout), name it explicitly in the design document and get a maintainer's
sign-off before implementing it.

## Working with coding agents

AI-assisted contributions are welcome. The same expectations apply whether an agent or a human
wrote the code; the points below address failure modes specific to generated work:

- **Design before agentic implementation.** For any non-trivial change, write the design document
  first and let it drive the implementation, not the reverse.
- **Alternatives are part of the design.** The document lists the approaches considered and
  rejected, with pros and cons for each. An agent's first proposal is a candidate, not a decision.
- **Refine the design with a fresh-context review.** Have an agent with no memory of the drafting
  session critically review the design, then revise. Iterate until the review stops finding
  substantive problems. Only then implement.
- **Passing tests is not a sufficient quality check.** After implementation, run the same
  fresh-context critical review on the code and refine it iteratively. Tests confirm the behavior
  you thought to test; the review looks for the behavior you did not.
- **Design document reflects final state.** The design document and the PR description are the
  record of what was intended and what was done. Each commit whose implementation departs from
  the design document updates the document in the same commit. Do not let design and code
  diverge.
- **You are accountable for every claim.** Run the tests and benchmarks yourself; verify every
  cited number and file reference. Unverified generated claims are grounds for rejection.

## Questions

Ask questions on the issue itself, or open a new issue if none fits.
