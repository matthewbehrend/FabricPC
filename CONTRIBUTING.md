# Contributing to FabricPC

## Development setup

```bash
git clone https://github.com/trueagi-io/FabricPC.git
cd FabricPC
pip install -e ".[dev,experiments]"   # what CI installs; [experiments] carries optuna for the tuner tests
pre-commit install
pytest -q
```

`[dev]` brings pytest, hypothesis, black, ruff, mypy, and pre-commit. The pre-commit hooks run
`ruff check` (lint) and `black` (format); both versions are pinned so the hooks and a local run
agree. The `[tfds]` and `[viz]` extras are optional — tests that need them skip via
`pytest.importorskip` when they are absent.

## What CI checks on a pull request

- **Tests** (`.github/workflows/test.yml`): `pytest -q` on Python 3.11 and 3.13 against the newest
  JAX, plus a leg on Python 3.11 pinned to `jax==0.7.0`, the declared floor in `pyproject.toml`. A
  change that raises the real JAX minimum fails that leg; raise the declared floor in the same PR
  and say why.
- **Lint** (`.github/workflows/lint.yml`): the pre-commit hooks. Running `pre-commit install`
  locally means CI never surprises you.
- **Doc snippets**: code blocks in `docs/user_guides/` and `README.md` are parse-, import-, and
  signature-checked by `tests/test_doc_snippets.py`. If your change alters the signature of an API
  a guide demonstrates, update the guide; the test fails otherwise. Behavior changes that keep the
  signature pass the check, so verify affected guides by hand.

## Pull-request expectations

- Develop on a branch named `username/your_feature_name`; rebase on `main` before opening the PR.
- New behavior comes with tests. Demos must match their baseline results (the `Results:` block in
  each demo's module docstring, e.g. `examples/mnist_demo.py`), or the PR explains the divergence.
- Every number, benchmark, and `file:line` reference in a PR description or design document must
  be real and reproducible. State the command that produced a measurement.
- Link the issue the PR resolves.

## Design-first workflow (issues labeled `design`)

Some issues ask for a design before any code. For these, a design document precedes
implementation, and both ship in one pull request:

1. Claim the issue by commenting on it.
2. Write the design document and open a **draft PR** containing only that document. Author it in
   `docs/dev_plans/` (this directory exists only on work-in-progress branches) or directly in
   `docs/dev_plans_archive/` — either is fine.
3. A maintainer reviews the design in the draft PR. Iterate there until sign-off. Do not start
   the implementation before sign-off; a rejected design would waste it.
4. Implement on the same branch.
5. In the final commit, make sure the document sits in `docs/dev_plans_archive/`, the record of
   completed designs, then mark the PR ready for review.

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

AI-assisted contributions are welcome. The expectations are about practice and content, and they
apply whether an agent or a human wrote the code:

- **Design before agentic implementation.** For any non-trivial change, write the design document
  (planning file) first and let it drive the implementation, not the reverse.
- **Alternatives are part of the design.** The document lists the approaches considered and
  rejected, with pros and cons for each. An agent's first proposal is a candidate, not a decision.
- **Refine the design with a fresh-context review.** Have an agent with no memory of the drafting
  session critically review the design, then revise. Iterate until the review stops finding
  substantive problems. Only then implement.
- **Passing tests is not a sufficient quality check.** After implementation, run the same
  fresh-context critical review on the code and refine it iteratively. Tests confirm the behavior
  you thought to test; the review looks for the behavior you did not.
- **Design document reflects final state.** The design document and the PR description are the 
  record of what was intended and what was done. Update design to reflect the final state in each
  commit that changes or closes gaps in the design. Do not let design and code diverge.
- **You are accountable for every claim.** Run the tests and benchmarks yourself; verify every
  cited number and file reference. Unverified generated claims are grounds for rejection.

## Questions and claiming work

- Claim an issue by commenting on it, so effort is not duplicated.
- Ask questions on the issue itself, or open a new issue if none fits.
