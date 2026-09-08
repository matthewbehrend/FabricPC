# Design Plan: GitHub Issue and Pull Request Templates

## Status
- **Issue**: #58
- **Date**: 2026-08-30
- **Author**: mhaye9545

## Problem Statement
Bug reports without diagnostic context (JAX version, backend, installed extras, device topology, reproduction scripts) cost maintainers multiple communication round-trips to triage. With the upcoming public hackathon release in October 2026, establishing standardized issue and pull request templates is essential to capture mandatory environment diagnostics up front and guide new contributors.

## Chosen Approach
1. **GitHub Issue Form for Bug Reports (`.github/ISSUE_TEMPLATE/bug_report.yml`)**:
   - Includes a reference to the troubleshooting guide (`docs/user_guides/16_troubleshooting.md`).
   - Fields for version, extras selected, JAX environment info (`jax.print_environment_info()`, `jax.default_backend()`, `jax.devices()`), environment variables, key package list, GPU/driver status, minimal reproduction script, and expected vs observed behavior.
2. **GitHub Issue Form for Feature Requests (`.github/ISSUE_TEMPLATE/feature_request.yml`)**:
   - Collects problem statement, proposed behavior, alternatives considered, and links to `CONTRIBUTING.md`.
3. **Issue Config (`.github/ISSUE_TEMPLATE/config.yml`)**:
   - Disables blank issues (`blank_issues_enabled: false`) and redirects questions to discussions and `CONTRIBUTING.md`.
4. **Pull Request Template (`.github/PULL_REQUEST_TEMPLATE.md`)**:
   - Mirrors `CONTRIBUTING.md` standards: linked issue, summary of changes, design document link, testing checklist (tests, linter, demo baseline fidelity, reference verification).

## Alternatives Considered
- **Standard Markdown Issue Templates**:
  - *Pros*: Simple markdown files.
  - *Cons*: Users frequently delete sections or leave required diagnostic details blank. Issue forms (`.yml`) allow structured fields, dropdowns, checkboxes, and required validation.
- **Requiring bot triage**:
  - *Pros*: Automated labeling.
  - *Cons*: Adds infrastructure overhead and external dependencies; native GitHub forms achieve the same validation natively.

## Testing & Verification
- Validated YAML schema syntax of issue templates and config.
- Checked template rendering and links against existing documentation (`docs/user_guides/16_troubleshooting.md` and `CONTRIBUTING.md`).
