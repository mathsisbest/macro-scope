<!--
House style: a small, single-concern PR (~1–5 files). Branch `<type>/<slug>`.
Title = conventional commit (feat:/fix:/refactor:/docs:/test:/build:). Link the issue with `Closes #NN`.
Keep the five sections below — they mirror CLAUDE.md › Dev workflow.
-->

## Concern
<!-- The one problem this PR addresses. Closes #NN -->

## What changed
<!-- The actual edits, file by file. Say whether it's config/data-only or touches logic. -->

## Risk
<!-- Blast radius + how to undo. Only write "Low" if you mean it. -->

## `make ci`
<!-- Paste the real result, e.g.:
PASS — ruff, ruff format, mypy, seed, portfolio, dbt build+tests, dashboard smoke, pytest (N passed).
Docs/config-only with no src/ transform/ dashboard/ tests/ paths touched? Write "N/A — no code paths touched" and say why. -->

## Questions
<!-- Open decisions for the reviewer, or "None". -->

---
- [ ] Single concern, ~1–5 files
- [ ] Conventional-commit title
- [ ] `make ci` result pasted above (or N/A justified)
- [ ] Reviewed in a fresh context (Claude 2 / `/review-pr`), not by the implementer
- [ ] No secrets/keys in the diff, logs, or UI; authored as `mathsisbest`
