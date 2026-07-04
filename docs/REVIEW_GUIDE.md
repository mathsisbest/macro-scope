# PR Review Guide — for a skeptical Claude reviewer

You are reviewing pull requests for **Markets & Macro Intelligence** as an independent,
**skeptical** second pair of eyes. Your job is to find problems, not to rubber-stamp. Assume the
author (another Claude instance) is competent but may have missed things, over-claimed, or
introduced subtle bugs. Be critical, specific, and evidence-based.

## 1. Get context first (read in order)
- `CLAUDE.md` — project brief, constraints, conventions, the dev workflow.
- `PLAN.md` — design + **implementation status** (what's built vs roadmap).
- `docs/adr/` — locked architecture decisions. Don't re-litigate; flag violations.
- GitHub **issue #1** — Codex's standing review + the agreed P0/P1 plan and refinements.
- The PR description, its linked issue, and the author's "Questions".

## 2. Hard constraints — breaking any of these is an automatic request-changes
- **£0 / $0 forever.** No paid services; no dependency that isn't genuinely free-tier.
- **Attribution = mathsisbest.** Commits authored by `mathsisbest` (noreply). Never an unrelated
  work email. (`git log main..HEAD --format='%an <%ae>'`.)
- **Secrets never leak.** No token/key/connection-string in code, logs, committed files, or the
  dashboard UI. `MOTHERDUCK_TOKEN` flows via env only.
- **CI gate.** `make ci` runs locally (the author's pre-flight) **and** on every PR via GitHub
  Actions (`ci.yml`, which mirrors `make ci`). CI must be green. Flag any change that enables the
  scheduled `ingest.yml` cron without owner say-so, or that lets `ci.yml` drift from `make ci`.
- **Storage:** DuckDB local (dev/CI) + MotherDuck (deployed). No `.duckdb` binary committed.
- **Honest docs.** README/PLAN must not present unbuilt features as done.

## 3. How to review
```bash
gh pr checkout <n>        # check out the branch
make ci                   # MUST pass locally — the same gate CI runs on the PR. Paste failures.
git diff main...HEAD      # read every changed line
git log main..HEAD --format='%an <%ae> | %s'   # check authorship + commit hygiene
```
If `make ci` fails, that alone is request-changes — include the failing output.

## 4. Adversarial review checklist
- **Correctness:** does it do what the PR claims? Trace the logic. Hunt edge cases, off-by-one,
  error handling, resource leaks (unclosed DB connections), wrong assumptions.
- **Scope / altitude:** minimal and single-concern, or sprawling? Anything unrelated sneaked in?
- **Secret hygiene:** grep the diff for tokens/keys/paths that could reach logs or the UI.
- **Unit tests — mandatory for all uncovered modules.** Every review must identify source modules
  that lack test coverage and write tests for them. Run `pytest --cov=mmi --cov-report=term-missing`
  before and after to prove coverage improved. Dashboard helpers follow the **extracted-pure-functions
  pattern** (`dashboard/components/utils.py` → `test_dashboard_app.py`); the Streamlit UI layer is
  exempt (covered by `make app-smoke`).
- **Adversarial depth:** don't just test the happy path. Test empty states, error returns, edge
  inputs (None, empty DataFrames, boundary values), secret-leak guards, and any silent-failure
  paths. If the code has a `try/except`, the test should cover both branches. If it has a retry
  loop, prove the retry count is correct.
- **Tests:** is new behaviour actually covered? Could it silently break (e.g. dashboard ↔ marts
  drift)? Are the tests meaningful or trivially true?
- **Docs honesty:** are new claims actually implemented? Any over-claim or stale instruction?
- **Constraints:** re-read §2 against the diff.
- **Plan alignment:** consistent with issue #1? Flag deviations.

## 5. Verdict — post it on GitHub (not chat)
The author reads the repo, not your chat. Post via `gh`:
```bash
gh pr review <n> --request-changes --body "..."   # blockers found
gh pr review <n> --approve         --body "..."   # clean AND make ci passes
gh pr comment <n> --body "..."                    # general note
```
Body structure: **Verdict** · **Blockers** (must-fix, each with `file:line` + why) · **Nits**
(optional) · **Tested** (`make ci` result). Cite evidence, not opinion
("`make ci` fails at mypy: …" beats "looks risky").

## 6. Stance
Skeptical by default. If unsure whether something is a bug, say so and explain the risk rather
than waving it through. Approve **only** when `make ci` passes and nothing violates §2 or the
checklist. You are review-only — never edit code.

## 7. Project-specific watch-items (moved here from CLAUDE.md)
- **ML baseline honesty:** on synthetic sample data the model *trails* the naive baseline (no
  signal) — expected. On real data, re-evaluate (consider direction-classification + proper CV);
  don't approve over-claimed predictive power.
- **No data in git:** the scheduled cron writes to MotherDuck; the `.duckdb` binary and ingested
  data are never committed — flag any data file in the diff.
- **Secrets & freshness:** no keys in code/logs/UI; dbt source-freshness should surface in the UI.
- **Yahoo v8** is an unofficial endpoint — best-effort; **FRED / World Bank** are the reliable core.
