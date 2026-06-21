---
description: Skeptically review an open PR — checkout, run `make ci`, scrutinise the diff, post a GitHub review
argument-hint: <pr-number>
---
Act as the **skeptical PR reviewer** for this repo. First read `docs/REVIEW_GUIDE.md` in full and
follow it exactly. Also read `CLAUDE.md`, `PLAN.md`, `docs/adr/`, and GitHub issue #1 for context.

Review **PR #$1** (if no number was given, run `gh pr list` and ask which one to review):

1. `gh pr checkout $1`
2. Run `make ci` — this is the gate. If it fails, that is an automatic request-changes; capture the failing output.
3. `git diff main...HEAD` and `git log main..HEAD --format='%an <%ae> | %s'` — scrutinise every changed line and the commit authorship against the REVIEW_GUIDE checklist and the hard constraints.
4. Post your verdict on GitHub with `gh pr review $1` — `--approve` only if `make ci` passes and you found nothing violating the constraints, otherwise `--request-changes` — with blockers cited by `file:line` and your `make ci` result included.

Be critical and specific; your goal is to catch problems, not to approve quickly. Do not edit code — you are review-only.
