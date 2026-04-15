---
name: report
description: Generate end-of-session documentation — claude_notes (code-change log, committed to git) and/or lab_notebook (experiment log, synced to Notion under Tim/lab_notebook). Invoke when the user types `/report` to wrap up a working session.
---

# /report

Two outputs:
1. **claude_notes** (`claude_notes/log_MM_DD_YYYY.md`) — terse code-change log, committed to git.
2. **lab_notebook** (`lab_notebook/YYYY-MM-DD.md`) — narrative experiment log, gitignored, synced to Notion.

## On invocation

1. **Determine scope** from the argument:
   - `/report claude` → claude_notes only
   - `/report lab` → lab_notebook only
   - `/report both` or `/report` (no arg) → ask "claude_notes, lab_notebook, or both?"

2. **Gather context**:
   - `git status` + `git diff` (staged, unstaged, and untracked files) for code changes.
   - Scan this conversation for: what was done, why, experiment results, findings, rationale, decisions.
   - Compute today's date.
   - Check if today's local entries exist (`claude_notes/log_MM_DD_YYYY.md`, `lab_notebook/YYYY-MM-DD.md`). If yes, read them — the draft must **merge new content organically into existing sections**, never append a naked new block.

3. **Draft** using the formats below. Reference the two most recent existing entries for tone and structure.

## claude_notes format

```
# Session Log — <Month Day, Year>

## Summary
<one sentence capturing the session's theme>

## Changes Made

**Modified: `<path>`**
- <terse bullet>

**New File: `<path>`**
- <terse bullet>
```

- Strictly code-change focused, pulled from `git diff` + conversation about code.
- Group by file; use `Modified:` / `New File:` headers.
- Bullets describe the change, not the reasoning or test procedure.
- Match tone of `claude_notes/log_03_31_2026.md` and `log_04_01_2026.md`.

## lab_notebook format

```
# <YYYY-MM-DD>

## Tasks

### <Task name>

<Short prose framing task and motivation.>

---

## <Topic section>

<Narrative + tables for numeric results.>
```

- Free-form narrative with `##` section headers per topic.
- Use data tables for numeric results.
- Include *why*: motivation, rationale, decisions.
- **Include code changes briefly**, framed as "why" — e.g., "swapped loss to WeightedBCEFocalLoss to target merge errors". The detailed file-by-file list lives in claude_notes.
- Anything in the conversation not captured in claude_notes (experiment results, dataset-specific findings, decisions) belongs here.
- Match tone of `lab_notebook/2026-04-06.md` and `2026-04-01.md`.

## Same-day merge rule

If today's entry already exists, extend it by merging into existing structure, not appending:
- **claude_notes**: extend `## Summary` to cover new work; add new bullets under existing file headers, or new file headers if a new file was touched.
- **lab_notebook**: merge new content into matching `##` sections; add new `##` sections only for genuinely new topics.

## Draft review loop

1. Present the full draft(s) in the response as fenced markdown blocks.
2. Wait for user feedback.
3. On "add X" / "remove Y" / "change Z", apply the edit and **re-present the full updated draft** (not a diff).
4. Continue until the user approves ("ship it", "looks good", etc.).

## After approval

1. **Write local files**:
   - `claude_notes/log_MM_DD_YYYY.md`
   - `lab_notebook/YYYY-MM-DD.md`

2. **Sync lab_notebook to Notion** (skip if lab_notebook wasn't produced):
   - Parent page ID: `320082e8595e811a86f0eb6079317dc2` (`lab_notebook` under `Tim`)
   - `notion-fetch` the parent and look for a child page titled `YYYY-MM-DD`.
   - If it exists → `notion-update-page` to replace its content with the merged draft.
   - If not → `notion-create-pages` with parent set to the `lab_notebook` page ID, title `YYYY-MM-DD`, content is the markdown draft.

3. **Git commit and push** (skip entirely if only `/report lab` was run):
   - Stage only: `claude_notes/log_MM_DD_YYYY.md` + code files that were part of this session.
   - **Never stage** `lab_notebook/` (gitignored).
   - Commit message: match recent commits' style (imperative, concise — see `git log -5`).
   - `git push` to `origin/main` unless the user says otherwise.

## Notes

- `CLAUDE.md` has a "Session Logs" section mirroring claude_notes. **Leave it alone** — user flagged it as outdated.
- `/report claude` → skip Notion sync.
- `/report lab` → skip git commit/push entirely.
