**Git**
1. Work on branch `refactor/v2`. Commit after every completed stage (and after large sub-steps). Run tests first.
   Stage files by explicit path, never `git add -A`.
2. **NEVER `git push`**, never add or change a remote, never force anything.
3. **NEVER add `Co-Authored-By` or any "Generated with Claude Code" line** to commit messages.
   Commits use the user's existing git identity only.
4. Commit style: `type: short summary` (`chore`, `feat`, `fix`, `docs`, `test`, `refactor`), one line, under 72 chars.
5. Never commit `data/`, `.venv/`, model binaries over 5 MB, `__pycache__`, or LaTeX build files.

**Requests, tokens, rate limit (gateway allows ~80 requests per minute)**
6. **One tool call at a time.** No parallel tool calls, no subagents/Task tool, no web search/fetch
   (single exception: the S9 citation audit, Section 9), no MCP tools.
7. Chain shell commands with `&&`. Write each whole file in one create call. Do not re-read a file you just wrote.
8. Cap output: `| head -n 40`, `| tail -n 20`. Never print a DataFrame over 15 rows.
9. Keep chat replies terse: at most 5 lines after each task. No recaps.
10. **On any API/gateway error** (429, 5xx, "overloaded", timeout): update `PROGRESS.md`, wait at least 60 s
    (`python -c "import time;time.sleep(60)"`), retry **once**, then STOP and tell the user what was last completed.
11. Any job expected to exceed 5 minutes: print the exact command and ask the user to run it in their own terminal.
    Never poll in loops.

**Science**
12. **The test set is touched exactly once** (S4). Record it in `PROGRESS.md`. If a bug forces a re-run, tell the
    user; no tuning of any kind after seeing test results.
13. Every number in the notebook, README and report must come from a file in `results/`. **Never type a metric by hand.**
14. Do not invent citations, results or causal explanations. Use only the verified references in Section 9.
15. **Log after every task** (Section 10). A task is not done until its `PROGRESS.md` entry exists.
