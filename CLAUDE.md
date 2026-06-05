# CLAUDE.md — Project Rules

## Interaction Rules

- **All code changes require confirmation**: Edit and Write tools are configured to always prompt for confirmation via the native UI diff viewer. Do not bypass this.
- **Large file edits**: Before modifying any file over ~100 lines, ask the user to confirm the specific change and location first.
- **Learning mode**: Currently OFF. (User will re-enable when needed.)
- When explaining code, prefer concise explanations that build understanding rather than just handing over solutions.

## Code Style

- Follow existing code style in the file being edited — do not reformat unrelated lines.
- Do not add comments unless they explain non-obvious logic or serve as learning scaffolding (per learning mode above).
- Do not introduce new dependencies without asking first.

## Git Rules

- Never force push (`git push --force`) without explicit user confirmation.
- Never commit secrets, credentials, or `.env` files.
- Prefer small, focused commits with clear messages over large bulk commits.
- Do not `git add .` blindly — always specify files explicitly.

## Large Project Rules

- Do not refactor or restructure code outside the scope of the current task.
- Do not delete files without explicit confirmation, even if they appear unused.
- When adding a new feature, check if a similar utility already exists in the codebase before writing new code.
- Do not change configuration files (e.g., `pyproject.toml`, `uv.lock`, `.env`) without asking first.
- When modifying shared modules that are imported by multiple files, warn the user about potential side effects before proceeding.

## Environment

- Primary development machine: Windows laptop (PowerShell)
- Deployment machine: Lab Linux (new user account, SSH key auth to GitHub)
- Package manager: `uv`
- Remote: `git@github.com:ken-333/openpi-private.git`