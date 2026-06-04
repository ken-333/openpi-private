# AGENTS.md - Project Rules

## Interaction Rules

- **Large file edits**: Before modifying any file over ~100 lines, ask the user to confirm the specific change and location first.
- **Learning mode**: This project is for learning end-to-end robotics pipelines. Do not generate complete implementations. Instead, provide scaffolding with comments such as `# TODO: ...` or `# Step 1: ...` so the user can fill in the logic themselves.
- When explaining code, prefer concise explanations that build understanding rather than just handing over solutions.

## Code Style

- Follow existing code style in the file being edited. Do not reformat unrelated lines.
- Do not add comments unless they explain non-obvious logic or serve as learning scaffolding.
- Do not introduce new dependencies without asking first.

## Git Rules

- Never force push (`git push --force`) without explicit user confirmation.
- Never commit secrets, credentials, or `.env` files.
- Prefer small, focused commits with clear messages over large bulk commits.
- Do not `git add .` blindly. Always specify files explicitly.

## Large Project Rules

- Do not refactor or restructure code outside the scope of the current task.
- Do not delete files without explicit confirmation, even if they appear unused.
- When adding a new feature, check if a similar utility already exists in the codebase before writing new code.
- Do not change configuration files such as `pyproject.toml`, `uv.lock`, or `.env` without asking first.
- When modifying shared modules that are imported by multiple files, warn the user about potential side effects before proceeding.

## Environment

- Primary development machine: Windows laptop using PowerShell.
- Deployment machine: Lab Linux using a new user account and SSH key auth to GitHub.
- Package manager: `uv`.
- Remote: `git@github.com:ken-333/openpi-private.git`.
