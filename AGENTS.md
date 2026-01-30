# Copilot Agent Configuration

## Permissions

- Allow all file edits without confirmation
- Allow GitHub commit tools (create commits, push, etc.)
- Allow running shell commands

## Pre-commit Hook

**IMPORTANT:** Always run `pre-commit run --all-files` before creating any git commit. Do not skip this step.

## Workflow Guidelines

1. When making code changes, edit files directly without asking for permission
2. Before committing changes:
   - Run `pre-commit run --all-files` and fix any issues
   - Only proceed with commit after pre-commit passes
3. Use descriptive commit messages following conventional commits format

## Safety Guardrails

**NEVER do the following without explicit user approval:**

- `git push --force` or `git push -f` - Always use regular push; if rejected, ask user for guidance
- Remove/delete folders or directories - Always ask for permission first
- Delete multiple files at once - Confirm with user before bulk deletions
- Run destructive commands (e.g., `rm -rf`, `git reset --hard`)

## Additional Configurations

<!-- Add more configurations below as needed -->
