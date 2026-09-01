# Contributing

## Branch Strategy
- `main` — production-ready code only
- `develop` — integration branch
- `feature/*` — new features
- `bugfix/*` — non-urgent fixes
- `hotfix/*` — urgent production fixes

## Workflow
1. Branch off `develop` using the naming convention above.
2. Make your changes, ensuring `pytest`, `ruff check .`, and `black --check .` all pass.
3. Open a Pull Request into `develop`. Direct pushes to `main`/`develop` are blocked.
4. At least one CI check must pass before merging.

## Code Style
Black and Ruff run automatically on save if you've installed the recommended VS Code extensions (`.vscode/extensions.json`).