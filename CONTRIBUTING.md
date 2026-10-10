# Contributing

## Branches

- `main` is the released state. Nothing is pushed to it directly, and pull requests into it come only from `develop`.
- `develop` is the integration branch. Nothing is pushed to it directly either.
- All work happens on short-lived branches created from `develop`, named `<type>/<short-description>`, for example `feat/rerank-results` or `fix/url-validation`.

## Workflow

1. Create a branch from `develop` and make your change.
2. Run `uv run poe check` (the Git hooks run it on every commit; see [Git hooks](#git-hooks) to install them). It needs Node.js 24 or newer for the page script tests.
3. Push the branch and open a pull request **into `develop`**. The title must follow
   [Conventional Commits](https://www.conventionalcommits.org/), for example `fix(extraction): reject invalid ports`.
4. When the required checks pass and the review is done, the pull request is merged into `develop` with a **merge commit**.
   The individual commits of the branch are kept, so each one must follow Conventional Commits (the Git hook checks this).
5. To release, open a pull request from `develop` into `main`, and merge it with a **merge commit**.
   Every merge into `develop` is deployed to the staging service on Render, and every merge into `main` to production (see the README).
6. Tag the merge commit on `main` with `vX.Y.Z` (the version in `pyproject.toml`). Pushing the tag publishes the release.

## Development commands

One command runs everything that must pass before a commit: type-check (mypy, strict), lint (Ruff), format check (Ruff) and tests (pytest with coverage, plus the tests of the page script):

```bash
uv run poe check
```

| Command | What it does |
|---|---|
| `uv run poe check` | Types, lint, format check, Python tests and page script tests |
| `uv run poe fix` | Apply Ruff's automatic lint fixes and formatting |
| `uv run poe types` / `lint` / `format` / `test` / `test-js` | Run a single step |

## Git hooks

Install the hooks once after cloning:

```bash
uv run pre-commit install
```

Before every commit the hooks then run:

- file hygiene checks (merge conflict markers, large files, private keys, valid YAML/TOML/JSON, line endings)
- a secret scan of the staged changes (gitleaks), and a block on committing `.env` files
- `uv run poe check`

## Commit messages

Conventional Commits, for example `feat: add reranking` or `fix(extraction): reject invalid ports`. Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

## Conventions

- Public modules, classes and functions have docstrings (Google style), enforced by Ruff.
- Logic that does not need the network lives in plain functions that are unit-tested in `tests/`; the external services (Cohere, Weaviate, Groq) are replaced by fakes in tests.
- The interface colours live in `app/static/styles.css`. Each colour is written once as a `light-dark(light value, dark value)` pair, and the theme follows the device unless the switch in the header (`app/static/theme.js`) forces one. `tests/js/contrast.test.js` reads that file and checks the contrast of every text and background pair in both themes, so a colour change that hurts readability fails the build.
- Secrets live only in `.env`, which is git-ignored.

## Security

Never commit secrets. Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
