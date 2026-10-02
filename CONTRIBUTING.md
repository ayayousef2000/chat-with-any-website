# Contributing

## Branches

- `main` is the released state. Nothing is pushed to it directly, and pull requests into it come only from `develop`.
- `develop` is the integration branch. Nothing is pushed to it directly either.
- All work happens on short-lived branches created from `develop`, named `<type>/<short-description>`, for example `feat/rerank-results` or `fix/url-validation`.

## Workflow

1. Create a branch from `develop` and make your change.
2. Run `uv run poe check` (the Git hooks run it on every commit; see the README to install them).
3. Push the branch and open a pull request **into `develop`**. The title must follow
   [Conventional Commits](https://www.conventionalcommits.org/), for example `fix(extraction): reject invalid ports`.
4. When the required checks pass and the review is done, the pull request is **squash-merged** into `develop`.
5. To release, open a pull request from `develop` into `main`, and merge it with a **merge commit**.
6. Tag the merge commit on `main` with `vX.Y.Z` (the version in `pyproject.toml`). Pushing the tag publishes the release.

## Commit messages

Conventional Commits: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

## Security

Never commit secrets. Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
