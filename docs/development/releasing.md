# Cutting a release

This page is for maintainers. It documents how a new `escape-fel` version
gets published to PyPI.

## How versioning works

The package version is **not** written anywhere in the source — it's derived
automatically from git tags by `setuptools-scm` (see `[tool.setuptools_scm]`
in `pyproject.toml`). A tag `v0.2.0` produces version `0.2.0`; commits after
a tag produce a `.devN+g<hash>` suffix automatically.

This means: never hand-edit a version number in `pyproject.toml`. To release,
you only need to create the right git tag.

## What triggers a publish

`.github/workflows/publish.yml` runs on any pushed tag matching `v*.*.*`. It
builds the sdist/wheel and publishes to PyPI using
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC — no
API token stored in the repo), via the `pypi` GitHub Environment.

Pushing commits to `main`, or pushing a tag that doesn't match `v*.*.*`,
does **not** publish anything.

```{warning}
Some old tags in this repo (`0.1.0`, `0.2.0`, `test1`, ...) predate this
convention and lack the `v` prefix — they don't trigger the workflow and
should not be reused or treated as release markers. Always tag with a
leading `v`.
```

## Auto-tagging on push (`.githooks/pre-push`)

Every push of `main` to `origin` auto-creates and pushes the next PATCH tag
(`vX.Y.Z` → `vX.Y.(Z+1)`), which in turn triggers the PyPI publish above —
so **pushing to `main` releases**, without a separate manual tag step. This
is a client-side hook, so each clone/machine needs to opt in once:

```bash
git config core.hooksPath .githooks
```

Behavior:

- Skips auto-tagging if the commit being pushed already carries a `v*.*.*`
  tag — so a deliberate MINOR/MAJOR release still works exactly like the
  manual flow below: tag it yourself (`git tag vX.Y.0`) *before* pushing,
  and the hook leaves it alone.
- Only fires on `refs/heads/main`; other branches are never auto-tagged.
- Bypass for a single push that shouldn't release (e.g. an experimental
  commit to `main`): `SKIP_AUTOTAG=1 git push`.
- Never blocks the underlying code push — if the tag push itself fails for
  any reason, it just warns and continues.

The hook only bumps the last digit. It has no way to know a push contains a
breaking change or new feature (see "Choosing the version number" below) —
that judgment call is still yours; make it by tagging manually beforehand.

## Release checklist (manual / MINOR+ releases)

1. Make sure `main` is green and up to date locally:

   ```bash
   git checkout main
   git pull
   ```

2. Pick the next version number (see below) and tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

   (`git push --tags` also works, but pushes *all* local tags — prefer
   pushing the single new tag explicitly.) If you tag before pushing `main`
   itself, the auto-tag hook above sees the commit is already tagged and
   won't also add a PATCH tag on top of it.

3. Watch the **Publish to PyPI** run under the repo's *Actions* tab. Once it
   finishes, the release is live — `pip install escape-fel` will pick it up
   immediately.

4. There is no separate GitHub Release step required for PyPI, but creating
   one (`gh release create vX.Y.Z --generate-notes`) gives users a readable
   changelog and is recommended.

## Choosing the version number

`escape-fel` is still pre-1.0, so by [semver](https://semver.org/#spec-item-4)
convention anything may change at any time, but in practice this project
treats the numbers like:

- **PATCH** (`0.1.2` → `0.1.3`): pure bugfixes, no API or default-behavior
  changes.
- **MINOR** (`0.1.x` → `0.2.0`): new features, and also any breaking change
  (renamed/removed parameter, changed default, changed return type). Since
  the major version is still `0`, breaking changes travel in a MINOR bump
  rather than a MAJOR one.
- **MAJOR** (`0.x` → `1.0.0`): reserved for the first stable API commitment;
  not relevant yet.

Before tagging a MINOR release that includes breaking changes, mention the
break in the GitHub release notes so downstream users notice it — there is
no automated changelog yet.
