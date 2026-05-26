# Vendored weavenet

This directory is a vendored snapshot of <https://github.com/omron-sinicx/weavenet>
`main`, installed editable into the project via `pip install -e external/weavenet`
(see `.devcontainer/post-create.sh`). The vendoring is **temporary** — once
upstream cuts the next release (v1.0.2, which will include the four PRs listed
below), the plan is to delete `external/weavenet/` and switch back to
`pip install weavenet>=1.0.2`.

## Why vendored, not installed from PyPI

The latest upstream tag is `v1.0.1`. The four upstream changes this project
depends on landed on `main` after that tag:

- [#3](https://github.com/omron-sinicx/weavenet/pull/3) fix(criteria):
  CPU/GPU device mismatch in fairness loss.
- [#4](https://github.com/omron-sinicx/weavenet/pull/4) fix(model): residual
  on side-b stream was silently dropped (`xba_t` typo).
- [#5](https://github.com/omron-sinicx/weavenet/pull/5) feat(layers): add
  `MeanAggregator` stream aggregator.
- [#6](https://github.com/omron-sinicx/weavenet/pull/6) feat(criteria): add
  `CriteriaPerAxisStableMatching` + extract `_BaseCriteriaStableMatching`.

This vendored copy tracks `main` post-#6, so the project does not have to
wait for the v1.0.2 tag to use the fixed/extended API.

## What is in this directory

- `src/weavenet/` — Python source, mirroring `main`.
- `tests/`, `pyproject.toml`, `README.md`, `CHANGELOG.md`, `LICENSE.txt`,
  `docs_source/` — upstream's files, copied verbatim.

Removed from upstream tree to keep the vendoring lightweight:
- `.git/` — having a nested git repo conflicts with this project's git tree.
- `docs/` — 16 MB of pre-built HTML; re-buildable from `docs_source/` if needed.

## Updating from upstream

Once a new upstream commit (or tag) appears that this project wants to
consume, refresh the vendored tree:

```sh
git clone --depth=1 https://github.com/omron-sinicx/weavenet /tmp/weavenet-fresh
rsync -a --delete \
    --exclude=__pycache__ --exclude=_version.py --exclude=.git --exclude=docs \
    /tmp/weavenet-fresh/ external/weavenet/
rm -rf /tmp/weavenet-fresh
# Run scripts/build_valtest_zip.py's smoke checks or your training to verify.
```

Once upstream is tagged `v1.0.2` (or later) and this project switches to
`pip install weavenet>=1.0.2`, delete `external/weavenet/` and remove the
`pip install -e` line from `.devcontainer/post-create.sh`.
