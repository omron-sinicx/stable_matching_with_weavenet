# Vendored weavenet

This directory is a vendored snapshot of
[omron-sinicx/weavenet@v1.1.0](https://github.com/omron-sinicx/weavenet/releases/tag/v1.1.0),
installed editable into the project via `pip install -e external/weavenet`
(see `.devcontainer/post-create.sh`).

**The vendoring is a transitional convenience and can be removed any
time.** Once we are happy switching to `pip install weavenet>=1.1.0`,
delete `external/weavenet/` and replace the editable install in
`.devcontainer/post-create.sh` with `pip install weavenet>=1.1.0`. The
upstream v1.1.0 release already carries all the fixes/features this
project depends on:

- [#3](https://github.com/omron-sinicx/weavenet/pull/3) fix(criteria):
  CPU/GPU device mismatch in fairness loss.
- [#4](https://github.com/omron-sinicx/weavenet/pull/4) fix(model): residual
  on side-b stream was silently dropped (`xba_t` typo).
- [#5](https://github.com/omron-sinicx/weavenet/pull/5) feat(layers): add
  `MeanAggregator` stream aggregator.
- [#6](https://github.com/omron-sinicx/weavenet/pull/6) feat(criteria): add
  `CriteriaPerAxisStableMatching` + extract `_BaseCriteriaStableMatching`.

The vendored copy currently exists only because PyPI / a Hugging Face
mirror may not yet have v1.1.0 on the install path your environment
hits — once that is sorted, the vendor is removable.

## What is in this directory

- `src/weavenet/` — Python source, mirroring upstream v1.1.0.
- `tests/`, `pyproject.toml`, `README.md`, `CHANGELOG.md`, `LICENSE.txt`,
  `docs_source/` — upstream's files, copied verbatim.

Removed from the upstream tree to keep the vendoring lightweight:
- `.git/` — having a nested git repo conflicts with this project's git tree.
- `docs/` — 16 MB of pre-built HTML; re-buildable from `docs_source/` if needed.

## Refreshing the vendored snapshot

If upstream advances and this project wants to consume the new commit
without waiting for the next PyPI release:

```sh
git clone --depth=1 https://github.com/omron-sinicx/weavenet /tmp/weavenet-fresh
rsync -a --delete \
    --exclude=__pycache__ --exclude=_version.py --exclude=.git --exclude=docs \
    /tmp/weavenet-fresh/ external/weavenet/
rm -rf /tmp/weavenet-fresh
# Smoke-test: re-run scripts/build_valtest_zip.py's import line or a
# 1-epoch training before committing.
```

## Removing the vendor (when ready)

```sh
rm -rf external/weavenet
# In .devcontainer/post-create.sh, replace
#   sudo pip install --no-deps --no-cache-dir -e /workspace/external/weavenet
# with
#   pip install --no-cache-dir weavenet>=1.1.0
# (or add `weavenet>=1.1.0` to requirements.txt and rely on the existing
#  `pip install -r requirements.txt` in the Dockerfile).
```
