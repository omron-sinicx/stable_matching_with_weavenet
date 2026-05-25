# Vendored weavenet

This directory is a vendored copy of <https://github.com/omron-sinicx/weavenet>,
installed editable into the project via `pip install -e external/weavenet` (see
`.devcontainer/Dockerfile`).

## Upstream provenance

- Upstream commit: `dd75c94c574fd0bd6229cc0dc0bc191b6a9d369f` (2023-12-05)
- Subset kept here: source (`src/weavenet/`), tests, `pyproject.toml`, README,
  CHANGELOG, license, `docs_source/`.
- Removed: `.git/` (would conflict with this project's git), `docs/` (16 MB of
  built HTML; re-buildable from `docs_source/`).

## Local modifications vs upstream

Two bug fixes plus one project-specific compatibility module. Re-apply these if
you re-pull from upstream.

### 1. `src/weavenet/criteria.py` — CPU/GPU device mismatch in fairness loss

`CriteriaStableMatching.generate_criterion` constructs an inline
`torch.tensor([not gate_fairness_loss])` without a `device=` arg, so on GPU
training with `fairness != None` it raises:

> RuntimeError: Expected all tensors to be on the same device, but found at
> least two devices, cuda:0 and cpu!

Fix: thread `device=l.device` through that constructor:

```diff
-            loss += fairness_weight * l * ((loss.detach()<=0).max(torch.tensor([not gate_fairness_loss])).to(l.dtype))
+            loss += fairness_weight * l * ((loss.detach()<=0).max(torch.tensor([not gate_fairness_loss], device=l.device)).to(l.dtype))
```

### 2. `src/weavenet/model.py` — `xba_t` typo silently disables residual on side b

`MatchingNet.forward` does a tuple-unpack to apply the residual addition to both
streams, but the second line writes to `xba` instead of `xba_t`. Result: the
residual addition for the `xba_t` stream is computed and discarded, while the
loop variable `xba_t` retains its non-residual value. Only the `xab` stream
benefits from residual connections.

```diff
                 if calc_res:
                     xab_keep, xab = xab, xab + xab_keep
-                    xba_t_keep, xba = xba_t, xba_t + xba_t_keep
+                    xba_t_keep, xba_t = xba_t, xba_t + xba_t_keep
```

This silently breaks any deep WeaveNet trained with residual connections — the
two streams diverge in depth-equivalent capacity.

### 3. `src/weavenet/legacy_compat.py` — new module (not in upstream)

Provides classes that translate the paper-time `MatcherWeaveNet`
(`docs/papers/legacy_code/src/networks.py` of this project) onto the current
PyPI weavenet abstractions:

- `LegacyMaxPoolEncoder` — port of legacy `EncoderMaxPool` to NHWC/Linear.
- `LegacyUnitListGenerator` — factory for `Unit` wrappers around the above.
- `LegacyWeaveNet` — `WeaveNet`-style backbone using the legacy encoder.
- `LegacyMeanAggregator` — raw-logit mean of two streams (alternative to
  `DualSoftmaxSqrt`; requires a legacy-compatible criterion to be useful).
- `legacy_residual_pattern(L)` — `[F, F, T, F, T, …]` mask matching the legacy
  `if use_resnet and i%2==0` rule.

See the module docstrings for details.

## Updating from upstream

```sh
cd /tmp && git clone https://github.com/omron-sinicx/weavenet
diff -ruN /tmp/weavenet/src external/weavenet/src \
    --exclude=__pycache__ --exclude=_version.py --exclude=legacy_compat.py \
    > /tmp/local.patch
# Inspect /tmp/local.patch — it should show the two bug fixes above. If it
# shows additional diffs, upstream has moved; merge those into our vendored
# tree before discarding /tmp/weavenet.
```
