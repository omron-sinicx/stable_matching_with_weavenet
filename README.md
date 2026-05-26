# learning-based stable matching solver
This repository is an implementation of stable matching solver with WeaveNet. [[ref]](https://openreview.net/forum?id=ktHKpsbsxx).

## Preparation

### weavenet

This project depends on
[omron-sinicx/weavenet ≥ v1.1.0](https://github.com/omron-sinicx/weavenet/releases/tag/v1.1.0),
which adds the `MeanAggregator` aggregator and the
`CriteriaPerAxisStableMatching` criterion on top of v1.0.1, along with
two bug fixes (`criteria` device mismatch, `model` residual side-b typo).

Install directly from the tagged commit on GitHub (preferred — avoids
PyPI supply-chain risk and pins to an immutable git ref):

```
% pip install "git+https://github.com/omron-sinicx/weavenet.git@v1.1.0"
```

### Validation / test data

The training and evaluation configs expect 1000-sample NPZ datasets per
`(distribution, agent count, split)` triple under `data/{validation,test}/`.

**Option 1 — download the prebuilt dataset (~403 MB compressed):**

```
% wget -O stable_matching_val-test.zip \
    "https://drive.google.com/uc?export=download&id=1-qUDe8f-9JVgXNuL0DHgmTEnr8xU-XVU"
% unzip stable_matching_val-test.zip -d data/shared/
```

Or open the [Google Drive link](https://drive.google.com/file/d/1-qUDe8f-9JVgXNuL0DHgmTEnr8xU-XVU/view?usp=drive_link)
in a browser. The zip contains 80 combinations × 1000 NPZs (5
distributions × 8 sizes × 2 splits) plus an in-zip `README.md`
describing the per-combo provenance.

**Option 2 — regenerate from scratch with the paper seeds:**

```
% python scripts/generate_valtest_data.py --paper-seeds --out data/shared
```

This re-seeds numpy with the paper's defaults (135789 for validation,
2456 for test) once per `(split, dist, size)` invocation, then evolves
the random state through the per-instance loop — the same recipe as
the paper's `generate_dataset.py`. Output is statistically equivalent
to the paper's dataset (not bit-exact, since numpy versions differ).
NPZ schema per file: `sab, sba, matches, fairness, satisfaction,
gs_matches, SexEqualityCost, EgalitarianCost`.

## Quickstart
- training
```
% cd src
% python train.py
```

- evaluation
```
% cd src
% python eval.py logs/.../checkpoints/epoch_???.ckpt
```


## How to run
### Prerequired knowledges
- Please checkout [this template](https://github.com/ashleve/lightning-hydra-template) to know the structure of this repository.
- train the model with sexequality loss.
```
% cd src
% python train.py +model/criteria=sexequal
```
- Train the model with problem size 20x20 and distribution type GG.
```
% python train.py +datamodule/stable_matching/training_data=GG20x20 +datamodule/stable_matching/val_test_data=GG20x20
```

- Use the bias-reducing model for type UD, whose inputs to side A and B are differently biased.
```
% python train.py +datamodule/stable_matching/training_data=UD20x20 +datamodule/stable_matching/val_test_data=UD20x20 +model/net/head=weavenet_anti_bias
```

- Disable jit scripting (mainly for debug purpose)
```
% python train.py +model/do_jit_scripting=false
```

### Paper-spec WN-60 reproduction (Exp 2 / Table 2, 3)
- 60-layer WeaveNet with raw-mean aggregator + per-axis-softmax criterion,
  matching the loss / aggregation recipe used in the paper.
- Train on GG30x30 for the paper's 1000 epochs:
```
% python src/train.py \
    trainer.max_epochs=1000 trainer.check_val_every_n_epoch=10 \
    trainer.accelerator=gpu trainer.devices=1 \
    model/net=weavenet_dense_60_paper \
    model/criteria=paper_sexequal \
    model.do_jit_scripting=false \
    datamodule/training_data=GG30x30 \
    datamodule/val_test_data=GG30x30
```
- Substitute `UU30x30` / `DD30x30` / `UD30x30` / `LL30x30` for other
  distributions in Table 2. For Table 3 use `*20x20` configs.

- Please check any other configuration ideas in `./configs/`.
