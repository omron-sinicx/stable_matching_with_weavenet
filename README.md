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

`stable_matching_val-test.zip` (1 GB after unzip) holds the 80 paper-
authentic + paper-seed-regenerated NPZ datasets the configs expect under
`data/{validation,test}/`. Download from [TBD link] and unzip into the
repo root:

```
% unzip stable_matching_val-test.zip       # produces data/validation/ and data/test/
```

The exact provenance per `(distribution, agent count, split)` triple and
the schema of each NPZ is documented inside the zip's `README.md`
(`scripts/zip_README.md` in this repo). If you cannot obtain the zip,
regenerate the data from scratch with:

```
% python scripts/generate_valtest_data.py --paper-seeds --out data/shared
```

This uses the paper's default seeds (135789 for val, 2456 for test) and
produces statistically equivalent (not bit-exact) datasets.

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
