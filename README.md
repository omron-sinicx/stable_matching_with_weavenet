# learning-based stable matching solver
This repository is an implementation of stable matching solver with WeaveNet. [ref](https://openreview.net/forum?id=ktHKpsbsxx).

## Preparation
- Install [weavenet package (v1.0.0)](https://github.com/omron-sinicx/weavenet/releases/tag/v1.0.0).
- Download [stable_matching_val-test.zip](https://drive.google.com/file/d/1MLIvUe4q_5kSNvfmn89Mfkb1TG2Z2Dj4/view?usp=sharing) and put it in `/data/` directory. Then, unzip it. You will get `data/validation` and `data/test` directories where {UU, DD, GG, UD, LL} type data with sizes {3,5,7,9,10,20,30,100}.
```
% unzip stable_matching_val-test.zip
```

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

- Please check any other configuration ideas in `./config/'.