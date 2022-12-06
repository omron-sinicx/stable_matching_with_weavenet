from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric

from .components import criteria
from functools import partial


def rreplace(s, src, tar, count=1):
    # replace src with tar, but search from back.
    return s[: : -1].replace(src[: : -1], tar[: : -1], count)[: : -1]

class WeaveNetLitModule(LightningModule):
    """Example of LightningModule for WeaveNet bipartite graph matching.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criteria: criteria.CriteriaStableMatching,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        
        # loss functions        
        self.criterion = torch.jit.script(criteria.generate_criterion())
        
        # metric objects for calculating and averaging accuracy across batches
        self.metric =  torch.jit.script(criteria.generate_metric())
        
        self.fairness = criteria.fairness
        for mode in ['train', 'val', 'test']:
            setattr(self, '{}_total_loss'.format(mode), MeanMetric())            
            for criterion in criteria.base_criterion_names:                 
                setattr(self, '{}_{}'.format(mode,criterion), MeanMetric())
            for metric in criteria.base_metric_names:
                setattr(self, '{}_{}'.format(mode,metric), MeanMetric())
                
            if self.fairness is None:
                continue                
            setattr(self, '{}_{}'.format(mode,criteria.fairness_criterion_name), MeanMetric())
            setattr(self, '{}_{}'.format(mode,criteria.fairness_metric_name), MeanMetric())

            
        self.val_success_rate_best = MaxMetric() # for selecting the stability-based best model
        if self.fairness:
            self.fairness_criterion_name = criteria.fairness_criterion_name 
            self.val_fairness_best = MinMetric()
        

    def forward(self, sab: torch.Tensor, sba:torch.Tensor):
        m = self.net([sab,sba])
        return m

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_success_rate_best.reset()
        if self.fairness:
            self.val_fairness_best.reset()

    def step(self, batch: Any):
        sab, sba = batch[:2]
        m, _, _ = self.forward(sab, sba)
        # sab: (batch_size, 1, N, M)
        # sba: (batch_size, 1, M, N)
        # m: (batch_size, N, M)
        loss, log = self.criterion(m, sab.squeeze(1), sba.squeeze(1))
        return m, sab.squeeze(1), sba.squeeze(1), loss, log
    
    def training_step(self, batch: Any, batch_idx: int):
        m, sab, sba, loss, log = self.step(batch)
        
        # update and log metrics
        self.train_total_loss(loss)
        self.log("train/total_loss", self.train_total_loss, on_step=False, on_epoch=True, prog_bar=True)
        mode = 'train'
        for k,v in log.items():
            _loss =  getattr(self, '{}_{}'.format(mode,k))
            _loss(v)
            self.log("{}/{}".format(mode,k), _loss, on_step=False, on_epoch=True, prog_bar=True)
        log = self.metric(m, sab, sba)
        for k,v in log.items():
            _metric =  getattr(self, '{}_{}'.format(mode,k))
            _metric(v.float().mean())
            self.log("{}/{}".format(mode,k), _metric, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}
    
    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        m, sab, sba, loss, log = self.step(batch)

        # update and log metrics
        self.val_total_loss(loss)        
        self.log("val/total_loss", self.val_total_loss, on_step=False, on_epoch=True, prog_bar=True)
        mode = 'val'
        for k,v in log.items():
            _loss =  getattr(self, '{}_{}'.format(mode,k))
            _loss(v)
            self.log("{}/{}".format(mode,k), _loss, on_step=False, on_epoch=True, prog_bar=True)
        log = self.metric(m, sab, sba)
        for k,v in log.items():
            _metric =  getattr(self, '{}_{}'.format(mode,k))
            _metric(v.float().mean())
            self.log("{}/{}".format(mode,k), _metric, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        success_rate = self.val_is_success.compute()
        self.val_success_rate_best(success_rate)
        self.log("val/success_rate_best", self.val_success_rate_best.compute())
        
        if self.fairness:
            fairness = getattr(self, 'val_{}'.format(self.fairness_criterion_name)).compute()
            self.val_fairness_best(fairness)
            self.log("val/fairness_best", self.val_fairness_best.compute())
        
        
        pass

    def test_step(self, batch: Any, batch_idx: int):
        m, sab, sba, loss, log = self.step(batch)

        # update and log metrics
        self.test_total_loss(loss)
        self.log("test/loss", self.test_total_loss, on_step=False, on_epoch=True, prog_bar=True)
        mode = 'test'
        for k,v in log.items():
            _loss =  getattr(self, '{}_{}'.format(mode,k))
            _loss(v)
            self.log("{}/{}".format(mode,k), _loss, on_step=False, on_epoch=True, prog_bar=True)
            
        log = self.metric(m, sab, sba)
        for k,v in log.items():
            _metric =  getattr(self, '{}_{}'.format(mode,k))
            _metric(v.float().mean())
            self.log("{}/{}".format(mode,k), _metric, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


    
if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf

    import pyrootutils
    root = pyrootutils.setup_root(__file__, pythonpath=True)    

    config_path= root / "configs"
    @hydra.main(version_base=None, config_path = config_path, config_name="train")
    def my_app(cfg: DictConfig) -> None:
        print(OmegaConf.to_yaml(cfg))
        _ = hydra.utils.instantiate(cfg.model)
    my_app()
    
