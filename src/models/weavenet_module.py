from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric

from .components import criteria
from functools import partial



# packages to log computational graph
from torchviz import make_dot
from tempfile import NamedTemporaryFile
from pathlib import Path
from PIL import Image
import numpy as np
def log_computational_graph(var, params, format='png'):
    # log computational graph as an image.
    dot = make_dot(var, params=params, show_attrs=True)#, show_saved=True)
    
    extname = '.'+format
    with NamedTemporaryFile(suffix=extname) as file:
        path = Path(file.name)

        dot.render(path.parent / path.stem, format=format)
        img = Image.open(path)
    
    try:
        import wandb
        img = wandb.Image(img)
        wandb.log({'computational_graph':img})
    except:
        pass
    

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
        self.computational_graph_logged = False
        
        # loss functions        
        self.criterion = torch.jit.script(criteria.generate_criterion())
        
        # metric objects for calculating and averaging accuracy across batches
        self.metric =  torch.jit.script(criteria.metric)
        
        self.fairness = criteria.fairness
        for mode in ['train', 'val', 'test']:
            setattr(self, '{}_total_loss'.format(mode), MeanMetric())            
            for criterion in criteria.base_criterion_names:                 
                setattr(self, '{}_{}'.format(mode,criterion), MeanMetric())
            for metric in criteria.metric_names:
                setattr(self, '{}_{}'.format(mode,metric), MeanMetric())

            if self.fairness is None:
                continue           
            print('{}_{}'.format(mode,criteria.fairness_criterion_name))
            setattr(self, '{}_{}'.format(mode,criteria.fairness_criterion_name), MeanMetric())
            self.val_fairness_w_penalty = MeanMetric()

            
        self.val_success_rate_best = MaxMetric() # for selecting the stability-based best model
        if self.fairness:
            self.larger_is_better = criteria.larger_is_better
            self.fairness_criterion_name = criteria.fairness_criterion_name 
            self.val_fairness_best = MaxMetric()
            self.offset = None

    def forward(self, sab: torch.Tensor, sba_t:torch.Tensor):
        if not isinstance(self.net, torch.jit.ScriptModule):
            self.net = torch.jit.trace(self.net, (sab, sba_t))
        return self.net(sab,sba_t)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_success_rate_best.reset()
        if self.fairness:
            self.val_fairness_best.reset()

    def step(self, batch: Any):
        sab, sba = batch[:2]
        # sab: (batch_size, N, M)
        # sba: (batch_size, M, N)
        
        sba_t = sba.transpose(-1,-2).contiguous() # (batch_size, N, M)
        m, mab, mba_t = self.forward(sab.unsqueeze(-1), sba_t.unsqueeze(-1))  # sab.shape == sba_t.shape == (batch_size, N, M, 1)
        # m: (batch_size, N, M, 1)        
        m = m.squeeze(-1)
        mab = mab.squeeze(-1)
        mba_t = mba_t.squeeze(-1)
        
        # m: (batch_size, N, M)      
        loss, log = self.criterion(m, mab, mba_t, sab, sba_t)
        loss = loss.mean()
        metric, _ = self.metric(m, sab, sba_t)

        return m, loss, log, metric
    
    def set_log(self, log, metric, mode):
        for k,v in log.items():
            _loss =  getattr(self, '{}_{}'.format(mode,k))
            _loss(v)
            self.log("{}/{}".format(mode,k), _loss, on_step=False, on_epoch=True, prog_bar=True)
        for k,v in metric.items():
            _metric =  getattr(self, '{}_{}'.format(mode,k))
            _metric(v.float().mean())
            self.log("{}/{}".format(mode,k), _metric, on_step=False, on_epoch=True, prog_bar=True)


            
    
    def training_step(self, batch: Any, batch_idx: int):
        m, loss, log, metric = self.step(batch)
        if not self.computational_graph_logged:
            log_computational_graph(m, dict(self.net.named_parameters()))
            self.computational_graph_logged = True

        # update and log metrics
        self.train_total_loss(loss)
        self.log("train/total_loss", self.train_total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.set_log(log,metric, 'train')
        return {"loss": loss}
    
    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass
    
    def fairness_w_penalty(self, 
                           fairness:torch.Tensor, 
                           is_success:torch.Tensor,
                           sab:torch.Tensor)->torch.Tensor:
        mask = (1.0-is_success.to(sab.dtype))
        penalty =  sab.max(dim=-1)[0].sum(dim=-1) * mask
        if self.larger_is_better:
            return fairness - penalty
        return fairness + penalty
    
    def validation_step(self, batch: Any, batch_idx: int):
        m, loss, log, metric = self.step(batch)

        # update and log metrics
        self.val_total_loss(loss)        
        self.log("val/total_loss", self.val_total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.set_log(log, metric, 'val')
        if self.fairness:
            fairness = metric[self.fairness]
            is_success = metric['is_success']
            self.val_fairness_w_penalty(self.fairness_w_penalty(fairness, is_success, batch[0]).mean())
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        if not self.fairness:            
            success_rate = self.val_is_success.compute()        
            self.val_success_rate_best(success_rate)
            self.log("val/best", self.val_success_rate_best.compute(), prog_bar=True)
        else:
            fairness_best = self.val_fairness_w_penalty.compute()
            if not self.larger_is_better:
                if self.offset is None:
                    self.offset = fairness_best
                fairness_best = -fairness_best + self.offset
            self.val_fairness_best(fairness_best)
            self.log("val/best", self.val_fairness_best.compute(), prog_bar=True)       

    def test_step(self, batch: Any, batch_idx: int):
        m, loss, log, metric = self.step(batch)

        # update and log metrics
        self.test_total_loss(loss)
        self.log("test/loss", self.test_total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.set_log(log,metric, 'test')

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

    
import random
class WeaveNetLPLitModule(WeaveNetLitModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criteria: criteria.CriteriaStableMatching,
    ):
        super().__init__(net,optimizer, scheduler, criteria)
        
    def step(self, batch: Any):
        sab, sba = batch[:2]
        m, m_binary,_, _ = self.forward(sab, sba)
        
        # sab: (batch_size, 1, N, M)
        # sba: (batch_size, 1, M, N)
        # m: (batch_size, N, M)
        sab, sba = sab.squeeze(1), sba.squeeze(1)
        loss, log = self.criterion(m, sab, sba)
        #loss_b, log_b = self.criterion(m_binary, sab.squeeze(1), sba.squeeze(1))
        #loss += loss_b
        #for k,v in log_b.items():
        #    log["{}_binary".format(k)] = v
        metric = self.metric(m_binary, sab, sba)
        
        #if random.random() > 0.99:
        #    print("hoge: ", m[0], m_binary[0], m_binary[0]-m[0])
        
        return m_binary, loss, log, metric

    def validation_epoch_end(self, outputs: List[Any]):
        success_rate = self.val_is_success.compute()
        self.val_success_rate_best(success_rate)
        self.log("val/success_rate_best", self.val_success_rate_best.compute())
        
        if self.fairness:
            fairness = getattr(self, 'val_{}'.format(self.fairness_criterion_name)).compute()
            self.val_fairness_best(fairness)
            self.log("val/fairness_best", self.val_fairness_best.compute())
        
        pass    
    
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
    
