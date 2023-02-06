from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import random
import json
import math

from pathlib import Path

class NpzDataset(Dataset):
    # WARNING: because the number of stable match differs for each sample, batch_size must always be one for ease of implementation.
    # test code for this class
    # ./test_validation_dataset_loader.ipynb
    def __init__(self, filelist, root_dir):
        # filelist: the filelist of npz given as `list` or `str`, where `str` must be a file with npz file per row.
        # root_dir: the root directory for the path of each entry in the filelist.
        self.root_dir = root_dir
        if isinstance(filelist,list):
            self.filelist = filelist
        else:
            self.filelist = self.load_list(filelist)
            
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self,idx):
        buf = self.load(os.path.join(self.root_dir,self.filelist[idx]))
        return buf

    def load_list(self,path):
        return [line.strip() for line in open(path)]

    @staticmethod
    def collate_fn(batch):
        # batch: sab, sba, N, M matches (with shape variance), fairness, satisfaction, gs_matches (with shape variance), SEq, Egal
        items = [None] * 10
        for i in range(2): # sab, sba (problem instances)
            items[i] = torch.stack( [torch.from_numpy(sample[i]) for sample in batch])
        for i in range(2, 4): # N, M (integers)
            items[i] = torch.Tensor([sample[i] for sample in batch])
        for i in range(4, 10): # some solutions and its scores (with variable shape)
            items[i] = [torch.from_numpy(sample[i]) for sample in batch]
        return tuple(items)
        
        
    def load(self,filename):        
        npz = np.load(filename)
        sab = npz['sab']
        N,M = sab.shape[-2:]
        return sab, npz['sba'], N, M, npz['matches'], npz['fairness'], npz['satisfaction'], npz['gs_matches'], npz['SexEqualityCost'], npz['EgalitarianCost']
        # sab, sba, N, M, matches, fairness, satisfaction, gs_matches, SEq, Egal


LibimSeTi_dist = np.array([[ 2525.,   542.,   236.,   237.,   414.,   383.,   511.,   507.,
          342.,  1092.],
       [  419.,  1096.,   181.,   144.,   257.,   240.,   307.,   294.,
          198.,   580.],
       [  238.,   335.,   546.,   150.,   235.,   198.,   263.,   242.,
          182.,   536.],
       [  198.,   199.,   183.,   432.,   292.,   212.,   301.,   291.,
          195.,   609.],
       [  218.,   197.,   227.,   273.,   860.,   369.,   536.,   478.,
          342.,  1088.],
       [  144.,   111.,   118.,   131.,   278.,   524.,   382.,   409.,
          298.,   913.],
       [  130.,   109.,   125.,   144.,   301.,   315.,   827.,   618.,
          487.,  1427.],
       [  119.,    72.,    88.,    97.,   242.,   223.,   355.,   792.,
          599.,  1734.],
       [   90.,    62.,    80.,    92.,   169.,   203.,   263.,   365.,
          772.,  2191.],
       [  262.,   206.,   159.,   142.,   360.,   332.,   574.,   838.,
         1305., 53376.]]).reshape((100,))
LibimSeTi_dist /= LibimSeTi_dist.sum()

class UniversalSMIGenerator(Dataset):
    # can freely choose male's preference, and female's preference, among U, D, G, and L(ibimSeTi)
    def __init__(self, distrib_m, distrib_w, sigma_m=0.4, sigma_w=0.4 ,N_range_m=(3,3), N_range_w=(3,3), transform=True, dtype=np.float32, samples_per_epoch=1000, *args, **kwargs):
        self.N_range_m=N_range_m
        self.N_range_w=N_range_w
        self.sigma_m=sigma_m
        self.sigma_w=sigma_w
        self.len=samples_per_epoch
        self.transform=transform
        self.dtype=dtype
        self.distrib_m=distrib_m
        self.distrib_w=distrib_w

    def __len__(self):
        return self.len

    def choice_n(self,N_range):
        if N_range[0] == N_range[1]:
            return N_range[0]
        else:
            # plan 1, randomly generate the group size according to N_range, cannot be used if batchsize>1
            return random.randint(N_range[0],N_range[1])


    def __getitem__(self,idx):
        na = self.choice_n(self.N_range_m)
        nb = self.choice_n(self.N_range_w)

        if self.distrib_m == 'L' or self.distrib_w == 'L':
            seed = np.random.choice(range(100),(na,nb),p=LibimSeTi_dist)
            sab = (seed%10).astype(self.dtype)
            sba = (seed//10).astype(self.dtype).transpose()

            bias_ab = np.random.rand(*sab.shape)/1.1
            bias_ba = np.random.rand(*sba.shape)/1.1
            pa = sba.sum(axis=0)/(sba.shape[-1]*11)
            pb = sab.sum(axis=0)/(sab.shape[-1]*11)
            for i in range(bias_ab.shape[0]):
                if np.random.rand()<0.2:
                    bias_ab[i] = pb
            for i in range(bias_ba.shape[0]):
                if np.random.rand()<0.2:
                    bias_ba[i] = pa

            sab += bias_ab
            sba += bias_ba
            sab/=10
            sba/=10
        else:
            if self.distrib_m == 'G':
                base_ab = np.tile(np.arange(1,nb+1),(na,1))
                sab = (np.random.normal(base_ab, self.sigma_m*nb)/nb).astype(self.dtype)
            elif self.distrib_m == 'D':
                nb1 = math.ceil(self.sigma_m*nb)
                nb2 = nb-nb1
                sab1 = np.random.uniform(low=0.5, high=1.0, size=(na, nb1)).astype(self.dtype) # popular
                sab2 = np.random.uniform(low=0.0, high=0.5, size=(na, nb2)).astype(self.dtype) # non-popular
                sab = np.hstack((sab2,sab1))
            else:
                sab = np.random.uniform(low=0.0, high=1.0, size=(na, nb)).astype(self.dtype)

            if self.distrib_w == 'G':
                base_ba = np.tile(np.arange(1,na+1),(nb,1))
                sba = (np.random.normal(base_ba, self.sigma_w*na)/na).astype(self.dtype)
            elif self.distrib_w == 'D':
                na1 = math.ceil(self.sigma_w*na)
                na2 = na-na1
                sba1 = np.random.uniform(low=0.5, high=1.0, size=(nb, na1)).astype(self.dtype)
                sba2 = np.random.uniform(low=0.0, high=0.5, size=(nb, na2)).astype(self.dtype)
                sba = np.hstack((sba2,sba1))
            else:
                sba = np.random.uniform(low=0.0, high=1.0, size=(nb, na)).astype(self.dtype)


        if self.transform:
            for i in range(sab.shape[0]) :
                sab[i,np.argsort(sab[i,:])] = np.linspace(0.1, 1, sab.shape[-1])
            for i in range(sba.shape[0]) :
                sba[i,np.argsort(sba[i,:])] = np.linspace(0.1, 1, sba.shape[-1])

        #sab = sab.reshape([na,nb])
        #sba = sba.reshape([nb,na])
        return sab,sba,na,nb    
    
class StableMatchingDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        training_data=None,
        val_test_data=None,
        data_dir: str = "data/",
        batch_size: int = 8,
        samples_per_epoch: int = 1000,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        if training_data:
            self.data_train = UniversalSMIGenerator(**training_data, samples_per_epoch=samples_per_epoch)
        else:
            self.data_train = None
        if val_test_data:
            data_dir = Path(data_dir)
            self.data_val = NpzDataset( data_dir / val_test_data.val_dat, data_dir / val_test_data.val_data_dir)
            self.data_test = NpzDataset(data_dir / val_test_data.test_dat, data_dir / val_test_data.test_data_dir)
        else:
            self.data_val = None
            self.data_test = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        #MNIST(self.hparams.data_dir, train=True, download=True)
        #MNIST(self.hparams.data_dir, train=False, download=True)
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        pass


    def train_dataloader(self):
        if self.data_train is None:
            raise RuntimeError("training data is not configurated.")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        if self.data_val is None:
            raise RuntimeError("validation data is not configurated.")
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn = NpzDataset.collate_fn,
        )

    def test_dataloader(self):
        if self.data_test is None:
            raise RuntimeError("test data is not configurated.")
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn = NpzDataset.collate_fn,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
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
        _ = hydra.utils.instantiate(cfg.datamodule)
    my_app()
    
