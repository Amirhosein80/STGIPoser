import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from typing import Callable, Optional
from logger import get_logger
import random

logger = get_logger(__name__)

def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Stack data to create a dictionary of tensors.
    
    Parameters:
        batch: list of dictionaries containing tensors
    
    Example:
    >>> batch = [
            {"key1": torch.tensor([1, 2, 3]), "key2": torch.tensor([4, 5, 6])},
            {"key1": torch.tensor([7, 8, 9]), "key2": torch.tensor([10, 11, 12])}
        ]
    >>> collate_fn(batch)
    {'key1': tensor([1, 7]), 'key2': tensor([4, 10])}
    """
    keys = batch[0].keys()
    out = {}
    for i, k in enumerate(keys):
        out[k] = torch.cat([b[k].unsqueeze(0) for b in batch], dim=0)
    return out


class RandomNoise:
    """
    Add random noise to data.
    
    Parameters:
        std: standard deviation of the noise
        p: probability to convert
    
    Example:
    >>> noise = RandomNoise(noise=0.1, p=0.5)
    >>> data = {"imu_acc": torch.tensor([1, 2, 3]), "uwb": torch.tensor([4, 5, 6])}
    >>> noise(data)
    """
    def __init__(self, std: float = 0.1, p: float = 0.5):
        self.p = p
        self.std = std

    def __call__(self, samples):
        """
        Callable function to add random noise to data.
        
        Parameters:
            samples: a dictionary of data
        """
        if random.random() < self.p:
            noise_acc = (self.noise**0.5)*torch.randn(*samples["imu_acc"].shape)
            samples["imu_acc"] += noise_acc
            
            if "uwb" in samples.keys():
                noise_uwb = (self.noise**0.5)*torch.randn(*samples["uwb"].shape)
                samples["uwb"] += noise_uwb
        return samples




class STIPoseDataset(Dataset):
    """
    Custom dataset for STIPose Model.
    
    Parameters:
        configs: configs for dataset
        phase: dataset for Train or Valid
        transform: optional transform to apply to the data
        
    Example:
    >>> configs = {
            "data_dir": "./data",
            "batch_size": 16,
            "num_worker": 4
        }
    >>> dataset = STIPoseDataset(configs, "Train")
    >>> dataloader = dataset.get_data_loader()
    """
    def __init__(self, configs: dict, phase: str, transform: Optional[Callable] = None) -> None:
        
        assert phase in ["Train",
                         "Valid"], "You should select phase between Train and Valid"

        if phase == "Train":
            self.files = glob.glob(
                os.path.join(configs["dataset"]["dir"], f"{phase}_seq", f"seq_{configs['dataset']['seq_length']}", "*.npz"))
        elif phase == "Valid":
            self.files = glob.glob(os.path.join(configs["dataset"]["dir"], f"{phase}", f"*/*", "*.npz"))

        self.phase = phase
        self.dir = configs["dataset"]["dir"]
        self.data_keys = ["imu_acc", "imu_ori", "grot", 
                          "jvel", "uwb", "trans",
                          "last_jvel", "last_trans"]

        self.batch_size = configs["dataset"]["batch_size"]
        self.num_worker = configs["dataset"]["num_worker"]

        self.transform = transform

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            length of the dataset
        """
        return len(self.files)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        """
        Get the item at the given index.
        
        Parameters:
            idx: index of the item
        Returns:
            item at the given index
        """ 
        file = self.files[idx]
        data = dict(np.load(file))

        x = {}

        for key, value in data.items():
            if key in self.data_keys:
                x[key] = torch.from_numpy(value).float()

        if self.phase == "Valid":
            if "imu_acc" not in x.keys():
                x['imu_acc'] = torch.from_numpy(data['vacc']).float()
                x['imu_ori'] = torch.from_numpy(data['vrot']).float()
        
        if self.transform is not None:
            x = self.transform(x)

        return x

    def get_data_loader(self) -> DataLoader:
        """
        Get the data loader for the dataset.
        
        Returns:
            data loader
        """
        if self.phase in ["Train", "All"]:
            sampler = RandomSampler(self)
            bs = self.batch_size
        else:
            sampler = SequentialSampler(self)
            bs = 1
        return DataLoader(self, batch_size=bs, sampler=sampler,
                          num_workers=self.num_worker, collate_fn=collate_fn,
                          pin_memory=True)


if __name__ == '__main__':
    pass
