import gc
import os
import random

import comet_ml
import numpy as np
import roma
import torch
import torch.nn as nn

from src.base_model import r6d2rot_mat
from src.logger import get_logger, get_train_logger, TRAIN_TIME

logger = get_logger(__name__)


def to_device(data: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    """
    Send data to device.

    Parameters:
        data: data to send
        device: device to send
    Returns:
        sent data

    Example:
    >>> data = {"imu_acc": torch.tensor([1, 2, 3]), "uwb": torch.tensor([4, 5, 6])}
    >>> to_device(data, "cuda")
    {'imu_acc': tensor([1, 2, 3]), 'uwb': tensor([4, 5, 6])}
    """
    keys = data.keys()
    for k in keys:
        data[k] = data[k].to(device)

    return data


def set_seed(seed: int) -> None:
    """
    Set random seed for modules.

    Parameters:
        seed: random seed

    Example:
    >>> set_seed(0)
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def setup_env() -> None:
    """
    Setup environment.

    Example:
    >>> setup_env()
    """
    torch.backends.cudnn.benchmark = True
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    set_seed(0)


def create_log_dir(configs: dict):
    """
    Create log directory.

    Parameters:
        configs: training configs
    """
    os.makedirs(f'./train_log/{configs["training"]["experimnet_name"]+ "_" + TRAIN_TIME}', exist_ok=True)
    os.makedirs(
        f'./train_log/{configs["training"]["experimnet_name"]+ "_" + TRAIN_TIME}/checkpoint',
        exist_ok=True,
    )
    configs["log_dir"] = f"./train_log/{configs['training']['experimnet_name']+ "_" + TRAIN_TIME}/"


def set_comet(configs: dict) -> comet_ml.Experiment:
    """
    Set comet ml experiment.

    Parameters:
        configs: training configs

    Returns:
        comet_ml.Experiment

    Example:
    >>> comet = set_comet(configs)
    """
    comet_ml.login()

    experiment = comet_ml.start(
        api_key=configs["training"]["comet_api_key"],
        project_name=configs["training"]["model_name"],
    )
    experiment.set_name(configs["training"]["experimnet_name"] + "_" + TRAIN_TIME)

    logger.info(f"Start comet ml for {configs['training']['experimnet_name']}")
    experiment.log_parameters(configs)
    return experiment


def resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    configs: dict,
) -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    int,
    float,
    float,
    dict[str, list[float]],
    dict[str, list[float]],
]:
    """
    Load parameters to continue training.

    Parameters:
        model: model
        optimizer: optimizer
        scheduler: scheduler
        configs: training configs

    Returns:
        model: model
        optimizer: optimizer
        scheduler: scheduler
        start_epoch: start epoch
        acc: accuracy
        best_acc: best accuracy

    Raises:
        Exception: if something is wrong
        Exception: if no checkpoint found

    Example:
    >>> model, optimizer, scheduler, start_epoch, acc, best_acc, train_hist, valid_hist = resume(model, optimizer, scheduler, configs)
    """
    last_ckpt = os.path.join(
        configs["log_dir"],
        f"checkpoint/last_{configs['training']['experiment_name']}.pth",
    )
    train_logger = get_train_logger(
        __name__, experiment_name=configs["training"]["experiment_name"]
    )

    if os.path.isfile(last_ckpt):
        try:
            loaded = torch.load(last_ckpt)
            model.load_state_dict(loaded["model"])
            optimizer.load_state_dict(loaded["optimizer"])
            scheduler.load_state_dict(loaded["scheduler"])
            start_epoch = loaded["epoch"] + 1
            train_hist = loaded["train_hist"]
            valid_hist = loaded["valid_hist"]
            acc = loaded["acc"]
            best_acc = loaded["best_acc"]
            train_logger.info(f"Load all parameters from last checkpoint :)")
            train_logger.info(
                f"Train start from epoch {start_epoch} epoch with accuracy {acc} and best accuracy {best_acc} :)"
            )

            logger.info(
                f"Load all parameters from last checkpoint for {configs['training']['experiment_name']} :)"
            )
            logger.info(
                f"Train start for {configs['training']['experiment_name']} from epoch {start_epoch} epoch with accuracy {acc} and best accuracy {best_acc} :)"
            )

            return (
                model,
                optimizer,
                scheduler,
                start_epoch,
                acc,
                best_acc,
                train_hist,
                valid_hist,
            )
        except Exception as error:
            train_logger.error(f"Something is wrong! :( ")
            train_logger.error(error)

            logger.error(
                f"Something is wrong! :( for {configs['training']['experiment_name']}"
            )
            logger.error(error)

            raise error

    else:
        train_logger.error(
            f"No checkpoint found for {configs['training']['experiment_name']} :)"
        )
        logger.error(
            f"No checkpoint found for {configs['training']['experiment_name']} :)"
        )
        raise Exception(
            f"No checkpoint found for {configs['training']['experiment_name']} :)"
        )


def save(
    model: torch.nn.Module,
    acc: float,
    best_acc: float,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    train_hist: dict,
    valid_hist: dict,
    name: str,
    log_dir: str,
) -> None:
    """
    Save model and other information.

    Parameters:
        model: model
        acc: last accuracy
        best_acc: best archived pose error
        optimizer: optimizer
        scheduler: scheduler
        epoch: last epoch
        train_hist: training accuracy and loss
        valid_hist: validation accuracy and loss
        name: experiment name
        log_dir: log directory

    Example:
    >>> save(model, acc, best_acc, optimizer, scheduler, epoch, train_hist, valid_hist, name, log_dir)
    """
    state = {
        "model": model.state_dict(),
        "acc": acc,
        "best_acc": best_acc,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_hist": train_hist,
        "valid_hist": valid_hist,
    }
    torch.save(state, os.path.join(log_dir, f"checkpoint/{name}.pth"))


def angle_between(rot1: torch.Tensor, rot2: torch.Tensor):
    """
    Calculate the angle between two rotations.

    Parameters:
        rot1: first rotation
        rot2: second rotation

    Returns:
        angle: angle between two rotations

    Example:
    >>> angle = angle_between(rot1, rot2)
    """
    rot1 = r6d2rot_mat(rot1).float()
    rot2 = r6d2rot_mat(rot2).float()
    offsets = rot1.transpose(-1, -2).matmul(rot2)
    angles = roma.rotmat_to_rotvec(offsets).norm(dim=-1) * 180 / np.pi
    return angles.mean()


class AverageMeter:
    """
    Save and calculate metric average.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Reset values.
        """
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: torch.Tensor, n: int = 1) -> None:
        """
        Update average.

        Parameters:
            val: metric value
            n: number of values
        """
        if type(val) == torch.Tensor:
            val = val.detach().cpu().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
