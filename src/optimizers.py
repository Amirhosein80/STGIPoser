import torch
import torch.amp as amp
import torch.optim as optim

from src.config_reader import read_config
from src.logger import get_logger

logger = get_logger(__name__)


def get_optimizer(
    model: torch.nn.Module, configs: dict
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    """
    Get optimizer and scheduler

    Parameters
    ----------
    model: torch.nn.Module
        The model to optimize
    configs: dict
        The configuration dictionary

    Returns
    -------
    optimizer: torch.optim.Optimizer
        The optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
        The scheduler

    Examples
    --------
    >>> optimizer, scheduler = get_optimizer(model, configs)

    Raises
    ------
    NotImplementedError
        If the optimizer or scheduler is not implemented
    """
    logger.info(
        f"Get optimizer : {configs['optimizer']['name']} with lr : {configs['optimizer']['lr']} and scheduler : {configs['optimizer']['schedular']}"
    )

    if hasattr(model, "get_params"):
        params = model.get_params(
            lr=configs["optimizer"]["lr"],
            weight_decay=configs["optimizer"]["weight_decay"],
        )
    else:
        params = model.parameters()

    if configs["optimizer"]["name"] == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=configs["optimizer"]["lr"],
            weight_decay=configs["optimizer"]["weight_decay"],
            betas=configs["optimizer"]["adamw_betas"],
        )

    elif configs["optimizer"]["name"] == "sgd":
        optimizer = optim.SGD(
            params,
            lr=configs["optimizer"]["lr"],
            weight_decay=configs["optimizer"]["weight_decay"],
            momentum=configs["optimizer"]["momentum"],
        )

    elif configs["optimizer"]["name"] == "adam":
        optimizer = optim.Adam(params, lr=configs["optimizer"]["lr"])

    else:
        logger.error(f"Invalid optimizer: {configs['optimizer']['name']}")
        raise NotImplementedError(f"Invalid optimizer: {configs['optimizer']['name']}")

    # add scheduler
    total_iters = (
        configs["training"]["epochs"] - configs["optimizer"]["warmup"]["epochs"]
    )
    warm_iters = configs["optimizer"]["warmup"]["epochs"]

    if configs["optimizer"]["schedular"] == "poly":
        main_lr_scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=total_iters, power=0.9
        )

    elif configs["optimizer"]["schedular"] == "cos":
        main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_iters
        )

    elif configs["optimizer"]["schedular"] == "constant":
        main_lr_scheduler = optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=total_iters, factor=1.0
        )

    else:
        logger.error(f"Invalid scheduler: {configs['optimizer']['schedular']}")
        raise NotImplementedError(
            f"Invalid scheduler: {configs['optimizer']['schedular']}"
        )

    # set warmup scheduler if you use
    if configs["optimizer"]["warmup"]["epochs"] > 0:
        warm_lr_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=configs["optimizer"]["warmup"]["factor"],
            total_iters=warm_iters,
        )

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warm_lr_scheduler, main_lr_scheduler],
            milestones=[warm_iters],
        )

    else:
        scheduler = main_lr_scheduler

    return optimizer, scheduler
