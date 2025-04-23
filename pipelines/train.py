from argparse import ArgumentParser

import comet_ml
import torch
import torchinfo

from src.config_reader import read_config
from src.dataset import RandomNoise, STIPoseDataset
from src.logger import get_logger
from src.loss import MotionLoss
from src.model import STIPoser
from src.optimizers import get_optimizer
from src.smpl_model import ParametricModel
from src.trainer import Trainer
from src.utils import set_comet, set_seed, setup_env, to_device, create_log_dir
from src.base_model import run_benchmark

logger = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=r"config\config.yaml")
    return parser.parse_args()


def main():
    """
    Train the model.
    """
    args = parse_args()

    logger.info(f"Start training with config: {args.config}")

    logger.info(f"Read config from {args.config}")
    configs = read_config(args.config)

    logger.info(f"Setup environment")
    setup_env()
    set_seed(0)

    create_log_dir(configs)
    experiment = set_comet(configs)
    # experiment.log_artifact(args.config)

    transform = RandomNoise(
        std=configs["dataset"]["random_noise"]["std"],
        p=configs["dataset"]["random_noise"]["p"],
    )

    train_dataset = STIPoseDataset(configs, "Train", transform)
    valid_dataset = STIPoseDataset(configs, "Valid")

    train_loader = train_dataset.get_data_loader()
    valid_loader = valid_dataset.get_data_loader()

    data_loader = {"train": train_loader, "valid": valid_loader}

    smpl_model = ParametricModel(configs["smpl"]["file"], configs["training"]["device"])

    model = STIPoser(configs, smpl_model=smpl_model)
    model.to(configs["training"]["device"])

    batch = next(iter(train_loader))
    batch = to_device(batch, configs["training"]["device"])
    torchinfo.summary(
        model=model, input_data=[batch], device=configs["training"]["device"], verbose=1
    )

    optimizer, scheduler = get_optimizer(configs=configs, model=model)
    criterion = MotionLoss(configs)

    trainer = Trainer(
        dataloaders=data_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        configs=configs,
        experiment=experiment,
    )

    trainer.train()


if __name__ == "__main__":
    main()
