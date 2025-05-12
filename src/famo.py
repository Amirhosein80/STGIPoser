import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config_reader import read_config
from src.logger import get_logger

logger = get_logger(__name__)


class FAMO:
    """
    FAMO: Fast Adaptive Multitask Optimization

    Parameters
    ----------
    config: dict
        The configuration dictionary
    model_optimizer: optim.Optimizer
        The model optimizer

    Examples
    --------
    >>> famo = FAMO(config, model_optimizer)

    Raises
    ------
    NotImplementedError
        If the optimizer is not implemented
    """

    def __init__(
        self,
        config: dict,
        model_optimizer: optim.Optimizer,
    ):

        self.model_optimizer = model_optimizer

        self.n_tasks = config["famo"]["n_tasks"]
        self.device = config["training"]["device"]

        self.min_losses = torch.zeros(self.n_tasks).to(self.device)
        self.weights = nn.Parameter(torch.zeros(self.n_tasks))
        self.softmax = nn.Softmax(dim=-1)
        self.prev_loss = None

        if config["optimizer"]["name"] == "adamw":
            self.loss_optimizer = optim.AdamW(
                self.weights,
                lr=config["famo"]["lr"],
                betas=config["optimizer"]["betas"],
                weight_decay=config["famo"]["gamma"],
            )
        elif config["optimizer"]["name"] == "sgd":
            self.loss_optimizer = optim.SGD(
                self.weights,
                lr=config["famo"]["lr"],
                momentum=config["optimizer"]["momentum"],
                weight_decay=config["famo"]["gamma"],
            )

        else:
            logger.error(f"Invalid optimizer: {config['optimizer']['name']}")
            raise ValueError(f"Invalid optimizer: {config['optimizer']['name']}")

    def set_min_losses(self, losses: torch.Tensor):
        """
        Set the minimum losses

        Parameters
        ----------
        losses: torch.Tensor
            The losses

        Examples
        --------
        >>> famo = FAMO(config, model_optimizer)
        >>> famo.set_min_losses(losses)
        """

        self.min_losses = losses

    def get_weighted_loss(self, losses: torch.Tensor):
        """
        Get the weighted loss

        Parameters
        ----------
        losses: torch.Tensor
            The losses

        Returns
        -------
        loss: torch.Tensor
            The weighted loss

        Examples
        --------
        >>> famo = FAMO(config, model_optimizer)
        >>> loss = famo.get_weighted_loss(losses)
        """
        self.prev_loss = losses
        z = self.softmax(self.weights)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss

    def update(self, curr_loss):
        """
        Update the weights

        Parameters
        ----------
        curr_loss: torch.Tensor
            The current loss

        Examples
        --------
        >>> famo = FAMO(config, model_optimizer)
        >>> famo.update(curr_loss)
        """
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - (
            curr_loss - self.min_losses + 1e-8
        ).log()

        with torch.enable_grad():
            d = torch.autograd.grad(
                self.softmax(self.weights), self.weights, grad_outputs=delta.detach()
            )[0]

        self.loss_optimizer.zero_grad()
        self.weights.grad = d
        self.loss_optimizer.step()

    def backward(
        self,
        losses: torch.Tensor,
    ):
        """
        Backward the loss

        Parameters
        ----------
        losses: torch.Tensor
            The losses

        Examples
        --------
        >>> famo = FAMO(config, model_optimizer)
        >>> loss = famo.backward(losses)
        """
        loss = self.get_weighted_loss(losses=losses)
        loss.backward()
        return loss

    def get_state_dicts(self):
        """
        Get the state dicts

        Returns
        -------
        state_dicts: dict
            The state dicts

        """
        return {
            "weights": self.weights.state_dict(),
            "min_losses": self.min_losses.state_dict(),
            "prev_loss": self.prev_loss.state_dict(),
            "loss_optimizer": self.loss_optimizer.state_dict(),
        }

    def load_state_dicts(self, state_dicts: dict):
        """
        Load the state dicts
        """
        self.weights = state_dicts["weights"]
        self.min_losses = state_dicts["min_losses"]
        self.prev_loss = state_dicts["prev_loss"]
        self.loss_optimizer.load_state_dict(state_dicts["loss_optimizer"])
