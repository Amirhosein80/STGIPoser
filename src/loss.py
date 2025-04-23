import torch
import torch.nn as nn

from src.base_model import rot_mat2r6d
from src.logger import get_logger


class MotionLoss(nn.Module):
    """
    Loss function for pose and translation between predicted and ground truth

    Parameters:
        configs: configs for loss functions

    Example:
    >>> configs = {
            "loss": {
                "smooth_l1_beta": 0.01,
                "trans_weight": 0.1,
                "aux_weight": 0.1
            }
        }
    >>> criterion = MotionLoss(configs)
    >>> criterion(predicts, targets)
    """

    def __init__(self, configs: dict) -> None:

        super().__init__()
        self.configs = configs
        self.pose_loss = nn.SmoothL1Loss(beta=self.configs["loss"]["pose_beta"])
        self.trans_loss = nn.SmoothL1Loss(beta=self.configs["loss"]["trans_beta"])
        self.aux_loss = nn.SmoothL1Loss(beta=self.configs["loss"]["aux_beta"])
        self.trans_weight = self.configs["loss"]["trans_weight"]
        self.aux_weight = self.configs["loss"]["aux_weight"]

    def forward(
        self, predicts: dict, targets: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate losses for pose and translation between predicted and ground truth

        Parameters:
            predicts: model predictions
            targets: ground truth
        """
        pose_loss = 0
        trans_loss = 0
        aux_loss = 0

        for key, val in predicts.items():
            inds = self.configs["loss"][key]["indexs"]
            tars = self.configs["loss"][key]["target"]

            if tars == "grot":
                gt = rot_mat2r6d(targets[tars])[:, :, inds]
            elif tars == "trans":
                gt = targets[tars]
            else:
                gt = targets[tars][:, :, inds]

            if "out_rot" in key:
                pose_loss += self.pose_loss(predicts[key], gt)
            elif "out_trans" in key:
                trans_loss += self.trans_loss(predicts[key], gt)
            else:
                aux_loss += self.aux_loss(
                    predicts[key], gt
                )

        main_loss = (
            pose_loss + (self.trans_weight * trans_loss) + (self.aux_weight * aux_loss)
        )
        return (
            main_loss,
            pose_loss,
            (trans_loss * self.trans_weight),
            (aux_loss * self.aux_weight),
        )
