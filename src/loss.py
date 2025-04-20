import torch
import torch.nn as nn

from logger import get_logger
from base_model import rot_mat2r6d


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
        self.sml1_loss = nn.SmoothL1Loss(beta=self.configs["loss"]["smooth_l1_beta"])
        self.trans_weight = self.configs["loss"]["trans_weight"]
        self.aux_weight = self.configs["loss"]["aux_weight"]

    def forward(self, predicts: dict, targets: dict) -> tuple[torch.Tensor, torch.Tensor]:
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
                    label = targets[tars]
            else:
                label = targets[tars][:, :, inds]

            if key == "out_rot":
                pose_loss += self.sml1_loss(predicts[key], label)
            elif key == "out_trn":
                trans_loss += self.sml1_loss(predicts[key], label)
            else:
                aux_loss += self.sml1_loss(predicts[key].reshape(-1, 3), label.reshape(-1, 3))

        return pose_loss, trans_loss, aux_loss
