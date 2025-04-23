import time

import torch
import torch.nn as nn
import torch.nn.functional as f

from src.blocks import CausalAvg
from src.smpl_model import ParametricModel
from src.logger import get_logger

logger = get_logger(__name__)

class BaseModel(nn.Module):
    """
    Base model class for motion-related tasks using SMPL body model.

    Handles core functionality including rotation conversions, parameter grouping,
    and integration with SMPL kinematics. Inherits from PyTorch Module.

    Parameters
    ----------
    configs : dict
        Configuration dictionary containing:
        - smpl: reduced/ignored/leaf joints configuration
        - training: device information
    smpl_model : ParametricModel
        Initialized SMPL body model instance

    Examples
    --------
    >>> config = {"smpl": {...}, "training": {"device": "cuda"}}
    >>> smpl = ParametricModel("smpl_file.pkl")
    >>> model = BaseModel(config, smpl)
    """

    def __init__(self, configs: dict, smpl_model: ParametricModel) -> None:

        super().__init__()

        self.smpl_model = smpl_model

        self.reduced = configs["smpl"]["reduced_joints"]
        self.ignored = configs["smpl"]["ignored_joints"]
        self.leaf = configs["smpl"]["imu_joints"]
        self.imu = configs["smpl"]["leaf_joints"]
        self.avg_kernel = configs["model"]["avg_kernel"]

        self.device = configs["training"]["device"]

        adj_dict = torch.load(configs["model"]["adj_path"])
        self.adj_imu = adj_dict["imu"].float().to(self.device)

        if configs["model"]["use_translation"]:
            self.adj_imu2parts = adj_dict["imu2parts_wt"].float().to(self.device)
        else:
            self.adj_imu2parts = adj_dict["imu2parts"].float().to(self.device)

        self.casual_avg = CausalAvg(self.avg_kernel)

    def glb_6d_to_full_local_mat(
        self, glb_pose: torch.Tensor, sensor_rot: torch.Tensor
    ) -> torch.Tensor:
        """
        Converts global 6D rotations to full local rotation matrices for SMPL joints.

        Parameters
        ----------
        glb_pose : torch.Tensor
            Global 6D rotations tensor of shape (T, N, 6)
        sensor_rot : torch.Tensor
            Sensor rotation matrices for specific joints, shape (T, 6, 3, 3)

        Returns
        -------
        torch.Tensor
            Full local rotation matrices of shape (T, 24, 3, 3)

        Examples
        --------
        >>> global_pose = torch.randn(100, 15, 6)
        >>> sensor_rot = torch.randn(100, 6, 3, 3)
        >>> local_mats = model.glb_6d_to_full_local_mat(global_pose, sensor_rot)
        """
        T, N, _ = glb_pose.shape

        glb_pose = r6d2rot_mat(glb_pose).view(T, N, 3, 3)
        ign_pose = (
            torch.eye(3, device=glb_pose.device)
            .reshape(1, 1, 3, 3)
            .repeat(1, len(self.ignored), 1, 1)
        )

        global_full_pose = torch.eye(3, device=glb_pose.device).repeat(T, 24, 1, 1)
        global_full_pose[:, self.reduced[1:]] = glb_pose

        global_full_pose[:, [0, 4, 5, 15, 18, 19]] = sensor_rot

        pose = self.smpl_model.inverse_kinematics_R(global_full_pose)
        pose[:, self.ignored] = ign_pose

        return pose

    def get_params(self, lr: float, weight_decay: float) -> list[dict]:
        """
        Organizes model parameters for optimizer configuration.

        Parameters
        ----------
        lr : float
            Learning rate for parameter groups
        weight_decay : float
            Weight decay value for parameters with dimension > 1

        Returns
        -------
        list[dict]
            List of parameter groups with learning rate and weight decay settings

        Examples
        --------
        >>> optimizer = AdamW(model.get_params(1e-4, 0.01))
        """
        params_wd = []
        params_nwd = []

        for p in self.parameters():
            if p.dim == 1:
                params_nwd.append(p)
            else:
                params_wd.append(p)

        params = [
            {"params": params_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": params_nwd, "lr": lr, "weight_decay": 0},
        ]

        return params

    def normalize(
        self,
        imu_acc: torch.Tensor,
        imu_ori: torch.Tensor,
        casual_state: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalizes the linear accelration and orientations.

        Parameters
        ----------
        imu_acc : torch.Tensor
            Sensors's linear accelration with shape(batch size, seq length, num sensors, 3)
        imu_ori : torch.Tensor
            Sensors's orientations with shape(batch size, seq length, num sensors, 3, 3)
        casual_state : Optional [torch.Tensor]
            The state of casual average pooling with shape(batch size, kernel size - 1, num sensors, 3). Default is None

        Returns
        -------
         tuple[torch.Tensor, torch.Tensor]
            A normalized tensor for input of our neural network and the casual average pooling state for offline mode.

        Examples
        --------
        >>> x, casual_state = normalize(imu_acc, imu_ori, casual_state)
        """

        imu_acc, casual_state = self.casual_avg(imu_acc, state=casual_state)

        imu_acc[:, :, 1:] = imu_acc[:, :, 1:] - imu_acc[:, :, 0:1]
        imu_acc = imu_acc.matmul(imu_ori[:, :, 0])
        imu_ori[:, :, 1:] = (
            imu_ori[:, :, :1].transpose(-1, -2).matmul(imu_ori[:, :, 1:])
        )
        return torch.cat((imu_acc, imu_ori.flatten(-2)), dim=-1), casual_state


def rot_mat2r6d(rot_mat: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D representation.

    Parameters
    ----------
    rot_mat : torch.Tensor
        Input rotation matrices of shape (..., 3, 3)

    Returns
    -------
    torch.Tensor
        6D rotation representation of shape (..., 6)

    Examples
    --------
    >>> rot_matrix = torch.randn(32, 24, 3, 3)
    >>> r6d = rot_mat2r6d(rot_matrix)
    """
    r = rot_mat[..., :2].transpose(-1, -2)
    return r.reshape(*rot_mat.shape[:-2], 6)


def r6d2rot_mat(r6d: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation back to rotation matrices.

    Parameters
    ----------
    r6d : torch.Tensor
        6D rotation representation of shape (..., 6)

    Returns
    -------
    torch.Tensor
        Rotation matrices of shape (..., 3, 3)

    Examples
    --------
    >>> r6d_pose = torch.randn(32, 24, 6)
    >>> rot_matrix = r6d2rot_mat(r6d_pose)
    """
    x_raw = r6d[..., 0:3]
    y_raw = r6d[..., 3:6]

    x = f.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = f.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack((x, y, z), dim=-1)
    return matrix


def run_benchmark(model: nn.Module, datas: tuple[torch.Tensor, torch.Tensor]) -> None:
    """
    Benchmarks model inference speed and computes latency/FPS metrics.

    Parameters
    ----------
    model : nn.Module
        Model to benchmark
    datas : tuple[torch.Tensor, torch.Tensor]
        Input data tuple for model inference

    Examples
    --------
    >>> test_data = (torch.randn(32, 100), torch.randn(32, 100))
    >>> run_benchmark(model, test_data)
    """
    elapsed = 0
    model.eval()
    num_batches = 100
    print("Start benchmarking...")
    with torch.inference_mode():
        for _ in range(10):
            model(*datas)
        print("Benchmarking...")
        for i in range(10):
            if i < num_batches:
                start = time.time()
                _ = model(*datas)
                end = time.time()
                elapsed = elapsed + (end - start)
            else:
                break
    num_images = 10
    latency = elapsed / num_images * 1000
    fps = 1000 / latency
    print(f"Elapsed time: {latency:.3} ms, FPS: {fps:.3}")
    logger.info(f"Elapsed time: {latency:.3} ms, FPS: {fps:.3}")


if __name__ == "__main__":
    pass
