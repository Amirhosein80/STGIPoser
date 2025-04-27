import gc
import glob
import os
import argparse

import numpy as np
import roma
import torch
import tqdm
import matplotlib.pyplot as plt

from src.logger import get_logger
from src.model import STIPoser
from src.smpl_model import ParametricModel
from src.config_reader import read_config

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=r"train_log/stiposer_dip/checkpoint/best_stiposer_dip.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default="data/Valid")
    return parser.parse_args()

def resume(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Resume the trained model from checkpoint
    
    Parameters
    ----------
    model: torch.nn.Module
        The model to be loaded
    path: str
        The path to the checkpoint
    
    Returns
    -------
    model: torch.nn.Module
        The loaded model
    
    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist
    """
    loaded = torch.load(path)
    model.load_state_dict(loaded["model"])

    acc = loaded["acc"]
    best_acc = loaded["best_acc"]
    logger.info(f"Load all parameters from last checkpoint :)")
    logger.info(f" accuracy {acc} and best accuracy {best_acc} :)")
    return model


def load_data(path: str, device: str) -> dict[str, torch.Tensor]:
    """
    Load the data from the npz path
    
    Parameters
    ----------
    path: str
        The path to the npz file
    device: str
        The device to load the data
    
    Returns
    -------
    data: dict[str, torch.Tensor]
        The data loaded from the npz file
    
    Raises
    ------
    FileNotFoundError
        If the npz file does not exist
    """
    data = dict(np.load(path))
    for key in data.keys():
        data[key] = torch.from_numpy(data[key]).float().to(device)
    return data


def joint_distance(joint1: torch.Tensor, joint2: torch.Tensor) -> torch.Tensor:
    """
    Calculate distance between prediction joints and ground truth 
    
    Parameters
    ----------
    joint1: torch.Tensor
        The prediction joints
    joint2: torch.Tensor
        The ground truth joints
    
    Returns
    -------
    distance: torch.Tensor
        The distance between the prediction joints and the ground truth joints in cm
    """
    offset_from_p_to_t = (joint2[:, 0] - joint1[:, 0]).unsqueeze(1)
    je = (joint1 + offset_from_p_to_t - joint2).norm(dim=2)
    return je.mean() * 100


def jitter_error(joint1: torch.Tensor, fps: float = 60.) -> torch.Tensor:
    """
    Calculate jitter error for prediction joints
    
    Parameters
    ----------
    joint1: torch.Tensor
        The prediction joints
    fps: float
        The frequency of the data
    
    Returns
    -------
    jitter: torch.Tensor
        The jitter error of the prediction joints in km/s^3
    """
    je = joint1[3:] - 3 * joint1[2:-1] + 3 * joint1[1:-2] - joint1[:-3]
    jitter = (je * (fps ** 3)).norm(dim=2)
    return jitter.mean() / 1000


def angle_between(rot1: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
    """
    Calculate angle between prediction joints and ground truth
    
    Parameters
    ----------
    rot1: torch.Tensor
        The prediction joints
    rot2: torch.Tensor
        The ground truth
    
    Returns
    -------
    angular_error: torch.Tensor
        The angular error of the prediction joints and the ground truth in degrees
    """
    offsets = rot1.transpose(-1, -2).matmul(rot2)
    angles = roma.rotmat_to_rotvec(offsets).norm(dim=-1) * 180 / np.pi
    return angles.mean().item()


def trans_error(trans1, trans2, end_index) -> torch.Tensor:
    """
    Calculate translation error for prediction and ground truth
    
    Parameters
    ----------
    trans1: torch.Tensor
        The prediction translation
    trans2: torch.Tensor
        The ground truth translation
    end_index: int
        The end index
    
    Returns
    -------
    te: torch.Tensor
        The translation error of the prediction and the ground truth in cm
    """
    te = (trans1[:end_index] - trans2[:end_index]).norm(dim=1)
    return te.mean() * 100
    

class PoseEvaluator:
    """
    Pose estimation evaluator class
    This class calculates pose estimation metrics (SIP, Angular error, Joint Distance,
                                                   Jitter Error, Translation Errors)
                                                   
    Parameters
    ----------
    model: torch.nn.Module
        The model to be evaluated
    data_files: list[str]
        The data files to be evaluated
    configs: dict
        The model config
    
    Examples
    --------
    >>> evaluator = PoseEvaluator(model, data_files, configs)
    >>> evaluator.run()
    
    Raises
    ------
    AssertionError
        If no data files are provided
    """

    def __init__(self, model: torch.nn.Module, data_files, configs: dict) -> None:
        
        assert len(data_files) != 0, "No data files provided"

        self.data_files = data_files
        self.model = model
        
        self.device = configs["training"]["device"]
        self.sip_idx = configs["smpl"]["sip_joints"]
        self.ignored_joint_mask = torch.tensor([0] + configs["smpl"]["ignored_joints"])
        self.use_translation = configs["model"]["use_translation"]
        self.use_uwb = configs["model"]["use_uwb"]

        self.pose_errors = []
        self.tran_cum_errors = {window_size: [] for window_size in list(range(1, 8))}


    def run(self) -> None:
        """
        calculate pose estimation metrics for each data in dataset :)
        """
        loop = tqdm.tqdm(self.data_files)
        for data_file in loop:
            self.eval(data_file)
        
        self.pose_errors = torch.stack(self.pose_errors, dim=0)

        print()
        print("=" * 100)
        print()
        logger.info("Model Result")
        mean, std = self.pose_errors.mean(dim=0), self.pose_errors.std(dim=0)
        logger.info(f"SIP Error (deg): {(mean[0].item(), std[0].item())}, Ang Error (deg): {(mean[1].item(), std[1].item())}")
        logger.info(f"Joint Error (cm): {(mean[2].item(), std[2].item())}, Jitter Error (km/s^3): {(mean[3].item(), std[3].item())}")
        
        if self.use_translation:
            logger.info(f"Translation 2s Error (cm): {(mean[4].item(), std[4].item())}, ")
            logger.info(f"Translation 5s Error (cm): {(mean[5].item(), std[5].item())}, ")
            logger.info(f"Translation 10s Error (cm): {(mean[6].item(), std[6].item())}, ")
            logger.info(f"Translation All Error (cm): {(mean[7].item(), std[7].item())}")
            
            plt.plot([0] + [error for error in self.tran_cum_errors.keys()], [0] + [torch.tensor(error).mean() for error in self.tran_cum_errors.values()])
            plt.legend(fontsize=15)
            plt.show()
            logger.info("Translation Cumulative Error from 1 to 7 meters")
            logger.info([error for error in self.tran_cum_errors.keys()])
            logger.info([torch.tensor(error).mean() for error in self.tran_cum_errors.values()])

        print()
        print("=" * 100)
        print()
        
        

        # torch.save({'acc': self.accs, 'ori': self.oris, 'pose': self.pose, 'tran': self.tran}, 'test.pt')


    def eval(self, file) -> None:
        """
        calculate pose estimation metrics for dataset :)
        :param file: data paths :)
        """
        self.model.eval()
        datas = load_data(file, device=self.device)

        gt_pose = datas["pose"]
        gt_trans = datas["trans"]

        # gt_pose[:, self.ignored_joint_mask] = torch.eye(3, device=gt_pose.device)
        # global_t, joint_t, _, _ = self.model.smpl_model.forward_kinematics(pose=gt_pose, tran=gt_trans)

        if "imu_acc" in datas.keys():
            imu_acc = datas["imu_acc"]
            imu_ori = datas["imu_ori"]

        else:
            imu_acc = datas["vacc"]
            imu_ori = datas["vrot"]
        
        if self.use_uwb:
            uwb=datas["uwb"]

        else:
            uwb = None
        
        with torch.inference_mode():

            preds = self.model.forward_offline(imu_acc=imu_acc, imu_ori=imu_ori, uwb=uwb)
            p_pose = preds[0].detach()
            if self.use_translation:
                p_trans = preds[1].detach()
            else:
                p_trans = None
                gt_trans = None

            p_pose[:, self.ignored_joint_mask] = gt_pose[:, self.ignored_joint_mask]
            grot_p, joint_p, _, _ = self.model.smpl_model.forward_kinematics(pose=p_pose, tran=p_trans)
            grot_t, joint_t, _, _ = self.model.smpl_model.forward_kinematics(pose=gt_pose, tran=gt_trans)

            metrics = [
                angle_between(grot_p.detach()[10:, self.sip_idx], grot_t[10:, self.sip_idx]),
                angle_between(grot_p.detach()[10:], grot_t[10:]),
                joint_distance(joint_p[10:], joint_t[10:]),
                jitter_error(joint_p[10:]),
            ]
            if self.use_translation:
                metrics.append(trans_error(p_trans.detach(), gt_trans, end_index=60 * 2))
                metrics.append(trans_error(p_trans.detach(), gt_trans, end_index=60 * 5))
                metrics.append(trans_error(p_trans.detach(), gt_trans, end_index=60 * 10))
                metrics.append(trans_error(p_trans.detach(), gt_trans, end_index=-1))
           
            

                move_distance_t = torch.zeros(gt_trans.shape[0])
                v = (gt_trans[1:] - gt_trans[:-1]).norm(dim=1)
                for j in range(len(v)):
                    move_distance_t[j + 1] = move_distance_t[j] + v[j]

                    for window_size in self.tran_cum_errors.keys():
                        # find all pairs of start/end frames where gt moves `window_size` meters
                        frame_pairs = []
                        start, end = 0, 1
                        while end < len(move_distance_t):
                            if move_distance_t[end] - move_distance_t[start] < window_size:
                                end += 1
                            else:
                                if len(frame_pairs) == 0 or frame_pairs[-1][1] != end:
                                    frame_pairs.append((start, end))
                                start += 1

                        # calculate mean distance error
                        errs = []
                        for start, end in frame_pairs:
                            vel_p = p_trans[end] - p_trans[start]
                            vel_t = gt_trans[end] - gt_trans[start]
                            errs.append((vel_t - vel_p).norm() / (move_distance_t[end] - move_distance_t[start]) * window_size)
                        if len(errs) > 0:
                            self.tran_cum_errors[window_size].append(sum(errs) / len(errs))
            
            self.pose_errors.append(torch.tensor(metrics))



        del preds, p_pose, joint_p, grot_p, metrics
        torch.cuda.empty_cache()
        gc.collect()
        
if __name__ == "__main__":
    args = parse_args()
    
    logger.info("Start Validationw")
    
    files = glob.glob(os.path.join(args.data_dir, "*/*/*.npz"))
    configs = read_config(args.config)
    configs["training"]["device"] = args.device
        
    smpl_model = ParametricModel(configs["smpl"]["file"], device=args.device)
    model = STIPoser(configs=configs, smpl_model=smpl_model)
    model = resume(model, args.checkpoint)
    model = model.to(args.device)
    
   
    
    evaluator = PoseEvaluator(model, files, configs)
    # print(evaluator.ignored_joint_mask)
    evaluator.run()
