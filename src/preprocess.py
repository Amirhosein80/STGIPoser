# codes from https://github.com/Xinyu-Yi/PIP.git and https://github.com/dx118/dynaip.git

import gc
import glob
import math
import os
import pickle

import numpy as np
import roma
import torch
from tqdm import tqdm

from logger import get_logger
from preprocess_utils import (
    AMASS_ROT,
    DIP_IMU_MASK,
    IG_JOINTS,
    TC_IMU_MASK,
    V_MASK,
    XSENS_IMU_MASK,
    _syn_acc,
    _syn_uwb,
    _syn_vel,
    read_mvnx,
    read_xlsx,
    virginia_clip_config,
)
from smpl_model import ParametricModel

logger = get_logger(__name__)


def process_dip(smpl_path: str, dip_path: str, folder: str = "Test") -> None:
    """
    Processes DIP IMU data files using a parametric body model and saves the processed outputs.

    This function reads DIP IMU data from .pkl files located under the specified DIP directory and
    processes the data by cleaning NaN values in the IMU orientation and acceleration arrays, trimming
    the sequences, and computing forward kinematics using a body model initialized with the SMPL file.
    It synchronizes computed joint velocities and UWB signals, then saves the processed results in a .npz
    file format into a target folder derived from the input file path.

    Parameters
    ----------
    smpl_path : str
        Path to the SMPL file used to initialize the parametric body model.
    dip_path : str
        Directory path where the DIP data files (.pkl) are stored.
    folder : str
        Folder name to be used in the path for the processed data output.
        Default is "Test".

    Returns
    -------
    None
        This function does not return any value. The processed data is saved directly to disk.

    Examples
    --------
    >>> process_dip("path/to/smpl_file", "path/to/dip_data", folder="Test")
    """
    discread_files = []
    body_model = ParametricModel(smpl_path, device="cuda")
    files = glob.glob(os.path.join(dip_path, "*/*.pkl"))
    loop = tqdm(files)
    for file in loop:

        with open(file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        ori = torch.from_numpy(data["imu_ori"]).float()[:, DIP_IMU_MASK].to("cuda")
        acc = torch.from_numpy(data["imu_acc"]).float()[:, DIP_IMU_MASK].to("cuda")
        pose = torch.from_numpy(data["gt"]).float().view(-1, 24, 3).to("cuda")
        for _ in range(4):
            acc[1:].masked_scatter_(
                torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])]
            )
            ori[1:].masked_scatter_(
                torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])]
            )
            acc[:-1].masked_scatter_(
                torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])]
            )
            ori[:-1].masked_scatter_(
                torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])]
            )

        if (
            torch.isnan(acc).sum() != 0
            or torch.isnan(ori).sum() != 0
            or torch.isnan(pose).sum() != 0
        ):
            print(f"DIP-IMU: {file} has too much nan! Discard!")
            continue

        acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
        length = pose.shape[0]
        trans = torch.zeros(pose.shape[0], 3).to(pose.device)

        rot_mat = roma.rotvec_to_rotmat(pose).view(-1, 24, 3, 3)
        rot_mat[:, IG_JOINTS] = torch.eye(3, device="cuda")

        grot, joint, vert, ljoint = body_model.forward_kinematics(
            pose=rot_mat, tran=trans, calc_mesh=True
        )

        jvel = _syn_vel(joint, grot[:, 0:1])
        out_uwb = _syn_uwb(vert[:, V_MASK])

        targets = {
            "imu_acc": acc.cpu().numpy(),
            "imu_ori": ori.cpu().numpy(),
            "pose": rot_mat.cpu().numpy(),
            "grot": grot.cpu().numpy(),
            "trans": trans.cpu().numpy(),
            "jvel": jvel.cpu().numpy(),
            "uwb": out_uwb.cpu().numpy(),
        }

        target_folder = (
            file.split(os.path.basename(file))[0]
            .replace("DIP_IMU", "DIP_process_data")
            .replace("raw data", folder)
        )
        os.makedirs(target_folder, exist_ok=True)

        np.savez(
            os.path.join(target_folder, os.path.basename(file).replace(".pkl", "")),
            **targets,
        )

        loop.set_description(
            f"Processed {file}, discrete num {len(discread_files)}, length {length}"
        )
        del pose, trans, rot_mat, grot, joint, vert, targets
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("DIP process done!")


def process_total_capture(smpl_path: str, tc_path: str, folder: str = "Valid") -> None:
    """
    Processes Total Capture IMU and pose data files using a parametric body model and saves the processed outputs.

    This function reads Total Capture data from .pkl files under the specified Total Capture directory.
    It loads IMU orientation and acceleration data, locates the corresponding pose files, and adjusts the data
    based on the captured framerate. The function computes a body model's forward kinematics, synchronizes the
    resulting joint velocities and UWB signals with the data, and finally saves the processed results in an
    .npz file format into a designated folder structure.

    Parameters
    ----------
    smpl_path : str
        Path to the SMPL file used to initialize the parametric body model.
    tc_path : str
        Directory path where the Total Capture data files (.pkl) are stored.
    folder : str
        Folder name to be used for the processed data output. Default is "Valid".

    Returns
    -------
    None
        This function does not return any value. All processed data is saved directly to disk.

    Examples
    --------
    >>> process_total_capture("path/to/smpl_file", "path/to/total_capture_data", folder="Valid")
    """
    discread_files = []
    body_model = ParametricModel(smpl_path, device="cuda")
    files = glob.glob(os.path.join(tc_path, "*/*.pkl"))
    loop = tqdm(files)
    for file in loop:

        with open(file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        ori = torch.from_numpy(data["ori"]).float()[:, TC_IMU_MASK].to("cuda")
        acc = torch.from_numpy(data["acc"]).float()[:, TC_IMU_MASK].to("cuda")

        pose_path = file.split("TotalCapture_Real_60FPS")[0]
        pose_file = os.path.basename(file).split("_")
        subject = pose_file[0].upper()
        pose_file = os.path.join(
            pose_path,
            "TotalCapture",
            pose_file[0].lower(),
            f"{pose_file[1].split('.')[0]}_poses.npz",
        )
        if not os.path.exists(pose_file):
            print(f"No trans file found for {file}")
            continue
        cdata = np.load(pose_file)
        framerate = int(cdata["mocap_framerate"])
        if framerate == 120 or framerate == 100:
            step = 2
        elif framerate == 60 or framerate == 59 or framerate == 50:
            step = 1
        elif framerate == 250:
            step = 5
        elif framerate == 150:
            step = 3
        else:
            print(f"framerate {framerate} not supported in {os.path.basename(file)}")
            discread_files.append(file)
            continue

        trans = cdata["trans"][::step].astype(np.float32)
        shape = torch.tensor(cdata["betas"][:10]).float().to("cuda")
        trans = torch.tensor(trans).to("cuda")
        trans = trans - trans[0:1]

        pose = cdata["poses"][::step].astype(np.float32).reshape(-1, 52, 3)
        pose[:, 23] = pose[:, 37]
        pose = pose[:, :24]
        pose = torch.tensor(pose, device="cuda")

        pose[:, 0] = roma.rotmat_to_rotvec(
            AMASS_ROT.matmul(roma.rotvec_to_rotmat(pose[:, 0]))
        )

        trans = AMASS_ROT.matmul(trans.unsqueeze(-1)).view_as(trans)

        if acc.shape[0] < trans.shape[0]:
            trans = trans[: acc.shape[0]]
            pose = pose[: acc.shape[0]]

        elif acc.shape[0] > pose.shape[0]:
            acc = acc[: pose.shape[0]]
            ori = ori[: pose.shape[0]]

        assert trans.shape[0] == acc.shape[0]

        length = acc.shape[0]
        rot_mat = roma.rotvec_to_rotmat(pose).view(-1, 24, 3, 3)

        rot_mat[:, IG_JOINTS] = torch.eye(3, device="cuda")

        grot, joint, vert, ljoint = body_model.forward_kinematics(
            pose=rot_mat, tran=trans, shape=shape, calc_mesh=True
        )

        vacc = _syn_acc(vert[:, V_MASK])
        d = vacc.mean(dim=0, keepdim=True) - acc.mean(dim=0, keepdim=True)
        acc = acc + d

        jvel = _syn_vel(joint, grot[:, 0:1])

        out_uwb = _syn_uwb(vert[:, V_MASK])

        targets = {
            "imu_acc": acc.cpu().numpy(),
            "imu_ori": ori.cpu().numpy(),
            "pose": rot_mat.cpu().numpy(),
            "trans": trans.cpu().numpy(),
            "grot": grot.cpu().numpy(),
            "jvel": jvel.cpu().numpy(),
            "uwb": out_uwb.cpu().numpy(),
        }

        target_folder = (
            file.split("TotalCapture_Real_60FPS")[0]
            .replace("TotalCapture", "TC_process_data")
            .replace("raw data", folder)
        )
        os.makedirs(os.path.join(target_folder, subject), exist_ok=True)

        np.savez(
            os.path.join(
                target_folder, subject, os.path.basename(file).replace(".pkl", "")
            ),
            **targets,
        )

        loop.set_description(
            f"Processed {file}, discrete num {len(discread_files)}, length {length}"
        )
        del pose, trans, rot_mat, grot, joint, vert, targets
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("Total Capture process done!")


def process_emonike(smpl_path: str, emonike_path: str, folder: str = "Train") -> None:
    """
    Processes Emokine dataset MVNX files using a parametric body model and saves processed outputs.

    This function reads motion capture data from .mvnx files, extracts IMU orientations and accelerations,
    computes forward kinematics using the SMPL body model, and generates synthetic joint velocities and UWB signals.
    The processed data is saved in .npz format with adjusted directory structure.

    Parameters
    ----------
    smpl_path : str
        Path to the SMPL model file for body kinematics computation.
    emonike_path : str
        Directory path containing Emokine dataset .mvnx files.
    folder : str
        Output folder name for processed data. Default is "Train".

    Returns
    -------
    None
        Data is saved to disk without returning a value.

    Examples
    --------
    >>> process_emonike("path/to/smpl_file", "path/to/emokine_data", folder="Train")
    """
    discread_files = []
    body_model = ParametricModel(smpl_path, device="cuda")
    files = glob.glob(os.path.join(emonike_path, "*.mvnx"))
    loop = tqdm(files)
    for file in loop:
        data = read_mvnx(file=file, smpl_file=smpl_path)
        pose = data["joint"]["orientation"].to("cuda")
        trans = data["joint"]["translation"].to("cuda")
        length = pose.shape[0]

        ori = data["imu"]["calibrated orientation"][:, XSENS_IMU_MASK].to("cuda")
        acc = data["imu"]["free acceleration"][:, XSENS_IMU_MASK].to("cuda")
        trans = trans - trans[0:1]

        grot, joint, vert, ljoint = body_model.forward_kinematics(
            pose=pose, tran=trans, calc_mesh=True
        )

        jvel = _syn_vel(joint, grot[:, 0:1])

        out_uwb = _syn_uwb(vert[:, V_MASK])

        targets = {
            "imu_acc": acc.cpu().numpy(),
            "imu_ori": ori.cpu().numpy(),
            "pose": pose.cpu().numpy(),
            "trans": trans.cpu().numpy(),
            "grot": grot.cpu().numpy(),
            "jvel": jvel.cpu().numpy(),
            "uwb": out_uwb.cpu().numpy(),
        }

        seq = os.path.basename(file).split("_")[1]
        target_folder = file.split("EmokineDataset_v1.0")[0].replace("raw data", folder)
        target_folder = os.path.join(target_folder, "EM_process_data", seq)
        os.makedirs(target_folder, exist_ok=True)

        np.savez(
            os.path.join(target_folder, os.path.basename(file).replace(".mvnx", "")),
            **targets,
        )

        loop.set_description(
            f"Processed {file}, discrete num {len(discread_files)}, length {length}"
        )
        del pose, trans, grot, joint, vert, targets, length
        torch.cuda.empty_cache()
        gc.collect()


def process_andy(smpl_path, mvnx_path, folder: str = "Train") -> None:
    """
    Processes ANDY dataset XLSX files using a parametric body model and saves outputs.

    Reads XLSX files containing IMU and motion data, computes body kinematics via the SMPL model,
    generates synthetic velocity/UWB signals, and saves results in .npz format. Skips calibration files
    and bug-marked entries during processing.

    Parameters
    ----------
    smpl_path : str
        Path to the SMPL model file.
    mvnx_path : str
        Directory path containing ANDY dataset XLSX files.
    folder : str
        Output folder name for processed data. Default is "Train".

    Returns
    -------
    None
        Processed data is written directly to disk.

    Examples
    --------
    >>> process_andy("path/to/smpl_file", "path/to/andy_data", folder="Train")
    """
    discread_files = []
    body_model = ParametricModel(smpl_path, device="cuda")
    files = glob.glob(os.path.join(mvnx_path, "*/*/*/*/*.xlsx"))
    loop = tqdm(files)
    for file in loop:
        if "calibration" in file:
            loop.set_description(f"Calibration file {os.path.basename(file)}")
            continue
        if file.split("\\")[-2][0] == "_":
            loop.set_description(f"Bug file {os.path.basename(file)}")
            discread_files.append(os.path.basename(file))
            continue
        data = read_xlsx(xsens_file_path=file, smpl_file=smpl_path)
        pose = data["joint"]["orientation"].to("cuda")
        trans = data["joint"]["translation"].to("cuda")
        length = pose.shape[0]

        trans = trans - trans[0:1]

        ori = data["imu"]["calibrated orientation"][:, XSENS_IMU_MASK].to("cuda")
        acc = data["imu"]["free acceleration"][:, XSENS_IMU_MASK].to("cuda")

        grot, joint, vert, ljoint = body_model.forward_kinematics(
            pose=pose, tran=trans, calc_mesh=True
        )

        jvel = _syn_vel(joint, grot[:, 0:1])

        out_uwb = _syn_uwb(vert[:, V_MASK])

        targets = {
            "imu_acc": acc.cpu().numpy(),
            "imu_ori": ori.cpu().numpy(),
            "pose": pose.cpu().numpy(),
            "trans": trans.cpu().numpy(),
            "grot": grot.cpu().numpy(),
            "jvel": jvel.cpu().numpy(),
            "uwb": out_uwb.cpu().numpy(),
        }

        target_folder = file.split("MTwAwinda")[0]
        split_foder = file.split("\\")[1]

        target_folder = os.path.join(
            target_folder, f"ANDY_process_data/{split_foder}"
        ).replace("raw data", folder)
        os.makedirs(target_folder, exist_ok=True)

        np.savez(
            os.path.join(target_folder, os.path.basename(file).replace(".xlsx", "")),
            **targets,
        )

        loop.set_description(
            f"Processed {file}, discrete num {len(discread_files)}, length {length}"
        )
        del pose, trans, grot, joint, vert, targets, length
        torch.cuda.empty_cache()
        gc.collect()
    logger.info("Emonike process done!")


def process_cip(smpl_path, mvnx_path, folder: str = "Train") -> None:
    """
    Processes CIP dataset MVNX files using a parametric body model.

    Extracts IMU data and joint orientations from .mvnx files, computes forward kinematics,
    synthesizes velocity/UWB signals, and saves processed data in .npz format with structured directories.

    Parameters
    ----------
    smpl_path : str
        Path to the SMPL model file.
    mvnx_path : str
        Directory path containing CIP dataset .mvnx files.
    folder : str
        Output folder name for processed data. Default is "Train".

    Returns
    -------
    None
        Data is saved to disk without returning a value.

    Examples
    --------
    >>> process_cip("path/to/smpl_file", "path/to/cip_data", folder="Train")
    """
    discread_files = []
    body_model = ParametricModel(smpl_path, device="cuda")
    files = glob.glob(os.path.join(mvnx_path, "*/*.mvnx"))
    loop = tqdm(files)
    for file in loop:
        data = read_mvnx(file=file, smpl_file=smpl_path)
        pose = data["joint"]["orientation"].to("cuda")
        trans = data["joint"]["translation"].to("cuda")
        length = pose.shape[0]

        ori = data["imu"]["calibrated orientation"][:, XSENS_IMU_MASK].to("cuda")
        acc = data["imu"]["free acceleration"][:, XSENS_IMU_MASK].to("cuda")
        trans = trans - trans[0:1]

        grot, joint, vert, ljoint = body_model.forward_kinematics(
            pose=pose, tran=trans, calc_mesh=True
        )

        jvel = _syn_vel(joint, grot[:, 0:1])

        out_uwb = _syn_uwb(vert[:, V_MASK])

        targets = {
            "imu_acc": acc.cpu().numpy(),
            "imu_ori": ori.cpu().numpy(),
            "pose": pose.cpu().numpy(),
            "trans": trans.cpu().numpy(),
            "grot": grot.cpu().numpy(),
            "jvel": jvel.cpu().numpy(),
            "uwb": out_uwb.cpu().numpy(),
        }

        seq = file.split("\\")[-2]
        target_folder = file.split("xens_mnvx")[0].replace("raw data", folder)
        target_folder = os.path.join(target_folder, "CIP_process_data", seq)
        os.makedirs(target_folder, exist_ok=True)

        np.savez(
            os.path.join(target_folder, os.path.basename(file).replace(".mvnx", "")),
            **targets,
        )

        loop.set_description(
            f"Processed {file}, discrete num {len(discread_files)}, length {length}"
        )
        del pose, trans, grot, joint, vert, targets, length
        torch.cuda.empty_cache()
        gc.collect()


def process_virginia(smpl_path, mvnx_path, folder: str = "Train") -> None:
    """
    Processes Virginia dataset MVNX files with sequence splitting.

    Reads motion data from .mvnx files, splits sequences into chunks based on predefined clip configurations,
    computes kinematics and synthetic signals, and saves results. Uses overlap and fixed sequence length
    for chunking.

    Parameters
    ----------
    smpl_path : str
        Path to the SMPL model file.
    mvnx_path : str
        Directory path containing Virginia dataset .mvnx files.
    folder : str
        Output folder name. Default is "Train".

    Returns
    -------
    None
        Processed data is saved to disk.

    Examples
    --------
    >>> process_virginia("path/to/smpl_file", "path/to/virginia_data", folder="Train")
    """
    overlap = 0
    seq_length = 24000

    discread_files = []
    body_model = ParametricModel(smpl_path, device="cuda")
    loop = tqdm(virginia_clip_config)
    for file in loop:
        name, start_idxs, end_idxs = file["name"], file["start"], file["end"]
        file = os.path.join(mvnx_path, f"{name}.mvnx")
        data = read_mvnx(file=file, smpl_file=smpl_path)
        for i, (start_id, end_id) in enumerate(zip(start_idxs, end_idxs)):
            pose_all = data["joint"]["orientation"][start_id:end_id].to("cuda")
            trans_all = data["joint"]["translation"][start_id:end_id].to("cuda")
            ori_all = data["imu"]["calibrated orientation"][
                start_id:end_id, XSENS_IMU_MASK
            ].to("cuda")
            acc_all = data["imu"]["free acceleration"][
                start_id:end_id, XSENS_IMU_MASK
            ].to("cuda")

            length = pose_all.shape[0]

            num_sequences = int(math.ceil((length - overlap) / (seq_length - overlap)))
            for idx in range(num_sequences):
                print(idx)
                start = idx * (seq_length - overlap)
                end = start + seq_length

                pose = pose_all[start:end]
                trans = trans_all[start:end]
                ori = ori_all[start:end]
                acc = acc_all[start:end]

                trans = trans - trans[0:1]

                grot, joint, vert, ljoint = body_model.forward_kinematics(
                    pose=pose, tran=trans, calc_mesh=True
                )

                jvel = _syn_vel(joint, grot[:, 0:1])

                out_uwb = _syn_uwb(vert[:, V_MASK])

                targets = {
                    "imu_acc": acc.cpu().numpy(),
                    "imu_ori": ori.cpu().numpy(),
                    "pose": pose.cpu().numpy(),
                    "trans": trans.cpu().numpy(),
                    "grot": grot.cpu().numpy(),
                    "jvel": jvel.cpu().numpy(),
                    "uwb": out_uwb.cpu().numpy(),
                }

                target_folder = file.split("mvnx-dataset")[0].replace(
                    "raw data", folder
                )
                target_folder = os.path.join(
                    target_folder, "VIR_process_data", "subjects"
                )
                os.makedirs(target_folder, exist_ok=True)

                np.savez(
                    os.path.join(
                        target_folder,
                        os.path.basename(file).replace(".mvnx", "") + f"_{i}_{idx}",
                    ),
                    **targets,
                )
                del pose, trans, grot, joint, vert

        loop.set_description(
            f"Processed {file}, discrete num {len(discread_files)}, length {length}"
        )
        # del pose, trans, grot, joint, vert, targets, length
        torch.cuda.empty_cache()
        gc.collect()


def process_unipd(smpl_path, mvnx_path, folder: str = "Train") -> None:
    """
    Processes UNIPD dataset MVNX files using a parametric body model.

    Extracts IMU data and joint information from .mvnx files, computes body kinematics,
    generates synthetic signals, and saves processed data in .npz format with organized directory structure.

    Parameters
    ----------
    smpl_path : str
        Path to the SMPL model file.
    mvnx_path : str
        Directory path containing UNIPD dataset .mvnx files.
    folder : str
        Output folder name. Default is "Train".

    Returns
    -------
    None
        Data is written directly to disk.

    Examples
    --------
    >>> process_unipd("path/to/smpl_file", "path/to/unipd_data", folder="Train")
    """
    discread_files = []
    body_model = ParametricModel(smpl_path, device="cuda")
    files = glob.glob(os.path.join(mvnx_path, "*/*.mvnx"))
    loop = tqdm(files)
    for file in loop:
        data = read_mvnx(file=file, smpl_file=smpl_path)
        pose = data["joint"]["orientation"].to("cuda")
        trans = data["joint"]["translation"].to("cuda")
        length = pose.shape[0]

        ori = data["imu"]["calibrated orientation"][:, XSENS_IMU_MASK].to("cuda")
        acc = data["imu"]["free acceleration"][:, XSENS_IMU_MASK].to("cuda")
        trans = trans - trans[0:1]

        grot, joint, vert, ljoint = body_model.forward_kinematics(
            pose=pose, tran=trans, calc_mesh=True
        )

        jvel = _syn_vel(joint, grot[:, 0:1])

        out_uwb = _syn_uwb(vert[:, V_MASK])

        targets = {
            "imu_acc": acc.cpu().numpy(),
            "imu_ori": ori.cpu().numpy(),
            "pose": pose.cpu().numpy(),
            "trans": trans.cpu().numpy(),
            "grot": grot.cpu().numpy(),
            "jvel": jvel.cpu().numpy(),
            "uwb": out_uwb.cpu().numpy(),
        }

        seq = file.split("\\")[-2]
        target_folder = file.split("unipd")[0].replace("raw data", folder)
        target_folder = os.path.join(target_folder, "MVNX_process_data", seq)
        os.makedirs(target_folder, exist_ok=True)

        np.savez(
            os.path.join(target_folder, os.path.basename(file).replace(".mvnx", "")),
            **targets,
        )

        loop.set_description(
            f"Processed {file}, discrete num {len(discread_files)}, length {length}"
        )
        del pose, trans, grot, joint, vert, targets, length
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    smpl_path = "./data/smpl/SMPL_MALE.pkl"

    dip_path = "./data/raw data/DIP_IMU"
    tc_path = "./data/raw data/TotalCapture"
    emonike_path = "./data/raw data/EmokineDataset_v1.0/Data/MVNX"
    mtw_path = "./data/raw data/MTwAwinda"
    mvnx_path = "./data/raw data/xens_mnvx/"
    virg_path = "./data/raw data/mvnx-dataset/"
    uni_path = "./data/raw data/unipd/"

    # process_dip(smpl_path=smpl_path, dip_path=dip_path)
    # process_total_capture(smpl_path=smpl_path, tc_path=tc_path)
    # process_emonike(smpl_path=smpl_path, emonike_path=emonike_path)
    # process_andy(smpl_path=smpl_path, mvnx_path=mtw_path)
    # process_cip(smpl_path=smpl_path, mvnx_path=mvnx_path)
    process_virginia(smpl_path=smpl_path, mvnx_path=virg_path)
    # process_unipd(smpl_path=smpl_path, mvnx_path=uni_path)
