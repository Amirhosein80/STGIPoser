# codes from https://github.com/Xinyu-Yi/PIP.git and
#            https://github.com/dx118/dynaip.git and


import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import roma
import torch

from smpl_model import ParametricModel

V_MASK = [3021, 1176, 4662, 411, 1961, 5424]
AMASS_ROT = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.0]]], device="cuda")
DIP_IMU_MASK = [2, 11, 12, 0, 7, 8]

TC_IMU_MASK = [5, 2, 3, 4, 0, 1]

XSENS_IMU_MASK = [0, 15, 12, 2, 9, 5]


virginia_clip_config = [
    {"name": "P1_Day_1_1", "start": [0], "end": [-1]},
    {"name": "P1_Day_1_3", "start": [21800, 35800], "end": [27000, 71800]},
    {"name": "P2_Day_1_1", "start": [0, 82000], "end": [69000, -1]},
    {"name": "P3_Day_1_1", "start": [0], "end": [50000]},
    {"name": "P3_Day_1_2", "start": [0], "end": [-1]},
    {"name": "P4_Day_1_1", "start": [0], "end": [18000]},
    {"name": "P4_Day_1_2", "start": [0], "end": [43000]},
    {"name": "P4_Day_1_3", "start": [0], "end": [18000]},
    {"name": "P5_Day_1_1", "start": [16000], "end": [34000]},
    {"name": "P6_Day_2_1", "start": [80000], "end": [110000]},
    {"name": "P10_Day_1_1", "start": [82000], "end": [100000]},
    {
        "name": "P11_Day_1_2",
        "start": [31000, 44000, 206300, 229300],
        "end": [33500, 51800, 210300, 238300],
    },
    {"name": "P13_Day_2_1", "start": [0], "end": [-1]},
    {"name": "P13_Day_2_2", "start": [0], "end": [22500]},
]

IG_JOINTS = [7, 8, 10, 11, 20, 21, 22, 23]


def _syn_acc(trans: torch.Tensor) -> torch.Tensor:
    """
    Synthesizes acceleration signals from translation data using finite difference approximation.

    Parameters
    ----------
    trans : torch.Tensor
        Input translation tensor of shape (N, 3) where N is sequence length.

    Returns
    -------
    torch.Tensor
        Synthetic acceleration tensor of same shape as input.

    Examples
    --------
    >>> acc = _syn_acc(joint_translations)
    """
    acc = torch.zeros_like(trans)
    acc[1:-1] = (trans[0:-2] + trans[2:] - 2 * trans[1:-1]) * 3600
    acc[4:-4] = (trans[0:-8] + trans[8:] - 2 * trans[4:-4]) * 3600 / 16
    return acc


def _syn_vel(trans: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
    """
    Computes synthetic velocity signals from translation data in root coordinate frame.

    Parameters
    ----------
    trans : torch.Tensor
        Translation tensor of shape (N, 3)
    root : torch.Tensor
        Root rotation matrices of shape (N, 1, 3, 3)

    Returns
    -------
    torch.Tensor
        Velocity tensor in root coordinate system, shape (N, 3)

    Examples
    --------
    >>> velocity = _syn_vel(translations, root_rotations)
    """
    vel = torch.zeros_like(trans)
    vel[1:] = trans[1:] - trans[:-1]
    vel = root.transpose(-1, -2) @ vel.unsqueeze(-1)
    return vel[..., 0] * 60


def _syn_uwb(p):
    """
    Generates synthetic UWB signals through pairwise distance computation.

    Parameters
    ----------
    p : torch.Tensor
        Vertex positions tensor of shape (N, 6, 3)

    Returns
    -------
    torch.Tensor
        Distance matrix tensor of shape (N, 6, 6)

    Examples
    --------
    >>> uwb_signals = _syn_uwb(vertex_positions)
    """
    return torch.cdist(p, p).view(-1, 6, 6)


def imu_calibration(imu_data, joints_data, n_frame, joint_mask):
    """
    Calibrates IMU orientations using reference joint data.

    Parameters
    ----------
    imu_data : torch.Tensor
        Raw IMU orientation quaternions (N, M, 4)
    joints_data : torch.Tensor
        Reference joint orientation quaternions (N, K, 4)
    n_frame : int
        Number of frames to use for calibration
    joint_mask : list
        Indices of joints corresponding to IMU positions

    Returns
    -------
    torch.Tensor
        Calibrated IMU orientation quaternions (N, M, 4)

    Examples
    --------
    >>> calibrated_ori = imu_calibration(raw_imu, joint_ori, 150, [0, 4, 6])
    """
    quat1 = imu_data[:n_frame]
    quat2 = joints_data[:n_frame, joint_mask]
    quat_off = roma.quat_product(roma.quat_inverse(quat1), quat2)
    ds = quat_off.abs().mean(dim=0).max(dim=-1)[1]
    for i, d in enumerate(ds):
        quat_off[:, i] = quat_off[:, i] * quat_off[:, i, d : d + 1].sign()

    quat_off = roma.quat_normalize(roma.quat_normalize(quat_off).mean(dim=0))
    return roma.quat_product(imu_data, quat_off.repeat(imu_data.shape[0], 1, 1))


def convert_quaternion_xsens(quat):
    """
    Converts Xsens quaternion format (w,x,y,z) to standard SMPL format (w,z,x,y).
    Modifies input tensor in-place.

    Parameters
    ----------
    quat : torch.Tensor
        Input quaternion tensor to modify

    Returns
    -------
    None

    Examples
    --------
    >>> convert_quaternion_xsens(xsens_quaternions)
    """
    old_quat = quat.reshape(-1, 4).clone()
    quat.view(-1, 4)[:, 1] = old_quat[:, 2]
    quat.view(-1, 4)[:, 2] = old_quat[:, 3]
    quat.view(-1, 4)[:, 3] = old_quat[:, 1]


def convert_point_xsens(point):
    """
    Converts Xsens 3D point coordinates (x,y,z) to SMPL format (y,z,x).
    Modifies input tensor in-place.

    Parameters
    ----------
    point : torch.Tensor
        Input point tensor to modify

    Returns
    -------
    None

    Examples
    --------
    >>> convert_point_xsens(xsens_points)
    """
    old_point = point.reshape(-1, 3).clone()
    point.view(-1, 3)[:, 0] = old_point[:, 1]
    point.view(-1, 3)[:, 1] = old_point[:, 2]
    point.view(-1, 3)[:, 2] = old_point[:, 0]


def convert_quaternion_excel(quat):
    """
    Converts Excel-stored quaternion format (w,x,y,z) to (w,z,x,y).

    Parameters
    ----------
    quat : torch.Tensor
        Input quaternion tensor (N, M, 4)

    Returns
    -------
    torch.Tensor
        Converted quaternion tensor (N, M, 4)

    Examples
    --------
    >>> converted_quat = convert_quaternion_excel(excel_quat_data)
    """
    quat[:, :, [1, 2, 3]] = quat[:, :, [2, 3, 1]]
    return quat


def convert_point_excel(point):
    """
    Converts Excel-stored 3D points (x,y,z) to (y,z,x).

    Parameters
    ----------
    point : torch.Tensor
        Input point tensor (N, M, 3)

    Returns
    -------
    torch.Tensor
        Converted point tensor (N, M, 3)

    Examples
    --------
    >>> converted_points = convert_point_excel(excel_point_data)
    """
    point[:, :, [0, 1, 2]] = point[:, :, [1, 2, 0]]
    return point


def read_mvnx(file: str, smpl_file: str):
    """
    Reads and processes MVNX motion capture files into SMPL-compatible format.

    Parameters
    ----------
    file : str
        Path to .mvnx input file
    smpl_file : str
        Path to SMPL model file

    Returns
    -------
    dict
        Processed data containing:
        - framerate: capture frequency
        - joint: orientations, positions, translations
        - imu: calibrated orientations, free acceleration

    Examples
    --------
    >>> mvnx_data = read_mvnx("motion.mvnx", "smpl_model.pkl")
    """
    model = ParametricModel(smpl_file)
    tree = ET.parse(file)

    # read framerate
    frameRate = int(tree.getroot()[2].attrib["frameRate"])

    segments = tree.getroot()[2][1]
    n_joints = len(segments)
    joints = []
    for i in range(n_joints):
        assert int(segments[i].attrib["id"]) == i + 1
        joints.append(segments[i].attrib["label"])

    sensors = tree.getroot()[2][2]
    n_imus = len(sensors)
    imus = []
    for i in range(n_imus):
        imus.append(sensors[i].attrib["label"])

    # read frames
    frames = tree.getroot()[2][-1]
    data = {
        "framerate": frameRate,
        "timestamp ms": [],
        "joint": {"orientation": [], "position": []},
        "imu": {"free acceleration": [], "orientation": []},
    }

    if frameRate != 60:
        step = int(frameRate // 60)
    else:
        step = 1

    for i in range(len(frames)):
        if frames[i].attrib["type"] in ["identity", "tpose", "tpose-isb"]:  # virginia
            continue

        elif ("index" in frames[i].attrib) and (
            frames[i].attrib["index"] == ""
        ):  # unipd
            continue

        orientation = torch.tensor(
            [float(_) for _ in frames[i][0].text.split(" ")]
        ).view(n_joints, 4)
        position = torch.tensor([float(_) for _ in frames[i][1].text.split(" ")]).view(
            n_joints, 3
        )
        sensorFreeAcceleration = torch.tensor(
            [float(_) for _ in frames[i][7].text.split(" ")]
        ).view(n_imus, 3)
        try:
            sensorOrientation = torch.tensor(
                [float(_) for _ in frames[i][9].text.split(" ")]
            ).view(n_imus, 4)
        except:
            sensorOrientation = torch.tensor(
                [float(_) for _ in frames[i][8].text.split(" ")]
            ).view(n_imus, 4)

        data["timestamp ms"].append(int(frames[i].attrib["time"]))
        data["joint"]["orientation"].append(orientation)
        data["joint"]["position"].append(position)
        data["imu"]["free acceleration"].append(sensorFreeAcceleration)
        data["imu"]["orientation"].append(sensorOrientation)

    data["timestamp ms"] = torch.tensor(data["timestamp ms"])
    for k, v in data["joint"].items():
        data["joint"][k] = torch.stack(v)
    for k, v in data["imu"].items():
        data["imu"][k] = torch.stack(v)

    data["joint"]["name"] = joints
    data["imu"]["name"] = imus

    # to smpl coordinate frame

    convert_quaternion_xsens(data["joint"]["orientation"])
    convert_point_xsens(data["joint"]["position"])
    convert_quaternion_xsens(data["imu"]["orientation"])
    convert_point_xsens(data["imu"]["free acceleration"])

    if step != 1:
        data["joint"]["orientation"] = data["joint"]["orientation"][::step].clone()
        data["joint"]["position"] = data["joint"]["position"][::step].clone()
        data["imu"]["free acceleration"] = data["imu"]["free acceleration"][
            ::step
        ].clone()
        data["imu"]["orientation"] = data["imu"]["orientation"][::step].clone()

    # use first 150 frames for calibration
    n_frames_for_calibration = 150
    imu_idx = [data["joint"]["name"].index(_) for _ in data["imu"]["name"]]

    imu_ori = imu_calibration(
        roma.quat_wxyz_to_xyzw(data["imu"]["orientation"]),
        roma.quat_wxyz_to_xyzw(data["joint"]["orientation"]),
        n_frames_for_calibration,
        imu_idx,
    )
    data["imu"]["calibrated orientation"] = imu_ori

    data["joint"]["orientation"] = roma.quat_normalize(
        roma.quat_wxyz_to_xyzw(data["joint"]["orientation"])
    )
    data["joint"]["orientation"] = roma.unitquat_to_rotmat(data["joint"]["orientation"])

    data["imu"]["calibrated orientation"] = roma.quat_normalize(
        data["imu"]["calibrated orientation"]
    )
    data["imu"]["calibrated orientation"] = roma.unitquat_to_rotmat(
        data["imu"]["calibrated orientation"]
    )

    data["imu"]["orientation"] = roma.quat_normalize(data["imu"]["orientation"])
    data["imu"]["orientation"] = roma.unitquat_to_rotmat(data["imu"]["orientation"])

    indices = [
        0,
        19,
        15,
        1,
        20,
        16,
        3,
        21,
        17,
        4,
        22,
        18,
        5,
        11,
        7,
        6,
        12,
        8,
        13,
        9,
        13,
        9,
        13,
        9,
    ]
    data["joint"]["orientation"] = data["joint"]["orientation"][:, indices]
    data["joint"]["orientation"] = model.inverse_kinematics_R(
        data["joint"]["orientation"]
    )
    data["joint"]["position"] = data["joint"]["position"][:, indices]
    data["joint"]["translation"] = data["joint"]["position"][:, 0]

    return data


def read_xlsx(xsens_file_path: str, smpl_file: str):
    """
    Processes Xsens Excel data files into SMPL-compatible format.

    Parameters
    ----------
    xsens_file_path : str
        Path to Xsens .xlsx data file
    smpl_file : str
        Path to SMPL model file

    Returns
    -------
    dict
        Processed data containing:
        - joint: orientations, positions, translations
        - imu: calibrated orientations, free acceleration
        - framerate: capture frequency

    Examples
    --------
    >>> xsens_data = read_xlsx("motion.xlsx", "smpl_model.pkl")
    """
    model = ParametricModel(smpl_file)
    pos3s_com, segments_pos3d, segments_quat, imus_ori, imus_free_acc = pd.read_excel(
        xsens_file_path,
        sheet_name=[
            "Center of Mass",
            "Segment Position",  # positions of joints in 3d space
            "Segment Orientation - Quat",  # segment global orientation
            "Sensor Orientation - Quat",  # sensor orientation
            "Sensor Free Acceleration",  # sensor free acceleration (accelerometer data without gravity vector)
        ],
        index_col=0,
    ).values()

    data = {
        "framerate": 60.0,
        "joint": {"orientation": [], "position": []},
        "imu": {"free acceleration": [], "orientation": []},
    }

    # add dim (S, [1], 3)  +  ignore com_vel / com_accel
    pos3s_com = np.expand_dims(pos3s_com.values, axis=1)[..., [0, 1, 2]]
    n_samples = len(pos3s_com)

    # assumes a perfect sampling freq of 60hz
    timestamps = np.arange(1, n_samples + 1) * (1 / 60.0)

    segments_pos3d = segments_pos3d.values.reshape(n_samples, -1, 3)
    segments_quat = segments_quat.values.reshape(n_samples, -1, 4)
    imus_free_acc = imus_free_acc.values.reshape(n_samples, -1, 3)
    imus_ori = imus_ori.values.reshape(n_samples, -1, 4)
    mask = torch.tensor([0, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21])
    imus_ori = imus_ori[:, mask, :]
    imus_free_acc = imus_free_acc[:, mask, :]

    data["joint"]["orientation"] = torch.tensor(
        segments_quat.astype(np.float32)
    ).clone()
    data["joint"]["position"] = torch.tensor(segments_pos3d.astype(np.float32)).clone()
    data["imu"]["orientation"] = torch.tensor(imus_ori.astype(np.float32)).clone()
    data["imu"]["free acceleration"] = torch.tensor(
        imus_free_acc.astype(np.float32)
    ).clone()

    data["joint"]["orientation"] = convert_quaternion_excel(
        data["joint"]["orientation"]
    )
    data["joint"]["position"] = convert_point_excel(data["joint"]["position"])
    data["imu"]["orientation"] = convert_quaternion_excel(data["imu"]["orientation"])
    data["imu"]["free acceleration"] = convert_point_excel(
        data["imu"]["free acceleration"]
    )

    data["joint"]["name"] = [
        "Pelvis",
        "L5",
        "L3",
        "T12",
        "T8",
        "Neck",
        "Head",
        "RightShoulder",
        "RightUpperArm",
        "RightForeArm",
        "RightHand",
        "LeftShoulder",
        "LeftUpperArm",
        "LeftForeArm",
        "LeftHand",
        "RightUpperLeg",
        "RightLowerLeg",
        "RightFoot",
        "RightToe",
        "LeftUpperLeg",
        "LeftLowerLeg",
        "LeftFoot",
        "LeftToe",
    ]
    data["imu"]["name"] = [
        "Pelvis",
        "T8",
        "Head",
        "RightShoulder",
        "RightUpperArm",
        "RightForeArm",
        "RightHand",
        "LeftShoulder",
        "LeftUpperArm",
        "LeftForeArm",
        "LeftHand",
        "RightUpperLeg",
        "RightLowerLeg",
        "RightFoot",
        "LeftUpperLeg",
        "LeftLowerLeg",
        "LeftFoot",
    ]

    # use first 150 frames for calibration
    n_frames_for_calibration = 150
    imu_idx = [data["joint"]["name"].index(_) for _ in data["imu"]["name"]]

    imu_ori = imu_calibration(
        roma.quat_wxyz_to_xyzw(data["imu"]["orientation"]),
        roma.quat_wxyz_to_xyzw(data["joint"]["orientation"]),
        n_frames_for_calibration,
        imu_idx,
    )
    data["imu"]["calibrated orientation"] = imu_ori

    data["joint"]["orientation"] = roma.quat_normalize(
        roma.quat_wxyz_to_xyzw(data["joint"]["orientation"])
    )
    data["joint"]["orientation"] = roma.unitquat_to_rotmat(data["joint"]["orientation"])

    data["imu"]["calibrated orientation"] = roma.quat_normalize(
        data["imu"]["calibrated orientation"]
    )
    data["imu"]["calibrated orientation"] = roma.unitquat_to_rotmat(
        data["imu"]["calibrated orientation"]
    )

    data["imu"]["orientation"] = roma.quat_normalize(data["imu"]["orientation"])
    data["imu"]["orientation"] = roma.unitquat_to_rotmat(data["imu"]["orientation"])

    indices = [
        0,
        19,
        15,
        1,
        20,
        16,
        3,
        21,
        17,
        4,
        22,
        18,
        5,
        11,
        7,
        6,
        12,
        8,
        13,
        9,
        13,
        9,
        13,
        9,
    ]
    data["joint"]["orientation"] = data["joint"]["orientation"][:, indices]
    data["joint"]["orientation"] = model.inverse_kinematics_R(
        data["joint"]["orientation"]
    )
    data["joint"]["position"] = data["joint"]["position"][:, indices]
    data["joint"]["translation"] = data["joint"]["position"][:, 0]

    return data
