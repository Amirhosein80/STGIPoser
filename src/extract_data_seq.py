import glob
import math
import os

import numpy as np
import tqdm

from src.config_reader import read_config
from src.logger import get_logger

logger = get_logger(__name__)


def extract_seq(phase, data_path, seq_length, overlap=0):
    """
    Extracts fixed-length sequences from motion data files with optional overlap.

    Processes .npz files in the specified phase directory, splits them into sequences
    of `seq_length` frames, and saves them in a new directory structure. Handles both
    IMU-sensorized and vertex-based data variants.

    Parameters
    ----------
    phase : str
        Dataset phase to process. Typically "Train" or "Valid".
        Determines input subdirectory and output folder naming.
    data_path : str
        Root directory containing phase-specific data subdirectories.
    seq_length : int
        Length of sequences to extract in frames.
    overlap : int, optional
        Number of overlapping frames between consecutive sequences.
        Default is 0.

    Returns
    -------
    None
        Processed sequences are saved to disk in NPZ format.

    Examples
    --------
    >>> extract_seq(phase="Train", data_path="./data", seq_length=120, overlap=20)
    """
    logger.info(f"Starting data sequences from the {phase} folder")
    data_dir = os.path.join(data_path, phase, "*/*/*.npz")
    data_dirs = glob.glob(data_dir)
    loop = tqdm.tqdm(data_dirs)
    for data_dir in loop:

        data = np.load(data_dir)
        length = data["pose"].shape[0]
        if length < seq_length:
            continue

        loop.set_description(
            f"Extracting sequences from {data_dir} with seq length {seq_length}"
        )

        target_folder = os.path.join(data_path, f"{phase}_seq", f"seq_{seq_length}")
        os.makedirs(target_folder, exist_ok=True)
        num_sequences = int(math.ceil((length - overlap) / (seq_length - overlap)))
        for idx in range(num_sequences):
            start_index = idx * (seq_length - overlap) + 1
            end_index = start_index + seq_length
            len_data = data["pose"][start_index:end_index].shape[0]
            if len_data == seq_length:
                targets = {
                    "grot": data["grot"][start_index:end_index],
                    "trans": data["trans"][start_index:end_index],
                    "jvel": data["jvel"][start_index:end_index],
                    "uwb": data["uwb"][start_index:end_index],
                    "last_trans": data["trans"][start_index - 1],
                    "last_jvel": data["jvel"][start_index - 1],
                }

                if "imu_acc" and "imu_ori" in data.keys():
                    targets["imu_acc"] = data["imu_acc"][start_index:end_index]
                    targets["imu_ori"] = data["imu_ori"][start_index:end_index]

                else:
                    targets["imu_acc"] = data["vacc"][start_index:end_index]
                    targets["imu_ori"] = data["vrot"][start_index:end_index]

                name = (
                    [f"Seq_{idx + 1}"]
                    + data_dir.split("\\")[-3:-1]
                    + [os.path.basename(data_dir).replace(".pt", "")]
                )
                name = "_".join(name)

                np.savez(os.path.join(target_folder, name), **targets)
    logger.info(f"All sequences are extracted the {target_folder} dir")


if __name__ == "__main__":
    pass
    # configs = read_config(r"config/config.yaml")
    # extract_seq(
    #     data_path=configs["dataset"]["dir"],
    #     phase="Train",
    #     seq_length=configs["dataset"]["seq_length"],
    #     overlap=configs["dataset"]["overlap"],
    # )
