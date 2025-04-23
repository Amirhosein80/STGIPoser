from src.config_reader import read_config
from src.extract_data_seq import extract_seq
from src.preprocess import (
    process_andy,
    process_cip,
    process_dip,
    process_emonike,
    process_total_capture,
    process_unipd,
    process_virginia,
)

configs = read_config(r"config\config.yaml")

smpl_path = configs["smpl"]["file"]


def preprocss_data():
    process_dip(
        smpl_path=smpl_path,
        dip_path=configs["data_paths"]["dip"]["path"],
        folder=configs["data_paths"]["dip"]["set"],
    )

    process_total_capture(
        smpl_path=smpl_path,
        tc_path=configs["data_paths"]["total capture"]["path"],
        folder=configs["data_paths"]["total capture"]["set"],
    )

    process_emonike(
        smpl_path=smpl_path,
        emonike_path=configs["data_paths"]["emonike"]["path"],
        folder=configs["data_paths"]["emonike"]["set"],
    )

    process_andy(
        smpl_path=smpl_path,
        mvnx_path=configs["data_paths"]["andy"]["path"],
        folder=configs["data_paths"]["andy"]["set"],
    )

    process_cip(
        smpl_path=smpl_path,
        mvnx_path=configs["data_paths"]["cip"]["path"],
        folder=configs["data_paths"]["cip"]["set"],
    )

    process_unipd(
        smpl_path=smpl_path,
        mvnx_path=configs["data_paths"]["unipd"]["path"],
        folder=configs["data_paths"]["unipd"]["set"],
    )

    process_virginia(
        smpl_path=smpl_path,
        mvnx_path=configs["data_paths"]["virginia"]["path"],
        folder=configs["data_paths"]["virginia"]["set"],
    )


if __name__ == "__main__":

    # preprocss_data()

    extract_seq(
        "Train",
        data_path=configs["dataset"]["dir"],
        seq_length=configs["dataset"]["seq_length"],
        overlap=configs["dataset"]["overlap"],
    )
