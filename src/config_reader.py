from pathlib import Path

import yaml

from logger import get_logger

logger = get_logger(__name__)


def read_config(config_path):
    """Read and parse a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Dictionary containing the parsed YAML configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the specified path.
    yaml.YAMLError
        If there is an error parsing the YAML file.

    Examples
    --------
    >>> config = read_config("config/config.yaml")
    >>> bucket_name = config["data_ingestion"]["bucket_name"]
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Config file not found at {config_path}")

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_file}")
        raise e


if __name__ == "__main__":
    config = read_config(r"config\config.yaml")
    print(config)
