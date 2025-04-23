import os
import logging
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")
TRAIN_LOG_DIR = Path("train_logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)

TODAY_TIME = datetime.now().strftime("%Y-%m-%d")
TRAIN_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M")

LOGS_FILE = LOGS_DIR / f"logs_{TODAY_TIME}.log"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOGS_FILE),  # File handler
        logging.StreamHandler()          # Console handler
    ]
)


def get_logger(name):
    """Gets a logger configured to write to the application's log file.

    This function utilizes the root logging configuration set up
    by this module.

    Parameters
    ----------
    name : str
        The name of the logger to retrieve. Typically __name__ for the
        calling module.

    Returns
    -------
    logging.Logger
        A logger instance configured according to the module's settings.

    Examples
    --------
    >>> from logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("This is an informational message.")
    """
    logger = logging.getLogger(name)
    return logger


def get_train_logger(name, experiment_name, log_dir):
    """Gets a logger configured to write to the train's log file.

    This function utilizes a new logging configuration set up
    by the experiment's name.

    Parameters
    ----------
    name : str
        The name of the logger to retrieve. Typically __name__ for the
        calling module.
    experiment_name : str
        The name of the training experiment

    Returns
    -------
    logging.Logger
        A logger instance configured according to the experiment's settings.

    Examples
    --------
    >>> from logger import get_train_logger
    >>> logger = get_logger(__name__, experiment1)
    >>> logger.info("This is an informational message.")
    """
    train_logger = logging.getLogger(name)
    train_handler = logging.FileHandler(
        os.path.join(log_dir, f"log_{experiment_name}_{TRAIN_TIME}.log")
    )
    train_logger.addHandler(train_handler)
    return train_logger



