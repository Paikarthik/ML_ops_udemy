import logging
import os
from datetime import datetime

LOGS_DIR = "logs"  # Directory for log files to be stored
os.makedirs(LOGS_DIR, exist_ok=True)  # if dir exists ok, else create

# create a log file for each day with the log directory
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime("%Y-%m-%d")}.log")

# logging configuration
logging.basicConfig(
    filename=LOG_FILE,
    format="%(asctime)s - %(levelname)s - %(message)s",
    # this will create log in datatime - severity (info, warning, error) - log message
    level=logging.INFO,  # logs below info level are ingnored
    # Info, warning and error are displayed
)


def get_logger(name):
    logger = logging.getLogger(name)
    # creates a logger with the given name
    logger.setLevel(logging.INFO)

    return logger
