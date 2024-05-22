#!/usr/bin/env python

import logging
import os
import configparser
import time
from typing import Tuple

# Initialize configparser object
config = configparser.ConfigParser()
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir =  os.path.dirname(os.path.dirname(current_dir))
config.read(os.path.join(parent_dir, "config", "config.ini"))
application_path = config.get("settings", "application_path")

def setup_logger(log_path:str, log_filename:str) -> logging.Logger:
    # Check if logging folder a specific module exists.
    if not os.path.exists(f"{application_path}/logs/{log_path}"):
        os.makedirs(f"{application_path}/logs/{log_path}")
    else:
        logging.basicConfig(
            filename=os.path.join(f"{application_path}/logs", log_path, log_filename),
            level=logging.INFO,
            filemode='a',
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    return logging.getLogger(__name__)

def compute_execution_time(start_time:float)->Tuple[float, float, float]:
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return (hours, minutes, seconds)
