#!/usr/bin/env python

import requests
import logging
import configparser
import os
import json
import sys
from datetime import datetime
from commons.custom_logger import setup_logger

def main(logger_object:logging.Logger):
    
    #========================================================
    # INITIALIZE MODULE VARIABLES
    #========================================================
    # Initialize configparser object
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    config.read(os.path.join(parent_dir, "config", "config.ini"))

    api_url = config.get("api-settings", "nyc_zones_api")
    geospatial_batch_size = int(config.get("api-settings", "geospatial_batch_size"))
    geospatial_data_folder = config.get("local-path-settings", "geospatial_data_folder")

    #========================================================
    # INITIALIZE GEOSPATIAL DATA STORAGE
    #========================================================
    geospatial_data_storage = os.path.join(parent_dir, geospatial_data_folder)
    if not os.path.exists(geospatial_data_storage):
        os.makedirs(geospatial_data_storage)
    
    #========================================================
    # NYC ZONES COLLECTION FROM EXTERNAL API
    #========================================================
    params = {
        "$limit": geospatial_batch_size, #i.e. 50_000
        "$$app_token": config.get("api-settings", "app_token")
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if not data:
            logger_object.info("No data found, please check the API connection.")
            sys.exit()
        with open("{0}/nyc_zone_districts_data.json".format(geospatial_data_storage), "w") as f:
            json.dump(data, f, indent=4)
    else:
        logger_object.error("API request failed.")
        logger_object.error("Error: {0}".format(response.status_code))
        logger_object.error(response.text)
    
    logger_object.info("EXTRACTION FINISHED - Data collection of NYC Zone Districts from the Socrata API completed.")

if __name__ == "__main__":
    project_folder = "nyc_zones_collection"
    log_filename = f"nyc_zones_collection_execution_logs_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
    logger = setup_logger(project_folder, log_filename)
    try:
        main(logger)
        logger.info("SUCCESS: NYC zones collection completed.")
    except Exception as e:
        logger.error(e)
        logger.error("FAIL: NYC zones collection failed.")