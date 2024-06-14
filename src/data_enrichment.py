#!/usr/bin/env python

import logging
import configparser
import os
import time
from argparse import ArgumentParser
from multiprocessing import Pool
from datetime import datetime
from commons.custom_logger import setup_logger, compute_execution_time
from commons.staging_modules import init_stg_path, \
    get_latest_partitioned_folder, \
    retrieve_latest_modified_file, \
    enrich_partition_samples

def main(logger_object:logging.Logger):
    
    #========================================================
    # INITIALIZE MODULE VARIABLES
    #========================================================
    # Initialize ConfigParser() class
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    config.read(os.path.join(parent_dir, "config", "config.ini"))
    
    # Initialize ArgumentParser() class
    parser = ArgumentParser()
    parser.add_argument(
        "-tt", "--trip_type", type=str, help="Select trip type to enrich with geospatial data", default="short_trip", required=True
    )
    args = parser.parse_args()

    # Import configuration variables
    application_path = config.get("settings", "application_path")
    stg_partitioned_loc = config.get("local-path-settings", "stg_partitioned_loc")
    stg_processed_loc = config.get("local-path-settings", "staging_processed_folder")
    geospatial_data_folder = config.get("local-path-settings", "geospatial_processed_folder")
    execution_timestamp = datetime.now().strftime('%Y%m%d')

    #===================================================================
    # INITIALIZE STAGING PROCESSED PATH FOR STORING THE ENRICHED DATA
    #===================================================================
    stg_processed_path = os.path.join(application_path, stg_processed_loc, execution_timestamp, args.trip_type)
    init_stg_path(stg_processed_path, logger_object)
    #========================================================
    # COLLECT THE LATEST PARTITIONED FOLDER
    #========================================================
    stg_partitioned_path = os.path.join(application_path, stg_partitioned_loc)
    latest_partitioned_folder = os.path.join(get_latest_partitioned_folder(stg_partitioned_path, logger_object), args.trip_type)

    #========================================================
    # LIST ALL PARTITION DIRECTORIES
    #========================================================
    partitions:list = [
        os.path.join(latest_partitioned_folder, d)
        for d in os.listdir(latest_partitioned_folder)
        if os.path.isdir(os.path.join(latest_partitioned_folder, d))
    ]
    #=================================================================
    # MAP NEW NAMES FOR THE GENERATED COLUMNS FROM GEOSPATIAL SAMPLES
    #=================================================================
    mapping_names:list = []
    for suffix in ['pu_', 'do_']:
        rename_dict = {
            "zone": f"{suffix}zone",
            "polygon_area": f"{suffix}polygon_area",
            "polygon_centroid": f"{suffix}polygon_centroid",
        }
        mapping_names.append(rename_dict)
    
    #=================================================================
    # LOAD THE SAVED GEOSPATIAL DATA
    #=================================================================
    geospatial_data_storage = os.path.join(parent_dir, geospatial_data_folder)
    geo_path:str = retrieve_latest_modified_file(geospatial_data_storage, "taxi_zones")

    #=================================================================
    # EXECUTE DATA ENRICHMENT WITH GEOSPATIAL DATA USING THREADING
    #=================================================================
    try:
        num_processes:int = 3
        task_args:list = [(partition, mapping_names, geo_path, stg_processed_path, logger_object) for partition in partitions]
        start_time:float = time.perf_counter()

        with Pool(num_processes) as pool:
            pool.map(enrich_partition_samples, task_args)
        hours, minutes, seconds = compute_execution_time(start_time)
        logger_object.info(f"DATA ENRICHMENT FINISHED - Taxi trips enriched with geospatial data. (Total execution time = {int(hours):02}:{int(minutes):02}:{int(seconds):06})")
    except Exception as e:
        logger_object.error(e)

if __name__ == "__main__":
    project_folder = "batch_enrichment"
    log_filename = f"{project_folder}_execution_logs_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
    logger = setup_logger(project_folder, log_filename)
    try:
        main(logger)
        logger.info("SUCCESS: Batch processing/cleaning/feature-engineering completed.")
    except Exception as e:
        logger.error(e)
        logger.error("FAIL: Batch processing/cleaning/feature-engineering failed.")