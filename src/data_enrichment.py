#!/usr/bin/env python

import polars as pl
import logging
import configparser
import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from commons.custom_logger import setup_logger
from commons.staging_modules import init_stg_path, \
    get_latest_partitioned_folder, \
    retrieve_latest_modified_file, \
    enrich_partition_samples

def main(logger_object:logging.Logger):
    
    #========================================================
    # INITIALIZE MODULE VARIABLES
    #========================================================
    # Initialize configparser object
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    config.read(os.path.join(parent_dir, "config", "config.ini"))
    
    # Import configuration variables
    application_path = config.get("settings", "application_path")
    stg_partitioned_loc = config.get("local-path-settings", "stg_partitioned_loc")
    stg_processed_loc = config.get("local-path-settings", "staging_processed_folder")
    geospatial_data_folder = config.get("local-path-settings", "geospatial_processed_folder")
    execution_timestamp = datetime.now().strftime('%Y%m%d')
    
    #===================================================================
    # INITIALIZE STAGING PROCESSED PATH FOR STORING THE ENRICHED DATA
    #===================================================================
    stg_processed_path = os.path.join(application_path, stg_processed_loc)
    init_stg_path(stg_processed_path, logger_object)

    #========================================================
    # COLLECT THE LATEST PARTITIONED FOLDER
    #========================================================
    stg_partitioned_path = os.path.join(application_path, stg_partitioned_loc)
    latest_partitioned_folder = get_latest_partitioned_folder(stg_partitioned_path, logger_object)
    print(latest_partitioned_folder)

    #========================================================
    # LIST ALL PARTITION DIRECTORIES
    #========================================================
    partitions = [
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
            # "multipolygons": f"{suffix}multipolygons",
            "polygon_area": f"{suffix}polygon_area",
            "polygon_centroid": f"{suffix}polygon_centroid",
        }
        mapping_names.append(rename_dict)
    
    #=================================================================
    # LOAD THE SAVED GEOSPATIAL DATA
    #=================================================================
    geospatial_data_storage = os.path.join(parent_dir, geospatial_data_folder)
    # geospatial_cols = ["objectid", "zone", "multipolygons", "polygon_area", "polygon_centroid"]
    geospatial_cols = ["objectid", "zone", "polygon_area", "polygon_centroid"]
    df_geo = pl.read_ndjson(retrieve_latest_modified_file(geospatial_data_storage, "taxi_zones")).select(*geospatial_cols)
    print(df_geo.head())
    
    #=================================================================
    # EXECUTE DATA ENRICHMENT WITH GEOSPATIAL DATA USING THREADING
    #=================================================================

    #Without Threading
    # for partition in partitions:
    #     items = Path(partition).rglob("*.parquet")
    #     for parquet_file in items:
    #         partitions = dict(part.split('=') for part in parquet_file.parts if '=' in part)
    #         for key, value in partitions.items():
    #             print(key, value)
    #             df_partition= pl.read_parquet(parquet_file).with_columns(pl.lit(value, dtype=pl.Utf8).alias(key))
    #         print(df_partition.columns)
    #         print(df_partition.head())
    #     break
    #     data = enrich_partition_samples(partition, mapping_names, df_geo)
    #     print(data.columns)
    # exit()
    # With Threading
    # Number of threads to use
    num_threads = 2

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_partition = [executor.submit(enrich_partition_samples, partition, mapping_names, df_geo, stg_processed_path) for partition in partitions]
    end_time = time.perf_counter()
    logger_object.info(f"DATA ENRICHMENT FINISHED - Taxi trips enriched with geospatial data. (Total execution time = {end_time - start_time})")

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