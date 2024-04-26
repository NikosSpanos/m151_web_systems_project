#!/usr/bin/env python

import polars as pl
import logging
import configparser
import os
import glob
from datetime import datetime
from commons.custom_logger import setup_logger
from commons.staging_modules import init_stg_path, \
    init_unprocessed_folder, \
    init_lnd_folder, \
    get_latest_processed_lnd_folder, \
    update_processed_metadata_file, \
    load_json_to_dataframe, \
    write_df_toJSON, \
    fix_data_type, \
    remove_abnormal_dates, \
    remove_negative_charges, \
    remove_equal_pickup_dropoff_times, \
    feature_engineer_trip_duration, \
    feature_engineer_trip_hour, \
    feature_engineer_trip_daytime, \
    retrieve_latest_modified_file, \
    write_df_toCSV, \
    write_df_toJSON_v2

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
    lnd_parent_loc = config.get("local-path-settings", "landing_folder")
    stg_unprocessed_loc = config.get("local-path-settings", "staging_unprocessed_folder")
    stg_processed_loc = config.get("local-path-settings", "staging_processed_folder")
    checkpoint_folder = os.path.join(application_path, config.get("local-path-settings", "metadata_folder"))
    checkpoint_file = os.path.join(application_path, checkpoint_folder, config.get("local-path-settings", "metadata_processed_file"))
    execution_timestamp = datetime.now().strftime('%Y%m%d')
    
    #========================================================
    # INITIALIZE STAGING STORAGE PATH (UNPROCESS + PROCESSED)
    #========================================================
    stg_unprocessed_path = os.path.join(application_path, stg_unprocessed_loc)
    stg_processed_path = os.path.join(application_path, stg_processed_loc)
    init_stg_path(stg_unprocessed_path, logger_object)
    init_stg_path(stg_processed_path, logger_object)

    #========================================================
    # COLLECT THE LATEST PROCESSED LANDING FOLER
    #========================================================
    latest_date_modified_lnd_folder, latest_unprocessed_date = get_latest_processed_lnd_folder(checkpoint_file)

    #=========================================================
    # FIND THE LANDING FOLDER TO PROCESS LATEST COLLECTED DATA
    #=========================================================
    lnd_path = init_lnd_folder(os.path.join(application_path, lnd_parent_loc))
    latest_lnd_date = datetime.strptime(lnd_path.split('/')[-1], '%Y%m%d').date()
    execute_processing:bool = False
    if latest_lnd_date > latest_date_modified_lnd_folder:
        logger_object.info(f"Process records for data collected on: {latest_lnd_date}")
        execute_processing = True
    else:
        if latest_unprocessed_date:
            logger_object.info(f"Latest collected data in landing folder {latest_lnd_date} have been processed on: {latest_unprocessed_date}")
        else:
            logger_object.error(f"Even though the landing folder has been processed there is no unprocessed folder logged. Please check for errors.")
        execute_processing = False
        return # Exit the python program if there is no new data to process.
    
    if execute_processing:
        logger_object.info(f"Loading collected data from latest modified landing path: {lnd_path}")
        
        #======================================================================
        # INITIALIZE UNPROCESSED STORAGE PATH FOR COMPACTING LANDING JSON FILES
        #======================================================================
        compact_flag, stg_loc = init_unprocessed_folder(stg_unprocessed_path, execution_timestamp)

        if compact_flag:
            #========================================================================
            # COMPACT THE CHUNKS OF JSON FILES AND LOAD THEM INTO A POLARS DATAFRAME
            #========================================================================
            logger_object.info("Start COMPACTING json files and LOADING the Polars dataframe")
            df = pl.DataFrame([])
            schema = {
                "tpep_pickup_datetime": pl.Utf8,
                "tpep_dropoff_datetime": pl.Utf8,
                "trip_distance": pl.Utf8,
                "pulocationid": pl.Utf8,
                "dolocationid": pl.Utf8,
                "fare_amount": pl.Utf8,
                "extra": pl.Utf8,
                "mta_tax": pl.Utf8,
                "tolls_amount": pl.Utf8,
                "improvement_surcharge": pl.Utf8
            }
            cols_list = list(schema.keys())
            logger_object.info(f"Selected columns to keep: {cols_list}")
            df = load_json_to_dataframe(lnd_path, cols_list, logger_object)
            write_df_toJSON(stg_loc, df, "yellow_taxi_trip_unprocessed_data", logger_object)
        else:
            #==================================================
            # READ THE COMPACT JSON FILE ALREADY SAVED IN DISK
            #==================================================
            logger_object.info("READ already COMPACT json file with all the collected data records from the latest LANDING folder.")
            json_file = glob.glob("{0}/*.json".format(stg_loc))
            df = pl.read_json(json_file[0])
        
        if latest_date_modified_lnd_folder != latest_lnd_date: 
            processed_dict = {
                "latest_lnd_folder":lnd_path.split('/')[-1],
                "latest_unprocessed_folder":stg_loc.split('/')[-1],
                "execution_dt": execution_timestamp
            }
            update_processed_metadata_file(checkpoint_file, processed_dict)

        #========================================================
        # CLEANING / PREPROCESSING  RAW DATA
        #========================================================
        #===========================
        # 1. FIX DATA TYPES
        #===========================
        cast_str = pl.Utf8
        cast_categ = pl.Categorical
        cast_int = pl.Int64
        cast_float = pl.Float64
        dt_format = "%Y-%m-%dT%H:%M:%S.000"

        dtype_map = {
            "tpep_pickup_datetime": "datetime",
            "tpep_dropoff_datetime": "datetime",
            "pulocationid": cast_str,
            "dolocationid": cast_str,
            "trip_distance": cast_float,
            "fare_amount": cast_float,
            "extra": cast_float,
            "mta_tax": cast_float,
            "tolls_amount": cast_float,
            "improvement_surcharge": cast_float
        }
        df = fix_data_type(df, dtype_map, dt_format)

        # ==================================================
        # 2. REMOVE ROWS NOT FOLLOWING GENERAL COLUMN RULES
        # ==================================================
        # Rows with pickup, dropoff datetimes after/before dataset year
        cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
        dataset_year = datetime.now()
        start_of_time = datetime(1970,1,1)
        df = remove_abnormal_dates(df, cols, dataset_year, start_of_time, logger_object)

        # Rows with numerical negative values (float columns)
        cols = ["fare_amount", "tolls_amount", "extra", "mta_tax", "improvement_surcharge", "trip_distance"]
        df = remove_negative_charges(df, cols, logger_object)

        # Rows with equal pickup == dropoff datetimes or pickup date > dropoff date
        df = remove_equal_pickup_dropoff_times(df, "tpep_pickup_datetime", "tpep_dropoff_datetime", logger_object)

        # Compute the number of null records per column
        df_nulls = df.select(pl.all().is_null().sum()).to_dicts()[0]
        null_column_names = [k for k, v in df_nulls.items() if v > 0]
        logger_object.info("Column names with null values: {0}".format(null_column_names))

        # ==================================================
        # 3. FEATURE ENGINEERING
        # ==================================================
        df = feature_engineer_trip_duration(df, "tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_duration")
        print(df.select(pl.col("trip_duration")).head(10))

        hour_tuple = [("tpep_pickup_datetime", "pickup"), ("tpep_dropoff_datetime", "dropoff")]
        df = feature_engineer_trip_hour(df, hour_tuple)
        
        daytime_mapper = {"Rush-Hour": 1, "Overnight": 2, "Daytime": 3}
        daytime_tuple = [("pickup_hour", "pickup"), ("dropoff_hour", "dropoff")]
        df = feature_engineer_trip_daytime(df, daytime_mapper, daytime_tuple)
        print(df.select(pl.col("pickup_daytime")).head(10))
        print(df.shape)

        exit()
        # ==================================================
        # 4. ENRICH DATA WITH NYC ZONE NAMES
        # ==================================================
        relative_path = "{0}/data/geospatial/".format(application_path)
        df_geo = pl.read_json(retrieve_latest_modified_file(relative_path, False))
        print(df_geo.head())

        merged_data = pl.DataFrame([])
        geo_cols = ["objectid", "zone"]

        # Merge on PULocationID
        merged_data = df.join(df_geo.select(geo_cols), left_on=pl.col("pulocationid"), right_on=pl.col("objectid"), how='left')
        merged_data = merged_data.rename({"zone": "puzone"})

        # Merge on DOLocationID
        merged_data = merged_data.join(df_geo.select(geo_cols), left_on=pl.col("dolocationid"), right_on=pl.col("objectid"), how='left')
        merged_data = merged_data.rename({"zone": "dozone"})
        print(merged_data.columns)
        print(list(zip(merged_data.dtypes, merged_data.columns)))

        # Re-Compute the number of null records per column
        df_nulls = merged_data.select(pl.all().is_null().sum()).to_dicts()[0]
        null_column_names = [k for k, v in df_nulls.items() if v > 0]
        logger_object.info("Column names with null values: {0}".format(null_column_names))

    # # #==================================================
    # # WRITE PROCESSED TABLE TO JSON
    # # #==================================================
    # write_df_toJSON("{0}/data/staging/processed/{1}".format(application_path, execution_timestamp), merged_data, "yellow_taxi_trip_processed_data", logger_object)
    # write_df_toCSV("{0}/data/staging/processed/{1}".format(application_path, execution_timestamp), merged_data, "yellow_taxi_trip_processed_data", logger_object)

    # # Shuffle the dataframe and filter top 200_000 rows
    # merged_data_shuffled = merged_data.select(pl.col("*").shuffle(123))
    # merged_data_shuffled = merged_data_shuffled.head(samples_value)
    # write_df_toJSON("{0}/data/staging/processed/{1}".format(application_path, execution_timestamp), merged_data_shuffled, "yellow_taxi_trip_processed_data_{0}".format(samples_str), logger_object)
    # write_df_toCSV("{0}/data/staging/processed/{1}".format(application_path, execution_timestamp), merged_data_shuffled, "yellow_taxi_trip_processed_data_{0}".format(samples_str), logger_object)

    # logger_object.info("ENRICHMENT COMPLETED - Geospatial data imported to dataframe.")

if __name__ == "__main__":
    project_folder = "batch_processing"
    log_filename = f"batch_processing_execution_logs_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
    logger = setup_logger(project_folder, log_filename)
    try:
        main(logger)
        logger.info("SUCCESS: Batch processing/cleaning/feature-engineering completed.")
    except Exception as e:
        logger.error(e)
        logger.error("FAIL: Batch processing/cleaning/feature-engineering failed.")