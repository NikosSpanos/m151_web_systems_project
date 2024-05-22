#!/usr/bin/env python

import polars as pl
import logging
import configparser
import os
import time
import holidays
from datetime import datetime, timedelta
from commons.custom_logger import setup_logger, compute_execution_time
from commons.staging_modules import init_stg_path, \
    init_unprocessed_folder, \
    init_partitioned_folder, \
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
    feature_engineer_time_to_seconds, \
    feature_engineer_trip_daytime, \
    feature_engineer_trip_distance, \
    feature_engineer_trip_cost, \
    is_holiday, \
    is_weekend, \
    one_hot_encode_daytime, \
    retrieve_latest_modified_file

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
    application_path:str = config.get("settings", "application_path")
    lnd_parent_loc:str = config.get("local-path-settings", "landing_folder")
    stg_unprocessed_loc:str = config.get("local-path-settings", "staging_unprocessed_folder")
    stg_partitioned_loc:str = config.get("local-path-settings", "stg_partitioned_loc")
    stg_processed_loc:str = config.get("local-path-settings", "staging_processed_folder")
    checkpoint_folder:str = os.path.join(application_path, config.get("local-path-settings", "metadata_folder"))
    checkpoint_file:str = os.path.join(application_path, checkpoint_folder, config.get("local-path-settings", "metadata_processed_file"))
    execution_timestamp:str = datetime.now().strftime('%Y%m%d')
    
    #========================================================
    # INITIALIZE STAGING STORAGE PATH (UNPROCESS + PROCESSED)
    #========================================================
    stg_unprocessed_path = os.path.join(application_path, stg_unprocessed_loc)
    stg_partitioned_path = os.path.join(application_path, stg_partitioned_loc)
    stg_processed_path = os.path.join(application_path, stg_processed_loc)
    init_stg_path(stg_unprocessed_path, logger_object)
    init_stg_path(stg_partitioned_path, logger_object)
    init_stg_path(stg_processed_path, logger_object)

    #========================================================
    # COLLECT THE LATEST PROCESSED LANDING FOLER
    #========================================================
    latest_date_modified_lnd_folder, latest_unprocessed_date, status = get_latest_processed_lnd_folder(checkpoint_file)

    #=========================================================
    # FIND THE LANDING FOLDER TO PROCESS LATEST COLLECTED DATA
    #=========================================================
    lnd_path = init_lnd_folder(os.path.join(application_path, lnd_parent_loc))
    latest_lnd_date = datetime.strptime(lnd_path.split('/')[-1], '%Y%m%d').date()
    execute_processing:bool = False

    # exit()
    if latest_lnd_date > latest_date_modified_lnd_folder:
        logger_object.info(f"Process records for data collected on: {latest_lnd_date}")
        execute_processing = True
    else:
        if latest_unprocessed_date and status == "unprocessed":
            logger_object.info(f"Latest collected data in landing folder {latest_lnd_date} have been processed on: {latest_unprocessed_date}")
            execute_processing = True
        else:
            if latest_unprocessed_date and status == "processed":
                logger_object.info(f"Latest collected data in landing folder {latest_lnd_date} have been processed on: {latest_unprocessed_date}")
            else:
                logger_object.error(f"Even though the landing folder has been processed there is no unprocessed folder logged. Please check for errors.")
            execute_processing = False
            return
    if execute_processing:
        logger_object.info(f"Loading collected data from latest modified landing path: {lnd_path}")
        #======================================================================
        # INITIALIZE UNPROCESSED STORAGE PATH FOR COMPACTING LANDING JSON FILES
        #======================================================================
        compact_flag, stg_loc = init_unprocessed_folder(stg_unprocessed_path, execution_timestamp)
        short_trip_loc, long_trip_loc = init_partitioned_folder(stg_partitioned_path, execution_timestamp)

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
            df = pl.read_ndjson(retrieve_latest_modified_file(stg_loc))
        
        if latest_date_modified_lnd_folder != latest_lnd_date:
            processed_dict = {
                "latest_lnd_folder":lnd_path.split('/')[-1],
                "latest_unprocessed_folder":stg_loc.split('/')[-1],
                "status": "unprocessed",
                "execution_dt": execution_timestamp
            }
            update_processed_metadata_file(checkpoint_file, processed_dict)

        #========================================================
        # CLEANING / PREPROCESSING  RAW DATA
        #========================================================
        try:
            #===========================
            # 1. FIX DATA TYPES
            #===========================
            cast_str = pl.Utf8
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
            # 2. FEATURE ENGINEERING
            # ==================================================

            #=========================================================================
            # FEATURE ENGINEER TRIP DURATION
            #=========================================================================
            df = feature_engineer_trip_duration(df, "tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_duration")

            #=========================================================================
            # FEATURE ENGINEER TIME OF DAY. GREATLY AFFECTS CHARGES.
            #=========================================================================
            daytime_mapper = {"Rush-Hour": 1, "Overnight": 2, "Daytime": 3}
            daytime_tuple = [("tpep_pickup_datetime", "pickup"), ("tpep_dropoff_datetime", "dropoff")]
            df = feature_engineer_trip_daytime(df, daytime_mapper, daytime_tuple)

            #=========================================================================
            # FEATURE ENGINEER TRIP DISTNACE FROM MILES TO KILOMETERS (KM)
            #=========================================================================
            df = feature_engineer_trip_distance(df, "trip_distance")

            #=========================================================================
            # FEATURE ENGINEER TOTAL TRIP COST BY ADDING/SUM RELEVANT FARES/CHARGES
            #=========================================================================
            cost_cols = ["fare_amount", "extra", "mta_tax", "tolls_amount", "improvement_surcharge"]
            df = feature_engineer_trip_cost(df, cost_cols)

            #=========================================================================
            # FEATURE ENGINEER PICKUP, DROPOFF SECONDS FROM BEGINNING OF EACH MONTH
            #=========================================================================
            df = feature_engineer_time_to_seconds(df, 'pickup')
            df = feature_engineer_time_to_seconds(df, 'dropoff')

            #=========================================================================
            # FEATURE ENGINEER A BINARY COLUMN TO DENOTE HOLIDAY PICKUP-DROPOFF DATES
            #=========================================================================
            us_holidays = holidays.country_holidays('US', years=range((datetime.now() - timedelta(days=10*365)).year, datetime.now().date().year))
            hol_dts = []
            for date, name in sorted(us_holidays.items()):
                hol_dts.append(date)
            df = is_holiday(df, 'pickup', hol_dts)
            df = is_holiday(df, 'dropoff', hol_dts)

            #=================================================================================
            # FEATURE ENGINEER A BINARY COLUMN TO DENOTE IF PICKUP-DROPOFF DATES ARE WEEKENDS
            #=================================================================================
            df = is_weekend(df, 'pickup')
            df = is_weekend(df, 'dropoff')

            #=================================================================================
            # FEATURE ENGINEER PICKUP QUARTERS OF THE DAY (1 IN 96 QUARTERS IN A DAY)
            #=================================================================================
            df = df.with_columns(
                pickup_quarter = ( (pl.col('tpep_pickup_datetime').dt.hour() * 60 + pl.col('tpep_pickup_datetime').dt.minute())/15 ).floor().cast(pl.Int8)
            ).with_columns(
                dropoff_quarter = ( (pl.col('tpep_dropoff_datetime').dt.hour() * 60 + pl.col('tpep_dropoff_datetime').dt.minute())/15 ).floor().cast(pl.Int8)
            )

            #=========================================================================
            # ONE-HOT ENCODE THE DAYTIME VALUES (RUSH-HOUR, OVERNIGHT, DAYTIME)
            #=========================================================================
            df = one_hot_encode_daytime(df, 'pickup_daytime')
            df = one_hot_encode_daytime(df, 'dropoff_daytime')

            # ==================================================
            # 3. REMOVE ROWS NOT FOLLOWING GENERAL COLUMN RULES
            # ==================================================
            
            #===========================================================================================================
            # FILTER OUT ROWS WITH PICKUP-DROPOFF DATETIMES AFTER CURRETN YEAR OR BEFORE 1970-01-01 (BEGINNING OF TIME)
            #===========================================================================================================
            cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
            dataset_year = datetime.now()
            start_of_time = datetime(1970,1,1)
            df = remove_abnormal_dates(df, cols, dataset_year, start_of_time, logger_object)

            #=========================================================================
            # FILTER OUT ROWS WITH NEGATIVE NUMERICAL VALUES OF CHARGES/FARES
            #=========================================================================
            cols = ["fare_amount", "tolls_amount", "extra", "mta_tax", "improvement_surcharge", "trip_distance"]
            df = remove_negative_charges(df, cols, logger_object)

            #==============================================================================================
            # FILTER OUT ROWS WITH (EQUAL) PICKUP == DROPOFF DATETIMES (= 0 DURATION) AND PICKUP > DROPOFF
            #==============================================================================================
            df = remove_equal_pickup_dropoff_times(df, "tpep_pickup_datetime", "tpep_dropoff_datetime", logger_object)

            #=========================================================================
            # FILTER OUT UBNORMAL KM/H (KILOMETERS PER HOUR - AVERAGE SPEED) RECORDS
            #=========================================================================
            df = df.with_columns(
                average_speed = pl.col("trip_distance") / (pl.col("trip_duration")/60).cast(pl.Float32)
            ).filter(
                (pl.col("average_speed").gt(1)) & (pl.col("average_speed").lt(240))
            )

            # ==================================================
            # 4. CREATE THE PARTITION COLUMN - PARTITION_DT
            # ==================================================
            df = df.with_columns(
                pl.col("tpep_pickup_datetime").dt.strftime(format="%Y%m").alias("partition_dt")
            )

            # ===========================================================
            # 5. CREATE THE PARTITION COLUMN - TRIP_FLAG
            # ===========================================================
            df = df.with_columns(
                pl.when(pl.col("trip_distance").gt(35.0)).then(pl.lit('long_trip')).otherwise(pl.lit('short_trip')).alias("trip_type")
            )

            # ===========================================================
            # 6. CREATE A HASH COLUMN FROM UNIQUE COMBINATION OF RECORDS
            # ===========================================================
            df = df.with_columns(
                df.select(["tpep_pickup_datetime", "pulocationid", "dolocationid", "trip_cost"]).hash_rows(seed=1234).alias("hashing_key")
            )

            # ==================================================
            # 7. COMPUTE AND REPORT NULL VALUES PER COLUMN
            # ==================================================
            df_nulls = df.select(pl.all().is_null().sum()).to_dicts()[0]
            null_column_names = [k for k, v in df_nulls.items() if v > 0]
            logger_object.info("Column names with null values: {0}".format(null_column_names))
            
            # ==============================================================
            # 8. SAVE THE PREPROCESSED DATA OF TAXI TRIPS TO PARQUET FILES
            # ==============================================================
            logger_object.info("Start saving table to partitions")
            start_time = time.perf_counter()

            df_short, df_long = (
                df.filter(pl.col("trip_type").eq(pl.lit('short_trip'))),
                df.filter(pl.col("trip_type").eq(pl.lit('long_trip')))
            )
            assert (df_short.select(pl.count()).item() > 0) and (df_long.select(pl.count()).item() > 0), "Empty dataframes. Please check logs for errors."
            df_short.write_parquet(
                short_trip_loc,
                compression="zstd",
                use_pyarrow=True,
                pyarrow_options={
                    "partition_cols": ["partition_dt"],
                    "existing_data_behavior": "overwrite_or_ignore"
                }
            )
            df_long.write_parquet(
                long_trip_loc,
                compression="zstd",
                use_pyarrow=True,
                pyarrow_options={
                    "partition_cols": ["partition_dt"],
                    "existing_data_behavior": "overwrite_or_ignore"
                }
            )
            hours, minutes, seconds = compute_execution_time(start_time)
            formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):06}"
            logger_object.info(f"Execution time to write partition files: {formatted_time}")
            logger_object.info("Finished saving table to partitions")
            processing_completed:bool = True
        except Exception as e:
            processing_completed:bool = False
            logger_object.error(e)
            return

        if processing_completed:
            update_processed_metadata_file(checkpoint_file, None, latest_unprocessed_date)

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