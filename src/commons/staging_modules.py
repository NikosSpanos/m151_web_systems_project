#!/usr/bin/env python
import hashlib
import glob
import polars as pl
import os
import logging
import json
from datetime import datetime, timedelta
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

def md5_hashing(value):
    return hashlib.md5(value.encode()).hexdigest()

def init_stg_path(stg_path:str, logger_object:logging.Logger):
    if not os.path.exists(stg_path):
        logger_object.info(f"Staging path: {stg_path} not found. Creating path...")
        os.makedirs(stg_path)
    else:
        logger_object.info(f"Staging path: {stg_path} already exists. Program will continue.")
    return

def init_unprocessed_folder(stg_path:str, subfolder:str) -> Tuple[bool, str]:
    compact_flag:bool = False
    available_directories = [os.path.join(stg_path, file) for file in os.listdir(stg_path)]
    latest_modified_directory = max(available_directories, key=os.path.getmtime)
    if os.listdir(latest_modified_directory):
        return compact_flag, latest_modified_directory
    else:
        new_stg_loc = os.path.join(stg_path, subfolder)
        os.makedirs(new_stg_loc, exist_ok=True) # Overwrite the folder if it's empty.
        compact_flag = True
        return compact_flag, new_stg_loc

def init_lnd_folder(lnd_parent_path:str) -> str:
    available_directories = [
        os.path.join(lnd_parent_path, directory) for directory in os.listdir(lnd_parent_path) if len(os.listdir(os.path.join(lnd_parent_path, directory))) !=0 
    ] # Retrieve the latest non-empty landing directory with collected data.
    return max(available_directories, key=os.path.getmtime)

def update_processed_metadata_file(metadata_file:str, metadata_dict:dict):
    try:
        with open(metadata_file, 'r') as file:
            data = json.load(file)
    except Exception as e:
        data = []
    data.append(metadata_dict)
    with open(metadata_file, "w") as file:
        json.dump(data, file, indent=4)

def get_latest_processed_lnd_folder(metadata_file:str) -> Tuple[datetime, Optional[datetime]]:
    if not os.path.exists(metadata_file):
        data = []
        with open(metadata_file, 'w') as file:
            json.dump(data, file)
        return (datetime(year=1970, month=1, day=1).date(), None)
    with open(metadata_file, 'r') as file:
        data = json.load(file)
        if len(data) != 0:
            max_checkpoint = sorted(data, key=lambda x: x["latest_lnd_folder"], reverse=True)
            latest_checkpoint:datetime = datetime.strptime(max_checkpoint[0]["latest_lnd_folder"], '%Y%m%d').date()
            unprocessed_folder:datetime = datetime.strptime(max_checkpoint[0]["execution_dt"], '%Y%m%d').date()
        else:
            latest_checkpoint:datetime = datetime(year=1970, month=1, day=1).date()
            unprocessed_folder = None
    return (latest_checkpoint, unprocessed_folder)

def daytime_value(hour_value):
    if (hour_value in range(7,11)) or (hour_value in range(16,20)):
        return "Rush-Hour"
    elif hour_value in [20,21,22,23,0,1,2,3,4,5,6]:
        return "Overnight"
    else:
        return "Daytime"

def create_folder(folder_path:str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder '{0}' has been created.".format(folder_path))

def collect_data(args:tuple)-> pl.DataFrame:
    lnd_file, cols_list, logger_obj = args
    logger_obj.info(f"Processing file: {lnd_file}")
    load_df = pl.read_json(lnd_file)
    if load_df.shape[1] == 18: #exclude the dataframes with a number of c0lumns < 18 (18 are the number of columns in the original dataset)
        return load_df.select(cols_list)
    return None

def load_json_to_dataframe(lnd_path:str, cols_list:list, logger_obj:logging.Logger) -> pl.DataFrame:
    json_files = glob.glob("{0}/*.json".format(lnd_path))
    dataframes:list = []

    tasks = [(file, cols_list, logger_obj) for file in json_files]

    with Pool(5) as pool:
        results = pool.map(collect_data, tasks)

    dataframes = [df for df in results if df is not None]
    if dataframes:
        df = pl.concat(dataframes)
        return df
    else:
        return pl.DataFrame()

def write_df_toJSON(relative_path:str, df:pl.DataFrame, filename:str, logger_obj:logging.Logger):
    create_folder(relative_path)
    filename = os.path.join(relative_path, "{0}.json".format(filename))
    df.write_json(filename)
    logger_obj.info(f"DataFrame saved as JSON under path: {filename}")

def write_df_toJSON_v2(relative_path:str, df:pl.DataFrame, filename:str, logger_obj:logging.Logger):
    create_folder(relative_path)
    filename = os.path.join(relative_path, "{0}.json".format(filename))
    with open(filename, "w") as f:
        json.dump(df, f)
    logger_obj.info(f"DataFrame saved as JSON under path: {filename}")

def write_df_toCSV(relative_path:str, df:pl.DataFrame, filename:str, logger_obj:logging.Logger):
    create_folder(relative_path)
    filename = os.path.join(relative_path, "{0}.csv".format(filename))
    df.write_csv(filename, has_header=True, separator=",")
    logger_obj.info(f"DataFrame saved as CSV under path: {filename}")

def fix_data_type(df:pl.DataFrame, type_mapping:dict, dt_format:str = None) -> pl.DataFrame:
    for column, dtype in type_mapping.items():
        if dtype == "datetime":
            df = df.with_columns(pl.col(column).str.to_datetime(dt_format))
        else:
            df = df.with_columns(pl.col(column).cast(dtype))
    return df

def remove_abnormal_dates(df:pl.DataFrame, cols:list, dataset_year:datetime, start_of_time:datetime, logger_obj:logging.Logger) -> pl.DataFrame:
    for col in cols:
        above_current_fyear = df.filter(pl.col(col).dt.year().gt(dataset_year.year))
        before_start_of_time = df.filter(pl.col(col).dt.year().lt(start_of_time.year))
        logger_obj.info("{0} dates after current fiscal year ({1}): {2}".format(col, dataset_year.year, above_current_fyear.height))
        logger_obj.info("{0} dates before current fiscal year ({1}): {2}".format(col, dataset_year.year, before_start_of_time.height))
        # Remove rows with year greater/lower than the dataset year
        df = df.filter(
            (pl.col(col).dt.year().le(dataset_year.year)) & (pl.col(col).dt.year().gt( start_of_time.year ))
        )
    return df

def remove_negative_charges(df:pl.DataFrame, cols:list, logger_obj:logging.Logger) -> pl.DataFrame:
    for col in cols:
        if col in ["tolls_amount", "extra", "mta_tax", "improvement_surcharge"]:
            negative_charges = df.filter(pl.col(col).lt(0))
            logger_obj.info("{0} with negative values (<0): {1}".format(col, negative_charges.height))
            df = df.filter(pl.col(col).ge(0))
        else:
            negative_charges = df.filter(pl.col(col).le(0))
            logger_obj.info("{0} with negative values (<=0): {1}".format(col, negative_charges.height))
            df = df.filter(pl.col(col).gt(0))
    return df

def remove_equal_pickup_dropoff_times(df:pl.DataFrame, pu_col:str, do_col:str, logger_obj:logging.Logger) -> pl.DataFrame:
    equal_pu_do_dt = df.filter(pl.col(pu_col).ge(pl.col(do_col)))
    logger_obj.info("Taxi trips without duration (pickup date >= drop-off date): {0}".format(equal_pu_do_dt.height))
    df = df.filter(pl.col(pu_col).lt(pl.col(do_col)))
    return df

def feature_engineer_trip_duration(df:pl.DataFrame,  pu_col:str, do_col:str, duratation_col_name:str) -> pl.DataFrame:
    df = df.with_columns(
        ( ( ( ( pl.col(do_col) - pl.col(pu_col) )/60 )/1_000_000 ).round(2)).cast(pl.Float64).alias(duratation_col_name)
    )
    df = df.filter(pl.col(duratation_col_name).ge(1.0)) # Remove rows with trip duration less than 1 minute.
    return df

def feature_engineer_trip_hour(df:pl.DataFrame, cols:list) -> pl.DataFrame:
    for col in cols:
        df = df.with_columns(
            pl.col(col[0]).dt.hour().cast(pl.Int64).alias("{0}_hour".format(col[1])),
        )
    return df

def feature_engineer_trip_daytime(df:pl.DataFrame, daytime_mapper:list, cols:tuple) -> pl.DataFrame:
    for col in cols:
        df = df.with_columns(
            pl.col(col[0]).map_elements(daytime_value, return_dtype=pl.Utf8).map_dict(daytime_mapper).cast(pl.Int64).alias("{0}_daytime".format(col[1]))
        )
    return df

def retrieve_latest_modified_file(relative_path:str, short_version:bool, samples_values:str=None):
    if short_version:
        json_files = glob.glob("{0}/*_data_{1}.json".format(relative_path, samples_values))
    else:
        json_files = glob.glob("{0}/*_data.json".format(relative_path))
    files_with_timestamps = []
    for file in json_files:
        modified_time = os.path.getmtime(file)
        files_with_timestamps.append((file, modified_time))
    latest_json_file = max(files_with_timestamps, key=lambda x: x[1])[0]
    return latest_json_file