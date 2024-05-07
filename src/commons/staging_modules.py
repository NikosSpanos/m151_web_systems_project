#!/usr/bin/env python
import hashlib
import glob
import polars as pl
import os
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Literal
from multiprocessing import Pool
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely import wkt

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

def init_partitioned_folder(stg_path:str, subfolder:str) -> str:
    new_stg_loc = os.path.join(stg_path, subfolder)
    os.makedirs(new_stg_loc, exist_ok=True)
    return new_stg_loc

def init_lnd_folder(lnd_parent_path:str) -> str:
    available_directories = [
        os.path.join(lnd_parent_path, directory) for directory in os.listdir(lnd_parent_path) if len(os.listdir(os.path.join(lnd_parent_path, directory))) !=0 
    ] # Retrieve the latest non-empty landing directory with collected data.
    return max(available_directories, key=os.path.getmtime)

def update_processed_metadata_file(metadata_file:str, metadata_dict:dict =None, latest_unprocessed_folder:datetime = None):
    try:
        with open(metadata_file, 'r') as file:
            data = json.load(file)
    except Exception as e:
        data = []
    if data != []:
        latest_checkpoint:dict = sorted(data, key=lambda x: x["execution_dt"], reverse=True)[0]
        if (latest_checkpoint["status"] == "unprocessed") and latest_checkpoint["latest_unprocessed_folder"] == latest_unprocessed_folder.strftime("%Y%m%d"):
            latest_checkpoint["status"] = "processed"
            data[data.index(latest_checkpoint)] = latest_checkpoint
            with open(metadata_file, "w") as file:
                json.dump(data, file, indent=4)
            return
    if not metadata_dict:
        return
    data.append(metadata_dict)
    with open(metadata_file, "w") as file:
        json.dump(data, file, indent=4)

def get_latest_processed_lnd_folder(metadata_file:str) -> Tuple[datetime, Optional[datetime], Optional[str]]:
    if not os.path.exists(metadata_file):
        data = []
        with open(metadata_file, 'w') as file:
            json.dump(data, file)
        return (datetime(year=1970, month=1, day=1).date(), None, None)
    with open(metadata_file, 'r') as file:
        data = json.load(file)
        if len(data) != 0:
            max_checkpoint = sorted(data, key=lambda x: x["latest_lnd_folder"], reverse=True)[0]
            latest_checkpoint:datetime = datetime.strptime(max_checkpoint["latest_lnd_folder"], '%Y%m%d').date()
            unprocessed_folder:datetime = datetime.strptime(max_checkpoint["latest_unprocessed_folder"], '%Y%m%d').date()
            status:str = max_checkpoint["status"]
        else:
            latest_checkpoint:datetime = datetime(year=1970, month=1, day=1).date()
            unprocessed_folder = None
            status = None
    return (latest_checkpoint, unprocessed_folder, status)

def get_latest_partitioned_folder(stg_path:str, logging_object:logging.Logger) -> str:
    if os.listdir(stg_path):
        available_directories = [os.path.join(stg_path, file) for file in os.listdir(stg_path)]
        latest_modified_directory = max(available_directories, key=os.path.getmtime)
        return latest_modified_directory
    else:
        logging_object.error("Partition directory is empty. Please verify that the data_preprocessing.py has been executed first.")
        return

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
    df.write_ndjson(filename)
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

def retrieve_latest_modified_file(relative_path:str, flag:str = "taxi_trip") -> str:
    file_pattern = "*_data.json" if flag == "taxi_trip" else "*_zones.json"
    pattern = os.path.join(relative_path, file_pattern) 
    target_files = glob.glob(pattern)
    files_with_timestamps = [(file, os.path.getmtime(file)) for file in target_files]
    latest_json_file = max(files_with_timestamps, key=lambda x: x[1])[0] if files_with_timestamps else None
    return latest_json_file

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

# ==================================================
# FEATURE ENGINEERING MODULES
# ==================================================
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

def feature_engineer_trip_distance(df:pl.DataFrame, cols:str) -> pl.DataFrame:
    df = df.with_columns(
        (df[cols] * 1.60934).round(2).alias(cols)
    )
    return df

def feature_engineer_trip_cost(df:pl.DataFrame, cols:list) -> pl.DataFrame:
    sum_expression = sum([pl.col(column) for column in cols])
    df = df.with_columns(
        sum_expression.alias("trip_cost")
    )
    return df

# ==================================================
# GEOSPATIAL DATA PROCESSING MODULES
# ==================================================
def identify_nyc_zones_with_non_float_values(df:pl.DataFrame, pattern:Literal[''], cols:list) -> pl.DataFrame:
    for column in cols:
        df = df.with_columns(pattern_matched = pl.col(column).str.contains(pattern)).filter(pl.col("pattern_matched")).drop("pattern_matched")
    return df

def identify_nyc_zones_with_multiple_spaced_values(df:pl.DataFrame, cols:list) -> pl.DataFrame:
    conditions = [df[col].str.split(" ").list.lengths() == 1 for col in cols]
    filter_expression = pl.reduce(lambda a, b: a & b, conditions)
    df = df.filter(filter_expression)
    return df

def identify_nyc_zones_with_non_integer_values(df:pl.DataFrame, cols:list) -> pl.DataFrame:
    conditions = [(~df[col].apply(lambda x: x.is_integer())) for col in cols]
    filter_expression = pl.reduce(lambda a, b: a & b, conditions)
    df = df.filter(filter_expression)
    return df

def identify_nyc_zones_with_non_string_values(df:pl.DataFrame, cols:list, pattern:Literal['']) -> pl.DataFrame:
    conditions = [(~df[col].str.contains(pattern)) for col in cols]
    filter_expression = pl.reduce(lambda a, b: a & b, conditions)
    df = df.filter(filter_expression)
    return df

def flatten_list_of_lists(data) -> list:
    return [subitem2 for subitem1 in data for subitem2 in subitem1]

def transform_polygons_to_multipolygons(flat_list:list) -> list:
    return [ MultiPolygon( [Polygon(coord) for coord in polygon]).wkt for polygon in  flat_list]

def compute_geo_areas(multipolygons:MultiPolygon) -> float:
    return wkt.loads(multipolygons).area

def compute_polygon_center(multipolygons:MultiPolygon) -> Point:
    return wkt.loads(multipolygons).centroid.wkt

def compute_longitute(centroid_value:Point):
    return wkt.loads(centroid_value).x

def compute_latitude(centroid_value:Point):
    return wkt.loads(centroid_value).y

# ==================================================
# DATA ENRICHMENT
# ==================================================
def enrich_partition_samples(partition:str, mapping_names:list, df_geo:pl.DataFrame, stg_processed_path:str):
    items = Path(partition).rglob("*.parquet")
    for parquet_file in items:
        # df_partition = pl.read_parquet(os.path.join(partition, "*.parquet"))
        partitions = dict(part.split('=') for part in parquet_file.parts if '=' in part)
        for key, value in partitions.items():
            df_partition= pl.read_parquet(parquet_file).with_columns(pl.lit(value, dtype=pl.Utf8).alias(key))
            merged_batch = df_partition.join(
                df_geo,
                left_on=pl.col("pulocationid"),
                right_on=pl.col("objectid"),
                how='left'
            ).rename(mapping_names[0])
            merged_batch = merged_batch.join(
                df_geo,
                left_on=pl.col("dolocationid"),
                right_on=pl.col("objectid"),
                how='left'
            ).rename(mapping_names[1])
            merged_batch.write_parquet(
                stg_processed_path,
                compression="zstd",
                use_pyarrow=True,
                pyarrow_options={
                    "partition_cols": ["partition_dt"],
                    "existing_data_behavior": "overwrite_or_ignore"
                }
            )
    # return merged_batch