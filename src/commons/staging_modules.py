#!/usr/bin/env python
import glob
import polars as pl
import os
import logging
import json
import numpy as np
import uuid
from datetime import datetime
from typing import Tuple, Optional, Literal
from multiprocessing import Pool
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely import wkt

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

def init_partitioned_folder(stg_path:str, subfolder:str) -> Tuple[str, str]:
    new_stg_loc = os.path.join(stg_path, subfolder)
    short_trip_loc,  long_trip_loc = (
        os.path.join(new_stg_loc, "short_trip"),
        os.path.join(new_stg_loc, "long_trip")
    )
    (
        os.makedirs(new_stg_loc, exist_ok=True),
        os.makedirs(short_trip_loc, exist_ok=True),
        os.makedirs(long_trip_loc, exist_ok=True),
    )
    return short_trip_loc, long_trip_loc

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

def daytime_value(hour_value:int):
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

def collect_data(args:Tuple[str,list,logging.Logger])-> pl.DataFrame:
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

def retrieve_latest_modified_folder(relative_path:str) -> Tuple[str, None]:
    files_with_timestamps = [(folder, os.path.getmtime(os.path.join(relative_path, folder))) for folder in os.listdir(relative_path)]
    latest_modified_folder = max(files_with_timestamps, key=lambda x: x[1])[0] if files_with_timestamps else None
    return os.path.join(relative_path, latest_modified_folder)

def fix_data_type(df:pl.LazyFrame, type_mapping:dict, dt_format:str = None) -> pl.LazyFrame:
    for column, dtype in type_mapping.items():
        if dtype == "datetime":
            df = df.with_columns(pl.col(column).str.to_datetime(dt_format))
        else:
            df = df.with_columns(pl.col(column).cast(dtype))
    return df

def remove_abnormal_dates(df:pl.LazyFrame, cols:list, dataset_year:datetime, start_of_time:datetime, logger_obj:logging.Logger) -> pl.LazyFrame:
    for col in cols:
        above_current_fyear = df.filter(pl.col(col).dt.year().gt(dataset_year.year))
        before_start_of_time = df.filter(pl.col(col).dt.year().lt(start_of_time.year))
        logger_obj.info("{0} dates after current fiscal year ({1}): {2}".format(col, dataset_year.year, above_current_fyear.collect().height))
        logger_obj.info("{0} dates before start of Unix time (1970-01-01) ({1}): {2}".format(col, start_of_time.year, before_start_of_time.collect().height))
        # Remove rows with year greater/lower than the dataset year
        df = df.filter(
            (pl.col(col).dt.year().le(dataset_year.year)) & (pl.col(col).dt.year().gt( start_of_time.year ))
        )
    return df

def remove_negative_charges(df:pl.LazyFrame, cols:list, logger_obj:logging.Logger) -> pl.LazyFrame:
    for col in cols:
        if col in ["tolls_amount", "extra", "mta_tax", "improvement_surcharge"]:
            negative_charges = df.filter(pl.col(col).lt(0))
            logger_obj.info("{0} with negative values (<0): {1}".format(col, negative_charges.collect().height))
            df = df.filter(pl.col(col).ge(0))
        else:
            negative_charges = df.filter(pl.col(col).le(0))
            logger_obj.info("{0} with negative values (<=0): {1}".format(col, negative_charges.collect().height))
            df = df.filter(pl.col(col).gt(0))
    return df

def remove_equal_pickup_dropoff_times(df:pl.LazyFrame, pu_col:str, do_col:str, logger_obj:logging.Logger) -> pl.LazyFrame:
    equal_pu_do_dt = df.filter(pl.col(pu_col).ge(pl.col(do_col)))
    logger_obj.info("Taxi trips without duration (pickup date >= drop-off date): {0}".format(equal_pu_do_dt.collect().height))
    df = df.filter(pl.col(pu_col).lt(pl.col(do_col)))
    return df

# ==================================================
# FEATURE ENGINEERING MODULES
# ==================================================
def feature_engineer_trip_duration(df:pl.LazyFrame,  pu_col:str, do_col:str, duratation_col_name:str) -> pl.LazyFrame:
    df = df.with_columns(
        ( ( ( ( pl.col(do_col) - pl.col(pu_col) )/60 )/1_000_000 ).round(2)).cast(pl.Float64).alias(duratation_col_name)
    )
    df = df.filter(pl.col(duratation_col_name).ge(1.0)) # Remove rows with trip duration less than 1 minute.
    return df

def feature_engineer_trip_hour(df:pl.LazyFrame, col:str) -> pl.LazyFrame:
    target_col:str = 'tpep_pickup_datetime' if col == 'pickup' else 'tpep_dropoff_datetime'
    df = df.with_columns(
        pl.col(target_col).dt.hour().cast(pl.Int8).alias(f"{col}_hour"),
    )
    return df

def feature_engineer_trip_month(df:pl.LazyFrame, col:str) -> pl.LazyFrame:
    target_col:str = 'tpep_pickup_datetime' if col == 'pickup' else 'tpep_dropoff_datetime'
    df = df.with_columns(
        pl.col(target_col).dt.month().cast(pl.Int8).alias(f"{col}_month"),
    )
    return df

def feature_engineer_trip_weekday(df:pl.LazyFrame, col:str) -> pl.LazyFrame:
    target_col:str = 'tpep_pickup_datetime' if col == 'pickup' else 'tpep_dropoff_datetime'
    df = df.with_columns(
            (pl.col(target_col).dt.weekday()).cast(pl.Int8).alias(f"{col}_weekday")
        )
    df = df.with_columns(
        pl.when(pl.col(f"{col}_weekday").eq(1)).then("Monday")
          .when(pl.col(f"{col}_weekday").eq(2)).then("Tuesday")
          .when(pl.col(f"{col}_weekday").eq(3)).then("Wednesday")
          .when(pl.col(f"{col}_weekday").eq(4)).then("Thursday")
          .when(pl.col(f"{col}_weekday").eq(5)).then("Friday")
          .when(pl.col(f"{col}_weekday").eq(6)).then("Saturday")
          .otherwise("Sunday")
          .alias(f"{col}_weekday_str")
    )
    return df

def feature_engineer_trip_daytime(df:pl.LazyFrame, daytime_mapper:list, cols:tuple) -> pl.LazyFrame:
    for col in cols:
        df = df.with_columns(
            # pl.col(col[0]).map_elements(daytime_value, return_dtype=pl.Utf8).map_dict(daytime_mapper).cast(pl.Int64).alias("{0}_daytime".format(col[1]))
            pl.col(col[0]).dt.hour().map_elements(daytime_value, return_dtype=pl.Utf8).map_dict(daytime_mapper).cast(pl.Int64).alias("{0}_daytime".format(col[1]))
        )
    return df

def feature_engineer_trip_distance(df:pl.LazyFrame, distance_col:str) -> pl.LazyFrame:
    # df:pl.DataFrame = df.collect()
    df:pl.LazyFrame = df.with_columns(
        (pl.col(distance_col) * 1.60934).round(2).alias(distance_col)
    )
    return df

def feature_engineer_trip_cost(df:pl.LazyFrame, cols:list) -> pl.LazyFrame:
    sum_expression = sum([pl.col(column) for column in cols])
    df = df.with_columns(
        sum_expression.alias("trip_cost")
    )
    return df

def feature_engineer_nearest_hour_quarter(df:pl.LazyFrame, col:str) -> pl.LazyFrame:
    target_col = 'tpep_pickup_datetime' if col == 'pickup' else 'tpep_dropoff_datetime'
    df = df.with_columns(
        pl.col(target_col).dt.truncate("15m", use_earliest=True).dt.strftime('%H:%M').cast(pl.Utf8).alias(f"{col}_quarter")
    )
    return df

def feature_engineer_time_to_seconds(df:pl.DataFrame, col:str):
    target_col:str = 'tpep_pickup_datetime' if col == 'pickup' else 'tpep_dropoff_datetime'
    df = df.with_columns(
            pl.col(target_col).dt.date().cast(pl.Datetime).alias(f'{col}_month_start')
        ).with_columns(
        (
            (pl.col(target_col) - pl.col(f'{col}_month_start')).dt.minutes()
        ).alias(f'{col}_seconds')
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
    return [ MultiPolygon( [Polygon(coord) for coord in polygon]).wkt for polygon in flat_list]

def compute_geo_areas(multipolygons:MultiPolygon) -> float:
    return wkt.loads(multipolygons).area

def compute_polygon_center(multipolygons:MultiPolygon) -> Point:
    return wkt.loads(multipolygons).centroid.wkt

def compute_longitude(centroid_value:Point):
    return wkt.loads(centroid_value).x

def compute_latitude(centroid_value:Point):
    return wkt.loads(centroid_value).y

def compute_coordinates(df:pl.LazyFrame, col:str) -> pl.LazyFrame:
    target_col:str = 'pu_polygon_centroid' if col == 'pickup' else 'do_polygon_centroid'
    location_data:str = f'{col}_location_cleaned'
    coordinates:str = f'{col}_coordinates'
    df = df.with_columns(
        pl.col(target_col).str.replace_all(r'POINT \(|\)', '').alias(location_data)
    ).with_columns(
        pl.col(location_data).str.split(' ').alias(coordinates)
    ).drop(location_data)
    return df

def compute_haversine_disntance(df:pl.LazyFrame, R:np.float64, coordinates:dict) -> pl.LazyFrame:
    pl.Config.set_fmt_float("full")
    multiplier:float = np.pi/180
    rad_lat1:pl.Expr = (pl.col(coordinates["pickup_points"]).list.last().cast(pl.Float64) * (multiplier))
    rad_lat2:pl.Expr = (pl.col(coordinates["dropoff_points"]).list.last().cast(pl.Float64) * (multiplier))
    rad_lng1:pl.Expr = (pl.col(coordinates["pickup_points"]).list.first().cast(pl.Float64) * (multiplier))
    rad_lng2:pl.Expr = (pl.col(coordinates["dropoff_points"]).list.first().cast(pl.Float64) * (multiplier))
    haversin:pl.Expr = (
        (rad_lat2 - rad_lat1).truediv(2).sin().pow(2) +
        ((rad_lat1.cos() * rad_lat2.cos()) * (rad_lng2 - rad_lng1).truediv(2).sin().pow(2))
    ).cast(pl.Float64)
    df = df.with_columns(
        (
            2 * R * (haversin.sqrt().arcsin())
        ).cast(pl.Float64).alias("haversine_centroid_distance")
    )
    return df

# ==================================================
# DATA ENRICHMENT
# ==================================================
def read_parquet_files(partition_path:str, stg_path:str, file_type:str="*.parquet") -> pl.LazyFrame:
    # files = glob.glob(os.path.join(stg_path, partition_path.split('/')[-1], "*.parquet"))
    files_stg_path:str = os.path.join(stg_path, partition_path.split('/')[-1])
    if os.path.exists(files_stg_path):
        # return pl.concat([pl.scan_parquet(file) for file in files])
        return pl.scan_parquet(
            os.path.join(files_stg_path, file_type)
        ).with_columns(
            pl.lit(partition_path.split("=")[1], dtype=pl.Utf8).alias("partition_dt")
        )
    else:
        return pl.LazyFrame([])

def enrich_geospatial_info(df_partition:pl.LazyFrame, df_geo:pl.LazyFrame, mapping_names:list) -> pl.LazyFrame:
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
    #========================================================
    # REMOVE NULL VALUES FROM COLUMNS PU_ZONE, DO_ZONE
    #========================================================
    merged_batch = merged_batch.drop_nulls(["pu_zone", "do_zone"])
    # during data exploration, identified location ids [264, 265] with no available data from the geospatial sample.
    #========================================================================================================================
    # COMPUTE TRIP DISTANCE USING CENTROID DATA OF PICKUP-DROPOFF ZONES (SUPPLEMENTARY FEATURE TO ORIGINAL TRIP DISTANCE)
    #========================================================================================================================
    merged_batch = compute_coordinates(merged_batch, 'pickup') #generate pickup_coordinates
    merged_batch = compute_coordinates(merged_batch, 'dropoff')
    coordinates:dict = {
        "pickup_points": "pickup_coordinates",
        "dropoff_points": "dropoff_coordinates"
    }
    merged_batch = compute_haversine_disntance(merged_batch, 6371.0087714150598, coordinates)
    return merged_batch

def enrich_partition_samples(args:Tuple[str, list, str, str, logging.Logger]):
    """
    Explaining the architecutral logic
    1. Read the file saved under the paritioned data file path. Those files are the preprocessed/clean data samples per partition.
    2. Read the already existing data saved under the /processed folder. Data enriched with geospatial info.
    Now we have two scenarios:
        Scenario 1: Enriched data samples already exist (bullet 2 returned non-empty LazyFrame) then compare the different hash keys.
        If no matching hash key then END program execution.
        Else compute geospatial info for the NON-MATCHING hash keys only.
        Scenario 2: If no existing data have been saved (i.e. /processed folder was empty), then compute geospatial info for all records.
    """
    partition, mapping_names, geo_path, stg_processed_path, logger = args
    parquet_files:str = os.path.join(partition, "*.parquet")  # Read all the parquet files under a specific partition folder
    # Read preprocessed/cleaned samples
    df_partition:pl.LazyFrame= pl.scan_parquet(parquet_files).with_columns(
        pl.lit(partition.split("=")[1], dtype=pl.Utf8).alias("partition_dt")
    )
    # Read existing parquet files under /processed directory
    existing_data:pl.LazyFrame = read_parquet_files(partition, stg_processed_path)
    # Read geospatial info
    geospatial_cols:list = ["objectid", "zone", "polygon_area", "polygon_centroid"]
    df_geo:pl.LazyFrame = pl.scan_ndjson(geo_path).select(*geospatial_cols)

    if not existing_data.limit(1).collect().is_empty():
        assert ("hashing_key" in existing_data.columns) and ("hashing_key" in df_partition), "Hashing column not found in existing AND new samples..Exiting"
        existing_hash_keys:pl.LazyFrame = existing_data.select(pl.col("hashing_key")).unique()
        new_hash_keys:pl.LazyFrame = df_partition.select(pl.col("hashing_key")).unique()
        #===========================
        # DEDUPLICATION
        #===========================
        non_matching_indices:pl.DataFrame = new_hash_keys.join(existing_hash_keys, on="hashing_key", how="anti").collect()
        #===========================
        if non_matching_indices.is_empty():
            logger.info(f"No new records to save for {partition.split('/')[-1]}")
            return
        df_partition_non_matched:pl.LazyFrame = df_partition.filter(pl.col("hashing_key").is_in(non_matching_indices["hashing_key"]))
        merged_batch:pl.LazyFrame = enrich_geospatial_info(df_partition_non_matched, df_geo, mapping_names)
        merged_batch.collect().write_parquet(
            stg_processed_path,
            compression="zstd",
            use_pyarrow=True,
            pyarrow_options={
                "basename_template": str(uuid.uuid4()) + '-{i}.parquet', #unique generation of names results in append behavior.
                "partition_cols": ["partition_dt"],
                "existing_data_behavior": "overwrite_or_ignore"
            }
        )
        return
    else:
        logger.info(
            f"Staging path: {os.path.join(stg_processed_path, partition.split('/')[-1])} is empty.\
            Ernriching and Writing all collected data per partition..."
        )
        # Geospatial info enrichment
        merged_batch:pl.LazyFrame = enrich_geospatial_info(df_partition, df_geo, mapping_names)
        # Write everything if the STAGING directory is empty.
        merged_batch.collect().write_parquet(
            stg_processed_path,
            compression="zstd",
            use_pyarrow=True,
            pyarrow_options={
                "basename_template": str(uuid.uuid4()) + '-{i}.parquet',
                "partition_cols": ["partition_dt"],
                "existing_data_behavior": "overwrite_or_ignore"
            }
        )
        return

# =====================================================
# CATEGORICAL DATA ENCODING TO NUMERIC REPRESENTATIONS
# =====================================================

def one_hot_encode_column(df:pl.LazyFrame, col:list) -> pl.LazyFrame:
    df_dummies = df.collect().to_dummies(col, separator='_', drop_first=False)
    return df_dummies.lazy()

def is_holiday(df:pl.LazyFrame, col:str, us_holidays:list) -> pl.LazyFrame:
    target_col = 'tpep_pickup_datetime' if col == 'pickup' else 'tpep_dropoff_datetime'
    df = df.with_columns(
        pl.col(target_col).cast(pl.Date).is_in(us_holidays).cast(pl.UInt8).alias(f"{col}_holiday")
    )
    return df

def is_weekend(df:pl.LazyFrame, col:str) -> pl.LazyFrame:
    target_col = 'tpep_pickup_datetime' if col == 'pickup' else 'tpep_dropoff_datetime'
    df = df.with_columns(
        pl.col(target_col).dt.weekday().is_in([6,7]).cast(pl.UInt8).alias(f"{col}_weekend")
    )
    return df