#!/usr/bin/env python

import logging
import configparser
import os
import polars as pl
from datetime import datetime
from commons.custom_logger import setup_logger
from commons.staging_modules import init_stg_path, \
    retrieve_latest_modified_file, \
    identify_nyc_zones_with_non_float_values, \
    identify_nyc_zones_with_multiple_spaced_values, \
    identify_nyc_zones_with_non_string_values, \
    flatten_list_of_lists, \
    compute_geo_areas, \
    compute_polygon_center, \
    transform_polygons_to_multipolygons, \
    compute_longitute, \
    fix_data_type, \
    write_df_toJSON

def main(logger_object:logging.Logger):
    
    #========================================================
    # INITIALIZE MODULE VARIABLES
    #========================================================
    # Initialize configparser object
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    config.read(os.path.join(parent_dir, "config", "config.ini"))

    geospatial_data_folder = config.get("local-path-settings", "geospatial_data_folder")
    geospatial_processed_folder = config.get("local-path-settings", "geospatial_processed_folder")

    #========================================================
    # CHECK EXISTENCE OF GEOSPATIAL DATA STORAGE
    #========================================================
    geospatial_data_storage = os.path.join(parent_dir, geospatial_data_folder)
    if not os.path.exists(geospatial_data_storage ) :
        logger_object.error("Geospatial directory with collected data not found. Application will exit().")
    
    geospatial_processed_storage = os.path.join(parent_dir, geospatial_processed_folder)
    init_stg_path(geospatial_processed_storage, logger_object)
    #========================================================
    # LOAD COLLECTED DATA
    #========================================================
    df_geo = pl.read_json(retrieve_latest_modified_file(geospatial_data_storage))

    #====================================================================================================
    # IDENTIFY AND DROP ROWS THAT HAVE SHAPE LENGTH/AREA NON EQUAL TO A VALID FLOAT NUMBER BETWEEN (0,1)
    #====================================================================================================
    pattern = r"^0\.\d+$"
    cols_list:list = ["shape_leng", "shape_area"]
    df_geo = identify_nyc_zones_with_non_float_values(df_geo, pattern, cols_list)
    #====================================================================================================
    # IDENTIFY AND DROP ROWS THAT HAVE MULTIPLE SHAPE LENGTH/AREA (> 1 VALUE PER ROW)
    #====================================================================================================
    cols_list:list = ["shape_leng", "shape_area", "objectid", "location_id"]
    df_geo = identify_nyc_zones_with_multiple_spaced_values(df_geo, cols_list)
    #====================================================================================================
    # IDENTIFY AND DROP ROWS THAT HAVE NUMERIC DATA FOR ZONE, BOROUGH
    #====================================================================================================
    pattern = r"\d"
    cols_list:list = ["zone", "borough"]
    df_geo = identify_nyc_zones_with_non_string_values(df_geo, cols_list, pattern)

    #====================================================================================================
    # TRANSFORM POLYGON OBSERVATIONS TO MULTIPOLYGONS USING GEO MODULE 'SHAPELY'
    #====================================================================================================
    df_geo = df_geo.unnest("the_geom")
    df_geo = df_geo.select(["objectid", "location_id", "zone", "borough", "coordinates"])
    flattened_lists:list = [flatten_list_of_lists(row) for row in df_geo["coordinates"].to_list()]
    multipolygons:list = transform_polygons_to_multipolygons(flattened_lists)

    df_geo = df_geo.with_columns(pl.Series("multipolygons", multipolygons)) \
             .with_columns(polygon_area=pl.col("multipolygons").map_elements(compute_geo_areas)) \
             .with_columns(polygon_centroid=pl.col("multipolygons").map_elements(compute_polygon_center)) \
             .drop("coordinates")
    #====================================================================================================
    # TRANSFORM THE DATATYPE OF NUMERICAL COLUMNS TO APPROPRIATE DATATYPE
    # The reason to apply this operation after the logical checks is because we used String operations over the abovementioned steps.
    #====================================================================================================
    cast_str = pl.Utf8
    cast_float = pl.Float64
    cast_object = pl.Object
    dtype_map = {
        "objectid": cast_str,
        "location_id": cast_str,
        "zone": cast_str,
        "borough": cast_str,
        "multipolygons": cast_str,
        "polygon_area": cast_float,
        "polygon_centroid": cast_str
    }
    df_geo = fix_data_type(df_geo, dtype_map)

    #====================================================================================================
    # COMPUTE NULL VALUES PER COLUMN
    #====================================================================================================
    df_nulls = df_geo.select(pl.all().is_null().sum()).to_dicts()[0]
    null_column_names = [k for k, v in df_nulls.items() if v > 0]
    logger_object.info("Column names with null values: {0}".format(null_column_names))

    #====================================================================================================
    # WRITE PROCESSED GEOSPATIAL DATA TO DISK
    #====================================================================================================
    logger_object.info(f"Final Dataframe has shape: {df_geo.shape}")
    write_df_toJSON(geospatial_processed_storage, df_geo, "nyc_taxi_zones", logger_object)
    logger_object.info("PROCESSING FINISHED - Data processing of NYC Zone Districts completed.")

if __name__ == "__main__":
    project_folder = "nyc_zones_processing"
    log_filename = f"{project_folder}_execution_logs_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
    logger = setup_logger(project_folder, log_filename)
    try:
        main(logger)
        logger.info("SUCCESS: NYC zones processing completed.")
    except Exception as e:
        logger.error(e)
        logger.error("FAIL: NYC zones processing failed.")