#!/usr/bin/env python

import sys
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from commons.custom_logger import setup_logger
from commons.landing_modules import validation_check_dt_threshold, \
    collect_data, \
    fetch_starting_point, \
    batch_size, \
    threads_size, \
    checkpoint_file

# Initialize module arguments
threshold_date:str = sys.argv[1] #extract and store records less than equal (<=) this date
current_year:int = datetime.now().year

def main(logger_object:logging.Logger, threshold_date:str):
    
    if len(sys.argv) != 2:
        logger_object.error("Insufficient number of arguments given - Usage: python data_collection.py date_value")
        sys.exit(1)
    
    #==============================================================
    # SETUP THRESHOLD DATE FOR DATA EXTRACTION - CHECK DATE FORMAT
    #==============================================================
    try:
        threshold_date:datetime = validation_check_dt_threshold(threshold_date, current_year, logger_object)
    except ValueError:
        logger_object.error("Invalid arguments. Please use YYYY-MM-DD format and for intervals use positive integer values.")
    
    #==============================================================
    # DATA COLLECTION FROM SODA API [USING MULTI_THREADING]
    #==============================================================
    logger_object.info("EXTRACTION STARTED - Data collection from the Socrata API started.")
    starting_checkpoint:datetime = fetch_starting_point(checkpoint_file, logger_object)

    with ThreadPoolExecutor(max_workers=threads_size) as executor:
        futures = [executor.submit(collect_data, i * batch_size, threshold_date, starting_checkpoint, logger_object) for i in range(threads_size)]
    logger_object.info("EXTRACTION FINISHED - Data collection from the Socrata API completed.")

    #==============================================================
    # DATA CONVERSION FROM JSON TO COLUMNAR TABLE
    #==============================================================
    # schema = pl.DataFrame(all_records).schema
    # df = pl.DataFrame(all_records, schema=schema)
    # df.columns = list(map(lambda x: x.lower(), df.columns))
    # logger_object.info("Total records retrieved: {0}".format(df.height))

    #==============================================================
    # CREATE THE PARTITION COLUMN(s) to split the raw .json files
    #==============================================================
    # df = df.with_columns(pl.col("tpep_pickup_datetime").str.to_datetime("%Y-%m-%dT%H:%M:%S.000").dt.date().alias("partition_dt"))
    # partition_values = df.select(pl.col("partition_dt")).unique().to_dict(as_series=False)
    # logger.info("Unique partitions :{0}".format(partition_values["partition_dt"]))

    #========================================================
    # SAVE RAW DATA TO LADNING ZONE
    #========================================================
    # Establish connection to S3 bucket resource
    # s3_bucket_resource = boto3.resource(
    #     service_name  = "s3",
    #     region_name = config.get("settings", "aws_bucket_region"),
    #     aws_access_key_id = config.get("settings", "aws_access_key"),
    #     aws_secret_access_key = config.get("settings", "aws_secret_key"),
    # )
    #------------------------------------------------------------------------------------
    # bucket_name = config.get("settings", "aws_bucket_name")
    # bucket = s3_bucket_resource.Bucket(bucket_name)
    # execution_timestamp = datetime.now().strftime('%Y_%m_%d_00_00_00')
    # destination = config.get("settings", "aws_landing_unprocessed_folder") + "yellow_taxi_trip_sample_" + str(execution_timestamp) + ".json"
    # json_buffer = StringIO()
    
    #===============================================================
    # BENCHMARK 1 (execution time: 33:36)
    # Command: python src/data_batch_collection.py 2017-07-26 10
    #===============================================================
    # df.write_json(json_buffer)
    # try:
    #     # Retrieve the existing files in the bucket data/landing/unprocessed folder/
    #     for object_summary in bucket.objects.filter(Prefix=config.get("settings", "aws_landing_unprocessed_folder")):
    #         logger_object.info(object_summary.key)
    #     logger_object.info("WRITE STARTED - Writing JSON file with collected under path : {0}".format(destination))
    #     bucket.put_object(Bucket = bucket_name, Key = destination, Body = json_buffer.getvalue())
    #     logger_object.info("WRITE COMPLETED - Data have been written to landing zone under path: {0}".format(destination))
    #     for object_summary in bucket.objects.filter(Prefix=config.get("settings", "aws_landing_unprocessed_folder")):
    #         logger_object.info(object_summary.key)
    # except Exception as e:
    #     logger_object.error(e)
    #     logger_object.error("WRITE TO LANDING ZONE FAILED - Check execution logs for errors.")

if __name__ == "__main__":
    project_folder = "data_collection"
    log_filename = f"batch_collection_execution_logs_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
    logger = setup_logger(project_folder, log_filename)

    try:
        main(logger, threshold_date)
        logger.info("SUCCESS: Batch collection process completed.")
    except Exception as e:
        logger.error(e)
        logger.error("FAIL: Batch collection process failed.")
