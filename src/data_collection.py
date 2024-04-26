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
    # DATA COLLECTION FROM SODA API [USING MULTI THREADING FOR CONCURRENT DOWNLOAD]
    #==============================================================
    logger_object.info("EXTRACTION STARTED - Data collection from the Socrata API started.")
    starting_checkpoint:datetime = fetch_starting_point(checkpoint_file, logger_object)

    with ThreadPoolExecutor(max_workers=threads_size) as executor:
        futures = [executor.submit(collect_data, i * batch_size, threshold_date, starting_checkpoint, logger_object) for i in range(threads_size)]
    logger_object.info("EXTRACTION FINISHED - Data collection from the Socrata API completed.")

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
