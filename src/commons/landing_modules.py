import logging
import sys
import requests
import json
import configparser
import os
import threading
from datetime import datetime, timedelta

# Initialize configparser object
config = configparser.ConfigParser()
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
config.read(os.path.join(parent_dir, "config", "config.ini"))

# Initialize configuration parameters for data collection pipeline
execution_date = datetime.now().strftime('%Y%m%d')
application_path = config.get("settings", "application_path")
api_url = config.get("api-settings", "collection_api")
api_token = config.get("api-settings", "app_token")
batch_size = int(config.get("api-settings", "batch_size"))
threads_size = int(config.get("api-settings", "threads_value"))
checkpoint_folder = os.path.join(application_path, config.get("local-path-settings", "metadata_folder"))
checkpoint_file = os.path.join(application_path, checkpoint_folder, config.get("local-path-settings", "metadata_file"))
# Check if metadata file exists
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)

# Initialize landing location path
data_loc:str = os.path.join(application_path, "data", "landing", execution_date)
# Check if landing path exists
if not os.path.exists(data_loc):
    os.makedirs(data_loc)

# Initialize threading lock for thread-safe read/write operations from same files
lock = threading.Lock()

# Initialize the value of global variable max_datetime to None for the first iteration.
max_datetime = None

def end_date_calculation(dt_start: datetime, dt_interval:int) -> datetime:
    ending_date = dt_start - timedelta(days=dt_interval)
    return ending_date

def date_calculation(starting_date:datetime, date_interval: int) -> datetime:
    end_date = end_date_calculation(starting_date, date_interval)
    return end_date

def validation_check_dt_threshold(threshold_date:str, current_year:int, logger_object:logging.Logger) -> str:
    dt_year = int(threshold_date.split("-")[0])
    dt_month = int(threshold_date.split("-")[1])
    dt_day = int(threshold_date.split("-")[2])

    if (dt_year > current_year):
        logger_object.error("Invalid year value. Year cannot be greater than the current fiscal year.")
        sys.exit(1)
    if (dt_month > 12):
        logger_object.error("Invalid month value. Month should be a positive integer between 1-12.")
        sys.exit(1)
    if (dt_day < 1 or dt_day > 31):
        logger_object.error("Invalid day value. Day should be a positive integer between 1-31.")
        sys.exit(1)
    dt_object = datetime.strptime(threshold_date, "%Y-%m-%d")

    #Apply the floating_timestamp format for querying the taxi trip data
    # threshold_date = dt_object.strftime("%Y-%m-%dT%H:%M:%S")
    threshold_date = dt_object.isoformat()
    logger_object.info(f"Extracting data until: {threshold_date}")

    return threshold_date

def calculate_thread(offset_value:int, batch_size_value:int, num_threads:int):
    # Calculate the batch index based on the batch size and the size of each batch
    batch_index = offset_value // batch_size_value
    
    # Calculate the thread number (1-based index)
    thread_number = (batch_index % num_threads) + 1
    
    return thread_number

def fetch_maximum_collected_date(chunk_of_records:list):
    with lock:
        sorted_records = sorted(chunk_of_records, key=lambda x: x["tpep_pickup_datetime"], reverse=True)
        return datetime.fromisoformat(sorted_records[0]["tpep_pickup_datetime"])

def compute_stored_date(checkpoint_file:str, logger_object:logging.Logger):
    if os.path.isfile(checkpoint_file):
        with open(checkpoint_file, 'r') as file:
            checkpoints = json.load(file)
        if len(checkpoints) != 0:
            max_checkpoint = sorted(checkpoints, key=lambda x: x["max_date"], reverse=True)
            latest_checkpoint = datetime.fromisoformat(max_checkpoint[0]["max_date"])
        else:
            latest_checkpoint = datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0)
    else:
        logger_object.info("Checkpoints file not found - Creating a new empty file for storing checkpoints.")
        data = []
        with open(checkpoint_file, 'w') as file:
            json.dump(data, file)
        latest_checkpoint = datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0) #Initialize a very small value for date
    return latest_checkpoint

def fetch_latest_collected_date(checkpoint_file:str, logger_object:logging.Logger):
    with lock:
        latest_checkpoint = compute_stored_date(checkpoint_file, logger_object)
        return latest_checkpoint

def fetch_starting_point(checkpoint_file:str, logger_object:logging.Logger):
    latest_checkpoint = compute_stored_date(checkpoint_file, logger_object)
    return latest_checkpoint

def update_max_datetime(new_stored_date:datetime, logger_object:logging.Logger):
    global max_datetime
    with lock:
        if max_datetime is None or (new_stored_date > max_datetime):
            max_datetime = new_stored_date
            reporting_dict = {
                    "max_date": max_datetime.isoformat(),
                    "execution_date": execution_date
                }
            save_latest_collected_date(checkpoint_file, reporting_dict, logger_object)

def save_latest_collected_date(checkpoint_file:str, reporting_dict:dict, logger_object:logging.Logger):
    # with lock:
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger_object.error(e)
        data = []
    data.append(reporting_dict)
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f, indent=4)

def collect_data(start_offset:int, threshold_date:datetime, starting_checkpoint:datetime, logger_object:logging.Logger):
    iteration_value:int = 1
    offset = start_offset

    # Compute the thread index for logging
    thread_indx = calculate_thread(offset, batch_size, threads_size)
    
    while True:
        # Fetch the latest checkpoint date
        stored_checkpoint_date = fetch_latest_collected_date(checkpoint_file, logger_object)

        params = {
            "$limit": batch_size,
            "$offset": offset,
            "$$app_token": api_token,
            "$where": f"tpep_pickup_datetime > '{starting_checkpoint.isoformat()}' and tpep_pickup_datetime <= '{threshold_date}'"
        }
        response = requests.get(api_url, params=params)

        if response.status_code == 200:
            # Collect response from json object
            data:list = response.json()

            # Fetch the maximum date from the batch of records collected
            collected_batch_date = fetch_maximum_collected_date(data)

            if (collected_batch_date <= starting_checkpoint and starting_checkpoint != datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0)):
                logger_object.info("Thread: {0}".format(thread_indx))
                logger_object.info("Up-to-date / Collected all the available data. Application will exit.")
                break

            if not data: # End the while loop based on three different scenarios
                if (max_datetime == stored_checkpoint_date):
                    logger_object.info("Collected all available records for the specified extraction query. Application will exit.")
                else:
                    logger_object.info("Request returned 0 records. Please check the correctness/validity of extraction query. Application will exit.")
                break

            # Update the maximum date
            update_max_datetime(collected_batch_date, logger_object)

            # Save the collected data of current batch
            with open("{0}/yellow_taxi_trip_data_offset_{1}.json".format(data_loc, offset + batch_size), "w") as f:
                json.dump(data, f, indent=4)

            # Log information of the current batch collected for better handling of collected data
            logger_object.info("Offset: {0}".format(offset + batch_size))
            logger_object.info("Response Status: {0}".format(response.status_code))
            logger_object.info("Thread: {0}".format(thread_indx))
            logger_object.info("Records downloaded: {0}".format(len(data)))
            logger_object.info("Records filtered: {0}".format(len(data)))
            logger_object.info("Iteration value: {0}".format(iteration_value))
            logger_object.info("-"*20)
            
            # Update offset and iteration_value for the correct computations of the next thread pool
            offset += batch_size * threads_size
            iteration_value +=1
        else:
            logger_object.error("API request failed.")
            logger_object.error("Error: {0}".format(response.status_code))
            logger_object.error(response.text)
            break