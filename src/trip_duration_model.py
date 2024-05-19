import polars as pl
import numpy as np
import xgboost as xgb
import logging
import configparser
import os
import mlflow
import mlflow.sklearn
from typing import List
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from commons.custom_logger import setup_logger
from commons.staging_modules import retrieve_latest_modified_folder, \
    one_hot_encode_daytime, \
    write_df_toJSON, \
    write_df_toCSV
from commons.ml_modules import label_encode_column, \
    train_linear_regressor, \
    train_randomforest_regressor, \
    train_xgboost_regressor, \
    make_predictions, \
    save_label_encoder

def duration_predictor(logger_object:logging.Logger):
    
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
    stg_processed_loc:str = config.get("local-path-settings", "staging_processed_folder")
    stg_processed_path:str = os.path.join(application_path, stg_processed_loc)
    if not stg_processed_path:
        logger_object.error("Path not found for processed and enriched tax-trips. Application willl exit...")
    execution_timestamp:datetime = datetime.now().strftime('%Y%m%d')

    # ml_model_name = config.get("ml-settings", "duration_model_name")
    # split_perce:float = float(config.get("ml-settings", "train_test_split_perce"))
    # artifact_path:str = os.path.join(application_path, "model_artifacts", execution_timestamp)
    # create_folder(artifact_path)

    RANDOM_SEED:int = 42
    np.random.seed(RANDOM_SEED)
    
    #=========================================================================================
    # READ THE PROCESSED-DATA PARQUET FILES FROM LATEST MODIFIED STAGING FOLDER OF TAXI_TRIPS
    #=========================================================================================
    # partitions = Path(stg_processed_path).rglob("*.parquet")
    latest_modified_stg_folder:str = retrieve_latest_modified_folder(stg_processed_path)
    partitions = Path(latest_modified_stg_folder)
    
    # parquet_directories = [x for x in partitions.iterdir() if x.is_dir()]
    parquet_directories = ["/home/nspanos/m151_web_systems_project/data/staging/processed/taxi_trips/20240519/partition_dt=202110"]
    print(parquet_directories)

    def read_parquet_files_in_chunks(parquet_files: List[str], chunk_size: int = 500_000):
        schema:dict={
            'trip_duration': pl.Float64,
            'trip_distance': pl.Float64,
            'pickup_daytime_2': pl.UInt8,
            'pickup_daytime_3': pl.UInt8,
            'pickup_quarter': pl.Int8,
            'pickup_holiday': pl.UInt8,
            'pickup_weekend': pl.UInt8,
            'haversine_centroid_distance': pl.Float64
        }
        current_chunk = pl.LazyFrame(schema = schema)
        current_chunk_rows = 0
        chunk_size: int = 500_000
        iteration = 1
        for directory in parquet_files:
            print(directory)
            df:pl.LazyFrame = pl.concat([pl.scan_parquet(os.path.join(directory, "*.parquet"))]).select(list(schema.keys()))
            df_rows:int = df.select(pl.count()).collect().item(0,0)
            while df_rows > 0:
                remaining_space = chunk_size - current_chunk_rows
                print(iteration)
                print(remaining_space)
                if df_rows <= remaining_space:
                    current_chunk = pl.concat([current_chunk, df], how='vertical')
                    current_chunk_rows += df_rows
                    df_rows = 0
                else:
                    current_chunk = pl.concat([current_chunk, df.slice(current_chunk_rows, remaining_space)], how='vertical')
                    df = df.slice(remaining_space, None)
                    df_rows -= remaining_space
                    yield current_chunk
                    current_chunk = pl.LazyFrame(schema=schema)
                    current_chunk_rows = 0
                iteration +=1

        if current_chunk_rows > 0:
            yield current_chunk

    for chunk_df in read_parquet_files_in_chunks(parquet_directories):
        # Here, you can process each chunk DataFrame as needed
        print(chunk_df.select(pl.count()).collect())
        print(chunk_df.collect().head())

    # You can also train your model on this chunk
    # train_model(chunk_df)
    # # preprocessed_df:pl.DataFrame = pl.DataFrame([])
    # preprocessed_df:pl.LazyFrame = pl.LazyFrame([])
    # lazy_frames = []
    # for parquet_dir in parquet_directories:
  
    #     # DataFrame API
    #     # df = pl.concat([pl.scan_parquet(os.path.join(parquet_dir, "*.parquet"))]).collect()
    #     # preprocessed_df = preprocessed_df.vstack(df)
        
    #     # LazyFrame API
    #     df = pl.concat([pl.scan_parquet(os.path.join(parquet_dir, "*.parquet"))])
    #     lazy_frames.append(df)
    
    # preprocessed_df = pl.concat(lazy_frames)

    # print(preprocessed_df.columns)
    # # print(preprocessed_df.shape)
    # print(preprocessed_df.select(pl.count()).collect())

    exit()





    write_df_toJSON("{0}/data/staging/processed/{1}".format(application_path, execution_timestamp), df, "nulls_pruned_yellow_taxi_trip_processed_data_{0}".format(samples_str), logger_object)
    write_df_toCSV("{0}/data/staging/processed/{1}".format(application_path, execution_timestamp), df, "nulls_pruned_yellow_taxi_trip_processed_data_{0}".format(samples_str), logger_object)

    #========================================================
    # POLARS TO PANDAS FOR BETTER HANDLING FROM SKLEARN/XGBOOST
    #========================================================
    df = df.to_pandas()

    #========================================================
    # LABEL ENCODE CATEGORICAL VARIABLES
    #========================================================
    label_encoder = LabelEncoder()
    encode_cols = ["puzone", "dozone"]
    fit_encoder = True
    for name in encode_cols:
        df, label_encoder, refit = label_encode_column(df, name, label_encoder, fit_encoder, logger_object)
        if refit:
            save_label_encoder(label_encoder, artifact_path + "/{0}_label_encoder.joblib".format(name))

    #========================================================
    # ISOLATE X, Y FEATURES AND SPLIT THEM TO TRAIN/TEST SAMPLES
    #========================================================
    # x_features = ["puzone_encoded", "dozone_encoded", "trip_distance", "pickup_daytime"]
    x_features = ["puzone_encoded", "dozone_encoded", "pickup_daytime"]
    y_features = ["trip_duration"]
    X = df[x_features]
    y = df[y_features]
    logger_object.info(y.describe())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_perce, random_state=RANDOM_SEED)

    custom_mlruns_path:str = os.path.join(application_path, "mlruns", execution_timestamp)
    mlflow.set_tracking_uri("file://{0}".format(custom_mlruns_path))
    mlflow.set_experiment("trip-{0}-prediction-model".format(ml_model_name))

    model_regressors = ["xgboost", "linear", "randomforest"]
    best_score:np.float64 = np.inf
    for model_name in model_regressors:
        if model_name == "xgboost":
            with mlflow.start_run(run_name="{0}-model".format(model_name), nested=False):
                logger_object.info("Strarted training/evaluating {0} regressor".format(model_name))
                dtrain = xgb.DMatrix(X_train, label=y_train)
                params = {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "learning_rate": 0.01,
                    "max_depth": 10,
                    "num_parallel_tree": 100
                }
                model = train_xgboost_regressor(params, dtrain)
                dtest = xgb.DMatrix(X_test)
                mlflow.log_params(params)
                rmse, second_metric_value, second_metric_name = make_predictions(model_name, model, dtest, y_test, logger_object)
                mlflow.log_metric("root-mean-squared-error", rmse)
                mlflow.log_metric("{0}".format(second_metric_name), second_metric_value)
                logger_object.info("Completed training/evaluating {0} regressor".format(model_name))
                if rmse < best_score:
                    logger_object.info("Found a model that improved RMSE from {0} to {1}".format(best_score, rmse))
                    best_score = rmse
                    mlflow.sklearn.log_model(model, "best_{0}_recommendation".format(ml_model_name), serialization_format = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
                logger_object.info("========================================================================================")
            mlflow.end_run()
        elif model_name == "linear":
            with mlflow.start_run(run_name="{0}-model".format(model_name), nested=False):
                logger_object.info("Strarted training/evaluating {0} regressor".format(model_name))
                dtrain = X_train.values
                dtest = X_test.values
                params = {
                    "fit_intercept": True,
                    "copy_X": True
                }
                model = train_linear_regressor(dtrain, y_train.values, params)
                mlflow.log_params(params)
                rmse, second_metric_value, second_metric_name = make_predictions(model_name, model, dtest, y_test, logger_object)
                mlflow.log_metric("root-mean-squared-error", rmse)
                mlflow.log_metric("{0}".format(second_metric_name), second_metric_value)
                logger_object.info("Completed training/evaluating {0} regressor".format(model_name))
                if rmse < best_score:
                    logger_object.info("Found a model that improved RMSE from {0} to {1}".format(best_score, rmse))
                    best_score = rmse
                    mlflow.sklearn.log_model(model, "best_{0}_recommendation".format(ml_model_name), serialization_format = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
                logger_object.info("========================================================================================")
            mlflow.end_run()
        else:
            with mlflow.start_run(run_name="{0}-model".format(model_name), nested=False):
                logger_object.info("Strarted training/evaluating {0} regressor".format(model_name))
                dtrain = X_train.values
                dtest = X_test.values
                params = {
                    "n_estimators": 100,
                    "criterion": "squared_error",
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": RANDOM_SEED
                }
                model = train_randomforest_regressor(dtrain, y_train.values.ravel(), params)
                mlflow.log_params(params)
                rmse, second_metric_value, second_metric_name = make_predictions(model_name, model, dtest, y_test, logger_object)
                mlflow.log_metric("root-mean-squared-error", rmse)
                mlflow.log_metric("{0}".format(second_metric_name), second_metric_value)
                logger_object.info("Completed training/evaluating {0} regressor".format(model_name))
                if rmse < best_score:
                    logger_object.info("Found a model that improved RMSE from {0} to {1}".format(best_score, rmse))
                    best_score = rmse
                    mlflow.sklearn.log_model(model, "best_{0}_recommendation".format(ml_model_name), serialization_format = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
                logger_object.info("========================================================================================")
            mlflow.end_run()
    logger_object.info("Completed training/evaluating {0}-model".format(ml_model_name))

if __name__ == "__main__":
    project_folder = "duration_recommendation_model"
    log_filename = f"{project_folder}_logs_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
    logger = setup_logger(project_folder, log_filename)
    try:
        duration_predictor(logger)
        logger.info("SUCCESS: duration recommendation model completed.")
    except Exception as e:
        logger.error(e)
        logger.error("FAIL: duration recommendation model failed.")