import polars as pl
import numpy as np
import xgboost as xgb
import logging
import configparser
import os
import holidays
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
from commons.custom_logger import setup_logger
from commons.staging_modules import feature_engineer_time_to_seconds, \
    one_hot_encode_daytime, \
    is_holiday, \
    is_weekend, \
    compute_coordinates, \
    compute_centroid_distance, \
    write_df_toJSON, \
    write_df_toCSV
from commons.ml_modules import remove_null_values, \
    label_encode_column, \
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
    # stg_processed_loc:str = config.get("local-path-settings", "staging_processed_folder")
    # stg_processed_path:str = os.path.join(application_path, stg_processed_loc)
    # if not stg_processed_path:
    #     logger_object.error("Path not found for processed and enriched tax-trips. Application willl exit...")
    execution_timestamp:datetime = datetime.now().strftime('%Y%m%d')

    # ml_model_name = config.get("ml-settings", "duration_model_name")
    # split_perce:float = float(config.get("ml-settings", "train_test_split_perce"))
    # artifact_path:str = os.path.join(application_path, "model_artifacts", execution_timestamp)
    # create_folder(artifact_path)

    RANDOM_SEED:int = 42
    np.random.seed(RANDOM_SEED)
    
    #=========================================================================
    # READ THE PROCESSED-DATA PARQUET FILES FROM STAGING FOLDER OF TAXI_TRIPS
    #=========================================================================
    # partitions = Path(stg_processed_path).rglob("*.parquet")
    # for parquet_file in partitions:
    #     df = pl.read_parquet(parquet_file, use_pyarrow=True)
    #     break
    # df = pl.read_parquet(stg_processed_path)
    # print(df.shape)
    # print(df.columns)

    df = pl.read_parquet("/home/nspanos/external_projects/m151_web_systems_project/data/93ae16fe18964d3e8b375273ff53e130-0.parquet", use_pyarrow=True).head(10_000)
    
    #========================================================
    # REMOVE NULL VALUES FROM COLUMNS PU_ZONE, DO_ZONE
    #========================================================
    df = df.drop_nulls(["pu_zone", "do_zone"]) # during data exploration, identified location ids [264, 265] with no available data from the geospatial sample.
    print(df.shape)

    #=========================================================================
    # FEATURE ENGINEER PICKUP, DROPOFF SECONDS FROM BEGINNING OF EACH MONTH
    #=========================================================================
    df = feature_engineer_time_to_seconds(df, 'pickup')
    df = feature_engineer_time_to_seconds(df, 'dropoff')

    #=========================================================================
    # ONE-HOT ENCODE THE DAYTIME VALUES (RUSH-HOUR, OVERNIGHT, DAYTIME)
    #=========================================================================
    df = one_hot_encode_daytime(df, 'pickup_daytime')
    df = one_hot_encode_daytime(df, 'dropoff_daytime')

    #=========================================================================
    # CREATE A BINARY COLUMN TO DENOTE HOLIDAY PICKUP-DROPOFF DATES
    #=========================================================================
    us_holidays = holidays.country_holidays('US', years=range((datetime.now() - timedelta(days=10*365)).year, datetime.now().date().year))
    hol_dts = []
    for date, name in sorted(us_holidays.items()):
        hol_dts.append(date)
    df = is_holiday(df, 'pickup', hol_dts)
    df = is_holiday(df, 'dropoff', hol_dts)

    #=========================================================================
    # CREATE A BINARY COLUMN TO DENOTE IF PICKUP-DROPOFF DATES ARE WEEKENDS
    #=========================================================================
    df = is_weekend(df, 'pickup')
    df = is_weekend(df, 'dropoff')

    #========================================================================================================================
    # COMPUTE TRIP DISTANCE USING CENTROID DATA OF PICKUP-DROPOFF ZONES (SUPPLEMENTARY FEATURE TO ORIGINAL TRIP DISTANCE)
    #========================================================================================================================
    df = compute_coordinates(df, 'pickup') #generate pickup_coordinates
    df = compute_coordinates(df, 'dropoff')
    df = df.with_columns(
        pl.struct(['pickup_coordinates', 'dropoff_coordinates']) \
        .map_elements(lambda x: compute_centroid_distance(x['pickup_coordinates'], x['dropoff_coordinates']), return_dtype=pl.Float32).alias("centroid_distance")
    )

    #=========================================================================
    # ONE-HOT ENCODE PICKUP-DROPOFF ZONES
    #=========================================================================


    #=========================================================================
    # FILTER OUT UBNORMAL KM/H (KILOMETERS PER HOUR - AVERAGE SPEED) RECORDS
    #=========================================================================
    

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