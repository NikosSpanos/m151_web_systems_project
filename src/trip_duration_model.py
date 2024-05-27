#!/usr/bin/env python
import polars as pl
import numpy as np
import logging
import configparser
import os
import time
import json
import joblib
from pathlib import Path
from argparse import ArgumentParser
from typing import List
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from commons.custom_logger import setup_logger, compute_execution_time
from commons.staging_modules import retrieve_latest_modified_folder, \
    one_hot_encode_column
from commons.ml_modules import init_model_artifacts, \
    ML_MODELING

def duration_predictor(logger_object:logging.Logger):
    
    #========================================================
    # INITIALIZE MODULE VARIABLES
    #========================================================
    # Initiate random seed for reprducibility.
    RANDOM_SEED:int = 42
    np.random.seed(RANDOM_SEED)

    # Initialize ConfigParser() class
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    config.read(os.path.join(parent_dir, "config", "config.ini"))
    
    # Initialize ArgumentParser() class
    parser = ArgumentParser()
    parser.add_argument(
        "-tt", "--trip_type", type=str, help="Select trip type to enrich with geospatial data", default="short_trip", required=True
    )
    parser.add_argument(
        "-mr", "--model_regressor", type=str, help="A valid model regressor name", default="random_forest_regressor", required=True
    )
    args = parser.parse_args()
    
    # Import configuration variables
    application_path:str = config.get("settings", "application_path")
    stg_processed_loc:str = config.get("local-path-settings", "staging_processed_folder")
    stg_processed_path:str = os.path.join(application_path, stg_processed_loc)
    if not stg_processed_path:
        logger_object.error("Path not found for processed and enriched tax-trips. Application willl exit...")
    execution_timestamp:datetime = datetime.now().strftime('%Y%m%d')
    chunk_size:int = int(config.get("ml-settings", "chunk_size"))
    train_test_split_perce:float = float(config.get("ml-settings", "train_test_split_perce"))

    # Initiate folder structure of the python process.
    model_artifacts_parent:str = config.get("ml-settings", "model_artifacts_path")
    model_artifacts_child:str = config.get("ml-settings", "duration_model_artifact")
    (model_artifacts_path, model_residuals_path, model_metadata_path) = (
        os.path.join(
            application_path, model_artifacts_parent, model_artifacts_child, execution_timestamp, args.trip_type, 'models'
        ),
        os.path.join(
            application_path, model_artifacts_parent, model_artifacts_child, execution_timestamp, args.trip_type, 'residuals'
        ),
        os.path.join(
            application_path, model_artifacts_parent, model_artifacts_child, execution_timestamp, args.trip_type, 'metadata'
        )
    )
    init_model_artifacts(model_artifacts_path, logger_object)
    init_model_artifacts(model_residuals_path, logger_object)
    init_model_artifacts(model_metadata_path, logger_object)

    # Initializa ML_MODELING() class
    ml_modeling = ML_MODELING(
        args.model_regressor,
        model_artifacts_path,
        model_metadata_path,
        model_residuals_path,
        10,
        RANDOM_SEED
    )
    assert (args.model_regressor in ml_modeling.valid_regressor_names) and args.trip_type in ("short_trip", "long_trip"), "Invalid arguments given. Check script input arguments."

    #=========================================================================================
    # READ THE PROCESSED-DATA PARQUET FILES FROM LATEST MODIFIED STAGING FOLDER OF TAXI_TRIPS
    #=========================================================================================
    latest_modified_stg_folder:str = os.path.join(retrieve_latest_modified_folder(stg_processed_path), args.trip_type)
    print(latest_modified_stg_folder)
    partitions:Path = Path(latest_modified_stg_folder)

    parquet_directories:list = [x for x in partitions.iterdir() if x.is_dir()]
    # parquet_directories = ["/home/nspanos/m151_web_systems_project/data/staging/processed/taxi_trips/20240526/short_trip/partition_dt=201101"]

    def read_parquet_files_in_chunks(parquet_files: List[str], chunk_size: int = 500_000):
        schema:dict = {
            'trip_duration': pl.Float64,
            'trip_distance': pl.Float64,
            'pickup_daytime': pl.Int64,
            'pickup_hour': pl.Int8,
            'pickup_weekday': pl.Int8,
            'pickup_month': pl.Int8,
            'pickup_quarter': pl.Int8,
            'pickup_seconds': pl.Int64,
            'pickup_holiday': pl.UInt8,
            'pickup_weekend': pl.UInt8,
            'haversine_centroid_distance': pl.Float64
        }
        current_chunk:pl.LazyFrame = pl.LazyFrame(schema = schema)
        current_chunk_rows: int = 0
        iteration:int = 1
        for directory in parquet_files:
            df:pl.LazyFrame = pl.concat([pl.scan_parquet(os.path.join(directory, "*.parquet"))]).select(list(schema.keys()))
            # print(df.columns)
            #.select(list(schema.keys()))
            df_rows:int = df.select(pl.count()).collect().item(0,0)
            while df_rows > 0:
                remaining_space = chunk_size - current_chunk_rows
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
    
    for iteration, chunk_df in enumerate(read_parquet_files_in_chunks(parquet_directories, chunk_size), 1):
        start_time:float = time.perf_counter()
        logger_object.info("\n")
        logger_object.info(f"Chunk {iteration} | Started model training process")
        # SHUFFLE ROWS
        shuffled_df:pl.DataFrame = chunk_df.collect().sample(fraction=1, with_replacement=False, shuffle=True, seed=1234)
        chunk_df:pl.LazyFrame = shuffled_df.lazy()
        # FILTER OUT OUTLIERS BASED ON PERCENTILES OF TRIP_DURATION
        lowest_quantile = 0.10
        highest_quantile = 0.95
        percentile_bottom_bound = chunk_df.select("trip_duration").quantile(lowest_quantile).collect().item()
        percentile_top_bound = chunk_df.select("trip_duration").quantile(highest_quantile).collect().item()
        rounded_bottom_bound = np.floor(percentile_bottom_bound)
        chunk_df = chunk_df.filter(pl.col("trip_duration") >= rounded_bottom_bound)
        chunk_df = chunk_df.filter(pl.col("trip_duration") <= percentile_top_bound)
        # COMPUTE TOTAL REMAINING ROWS AFTER QUANTIZATION
        chunk_df_height:int = chunk_df.select(pl.count()).collect().item()
        # print(chunk_df_height)
        # ADD ROW INDEX PER ROW FOR QUICK & SIMPLE TRAIN/TEST SPLIT
        chunk_df = chunk_df.with_row_count("row_index")
        # # ONE-HOT ENCODE (PICKUP DAYTIME) COLUMN
        # print("here")
        # chunk_df = one_hot_encode_column(chunk_df, "pickup_daytime")
        # chunk_df = one_hot_encode_column(chunk_df, "dropoff_daytime")
        # print(chunk_df.columns)
        # print(chunk_df.select(["pickup_daytime_1"]).unique().collect())
        # exit()
        # TRAIN-TEST SPLIT LAZYFRAMES
        chunk_train_data:pl.LazyFrame = chunk_df.filter(
            pl.col("row_index").le(int(train_test_split_perce * chunk_df_height))
        )
        chunk_test_data:pl.LazyFrame = chunk_df.filter(
            pl.col("row_index").gt(int(train_test_split_perce * chunk_df_height))
        )
        assert (chunk_train_data.select(pl.count()).collect().item() > 0) and (chunk_test_data.select(pl.count()).collect().item() > 0), "Train-Test samples are empty. Check logs for errors."
        # COMPUTE TARGET COLUMN STANDARD DEVIATION FOR COMPARISON WITH COMPUTED EVALUATION METRIC
        target_column_std = chunk_df.collect().describe().to_dict()['trip_duration'][3]
        # SELECT FEATURES COLUMN(s) AND TARGET VALUE
        X_features:list = [
            "trip_distance", # Will be scaled
            "pickup_daytime",
            "pickup_hour", 
            "pickup_weekday",
            "pickup_month", 
            "pickup_quarter", # Will be scaled
            "pickup_seconds", # Will be scaled
            "pickup_holiday",
            "pickup_weekend",
            "haversine_centroid_distance" # Will be scaled
        ]
        y_features:list = ["trip_duration"]
        # ISOLATED SELECTED FEATURES COLUMN(s) AND TARGET VALUE
        X_train, y_train = (
            chunk_train_data.select(X_features),
            chunk_train_data.select(y_features)
        )
        X_test, y_test = (
            chunk_test_data.select(X_features),
            chunk_test_data.select(y_features)
        )
        # FROM LAZYFRAMES TO NUMPY VECTORS (SKLEARN ACCEPTS EITHER NUMPY VECTORS OR PANDAS DATAFRAMES FOR OPTIMIZED COMPUTATIONS)
        X_train_vectors, y_train_vectors, X_test_vectors, y_test_vectors = (
            X_train.collect().to_numpy(),
            y_train.collect().to_numpy().ravel(),
            X_test.collect().to_numpy(),
            y_test.collect().to_numpy().ravel()
        )
        # LOAD BEST MODEL PIPELINE & RMSE SCORE
        if iteration == 1: # At first iteration initialize the model regressor weights from the ML_MODELING() class.
            pipeline_model = ml_modeling.pipeline[args.model_regressor] # Retrieve the selected pipeline model from the Class() instance
            best_rmse:float = float('inf')
            model_scores:dict = {}
            patience:int = 0
            with open(ml_modeling.model_scores, 'w') as f:
                json.dump({}, f, indent=4)
        else: # Otherwise the model will be loaded from the SAVED model after the first iteration.
            if args.model_regressor != "linear_regressor":
                if patience >= ml_modeling.patience:
                    patience = 0  # Reset patience
                    if args.model_regressor == "randomforest_regressor":
                        for key in ml_modeling.params_grid: # Adjust Hyper-Parameter Grid by 5%
                            if isinstance(ml_modeling.params_grid[key], int):
                                ml_modeling.params_grid[key] = int(ml_modeling.params_grid[key] * ml_modeling.hyper_parameters_adjustment)
                            else:
                                if key == "max_samples":
                                    ml_modeling.params_grid[key] *= ml_modeling.hyper_parameters_adjustment
                                    if ml_modeling.params_grid[key] > 1.0:
                                        ml_modeling.params_grid[key] = 0.5
                                    assert 0.0 <= ml_modeling.params_grid[key] <= 1.0, f"max_samples is out of bounds: {ml_modeling.params_grid[key]}"
                                else:
                                    ml_modeling.params_grid[key] *= ml_modeling.hyper_parameters_adjustment
                        logger_object.info(f"No improvement for 5 iterations, adjusting hyperparameters for RandomForest regressor: {ml_modeling.params_grid}")
                        pipeline_model:Pipeline = Pipeline(steps=[
                                ('preprocessor', ml_modeling.preprocessor_forests),
                                ('model', RandomForestRegressor(**ml_modeling.params_grid, criterion = "squared_error", random_state = RANDOM_SEED, n_jobs = -1))
                            ])
                    else:
                        for key in ml_modeling.params_rr: # Adjust Hyper-Parameter Grid by 5%
                            if isinstance(ml_modeling.params_rr[key], int):
                                ml_modeling.params_rr[key] = int(ml_modeling.params_rr[key] * ml_modeling.hyper_parameters_adjustment)
                            else:
                                ml_modeling.params_rr[key] *= ml_modeling.hyper_parameters_adjustment
                        for key in ml_modeling.params_elr: # Adjust Hyper-Parameter Grid by 5%
                            if isinstance(ml_modeling.params_elr[key], int):
                                ml_modeling.params_elr[key] = int(ml_modeling.params_elr[key] * ml_modeling.hyper_parameters_adjustment)
                            else:
                                ml_modeling.params_elr[key] *= ml_modeling.hyper_parameters_adjustment
                        pipeline_model:Pipeline = Pipeline(steps=[
                            ('preprocessor', ml_modeling.preprocessor_lr),
                            ('model', VotingRegressor(
                                estimators=[
                                    ("lr", LinearRegression(**ml_modeling.params_grid)),
                                    ("rr", Ridge(**ml_modeling.params_rr, fit_intercept=True, random_state=RANDOM_SEED, solver='auto')),
                                    ("elr", ElasticNet(**ml_modeling.params_elr, fit_intercept=True, random_state=RANDOM_SEED))
                                ],
                                n_jobs=-1,
                                verbose=False
                            ))
                        ])
                        logger_object.info(f"No improvement for 5 iterations, adjusting hyperparameters for Ridge Regressor: {ml_modeling.params_rr}")
                        logger_object.info(f"No improvement for 5 iterations, adjusting hyperparameters for ElasticNet Regressor: {ml_modeling.params_elr}")
                else:
                    # Load the already saved model if Patience < 5 and iteration !=1
                    pipeline_model:Pipeline = joblib.load(ml_modeling.model_artifact)
            with open(ml_modeling.model_scores, 'r') as f:
                model_scores:dict = json.load(f)
            best_rmse:float = sorted(model_scores.items(), key=lambda x: x[1]['rmse'])[0][1]['rmse']
            starting_rmse = None
        # MODEL FIT WITH K-FOLD CROSS VALIDATION
        rmses:list = []
        for train_index, val_index in ml_modeling.kf.split(X_train_vectors):
            X_train_batch, X_val_batch = X_train_vectors[train_index], X_train_vectors[val_index]
            y_train_batch, y_val_batch = y_train_vectors[train_index], y_train_vectors[val_index]

            pipeline_model.fit(X_train_batch, y_train_batch)
            y_pred = pipeline_model.predict(X_val_batch)
            rmse = ml_modeling.evaluation_metric(y_val_batch, y_pred)
            rmses.append(rmse)
        # MODEL EVALUATION ON OUT-OF-SAMPLE DATA
        y_pred = pipeline_model.predict(X_test_vectors)
        residuals = y_test_vectors - y_pred
        chunk_rmse = ml_modeling.evaluation_metric(y_test_vectors, y_pred)
        logger_object.info(f'Chunk {iteration} | {args.model_regressor} RMSE: {chunk_rmse}')
        
        if chunk_rmse < best_rmse:
            # Residuals Distribution Plot
            logger_object.info(f"Chunk {iteration} | Plot the residuals and save them to PNG file.")
            ml_modeling.plot_residuals(residuals)
            # Overwrite the saved model with the new best-fitted model
            joblib.dump(pipeline_model, ml_modeling.model_artifact)
            patience:int = 0 # Restart patience
            if chunk_rmse < target_column_std:
                logger_object.info(f"Chunk {iteration} | Model training has a good fit on never-seen-before data (RMSE: {chunk_rmse} < STD: {target_column_std}).")
            starting_rmse:float = float(model_scores["chunk_1"]["rmse"]) if iteration!=1 else chunk_rmse
        cummulative_percentage_improvement:float = np.round((((chunk_rmse/starting_rmse) - 1)*-1)*100, 2) if starting_rmse else 0.00
    
        logger_object.info(f"Chunk {iteration} | Patience = {patience}")
        patience += 1 # increase patience if no improvement achieved.
        hours, minutes, seconds = compute_execution_time(start_time)
        logger_object.info(f"Chunk {iteration} | Completed model training process after: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
        logger_object.info(100*"-")

        model_scores[f"chunk_{iteration}"] = {
            "model_path": ml_modeling.model_artifact,
            "rmse": chunk_rmse,
            "percentage_improvement": cummulative_percentage_improvement if iteration!=1 else 0.00
        }
        if iteration == 1:
            with open(ml_modeling.model_scores, 'w') as f:
                json.dump(model_scores, f, indent=4)
        else:
            with open(ml_modeling.model_scores, 'w+') as f:
                json.dump(model_scores, f, indent=4)

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