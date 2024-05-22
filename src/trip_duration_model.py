import polars as pl
import numpy as np
import xgboost as xgb
import logging
import configparser
import os
import mlflow
import mlflow.sklearn
from argparse import ArgumentParser
from typing import List, Dict
from pathlib import Path
from datetime import datetime
from commons.custom_logger import setup_logger
from commons.staging_modules import retrieve_latest_modified_folder, \
    write_df_toJSON, \
    write_df_toCSV
from commons.ml_modules import init_model_artifacts, \
    train_linear_regressor, \
    train_randomforest_regressor, \
    train_xgboost_regressor, \
    make_predictions

def duration_predictor(logger_object:logging.Logger):
    
    #========================================================
    # INITIALIZE MODULE VARIABLES
    #========================================================
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

    model_artifacts_parent:str = config.get("ml-settings", "model_artifacts_path")
    model_artifacts_child:str = config.get("ml-settings", "duration_model_artifact")
    model_artifact_path:str = (
        os.path.join(
            application_path, model_artifacts_parent, model_artifacts_child, execution_timestamp, args.trip_type
        )
    )
    init_model_artifacts(model_artifact_path, logger_object)
    RANDOM_SEED:int = 42
    np.random.seed(RANDOM_SEED)
    
    #=========================================================================================
    # READ THE PROCESSED-DATA PARQUET FILES FROM LATEST MODIFIED STAGING FOLDER OF TAXI_TRIPS
    #=========================================================================================
    # partitions = Path(stg_processed_path).rglob("*.parquet")
    latest_modified_stg_folder:str = os.path.join(retrieve_latest_modified_folder(stg_processed_path), args.trip_type)
    print(latest_modified_stg_folder)
    partitions: Path = Path(latest_modified_stg_folder)
    
    # parquet_directories:list = [x for x in partitions.iterdir() if x.is_dir()]
    parquet_directories:list = ["/home/nspanos/m151_web_systems_project/data/staging/processed/taxi_trips/20240521/short_trip/partition_dt=202110"]

    def read_parquet_files_in_chunks(parquet_files: List[str], chunk_size: int = 500_000):
        schema:dict={
            'trip_duration': pl.Float64,
            'trip_distance': pl.Float64,
            'pickup_daytime_2': pl.UInt8,
            'pickup_daytime_3': pl.UInt8,
            # 'pickup_quarter': pl.Int8,
            'pickup_seconds':pl.Int64,
            'pickup_holiday': pl.UInt8,
            'pickup_weekend': pl.UInt8,
            'haversine_centroid_distance': pl.Float64
        }
        current_chunk:pl.LazyFrame = pl.LazyFrame(schema = schema)
        current_chunk_rows: int = 0
        iteration:int = 1
        for directory in parquet_files:
            df:pl.LazyFrame = pl.concat([pl.scan_parquet(os.path.join(directory, "*.parquet"))]).select(list(schema.keys()))
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
 
    for iteration, chunk_df in enumerate(read_parquet_files_in_chunks(parquet_directories, chunk_size)):
        chunk_df_height:int = chunk_df.select(pl.count()).collect().item()
        # SHUFFLE ROWS
        shuffled_df:pl.DataFrame = chunk_df.collect().sample(fraction=1, with_replacement=False, shuffle=True, seed=1234)
        chunk_df:pl.LazyFrame = shuffled_df.lazy().with_row_count("row_index")
        # FILTER OUT OUTLIERS BASED ON PERCENTILES OF TRIP_DURATION
        lowest_quantile = 0.15
        highest_quantile = 0.95
        percentile_bottom_bound = chunk_df.select("trip_duration").quantile(lowest_quantile).collect().item()
        percentile_top_bound = chunk_df.select("trip_duration").quantile(highest_quantile).collect().item()
        rounded_bottom_bound = np.floor(percentile_bottom_bound)
        chunk_df = chunk_df.filter(pl.col("trip_duration") >= rounded_bottom_bound)
        chunk_df = chunk_df.filter(pl.col("trip_duration") <= percentile_top_bound)
        # TRAIN-TEST SPLIT LAZYFRAMES
        chunk_train_data:pl.LazyFrame = chunk_df.slice(0, int(train_test_split_perce * chunk_df_height))
        chunk_test_data:pl.LazyFrame = chunk_df.slice(
            int(train_test_split_perce * chunk_df_height), chunk_df_height - int(train_test_split_perce * chunk_df_height)
        )

        print(chunk_df.columns)
        # print(chunk_df.collect().describe())
        # print(chunk_df.select(pl.count()).collect().item())
        # exit()
        X_features:list = ["trip_distance", "pickup_daytime_2", "pickup_daytime_3", "pickup_seconds", "pickup_holiday", "pickup_weekend", "haversine_centroid_distance"]
        y_features:list = ["trip_duration"]

        X_train, y_train = (
            chunk_train_data.select(X_features),
            chunk_train_data.select(y_features)
        )
        X_test, y_test = (
            chunk_test_data.select(X_features),
            chunk_test_data.select(y_features)
        )
        print(X_train.columns)
        print(y_train.columns)

        # columns_to_normalize = ['trip_distance', 'pickup_quarter', 'haversine_centroid_distance']
        columns_to_normalize:list = [0, 3, 6]
        
        import joblib
        import xgboost as xgb
        from sklearn.model_selection import KFold
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
        
        preprocessor_lr:ColumnTransformer = ColumnTransformer(
            transformers=[
                ('standardized', StandardScaler(), columns_to_normalize)
            ],
            remainder='passthrough'
        )
        preprocessor_forests:ColumnTransformer = ColumnTransformer(
            transformers=[
                ('normalized', MinMaxScaler(), columns_to_normalize)
            ],
            remainder='passthrough'
        )
        params_lr:dict = {
            "fit_intercept": True,
            "copy_X": True
        }
        params_rf:dict = {
            "n_estimators": 100,
            "criterion": "squared_error",
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": RANDOM_SEED
        }

        pipelines:Dict[str, Pipeline] = {
            'linear_regression': Pipeline(steps=[
                ('preprocessor', preprocessor_lr),
                ('model', LinearRegression(**params_lr))
            ]),
            'random_forest': Pipeline(steps=[
                ('preprocessor', preprocessor_forests),
                ('model', RandomForestRegressor(**params_rf))
            ]),
            # 'xgboost': Pipeline(steps=[
            #     ('preprocessor', preprocessor_forests),
            #     ('model', xgb.XGBRegressor(**params_xgb))
            # ])
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        best_rmse = float('inf')
        best_model = None
        X_train_vectors, y_train_vectors, X_test_vectors, y_test_vectors = (
            X_train.collect().to_numpy(),
            y_train.collect().to_numpy().ravel(),
            X_test.collect().to_numpy(),
            y_test.collect().to_numpy().ravel()
        )
        for name, pipeline in pipelines.items():
            rmses:list = []
            for train_index, val_index in kf.split(X_train_vectors):
                X_train_batch, X_val_batch = X_train_vectors[train_index], X_train_vectors[val_index]
                y_train_batch, y_val_batch = y_train_vectors[train_index], y_train_vectors[val_index]

                pipeline.fit(X_train_batch, y_train_batch)
                y_pred = pipeline.predict(X_val_batch)
                rmse = mean_squared_error(y_val_batch, y_pred, squared=False)
                rmses.append(rmse)

            mean_rmse = np.mean(rmses)
            print(f'{name} RMSE: {mean_rmse}')

            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_model = pipeline
                joblib.dump(best_model, os.path.join(model_artifact_path, f"chunk_{iteration}_best_model.joblib"))
        break

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