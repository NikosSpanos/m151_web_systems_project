#!/usr/bin/env python
import os
import numpy as np
import logging
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple

def init_model_artifacts(path:str, logger_object:logging.Logger):
    if not os.path.exists(path):
        logger_object.info(f"Staging path: {path} not found. Creating path...")
        os.makedirs(path, exist_ok=True)
    else:
        logger_object.info(f"Staging path: {path} already exists. Program will continue.")
    return

def save_model_regressor(model, filename:str):
    model_filename = "linear_regression_model.joblib"
    joblib.dump(model, filename)

def train_xgboost_regressor(params, dtrain) -> xgb.Booster:
    model = xgb.train(params, dtrain)
    return model

def train_linear_regressor(train_x, train_y, params):
    model = LinearRegression(**params)
    model.fit(train_x, train_y)
    return model

def train_randomforest_regressor(train_x, train_y, params):
    model = RandomForestRegressor(**params)
    model.fit(train_x, train_y)
    return model

def make_predictions(model_name, model, dtest, y_test, logger_obj:logging.Logger) -> Tuple[float, float, str]:
    if model_name == "linear":
        y_test = y_test.values
    elif model_name == "randomforest":
        y_test = y_test.values.ravel()
    y_pred = model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    if np.any(y_pred < 0):
        # mape = mean_absolute_percentage_error(y_test, y_pred)
        mape = mean_absolute_error(y_test, y_pred)
        metric_selected_value = mape
        # metric_selected_name = "mean-absolute-percentage-error"
        metric_selected_name = "mean-absolute-error"
    else:
        msle = mean_squared_log_error(y_test, y_pred)
        metric_selected_value = msle
        metric_selected_name = "mean-squared-logarithmic-error"
    logger_obj.info("root-mean-squared-error: %f" % (rmse))
    logger_obj.info("%s: %f" % (metric_selected_name, metric_selected_value))
    return (rmse, metric_selected_value, metric_selected_name)