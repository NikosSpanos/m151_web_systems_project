#!/usr/bin/env python
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

def init_model_artifacts(path:str, logger_object:logging.Logger):
    if not os.path.exists(path):
        logger_object.info(f"Staging path: {path} not found. Creating path...")
        os.makedirs(path, exist_ok=True)
    else:
        logger_object.info(f"Staging path: {path} already exists. Program will continue.")
    return

class ML_MODELING():

    def __init__(self, model_type:str, ml_model:str, model_artifacts_path:str, model_metadata_path:str, model_residuals_path:str, kfold_splits:int, RANDOM_SEED:int):
        self.columns_to_normalize:list = [0, 5, 6, 8] if model_type=="trip_duration" else [0, 7, 8, 9, 10, 15]  #Integres =numppy index matching to dataframe columns
        self.valid_regressor_names:list = ["linear_regressor", "randomforest_regressor", "voting_regressor"]
        self.patience = 5
        self.hyper_parameters_adjustment = 1.15 # positive percentage increase. Thus 1.05 means 5%, 1.10 means 10% and so on so forth.
        self.preprocessor_lr:ColumnTransformer = ColumnTransformer(
            transformers=[
                ('standardized', StandardScaler(), self.columns_to_normalize)
            ],
            remainder='passthrough'
        )
        self.preprocessor_forests:ColumnTransformer = ColumnTransformer(
            transformers=[
                ('normalized', MinMaxScaler(), self.columns_to_normalize)
            ],
            remainder='passthrough'
        )
        if ml_model == "linear_regressor":
            self.params_grid:dict = {
                "fit_intercept": True,
                "copy_X": True,
                "n_jobs": -1
            }
            self.pipeline:Dict[str, Pipeline] = {
                ml_model: Pipeline(steps=[
                    ('preprocessor', self.preprocessor_lr),
                    ('model', LinearRegression(**self.params_grid))
                ])
            }
        elif ml_model == "randomforest_regressor":
            self.params_grid:dict = {
                "n_estimators": 100,
                "max_depth": 25,
                "max_samples": 0.25,
                "min_samples_split": 3,
                "min_samples_leaf": 5
            } # Ηyper-parameters n_estimators, max_depth, max_samples, min_samples_leaf greatly affect the RMSE score (after experimentation).
            self.pipeline:Dict[str, Pipeline] = {
                ml_model: Pipeline(steps=[
                    ('preprocessor', self.preprocessor_forests),
                    ('model', RandomForestRegressor(**self.params_grid, criterion = "squared_error", random_state = RANDOM_SEED, n_jobs = -1))
                ])
            }
        else:
            self.params_grid:dict = {
                "fit_intercept": True,
                "copy_X": True
            }
            self.params_rr:dict = {
                "tol": 0.0001,
                "alpha": 0.1,
                "max_iter": 2000,
            }
            self.params_elr:dict = {
                "alpha": 0.1,
                "l1_ratio": 0.1,
                "max_iter": 2000,
            }
            self.pipeline:Dict[str, Pipeline] = {
                ml_model: Pipeline(steps=[
                    ('preprocessor', self.preprocessor_lr),
                    ('model', VotingRegressor(
                        estimators=[
                            ("lr", LinearRegression(**self.params_grid)),
                            ("rr", Ridge(**self.params_rr, fit_intercept=True, random_state=RANDOM_SEED, solver='auto')),
                            ("elr", ElasticNet(**self.params_elr, fit_intercept=True, random_state=RANDOM_SEED))
                        ],
                        n_jobs=-1,
                        verbose=False
                    ))
                ])
            }

        self.kf = KFold(n_splits=kfold_splits, shuffle=True, random_state=RANDOM_SEED)
        self.evaluation_metric = root_mean_squared_error
        self.r_squared = r2_score
        self.mae = mean_absolute_error
        self.model_artifact:str = os.path.join(model_artifacts_path, f"{ml_model}_best_model.joblib")
        self.model_scores:str = os.path.join(model_metadata_path, f"{ml_model}_scores.json")
        self.model_residuals:str = os.path.join(model_residuals_path, f"{ml_model}_residuals_plot.png")

    def plot_residuals(self, residuals:np.ndarray):
        # Residuals Distribution Plot
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, bins=30, kde=True)
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Residuals Distribution")
        plt.grid(True)
        plt.savefig(self.model_residuals)