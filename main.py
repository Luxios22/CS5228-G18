import os
import pickle
import optuna
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRFRegressor, XGBRegressor
from lightgbm import LGBMRegressor
from optuna.visualization import (
    plot_contour, plot_edf, plot_intermediate_values, plot_optimization_history,
    plot_parallel_coordinate, plot_param_importances, plot_slice
)
# sklearn acceleration
from sklearnex import patch_sklearn
patch_sklearn()

# logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("optuna.log", mode="w"))
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()


# Read datasets
def concat_features(df_train, df_test):
    folder_path = 'data/auxiliary-data-preprocessed/'

    train_files = [f for f in os.listdir(folder_path) if f.endswith("03.csv") and not f.endswith("_test.csv")]
    test_files = [f for f in os.listdir(folder_path) if f.endswith("03_test.csv")]

    train_dataframes = [pd.read_csv(os.path.join(folder_path, f), sep=',') for f in train_files]
    test_dataframes = [pd.read_csv(os.path.join(folder_path, f), sep=',') for f in test_files]

    train_dataframes.insert(0, df_train[df_train.columns.values[:-1]])
    train_dataframes.append(df_train['resale_price'])
    
    train_data = pd.concat(train_dataframes, axis=1)
    test_data = pd.concat([df_test] + test_dataframes, axis=1)

    return train_data, test_data

train = pd.read_csv('data/train_cleaned.csv')
test = pd.read_csv('data/test_cleaned.csv')

train, test = concat_features(train, test)
# 1-room, 2-room, 3-room, 4-room, 5-room, executive, multi generation
# mapping = {'1-room': 1, '2-room': 2, '3-room': 3, '4-room': 4, '5-room': 5, 'executive': 6, 'multi generation': 7}

# train['flat_type_numerical'] = train['flat_type'].replace(mapping)
train['year'] = pd.DatetimeIndex(train['month']).year
train['month_num'] = pd.DatetimeIndex(train['month']).month
train['flat'] = pd.concat([train['flat_type'], train['flat_model']], axis=1).apply(lambda x: ' '.join(x), axis=1)

# test['flat_type_numerical'] = test['flat_type'].replace(mapping)
test['year'] = pd.DatetimeIndex(test['month']).year
test['month_num'] = pd.DatetimeIndex(test['month']).month
test['flat'] = pd.concat([test['flat_type'], test['flat_model']], axis=1).apply(lambda x: ' '.join(x), axis=1)

train.drop(columns=['month', 'flat_type', 'flat_model'], inplace=True)
test.drop(columns=['month', 'flat_type', 'flat_model'], inplace=True)

categorical_features = ['town', 'street_name', 'storey_range', 'subzone', 'flat']
numerical_features = ['year', 'month_num', 'floor_area_sqm', 'lease_commence_date', "latitude", "longitude"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough')

# Model selection
models = {
    "Linear Regression": LinearRegression(),
    "Gradient Boosting Tree": GradientBoostingRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42),
    "XGBRFRegressor": XGBRFRegressor(tree_method="gpu_hist", random_state=42, gpu_id=1),
    "XGBRegressor": XGBRegressor(tree_method="gpu_hist", random_state=42, gpu_id=1),
}

# Cross-validation and model evaluation
X = train.drop("resale_price", axis=1)
y = train["resale_price"]

for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    scores = -1 * cross_val_score(pipeline, X, y, cv=5, scoring="neg_mean_absolute_error")
    logger.info(f"{name} Mean Absolute Error: {np.mean(scores):.2f}")


# Hyperparameter tuning
def model_objective(trial, X, y, model_name):
    if model_name == "lr":
        model = LinearRegression()
    elif model_name == "lgbm":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        model = LGBMRegressor(**params, random_state=42)
    elif model_name == "gbt":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        }
        model = GradientBoostingRegressor(**params, random_state=42)
    elif model_name == "xgb":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 700, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.3, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.7),
        }
        model = XGBRegressor(**params, random_state=42, tree_method='gpu_hist', gpu_id=1, predictor="gpu_predictor")
    elif model_name == "xgbrf":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1),
        }
        model = XGBRFRegressor(**params, random_state=42, tree_method='gpu_hist', gpu_id=1, predictor="gpu_predictor")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    score = -1 * cross_val_score(pipeline, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    return np.mean(score)

model_studies = {}
studies_json = {}
for model_name in ["lr", "lgbm", "gbt", "xgb", "xgbrf"]:
    study_name = f"{model_name}-study"  # Unique identifier of the study.
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_name)
    logger.info(f"Start optimization for {model_name}.")
    study.optimize(lambda trial: model_objective(trial, X, y, model_name), n_trials=20, gc_after_trial=True)
    model_studies[model_name] = study
    studies_json[f'{model_name}_{study.best_value}'] = study.best_params

# Save converted_studies to a json file
import json
with open("studies_json.json", "w") as f:
    json.dump(studies_json, f)

# Select the best model
best_study = min(model_studies.values(), key=lambda s: s.best_value)
best_value = best_study.best_value
best_study_name = best_study.study_name
best_params = best_study.best_params
if best_study_name.startswith("lr"):
    best_model = LinearRegression()
elif best_study_name.startswith("lgbm"):
    best_model = LGBMRegressor(**best_params, random_state=42)
elif best_study_name.startswith("gbt"):
    best_model = GradientBoostingRegressor(**best_params, random_state=42)
elif best_study_name.startswith("xgb"):
    best_model = XGBRegressor(**best_params, random_state=42, tree_method='gpu_hist', gpu_id=1)
elif best_study_name.startswith("xgbrf"):
    best_model = XGBRFRegressor(**best_params, random_state=42, tree_method='gpu_hist', gpu_id=1)

# Train and predict with the best model
best_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", best_model)])
best_pipeline.fit(X, y)

# save best_model parameters to a json file
with open(f"result/{best_study_name}_{best_value}_best_model_params.json", "w") as f:
    json.dump(best_model.get_params(), f)

# Save the best pipeline as a pickle file
with open(f"result/{best_study_name}_{best_value}_best_pipeline.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)
logger.info("Best model saved as 'best_pipeline.pkl'.")

# Generate submission file
predictions = best_pipeline.predict(test)
submission = pd.DataFrame({"Id": test.index, "Predicted": predictions})
submission.to_csv(f"result/{best_study_name}_{best_value}_prediction.csv", index=False)
logger.info("Predictions saved to 'prediction.csv'.")

def save_visualizations(study, model_name):
    visualization_functions = {
        f"{model_name}_{study.best_value}_optimization_history": plot_optimization_history,
        f"{model_name}_{study.best_value}_intermediate_values": plot_intermediate_values,
        f"{model_name}_{study.best_value}_parallel_coordinate": plot_parallel_coordinate,
        f"{model_name}_{study.best_value}_contour": plot_contour,
        f"{model_name}_{study.best_value}_slice": plot_slice,
        f"{model_name}_{study.best_value}_param_importances": plot_param_importances,
        f"{model_name}_{study.best_value}_edf": plot_edf
    }

    for file_name, vis_function in visualization_functions.items():
        vis_plot = vis_function(study)
        vis_plot.write_image(f"result/{file_name}.png")
        # vis_plot.show()

for model_name, study in model_studies.items():
    save_visualizations(study, model_name)
