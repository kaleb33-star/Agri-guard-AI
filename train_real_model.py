
"""
AgriGuard Ethiopia
REAL CROP YIELD MODEL TRAINING

Data:
    data/etho_agri_real_matched.csv

Target:
    yield_kg_ha

Features:
    region
    crop_type
    year
    area_cultivated_ha
    NDVI_mean
    NDVI_peak
    temperature_mean
    rainfall

Models compared:
    1. Random Forest
    2. Extra Trees
    3. Gradient Boosting
    4. Random Forest with different settings
    5. XGBoost

Validation:
    Time-aware cross-validation.

The best model is selected automatically using
cross-validation RMSE.

IMPORTANT:
    No synthetic data is generated in this script.
"""

import os
import json
import joblib
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_FILE = "data/etho_agri_real_matched.csv"

MODEL_DIR = "models"

BEST_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "real_yield_model.joblib"
)

METADATA_FILE = os.path.join(
    MODEL_DIR,
    "real_yield_metadata.json"
)

CV_RESULTS_FILE = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

PREDICTIONS_FILE = os.path.join(
    MODEL_DIR,
    "test_predictions.csv"
)


# ============================================================
# 2. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


print("=" * 75)
print("AGRIGUARD ETHIOPIA - REAL YIELD MODEL")
print("=" * 75)


# ============================================================
# 3. LOAD REAL DATA
# ============================================================

print("\n[1/8] Loading real matched dataset...")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"\nCould not find:\n{DATA_FILE}\n\n"
        "Run prepare_real_dataset.py first."
    )

df = pd.read_csv(DATA_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "region",
    "crop_type",
    "year",
    "area_cultivated_ha",
    "NDVI_mean",
    "NDVI_peak",
    "temperature_mean",
    "rainfall",
    "yield_kg_ha"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# 5. CLEAN DATA
# ============================================================

print("\n[2/8] Checking data quality...")

df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce"
)

numeric_columns = [
    "area_cultivated_ha",
    "NDVI_mean",
    "NDVI_peak",
    "temperature_mean",
    "rainfall",
    "yield_kg_ha"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(
    subset=required_columns
).copy()

df = df.sort_values(
    "year"
).reset_index(drop=True)

print(f"Usable rows: {len(df):,}")
print(
    f"Year range: "
    f"{int(df['year'].min())} - "
    f"{int(df['year'].max())}"
)


# ============================================================
# 6. DEFINE FEATURES AND TARGET
# ============================================================

print("\n[3/8] Preparing features...")

FEATURES = [
    "region",
    "crop_type",
    "year",
    "area_cultivated_ha",
    "NDVI_mean",
    "NDVI_peak",
    "temperature_mean",
    "rainfall"
]

TARGET = "yield_kg_ha"

X = df[FEATURES].copy()
y = df[TARGET].copy()


CATEGORICAL_FEATURES = [
    "region",
    "crop_type"
]

NUMERIC_FEATURES = [
    "year",
    "area_cultivated_ha",
    "NDVI_mean",
    "NDVI_peak",
    "temperature_mean",
    "rainfall"
]


# ============================================================
# 7. TIME-AWARE TRAIN / TEST SPLIT
# ============================================================
#
# We reserve the latest year for final testing.
#
# Training:
#     2015 -> 2021
#
# Final test:
#     2022
#
# This simulates the real-world situation:
#
#     "Can the model use previous years to predict a future year?"
#

print("\n[4/8] Creating time-aware test set...")

latest_year = int(df["year"].max())

train_mask = df["year"] < latest_year
test_mask = df["year"] == latest_year

X_train = X.loc[train_mask].copy()
y_train = y.loc[train_mask].copy()

X_test = X.loc[test_mask].copy()
y_test = y.loc[test_mask].copy()

print(f"Training rows: {len(X_train):,}")
print(f"Final test rows: {len(X_test):,}")
print(f"Training years: {sorted(X_train['year'].unique())}")
print(f"Test year: {latest_year}")


# ============================================================
# 8. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    )
                ]
            ),
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            ),
            CATEGORICAL_FEATURES
        )
    ]
)


# ============================================================
# 9. TIME-AWARE CROSS-VALIDATION
# ============================================================
#
# TimeSeriesSplit ensures that validation always happens
# AFTER the training period.
#
# Example:
#
# Fold 1:
# train -> early years
# test  -> next year
#
# Fold 2:
# train -> more years
# test  -> next year
#
# This is much more realistic for agricultural forecasting.
#

print("\n[5/8] Setting up time-aware cross-validation...")

unique_train_years = sorted(
    X_train["year"].unique()
)

if len(unique_train_years) < 4:
    raise ValueError(
        "Not enough unique training years for "
        "time-aware cross-validation."
    )

# Use at most 4 splits because we only have
# a limited number of yearly observations.
n_splits = min(
    4,
    len(unique_train_years) - 1
)

tscv = TimeSeriesSplit(
    n_splits=n_splits
)

print(
    f"Using {n_splits}-fold TimeSeriesSplit."
)


# ============================================================
# 10. DEFINE CANDIDATE MODELS
# ============================================================

print("\n[6/8] Comparing candidate models...")

models = {

    "Random Forest": (
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ),
        {
            "model__n_estimators": [
                200,
                400
            ],
            "model__max_depth": [
                None,
                10,
                20
            ],
            "model__min_samples_leaf": [
                1,
                2,
                4
            ]
        }
    ),

    "Extra Trees": (
        ExtraTreesRegressor(
            random_state=42,
            n_jobs=-1
        ),
        {
            "model__n_estimators": [
                200,
                400
            ],
            "model__max_depth": [
                None,
                10,
                20
            ],
            "model__min_samples_leaf": [
                1,
                2,
                4
            ]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingRegressor(
            random_state=42
        ),
        {
            "model__n_estimators": [
                100,
                200
            ],
            "model__learning_rate": [
                0.03,
                0.05,
                0.1
            ],
            "model__max_depth": [
                2,
                3
            ]
        }
    ),

    "XGBoost": (
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        ),
        {
            "model__n_estimators": [
                200,
                400
            ],
            "model__learning_rate": [
                0.03,
                0.05,
                0.1
            ],
            "model__max_depth": [
                2,
                3,
                5
            ],
            "model__subsample": [
                0.8,
                1.0
            ],
            "model__colsample_bytree": [
                0.8,
                1.0
            ]
        }
    )
}


# ============================================================
# 11. RUN CROSS-VALIDATION
# ============================================================

comparison_results = []

best_name = None
best_pipeline = None
best_cv_rmse = float("inf")


for model_name, (model, parameters) in models.items():

    print("\n" + "-" * 70)
    print(f"Testing: {model_name}")
    print("-" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
        refit=True
    )

    search.fit(
        X_train,
        y_train
    )

    cv_rmse = -search.best_score_

    print(
        f"Best CV RMSE: "
        f"{cv_rmse:,.2f} kg/ha"
    )

    print(
        f"Best parameters: "
        f"{search.best_params_}"
    )

    comparison_results.append(
        {
            "model": model_name,
            "cv_rmse": cv_rmse,
            "best_parameters": str(
                search.best_params_
            )
        }
    )

    if cv_rmse < best_cv_rmse:
        best_cv_rmse = cv_rmse
        best_name = model_name
        best_pipeline = search.best_estimator_


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    comparison_results
).sort_values(
    "cv_rmse"
)

comparison_df.to_csv(
    CV_RESULTS_FILE,
    index=False
)

print("\n" + "=" * 75)
print("CROSS-VALIDATION RESULTS")
print("=" * 75)

print(
    comparison_df[
        ["model", "cv_rmse"]
    ].to_string(index=False)
)

print("\nBEST MODEL:")
print(best_name)

print(
    f"Best CV RMSE: "
    f"{best_cv_rmse:,.2f} kg/ha"
)


# ============================================================
# 13. FINAL TEST EVALUATION
# ============================================================

print("\n[7/8] Evaluating the selected model on 2022...")

best_pipeline.fit(
    X_train,
    y_train
)

predictions = best_pipeline.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\nFINAL TEST RESULTS")
print("-" * 50)

print(
    f"Test year: {latest_year}"
)

print(
    f"MAE:  {mae:,.2f} kg/ha"
)

print(
    f"RMSE: {rmse:,.2f} kg/ha"
)

print(
    f"R²:   {r2:.4f}"
)


# ============================================================
# 14. SAVE TEST PREDICTIONS
# ============================================================

test_predictions = X_test.copy()

test_predictions[
    "actual_yield_kg_ha"
] = y_test.values

test_predictions[
    "predicted_yield_kg_ha"
] = predictions

test_predictions[
    "absolute_error_kg_ha"
] = np.abs(
    test_predictions[
        "actual_yield_kg_ha"
    ]
    -
    test_predictions[
        "predicted_yield_kg_ha"
    ]
)

test_predictions.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# 15. SAVE BEST MODEL
# ============================================================

print("\n[8/8] Saving best model...")

joblib.dump(
    best_pipeline,
    BEST_MODEL_FILE
)


# ============================================================
# 16. SAVE METADATA
# ============================================================

best_parameters = {}

for result in comparison_results:
    if result["model"] == best_name:
        best_parameters = result[
            "best_parameters"
        ]

metadata = {

    "project": "AgriGuard Ethiopia",

    "model_type": best_name,

    "target": TARGET,

    "features": FEATURES,

    "categorical_features":
        CATEGORICAL_FEATURES,

    "numeric_features":
        NUMERIC_FEATURES,

    "training_rows":
        int(len(X_train)),

    "test_rows":
        int(len(X_test)),

    "training_years":
        [int(x) for x in unique_train_years],

    "test_year":
        latest_year,

    "cross_validation":
        "TimeSeriesSplit",

    "cv_splits":
        n_splits,

    "best_cv_rmse":
        float(best_cv_rmse),

    "final_test_mae":
        float(mae),

    "final_test_rmse":
        float(rmse),

    "final_test_r2":
        float(r2),

    "best_parameters":
        best_parameters,

    "data_source":
        "Etho-Agri + Sentinel-2 + Open-Meteo",

    "synthetic_data_used":
        False
}

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("REAL MODEL TRAINING COMPLETE")
print("=" * 75)

print(
    f"\nSelected model: {best_name}"
)

print(
    f"Cross-validation RMSE: "
    f"{best_cv_rmse:,.2f} kg/ha"
)

print(
    f"2022 test MAE: "
    f"{mae:,.2f} kg/ha"
)

print(
    f"2022 test RMSE: "
    f"{rmse:,.2f} kg/ha"
)

print(
    f"2022 test R²: "
    f"{r2:.4f}"
)

print("\nFiles created:")

print(
    f"  {BEST_MODEL_FILE}"
)

print(
    f"  {METADATA_FILE}"
)

print(
    f"  {CV_RESULTS_FILE}"
)

print(
    f"  {PREDICTIONS_FILE}"
)

print("\nNo synthetic data was used.")
print("=" * 75)
