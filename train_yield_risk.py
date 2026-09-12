"""
============================================================
AGRIGUARD ETHIOPIA - MODEL 3
YIELD RISK INTELLIGENCE
============================================================

Purpose:
    Predict the risk that a crop will have unusually low yield.

Classes:
    0 = Low Risk
    1 = Moderate Risk
    2 = High Risk

Data:
    Real Ethiopian crop observations matched with:
        - Sentinel-2 NDVI
        - Open-Meteo temperature
        - Open-Meteo rainfall

NO synthetic data is used.

The model learns from historical yield outcomes.

Validation:
    Expanding year-based validation.

The validation year always comes AFTER the training years.
This helps prevent future information from leaking into training.

Target used to create risk labels:
    yield_kg_ha

Important:
    Production is NOT used as an input feature because
    production and yield are directly related. Using production
    could cause target leakage.

============================================================
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

DATA = Path("data/etho_agri_real_matched.csv")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. FEATURES
# ============================================================

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
# 3. RISK CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk"
}


# ============================================================
# 4. CREATE RISK LABELS
# ============================================================

def make_risk_labels(data, baseline):
    """
    Create yield-risk labels from REAL historical yield.

    The labels are based on the historical yield distribution
    for the same region and crop.

    Method:

        Above or equal to the 33rd percentile
            -> Low Risk

        Between the 10th and 33rd percentile
            -> Moderate Risk

        Below the 10th percentile
            -> High Risk

    If there are too few observations for a region + crop,
    the function falls back to:

        1. Crop-level historical data
        2. Overall historical data

    IMPORTANT:

    The baseline is supplied separately.

    This means that when predicting a future year,
    only historical information available before that year
    is used to create the risk thresholds.
    """

    # --------------------------------------------------------
    # Historical yield grouped by region + crop
    # --------------------------------------------------------

    group_values = (
        baseline
        .groupby(
            ["region", "crop_type"]
        )["yield_kg_ha"]
        .apply(list)
        .to_dict()
    )

    # --------------------------------------------------------
    # Historical yield grouped by crop
    # --------------------------------------------------------

    crop_values = (
        baseline
        .groupby("crop_type")["yield_kg_ha"]
        .apply(list)
        .to_dict()
    )

    # --------------------------------------------------------
    # Overall historical yield
    # --------------------------------------------------------

    global_values = (
        baseline["yield_kg_ha"]
        .tolist()
    )

    labels = []

    # ========================================================
    # Process every observation
    # ========================================================

    for _, row in data.iterrows():

        # ----------------------------------------------------
        # First try region + crop history
        # ----------------------------------------------------

        values = group_values.get(
            (
                row["region"],
                row["crop_type"]
            ),
            []
        )

        # ----------------------------------------------------
        # If too little data, use crop history
        # ----------------------------------------------------

        if len(values) < 3:

            values = crop_values.get(
                row["crop_type"],
                []
            )

        # ----------------------------------------------------
        # If still too little data, use global history
        # ----------------------------------------------------

        if len(values) < 3:

            values = global_values

        # ----------------------------------------------------
        # Calculate historical thresholds
        # ----------------------------------------------------

        q10, q33 = np.quantile(
            values,
            [
                0.10,
                1 / 3
            ]
        )

        current_yield = row["yield_kg_ha"]

        # ----------------------------------------------------
        # Assign risk category
        # ----------------------------------------------------

        if current_yield < q10:

            # Extremely low compared with history
            labels.append(2)

        elif current_yield < q33:

            # Below normal but not extremely low
            labels.append(1)

        else:

            # Not unusually low
            labels.append(0)

    return np.array(
        labels,
        dtype=int
    )


# ============================================================
# 5. CREATE MACHINE LEARNING PIPELINE
# ============================================================

def make_pipeline(model):
    """
    Prepare categorical variables and numerical variables,
    then train the selected classifier.
    """

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                CATEGORICAL_FEATURES
            )
        ],

        remainder="passthrough"
    )

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

    return pipeline


# ============================================================
# 6. MODELS TO COMPARE
# ============================================================

MODELS = {

    "Random Forest":

        RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),


    "Extra Trees":

        ExtraTreesClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),


    "Gradient Boosting":

        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            random_state=42
        ),


    "XGBoost":

        XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,

            objective="multi:softprob",

            eval_metric="mlogloss",

            random_state=42,

            n_jobs=4
        )
}


# ============================================================
# 7. MAIN TRAINING FUNCTION
# ============================================================

def main():

    print()

    print("=" * 60)

    print(
        "AGRIGUARD ETHIOPIA - MODEL 3"
    )

    print(
        "YIELD RISK INTELLIGENCE"
    )

    print("=" * 60)

    # ========================================================
    # Load dataset
    # ========================================================

    if not DATA.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA}"
        )

    df = pd.read_csv(DATA)

    print(
        f"Total dataset rows: {len(df)}"
    )

    # ========================================================
    # Check required columns
    # ========================================================

    required_columns = (
        FEATURES +
        ["yield_kg_ha"]
    )

    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    # Historical years are used for training.

    train = (

        df[
            df["year"] <= 2021
        ]

        .copy()

        .sort_values("year")
    )

    # The latest year is kept as an unseen test year.

    test = (

        df[
            df["year"] == 2022
        ]

        .copy()

        .sort_values("year")
    )

    if train.empty:

        raise ValueError(
            "Training dataset is empty."
        )

    if test.empty:

        raise ValueError(
            "2022 test dataset is empty."
        )

    print(
        f"Training rows: {len(train)}"
    )

    print(
        f"Test rows: {len(test)}"
    )

    print(
        "Training years:",
        sorted(
            train["year"].unique()
        )
    )

    print(
        "Test year: 2022"
    )

    # ========================================================
    # CREATE TRAINING AND TEST RISK LABELS
    # ========================================================

    # Training labels use training history.

    y_train = make_risk_labels(
        train,
        train
    )

    # Test labels use ONLY training history.
    #
    # This is important because the 2022 actual yields
    # must not be used to define their own risk thresholds.

    y_test = make_risk_labels(
        test,
        train
    )

    # ========================================================
    # DISPLAY CLASS DISTRIBUTION
    # ========================================================

    print()

    print(
        "Training risk distribution:"
    )

    train_distribution = (

        pd.Series(y_train)

        .map(CLASS_NAMES)

        .value_counts()
    )

    print(
        train_distribution
    )

    print()

    print(
        "2022 test risk distribution:"
    )

    test_distribution = (

        pd.Series(y_test)

        .map(CLASS_NAMES)

        .value_counts()
    )

    print(
        test_distribution
    )

    # ========================================================
    # EXPANDING YEAR-BASED CROSS VALIDATION
    # ========================================================

    years = sorted(
        train["year"].unique()
    )

    # Example:
    #
    # 2015-2017 -> 2018
    # 2015-2018 -> 2019
    # 2015-2019 -> 2020
    # 2015-2020 -> 2021

    folds = [

        (
            years[:i],
            [years[i]]
        )

        for i in range(
            3,
            len(years)
        )
    ]

    comparison = []

    # ========================================================
    # TRAIN EVERY MODEL
    # ========================================================

    for model_name, model in MODELS.items():

        print()

        print("-" * 60)

        print(
            f"Training: {model_name}"
        )

        print("-" * 60)

        fold_accuracy = []

        fold_macro_f1 = []

        # ====================================================
        # CROSS VALIDATION
        # ====================================================

        for train_years, valid_years in folds:

            fold_train = (

                train[
                    train["year"]
                    .isin(train_years)
                ]
            )

            fold_valid = (

                train[
                    train["year"]
                    .isin(valid_years)
                ]
            )

            # -----------------------------------------------
            # Create labels using ONLY fold training history
            # -----------------------------------------------

            fold_y_train = make_risk_labels(
                fold_train,
                fold_train
            )

            fold_y_valid = make_risk_labels(
                fold_valid,
                fold_train
            )

            # -----------------------------------------------
            # Build pipeline
            # -----------------------------------------------

            pipeline = make_pipeline(
                model
            )

            # -----------------------------------------------
            # Train
            # -----------------------------------------------

            pipeline.fit(
                fold_train[FEATURES],
                fold_y_train
            )

            # -----------------------------------------------
            # Predict validation year
            # -----------------------------------------------

            predictions = pipeline.predict(
                fold_valid[FEATURES]
            )

            # -----------------------------------------------
            # Metrics
            # -----------------------------------------------

            accuracy = accuracy_score(
                fold_y_valid,
                predictions
            )

            macro_f1 = f1_score(
                fold_y_valid,
                predictions,
                average="macro",
                zero_division=0
            )

            fold_accuracy.append(
                accuracy
            )

            fold_macro_f1.append(
                macro_f1
            )

        # ====================================================
        # FINAL TRAINING FOR THIS MODEL
        # ====================================================

        final_pipeline = make_pipeline(
            model
        )

        final_pipeline.fit(
            train[FEATURES],
            y_train
        )

        # ====================================================
        # TEST ON 2022
        # ====================================================

        test_predictions = (
            final_pipeline.predict(
                test[FEATURES]
            )
        )

        test_accuracy = accuracy_score(
            y_test,
            test_predictions
        )

        test_macro_f1 = f1_score(
            y_test,
            test_predictions,
            average="macro",
            zero_division=0
        )

        test_precision = precision_score(
            y_test,
            test_predictions,
            average="macro",
            zero_division=0
        )

        test_recall = recall_score(
            y_test,
            test_predictions,
            average="macro",
            zero_division=0
        )

        # ====================================================
        # SAVE RESULTS
        # ====================================================

        comparison.append({

            "model":
                model_name,

            "cv_accuracy":
                float(
                    np.mean(
                        fold_accuracy
                    )
                ),

            "cv_macro_f1":
                float(
                    np.mean(
                        fold_macro_f1
                    )
                ),

            "test_accuracy":
                float(
                    test_accuracy
                ),

            "test_macro_f1":
                float(
                    test_macro_f1
                ),

            "test_macro_precision":
                float(
                    test_precision
                ),

            "test_macro_recall":
                float(
                    test_recall
                )
        })

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison_df = pd.DataFrame(
        comparison
    )

    comparison_df = (
        comparison_df
        .sort_values(
            "cv_macro_f1",
            ascending=False
        )
    )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_model_name = (

        comparison_df
        .iloc[0]["model"]
    )

    best_model = MODELS[
        best_model_name
    ]

    print()

    print("=" * 60)

    print(
        "MODEL COMPARISON"
    )

    print("=" * 60)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"BEST MODEL: {best_model_name}"
    )

    # ========================================================
    # TRAIN BEST MODEL
    # ========================================================

    best_pipeline = make_pipeline(
        best_model
    )

    best_pipeline.fit(
        train[FEATURES],
        y_train
    )

    # ========================================================
    # FINAL 2022 PREDICTIONS
    # ========================================================

    final_predictions = (

        best_pipeline.predict(
            test[FEATURES]
        )
    )

    final_probabilities = (

        best_pipeline
        .predict_proba(
            test[FEATURES]
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (

        MODEL_DIR /
        "yield_risk_model.joblib"
    )

    joblib.dump(
        best_pipeline,
        model_path
    )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    predictions_df = test[
        [
            "region",
            "crop_type",
            "year",
            "yield_kg_ha"
        ]
    ].copy()

    predictions_df[
        "actual_risk"
    ] = [

        CLASS_NAMES[value]

        for value in y_test
    ]

    predictions_df[
        "predicted_risk"
    ] = [

        CLASS_NAMES[value]

        for value in final_predictions
    ]

    predictions_df[
        "confidence"
    ] = (

        final_probabilities
        .max(axis=1)
    )

    predictions_path = (

        MODEL_DIR /
        "yield_risk_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # ========================================================
    # SAVE MODEL COMPARISON
    # ========================================================

    comparison_path = (

        MODEL_DIR /
        "yield_risk_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    final_accuracy = accuracy_score(
        y_test,
        final_predictions
    )

    final_macro_f1 = f1_score(
        y_test,
        final_predictions,
        average="macro",
        zero_division=0
    )

    final_precision = precision_score(
        y_test,
        final_predictions,
        average="macro",
        zero_division=0
    )

    final_recall = recall_score(
        y_test,
        final_predictions,
        average="macro",
        zero_division=0
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        y_test,
        final_predictions,
        labels=[
            0,
            1,
            2
        ]
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {

        "project":
            "AgriGuard Ethiopia",


        "model_name":
            "Yield Risk Intelligence",


        "model_type":
            best_model_name,


        "task":
            "3-class classification",


        "classes":
            CLASS_NAMES,


        "risk_definition":
            (
                "Yield risk is based on the historical "
                "yield distribution for the same region "
                "and crop."
            ),


        "thresholds":
            (
                "Below 10th percentile = High Risk; "
                "10th to 33rd percentile = Moderate Risk; "
                "33rd percentile and above = Low Risk."
            ),


        "features":
            FEATURES,


        "categorical_features":
            CATEGORICAL_FEATURES,


        "numeric_features":
            NUMERIC_FEATURES,


        "data_source":
            (
                "Etho-Agri + Sentinel-2 + Open-Meteo"
            ),


        "synthetic_data_used":
            False,


        "training_years":
            [
                int(year)

                for year in sorted(
                    train["year"].unique()
                )
            ],


        "test_year":
            2022,


        "training_rows":
            int(len(train)),


        "test_rows":
            int(len(test)),


        "cross_validation":
            (
                "Expanding year-based validation. "
                "Validation year always follows "
                "training years."
            ),


        "selection_metric":
            "CV macro F1",


        "best_model":
            best_model_name,


        "best_cv_macro_f1":
            float(
                comparison_df
                .iloc[0]["cv_macro_f1"]
            ),


        "test_accuracy":
            float(
                final_accuracy
            ),


        "test_macro_f1":
            float(
                final_macro_f1
            ),


        "test_macro_precision":
            float(
                final_precision
            ),


        "test_macro_recall":
            float(
                final_recall
            ),


        "confusion_matrix_labels":
            [
                "Low Risk",
                "Moderate Risk",
                "High Risk"
            ],


        "confusion_matrix":
            matrix.tolist(),


        "important_note":
            (
                "Risk labels are derived from observed "
                "historical yield. The model therefore "
                "predicts the likelihood of unusually "
                "low yield based on historical outcomes "
                "and environmental features."
            )
    }

    metadata_path = (

        MODEL_DIR /
        "yield_risk_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()

    print("=" * 60)

    print(
        "FINAL MODEL 3 RESULTS"
    )

    print("=" * 60)

    print(
        f"Best model: {best_model_name}"
    )

    print(
        f"CV Macro F1: "
        f"{comparison_df.iloc[0]['cv_macro_f1']:.4f}"
    )

    print(
        f"2022 Accuracy: "
        f"{final_accuracy:.4f}"
    )

    print(
        f"2022 Macro F1: "
        f"{final_macro_f1:.4f}"
    )

    print(
        f"2022 Macro Precision: "
        f"{final_precision:.4f}"
    )

    print(
        f"2022 Macro Recall: "
        f"{final_recall:.4f}"
    )

    print()

    print(
        "Confusion Matrix:"
    )

    print(
        matrix
    )

    print()

    print(
        "Files saved:"
    )

    print(
        f"  {model_path}"
    )

    print(
        f"  {metadata_path}"
    )

    print(
        f"  {comparison_path}"
    )

    print(
        f"  {predictions_path}"
    )

    print()

    print(
        "Model 3 training completed successfully."
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
