from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

"""
AgriGuard Ethiopia - Model 2
Crop Performance Intelligence

Purpose:
    Classify Ethiopian crop performance into three categories:

        0 = Usually Low
        1 = Usually Normal
        2 = Usually Above

The labels are created from real historical yield observations.
No synthetic data or randomly generated labels are used.

Data source:
    Etho-Agri + Sentinel-2 + Open-Meteo

Validation:
    Expanding year-based validation.
    The validation year always comes after the training years.
"""


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Real matched dataset created by prepare_real_dataset.py
DATA = Path("data/etho_agri_real_matched.csv")

# Folder where Model 2 files will be saved
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
# 3. PERFORMANCE CLASSES
# ============================================================

CLASS_NAMES = {
    0: "Usually Low",
    1: "Usually Normal",
    2: "Usually Above"
}


# ============================================================
# 4. CREATE PERFORMANCE LABELS
# ============================================================

def make_labels(data, baseline):
    """
    Create crop-performance labels using historical yield.

    The model does NOT randomly assign labels.

    For every region + crop combination:

        Below 33rd percentile
            -> Usually Low

        33rd to 67th percentile
            -> Usually Normal

        Above 67th percentile
            -> Usually Above

    If a region/crop combination has fewer than 3 historical
    observations, the function falls back to:

        1. Crop-level history
        2. Overall historical history

    IMPORTANT:
    The baseline is supplied separately so that validation/test
    labels are calculated only from information available before
    that year.
    """

    # Historical yield for each region + crop
    group_values = (
        baseline
        .groupby(["region", "crop_type"])["yield_kg_ha"]
        .apply(list)
        .to_dict()
    )

    # Historical yield for each crop
    crop_values = (
        baseline
        .groupby("crop_type")["yield_kg_ha"]
        .apply(list)
        .to_dict()
    )

    # Overall historical yield
    global_values = baseline["yield_kg_ha"].tolist()

    labels = []

    for _, row in data.iterrows():

        # First try region + crop history
        values = group_values.get(
            (row["region"], row["crop_type"]),
            []
        )

        # If there are too few observations,
        # use crop-level history
        if len(values) < 3:
            values = crop_values.get(
                row["crop_type"],
                []
            )

        # If crop history is also too small,
        # use all historical observations
        if len(values) < 3:
            values = global_values

        # Calculate the 33rd and 67th percentiles
        q1, q2 = np.quantile(
            values,
            [1 / 3, 2 / 3]
        )

        current_yield = row["yield_kg_ha"]

        # Assign performance category
        if current_yield < q1:

            labels.append(0)

        elif current_yield > q2:

            labels.append(2)

        else:

            labels.append(1)

    return np.array(labels, dtype=int)


# ============================================================
# 5. CREATE MACHINE-LEARNING PIPELINE
# ============================================================

def make_pipeline(model):
    """
    Prepare categorical variables and numerical variables.

    Region and crop type are converted using OneHotEncoder.

    Numerical features are passed through unchanged.
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
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


# ============================================================
# 6. MODELS TO COMPARE
# ============================================================

MODELS = {

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
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
    print("AGRIGUARD ETHIOPIA - MODEL 2")
    print("CROP PERFORMANCE INTELLIGENCE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load real dataset
    # --------------------------------------------------------

    if not DATA.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA}"
        )

    df = pd.read_csv(DATA)

    print(f"Total dataset rows: {len(df)}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = FEATURES + [
        "yield_kg_ha"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Time-based train/test split
    # --------------------------------------------------------
    #
    # 2015-2021 -> training
    # 2022      -> final test
    #
    # This prevents the model from being tested on a year
    # it has already seen during training.
    # --------------------------------------------------------

    train = (
        df[df["year"] <= 2021]
        .copy()
        .sort_values("year")
    )

    test = (
        df[df["year"] == 2022]
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

    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")

    print(
        "Training years:",
        sorted(train["year"].unique())
    )

    print("Test year: 2022")

    # ========================================================
    # 8. CREATE TRAINING AND TEST LABELS
    # ========================================================

    # Training labels are based on training history
    y_train = make_labels(
        train,
        train
    )

    # Test labels are also based ONLY on historical
    # training information.
    #
    # We do not use 2022 yield to calculate the threshold.
    y_test = make_labels(
        test,
        train
    )

    # ========================================================
    # 9. EXPANDING YEAR-BASED CROSS VALIDATION
    # ========================================================

    years = sorted(
        train["year"].unique()
    )

    # Example:
    #
    # Train 2015-2017 -> Validate 2018
    # Train 2015-2018 -> Validate 2019
    # Train 2015-2019 -> Validate 2020
    # Train 2015-2020 -> Validate 2021
    #
    # This is safer than randomly splitting rows because
    # agriculture changes over time.

    folds = [
        (
            years[:i],
            [years[i]]
        )
        for i in range(3, len(years))
    ]

    # ========================================================
    # 10. TRAIN AND COMPARE MODELS
    # ========================================================

    comparison = []

    for model_name, model in MODELS.items():

        print()
        print("-" * 60)
        print(f"Training: {model_name}")
        print("-" * 60)

        fold_accuracy = []
        fold_macro_f1 = []

        # ----------------------------------------------------
        # Cross-validation
        # ----------------------------------------------------

        for train_years, valid_years in folds:

            fold_train = train[
                train["year"].isin(train_years)
            ]

            fold_valid = train[
                train["year"].isin(valid_years)
            ]

            # Create labels using ONLY fold training history
            fold_y_train = make_labels(
                fold_train,
                fold_train
            )

            fold_y_valid = make_labels(
                fold_valid,
                fold_train
            )

            pipeline = make_pipeline(
                model
            )

            pipeline.fit(
                fold_train[FEATURES],
                fold_y_train
            )

            predictions = pipeline.predict(
                fold_valid[FEATURES]
            )

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

        # ----------------------------------------------------
        # Train on all training years
        # ----------------------------------------------------

        final_pipeline = make_pipeline(
            model
        )

        final_pipeline.fit(
            train[FEATURES],
            y_train
        )

        # ----------------------------------------------------
        # Test on 2022
        # ----------------------------------------------------

        test_predictions = final_pipeline.predict(
            test[FEATURES]
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

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        comparison.append({

            "model": model_name,

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
    # 11. SELECT BEST MODEL
    # ========================================================

    comparison_df = pd.DataFrame(
        comparison
    )

    # Macro F1 is used because it treats all three classes
    # more fairly than accuracy alone.

    comparison_df = comparison_df.sort_values(
        "cv_macro_f1",
        ascending=False
    )

    best_model_name = (
        comparison_df
        .iloc[0]["model"]
    )

    best_model = MODELS[
        best_model_name
    ]

    print()
    print("=" * 60)
    print("MODEL COMPARISON")
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
    # 12. TRAIN THE BEST MODEL
    # ========================================================

    best_pipeline = make_pipeline(
        best_model
    )

    best_pipeline.fit(
        train[FEATURES],
        y_train
    )

    # ========================================================
    # 13. FINAL 2022 PREDICTIONS
    # ========================================================

    final_predictions = best_pipeline.predict(
        test[FEATURES]
    )

    final_probabilities = (
        best_pipeline
        .predict_proba(
            test[FEATURES]
        )
    )

    # ========================================================
    # 14. CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        y_test,
        final_predictions,
        labels=[0, 1, 2]
    )

    # ========================================================
    # 15. SAVE BEST MODEL
    # ========================================================

    model_path = (
        MODEL_DIR /
        "crop_performance_model.joblib"
    )

    joblib.dump(
        best_pipeline,
        model_path
    )

    # ========================================================
    # 16. SAVE PREDICTIONS
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
        "actual_performance"
    ] = [
        CLASS_NAMES[value]
        for value in y_test
    ]

    predictions_df[
        "predicted_performance"
    ] = [
        CLASS_NAMES[value]
        for value in final_predictions
    ]

    predictions_df[
        "confidence"
    ] = final_probabilities.max(
        axis=1
    )

    predictions_path = (
        MODEL_DIR /
        "crop_performance_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # ========================================================
    # 17. SAVE MODEL COMPARISON
    # ========================================================

    comparison_path = (
        MODEL_DIR /
        "crop_performance_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    # ========================================================
    # 18. FINAL METRICS
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
    # 19. SAVE MODEL METADATA
    # ========================================================

    metadata = {

        "project":
            "AgriGuard Ethiopia",

        "model_name":
            "Crop Performance Intelligence",

        "model_type":
            best_model_name,

        "task":
            "3-class classification",

        "classes":
            CLASS_NAMES,

        "label_definition":
            (
                "Historical yield performance relative "
                "to the historical yield distribution "
                "for the same region and crop."
            ),

        "thresholds":
            (
                "33rd percentile and 67th percentile "
                "of the historical baseline."
            ),

        "features":
            FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "numeric_features":
            NUMERIC_FEATURES,

        "data_source":
            "Etho-Agri + Sentinel-2 + Open-Meteo",

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
                "Usually Low",
                "Usually Normal",
                "Usually Above"
            ],

        "confusion_matrix":
            matrix.tolist(),

        "important_note":
            (
                "Performance labels are derived from "
                "observed historical yield. Therefore, "
                "this model predicts a yield-performance "
                "category rather than independently "
                "measuring crop health."
            )
    }

    metadata_path = (
        MODEL_DIR /
        "crop_performance_metadata.json"
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
    # 20. PRINT FINAL RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL MODEL 2 RESULTS")
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
    print("Confusion Matrix:")
    print(matrix)

    print()
    print("Files saved:")
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
    print("Model 2 training completed successfully.")


# ============================================================
# 21. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
