"""
============================================================
AGRIGUARD ETHIOPIA
SHARED INTELLIGENCE ENGINE
============================================================

Purpose:
    This file is the central "brain" of AgriGuard.

    It connects:

        Farmer inputs
              |
              v
        Region + Crop + Area + Year
              |
              v
        Google Earth Engine
        Sentinel-2 NDVI
              |
              v
        Open-Meteo Weather
              |
              v
        Real environmental features
              |
              +----------------------+
              |                      |
              v                      v
        Model 1                 Model 2
        Yield Prediction        Crop Performance
              |                      |
              +----------+-----------+
                         |
                         v
                       Model 3
                     Yield Risk
                         |
                         v
                 AgriGuard Result
                         |
                         v
                  Amharic Summary

IMPORTANT:
    - The farmer does NOT enter NDVI.
    - NDVI is obtained automatically from Google Earth Engine.
    - Weather is obtained automatically from Open-Meteo.
    - No synthetic data is created here.
    - Production is calculated from predicted yield x area.
    - The three already-trained models are loaded from models/.
    - This file is designed to later be used by:
          * Streamlit web app
          * USSD demonstration
          * Real USSD gateway
          * SMS service

The environmental feature definitions match the real-data
training pipeline:

    region
    crop_type
    year
    area_cultivated_ha
    NDVI_mean
    NDVI_peak
    temperature_mean
    rainfall
============================================================
"""

import os
import json
from pathlib import Path
from datetime import date

import joblib
import numpy as np
import pandas as pd
import requests
import ee


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "data" / "real_data_cache"

YIELD_MODEL_FILE = MODEL_DIR / "real_yield_model.joblib"
PERFORMANCE_MODEL_FILE = MODEL_DIR / "crop_performance_model.joblib"
RISK_MODEL_FILE = MODEL_DIR / "yield_risk_model.joblib"

YIELD_METADATA_FILE = MODEL_DIR / "real_yield_metadata.json"
PERFORMANCE_METADATA_FILE = (
    MODEL_DIR / "crop_performance_metadata.json"
)
RISK_METADATA_FILE = MODEL_DIR / "yield_risk_metadata.json"

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. GOOGLE EARTH ENGINE PROJECT
# ============================================================

GEE_PROJECT = "agriguard-ethiopia"


# ============================================================
# 3. REGION COORDINATES
# ============================================================
#
# These are the same representative coordinates used by the
# real-data preparation pipeline.
#
# IMPORTANT:
# They represent regional analysis points, NOT individual
# farmer field boundaries.
#
# Therefore, NDVI returned here should be interpreted as a
# regional/crop-season indicator.
# ============================================================

REGION_COORDS = {
    "Tigray": (14.10, 38.30),
    "Afar": (11.75, 40.80),
    "Amhara": (11.50, 38.50),
    "Oromiya": (8.50, 39.00),
    "Somalia": (6.00, 44.00),
    "Benushangul Gumuz": (10.78, 35.70),
    "S.N.N.P.R": (7.00, 37.50),
    "Gambela": (8.25, 34.60),
    "Harari": (9.31, 42.13),
    "DIRE DAWA": (9.60, 41.85),
}


# ============================================================
# 4. CROP GROWING SEASONS
# ============================================================
#
# These are the same approximate crop-season windows used
# when creating the real training dataset.
# ============================================================

CROP_SEASONS = {
    "Teff": ("06-01", "11-30"),
    "Barely": ("06-01", "11-30"),
    "Wheat": ("06-01", "11-30"),
    "Millet": ("06-01", "11-30"),
    "Oats": ("06-01", "11-30"),
    "Maize": ("05-01", "10-31"),
    "Sorghum": ("05-01", "11-30"),
}


# ============================================================
# 5. MODEL FEATURES
# ============================================================
#
# These MUST remain consistent with the trained models.
#
# Production is intentionally NOT included because production
# is directly related to yield and could cause target leakage.
# ============================================================

FEATURES = [
    "region",
    "crop_type",
    "year",
    "area_cultivated_ha",
    "NDVI_mean",
    "NDVI_peak",
    "temperature_mean",
    "rainfall",
]


# ============================================================
# 6. CLASS NAMES
# ============================================================

PERFORMANCE_CLASSES = {
    0: "Usually Low",
    1: "Usually Normal",
    2: "Usually Above",
}


RISK_CLASSES = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk",
}


# ============================================================
# 7. AMHARIC TRANSLATIONS
# ============================================================
#
# These are short farmer-friendly explanations.
# They can later be replaced with improved professional
# agricultural translations.
# ============================================================

AMHARIC_PERFORMANCE = {
    "Usually Low": "በአብዛኛው ዝቅተኛ",
    "Usually Normal": "በአብዛኛው መደበኛ",
    "Usually Above": "በአብዛኛው ከፍተኛ",
}


AMHARIC_RISK = {
    "Low Risk": "ዝቅተኛ አደጋ",
    "Moderate Risk": "መካከለኛ አደጋ",
    "High Risk": "ከፍተኛ አደጋ",
}


# ============================================================
# 8. MODEL CACHE
# ============================================================
#
# Models are loaded once and reused.
# This prevents repeatedly loading large files every time a
# farmer makes a prediction.
# ============================================================

_yield_model = None
_performance_model = None
_risk_model = None

_gee_initialized = False


# ============================================================
# 9. BASIC VALIDATION
# ============================================================

def validate_farmer_input(
    region,
    crop_type,
    area_cultivated_ha,
    year,
):
    """
    Validate the information supplied by the farmer.

    Farmer-facing systems should collect only information the
    farmer can reasonably know.

    NDVI is NOT requested here.
    Weather is NOT requested here.
    """

    if region not in REGION_COORDS:
        raise ValueError(
            f"Unknown region: {region}. "
            f"Available regions: {', '.join(REGION_COORDS.keys())}"
        )

    if crop_type not in CROP_SEASONS:
        raise ValueError(
            f"Unknown crop: {crop_type}. "
            f"Available crops: {', '.join(CROP_SEASONS.keys())}"
        )

    try:
        area = float(area_cultivated_ha)
    except (TypeError, ValueError):
        raise ValueError(
            "Cultivated area must be a number."
        )

    if area <= 0:
        raise ValueError(
            "Cultivated area must be greater than zero."
        )

    try:
        year = int(year)
    except (TypeError, ValueError):
        raise ValueError(
            "Year must be an integer."
        )

    if year < 2015:
        raise ValueError(
            "The current real-data intelligence pipeline "
            "starts from 2015 because Sentinel-2 data was "
            "not used for older years."
        )

    return region, crop_type, area, year


# ============================================================
# 10. LOAD MODELS
# ============================================================

def load_models():
    """
    Load the three already-trained AgriGuard models.

    Model 1:
        Real Yield Prediction

    Model 2:
        Crop Performance Intelligence

    Model 3:
        Yield Risk Intelligence
    """

    global _yield_model
    global _performance_model
    global _risk_model

    if _yield_model is None:

        if not YIELD_MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Could not find Model 1:\n"
                f"{YIELD_MODEL_FILE}"
            )

        _yield_model = joblib.load(
            YIELD_MODEL_FILE
        )

    if _performance_model is None:

        if not PERFORMANCE_MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Could not find Model 2:\n"
                f"{PERFORMANCE_MODEL_FILE}"
            )

        _performance_model = joblib.load(
            PERFORMANCE_MODEL_FILE
        )

    if _risk_model is None:

        if not RISK_MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Could not find Model 3:\n"
                f"{RISK_MODEL_FILE}"
            )

        _risk_model = joblib.load(
            RISK_MODEL_FILE
        )

    return (
        _yield_model,
        _performance_model,
        _risk_model,
    )


# ============================================================
# 11. LOAD MODEL METADATA
# ============================================================

def load_metadata_file(path):
    """
    Safely load a model metadata JSON file.

    Metadata is useful for displaying model information in
    the future web application.
    """

    if not path.exists():
        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:
        return {}


def get_all_model_metadata():
    """
    Return metadata for all three models.
    """

    return {
        "yield_model": load_metadata_file(
            YIELD_METADATA_FILE
        ),

        "performance_model": load_metadata_file(
            PERFORMANCE_METADATA_FILE
        ),

        "risk_model": load_metadata_file(
            RISK_METADATA_FILE
        ),
    }


# ============================================================
# 12. INITIALIZE GOOGLE EARTH ENGINE
# ============================================================

def initialize_gee():
    """
    Connect to Google Earth Engine only when needed.

    This is called lazily instead of when this module is
    imported.

    That is important because future systems such as a local
    USSD simulator may import this file before making a request.
    """

    global _gee_initialized

    if _gee_initialized:
        return

    try:

        ee.Initialize(
            project=GEE_PROJECT
        )

        _gee_initialized = True

    except Exception as error:

        raise RuntimeError(
            "Google Earth Engine could not be initialized.\n\n"
            "Make sure you have authenticated with:\n"
            "    earthengine authenticate\n\n"
            f"Original error: {error}"
        )


# ============================================================
# 13. GET REAL SENTINEL-2 NDVI
# ============================================================

def get_sentinel_ndvi(
    region,
    crop_type,
    year,
):
    """
    Get real Sentinel-2 NDVI from Google Earth Engine.

    Returns:
        NDVI_mean
        NDVI_peak

    NDVI formula:

        NDVI = (NIR - RED) / (NIR + RED)

    Sentinel-2 bands:

        B8 = Near Infrared
        B4 = Red

    The implementation follows the same approach used during
    creation of the real training dataset.
    """

    if region not in REGION_COORDS:
        raise ValueError(
            f"No coordinates available for {region}."
        )

    if crop_type not in CROP_SEASONS:
        raise ValueError(
            f"No growing season available for {crop_type}."
        )

    cache_file = (
        CACHE_DIR
        / f"ndvi_{region}_{crop_type}_{year}.csv"
    )

    # --------------------------------------------------------
    # Check cache first
    # --------------------------------------------------------

    if cache_file.exists():

        try:

            cached = pd.read_csv(
                cache_file
            )

            if len(cached) > 0:

                return (
                    float(
                        cached.iloc[0]["NDVI_mean"]
                    ),
                    float(
                        cached.iloc[0]["NDVI_peak"]
                    ),
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # Initialize GEE
    # --------------------------------------------------------

    initialize_gee()

    latitude, longitude = REGION_COORDS[region]

    start_month_day, end_month_day = (
        CROP_SEASONS[crop_type]
    )

    start_date = (
        f"{year}-{start_month_day}"
    )

    end_date = (
        f"{year}-{end_month_day}"
    )

    # --------------------------------------------------------
    # Create regional analysis area
    # --------------------------------------------------------

    point = ee.Geometry.Point(
        [
            longitude,
            latitude,
        ]
    )

    region_geometry = point.buffer(
        5000
    )

    # --------------------------------------------------------
    # Sentinel-2 collection
    # --------------------------------------------------------

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterDate(
            start_date,
            end_date
        )
        .filterBounds(
            region_geometry
        )
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                70
            )
        )
        .map(
            lambda image:
            image.normalizedDifference(
                [
                    "B8",
                    "B4",
                ]
            ).rename(
                "NDVI"
            )
        )
    )

    # --------------------------------------------------------
    # Check whether Sentinel-2 images exist
    # --------------------------------------------------------

    count = (
        collection
        .size()
        .getInfo()
    )

    if count == 0:

        raise RuntimeError(
            f"No usable Sentinel-2 images were found "
            f"for {region}, {crop_type}, {year}."
        )

    # --------------------------------------------------------
    # Mean NDVI
    # --------------------------------------------------------

    mean_image = (
        collection.mean()
    )

    mean_result = (
        mean_image
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region_geometry,
            scale=10,
            bestEffort=True,
            maxPixels=1e8,
        )
        .getInfo()
    )

    # --------------------------------------------------------
    # Peak NDVI
    # --------------------------------------------------------
    #
    # First calculate the spatial mean for every image.
    # Then select the maximum seasonal value.
    # --------------------------------------------------------

    def image_mean(image):

        value = (
            image
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region_geometry,
                scale=10,
                bestEffort=True,
                maxPixels=1e8,
            )
            .get("NDVI")
        )

        return image.set(
            "regional_mean_ndvi",
            value
        )

    with_means = (
        collection.map(
            image_mean
        )
    )

    peak = (
        with_means
        .aggregate_max(
            "regional_mean_ndvi"
        )
        .getInfo()
    )

    mean_ndvi = mean_result.get(
        "NDVI"
    )

    if mean_ndvi is None or peak is None:

        raise RuntimeError(
            f"NDVI could not be calculated for "
            f"{region}, {crop_type}, {year}."
        )

    mean_ndvi = float(
        mean_ndvi
    )

    peak = float(
        peak
    )

    # --------------------------------------------------------
    # Save cache
    # --------------------------------------------------------

    result = pd.DataFrame(
        [
            {
                "NDVI_mean": mean_ndvi,
                "NDVI_peak": peak,
            }
        ]
    )

    result.to_csv(
        cache_file,
        index=False
    )

    return (
        mean_ndvi,
        peak,
    )


# ============================================================
# 14. GET REAL HISTORICAL WEATHER
# ============================================================

def get_historical_weather(
    region,
    crop_type,
    year,
):
    """
    Retrieve real historical weather from Open-Meteo.

    Returns:

        temperature_mean
        rainfall

    temperature_mean:
        Mean daily temperature during the crop season.

    rainfall:
        Total precipitation during the crop season.

    This matches the weather feature definition used when
    creating the real training dataset.
    """

    if region not in REGION_COORDS:
        raise ValueError(
            f"No coordinates available for {region}."
        )

    if crop_type not in CROP_SEASONS:
        raise ValueError(
            f"No growing season available for {crop_type}."
        )

    cache_file = (
        CACHE_DIR
        / f"weather_{region}_{crop_type}_{year}.csv"
    )

    # --------------------------------------------------------
    # Check cache
    # --------------------------------------------------------

    if cache_file.exists():

        try:

            cached = pd.read_csv(
                cache_file
            )

            if len(cached) > 0:

                return (
                    float(
                        cached.iloc[0][
                            "temperature_mean"
                        ]
                    ),
                    float(
                        cached.iloc[0][
                            "rainfall"
                        ]
                    ),
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    latitude, longitude = (
        REGION_COORDS[region]
    )

    start_month_day, end_month_day = (
        CROP_SEASONS[crop_type]
    )

    start_date = (
        f"{year}-{start_month_day}"
    )

    end_date = (
        f"{year}-{end_month_day}"
    )

    # --------------------------------------------------------
    # Open-Meteo historical archive
    # --------------------------------------------------------

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": (
            "temperature_2m_mean,"
            "precipitation_sum"
        ),
        "timezone": "Africa/Addis_Ababa",
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        daily = data.get(
            "daily"
        )

        if not daily:
            raise RuntimeError(
                "Open-Meteo returned no daily weather data."
            )

        temperatures = daily.get(
            "temperature_2m_mean"
        )

        rainfall = daily.get(
            "precipitation_sum"
        )

        if temperatures is None:
            raise RuntimeError(
                "Temperature data was not returned."
            )

        if rainfall is None:
            raise RuntimeError(
                "Rainfall data was not returned."
            )

        temperatures = pd.to_numeric(
            pd.Series(
                temperatures
            ),
            errors="coerce",
        )

        rainfall = pd.to_numeric(
            pd.Series(
                rainfall
            ),
            errors="coerce",
        )

        temperature_mean = (
            temperatures.mean()
        )

        rainfall_total = (
            rainfall.sum()
        )

        if pd.isna(
            temperature_mean
        ):

            raise RuntimeError(
                "Temperature data contains no usable values."
            )

        if pd.isna(
            rainfall_total
        ):

            raise RuntimeError(
                "Rainfall data contains no usable values."
            )

        temperature_mean = float(
            temperature_mean
        )

        rainfall_total = float(
            rainfall_total
        )

        # ----------------------------------------------------
        # Save cache
        # ----------------------------------------------------

        result = pd.DataFrame(
            [
                {
                    "temperature_mean":
                        temperature_mean,

                    "rainfall":
                        rainfall_total,
                }
            ]
        )

        result.to_csv(
            cache_file,
            index=False
        )

        return (
            temperature_mean,
            rainfall_total,
        )

    except requests.RequestException as error:

        raise RuntimeError(
            "Open-Meteo weather request failed.\n"
            f"Details: {error}"
        )

    except Exception as error:

        raise RuntimeError(
            "Weather calculation failed.\n"
            f"Details: {error}"
        )


# ============================================================
# 15. COLLECT ENVIRONMENTAL FEATURES
# ============================================================

def collect_environmental_features(
    region,
    crop_type,
    year,
):
    """
    Automatically collect the environmental information
    required by the three models.

    The farmer never needs to know or enter these values.

    Returns:

        NDVI_mean
        NDVI_peak
        temperature_mean
        rainfall
    """

    # --------------------------------------------------------
    # Satellite
    # --------------------------------------------------------

    ndvi_mean, ndvi_peak = (
        get_sentinel_ndvi(
            region,
            crop_type,
            year,
        )
    )

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    temperature_mean, rainfall = (
        get_historical_weather(
            region,
            crop_type,
            year,
        )
    )

    return {
        "NDVI_mean": float(
            ndvi_mean
        ),

        "NDVI_peak": float(
            ndvi_peak
        ),

        "temperature_mean": float(
            temperature_mean
        ),

        "rainfall": float(
            rainfall
        ),
    }


# ============================================================
# 16. CREATE MODEL INPUT ROW
# ============================================================

def create_model_input(
    region,
    crop_type,
    year,
    area_cultivated_ha,
    environmental_features,
):
    """
    Build exactly one model input row.

    This is the bridge between:
        external data
    and:
        machine-learning models.
    """

    row = {
        "region": region,
        "crop_type": crop_type,
        "year": int(year),
        "area_cultivated_ha": float(
            area_cultivated_ha
        ),
        "NDVI_mean": float(
            environmental_features[
                "NDVI_mean"
            ]
        ),
        "NDVI_peak": float(
            environmental_features[
                "NDVI_peak"
            ]
        ),
        "temperature_mean": float(
            environmental_features[
                "temperature_mean"
            ]
        ),
        "rainfall": float(
            environmental_features[
                "rainfall"
            ]
        ),
    }

    frame = pd.DataFrame(
        [row],
        columns=FEATURES,
    )

    return frame


# ============================================================
# 17. MODEL 1 — YIELD PREDICTION
# ============================================================

def predict_yield(
    model_input,
):
    """
    Run Model 1.

    Model 1 predicts:

        yield_kg_ha

    This is yield per hectare.

    Production is calculated separately as:

        predicted yield x cultivated area
    """

    yield_model, _, _ = (
        load_models()
    )

    prediction = (
        yield_model
        .predict(
            model_input
        )
    )

    predicted_yield = float(
        prediction[0]
    )

    if predicted_yield < 0:
        predicted_yield = 0.0

    return predicted_yield


# ============================================================
# 18. MODEL 2 — CROP PERFORMANCE
# ============================================================

def predict_performance(
    model_input,
):
    """
    Run Model 2.

    Classes:

        0 = Usually Low
        1 = Usually Normal
        2 = Usually Above

    The model's confidence is the highest class probability.
    """

    _, performance_model, _ = (
        load_models()
    )

    prediction = (
        performance_model
        .predict(
            model_input
        )
    )

    class_number = int(
        prediction[0]
    )

    class_name = (
        PERFORMANCE_CLASSES.get(
            class_number,
            "Unknown",
        )
    )

    confidence = None

    if hasattr(
        performance_model,
        "predict_proba",
    ):

        probabilities = (
            performance_model
            .predict_proba(
                model_input
            )[0]
        )

        confidence = float(
            np.max(
                probabilities
            )
        )

    return {
        "class_number":
            class_number,

        "class_name":
            class_name,

        "confidence":
            confidence,
    }


# ============================================================
# 19. MODEL 3 — YIELD RISK
# ============================================================

def predict_risk(
    model_input,
):
    """
    Run Model 3.

    Classes:

        0 = Low Risk
        1 = Moderate Risk
        2 = High Risk

    The risk model estimates risk based on historical yield
    outcomes and environmental features.
    """

    _, _, risk_model = (
        load_models()
    )

    prediction = (
        risk_model
        .predict(
            model_input
        )
    )

    class_number = int(
        prediction[0]
    )

    class_name = (
        RISK_CLASSES.get(
            class_number,
            "Unknown",
        )
    )

    confidence = None

    if hasattr(
        risk_model,
        "predict_proba",
    ):

        probabilities = (
            risk_model
            .predict_proba(
                model_input
            )[0]
        )

        confidence = float(
            np.max(
                probabilities
            )
        )

    return {
        "class_number":
            class_number,

        "class_name":
            class_name,

        "confidence":
            confidence,
    }


# ============================================================
# 20. CALCULATE PREDICTED PRODUCTION
# ============================================================

def calculate_production(
    predicted_yield_kg_ha,
    area_cultivated_ha,
):
    """
    Convert predicted yield into predicted total production.

    Formula:

        Production =
            Yield per hectare x cultivated area
    """

    production = (
        float(predicted_yield_kg_ha)
        * float(area_cultivated_ha)
    )

    return float(
        production
    )


# ============================================================
# 21. CREATE AMHARIC SUMMARY
# ============================================================

def create_amharic_summary(
    region,
    crop_type,
    area_cultivated_ha,
    predicted_yield_kg_ha,
    predicted_production_kg,
    performance,
    risk,
):
    """
    Create a short farmer-friendly Amharic result.

    This is intentionally kept simple because it may later
    be transmitted through SMS, where long messages are not
    ideal.
    """

    performance_amharic = (
        AMHARIC_PERFORMANCE.get(
            performance["class_name"],
            performance["class_name"],
        )
    )

    risk_amharic = (
        AMHARIC_RISK.get(
            risk["class_name"],
            risk["class_name"],
        )
    )

    summary = (
        f"አግሪጋርድ የእርሻ ውጤት\n"
        f"ክልል: {region}\n"
        f"ሰብል: {crop_type}\n"
        f"መሬት: {area_cultivated_ha:,.2f} ሄክታር\n"
        f"የሚጠበቅ ምርት/ሄክታር: "
        f"{predicted_yield_kg_ha:,.0f} ኪ.ግ\n"
        f"የሚጠበቅ ጠቅላላ ምርት: "
        f"{predicted_production_kg:,.0f} ኪ.ግ\n"
        f"የሰብል አፈጻጸም: "
        f"{performance_amharic}\n"
        f"የምርት አደጋ: "
        f"{risk_amharic}"
    )

    return summary


# ============================================================
# 22. MAIN AGRIGUARD ANALYSIS
# ============================================================

def analyze_farm(
    region,
    crop_type,
    area_cultivated_ha,
    year,
):
    """
    Run the COMPLETE AgriGuard intelligence pipeline.

    This is the most important function in this file.

    Future systems can simply call:

        analyze_farm(
            region="Oromiya",
            crop_type="Maize",
            area_cultivated_ha=2,
            year=2022
        )

    The function then automatically:

        1. Validates farmer input
        2. Gets NDVI from GEE
        3. Gets weather from Open-Meteo
        4. Builds model features
        5. Runs Model 1
        6. Runs Model 2
        7. Runs Model 3
        8. Calculates expected production
        9. Creates Amharic summary

    Returns a dictionary containing all results.
    """

    # --------------------------------------------------------
    # STEP 1 — Validate farmer input
    # --------------------------------------------------------

    (
        region,
        crop_type,
        area_cultivated_ha,
        year,
    ) = validate_farmer_input(
        region,
        crop_type,
        area_cultivated_ha,
        year,
    )

    # --------------------------------------------------------
    # STEP 2 — Automatically collect satellite + weather
    # --------------------------------------------------------

    environmental_features = (
        collect_environmental_features(
            region,
            crop_type,
            year,
        )
    )

    # --------------------------------------------------------
    # STEP 3 — Build the ML input
    # --------------------------------------------------------

    model_input = create_model_input(
        region=region,
        crop_type=crop_type,
        year=year,
        area_cultivated_ha=area_cultivated_ha,
        environmental_features=environmental_features,
    )

    # --------------------------------------------------------
    # STEP 4 — Model 1
    # --------------------------------------------------------

    predicted_yield = (
        predict_yield(
            model_input
        )
    )

    # --------------------------------------------------------
    # STEP 5 — Calculate expected production
    # --------------------------------------------------------

    predicted_production = (
        calculate_production(
            predicted_yield,
            area_cultivated_ha,
        )
    )

    # --------------------------------------------------------
    # STEP 6 — Model 2
    # --------------------------------------------------------

    performance = (
        predict_performance(
            model_input
        )
    )

    # --------------------------------------------------------
    # STEP 7 — Model 3
    # --------------------------------------------------------

    risk = (
        predict_risk(
            model_input
        )
    )

    # --------------------------------------------------------
    # STEP 8 — Amharic summary
    # --------------------------------------------------------

    amharic_summary = (
        create_amharic_summary(
            region=region,
            crop_type=crop_type,
            area_cultivated_ha=area_cultivated_ha,
            predicted_yield_kg_ha=predicted_yield,
            predicted_production_kg=predicted_production,
            performance=performance,
            risk=risk,
        )
    )

    # --------------------------------------------------------
    # STEP 9 — Return complete intelligence result
    # --------------------------------------------------------

    return {
        "farmer_input": {
            "region": region,
            "crop_type": crop_type,
            "area_cultivated_ha":
                float(
                    area_cultivated_ha
                ),
            "year": int(year),
        },

        "environmental_features":
            environmental_features,

        "model_input":
            model_input.to_dict(
                orient="records"
        )[0],

        "yield_prediction": {
            "yield_kg_ha":
                float(
                    predicted_yield
                ),

            "production_kg":
                float(
                    predicted_production
                ),
        },

        "crop_performance":
            performance,

        "yield_risk":
            risk,

        "amharic_summary":
            amharic_summary,

        "data_sources": [
            "Etho-Agri",
            "Sentinel-2 via Google Earth Engine",
            "Open-Meteo",
        ],

        "synthetic_data_used":
            False,
    }


# ============================================================
# 23. SMS-FRIENDLY SUMMARY
# ============================================================

def create_sms_summary(
    result,
):
    """
    Convert the complete result into a short SMS-style message.

    This is for the future SMS/USSD system.

    It deliberately does not expose NDVI numbers unless needed.
    The farmer does not need to understand satellite data to use
    the service.
    """

    input_data = result[
        "farmer_input"
    ]

    yield_data = result[
        "yield_prediction"
    ]

    performance = result[
        "crop_performance"
    ]

    risk = result[
        "yield_risk"
    ]

    performance_amharic = (
        AMHARIC_PERFORMANCE.get(
            performance["class_name"],
            performance["class_name"],
        )
    )

    risk_amharic = (
        AMHARIC_RISK.get(
            risk["class_name"],
            risk["class_name"],
        )
    )

    message = (
        f"AgriGuard\n"
        f"{input_data['crop_type']} - "
        f"{input_data['region']}\n"
        f"የሚጠበቅ ምርት: "
        f"{yield_data['production_kg']:,.0f} ኪ.ግ\n"
        f"አፈጻጸም: "
        f"{performance_amharic}\n"
        f"አደጋ: "
        f"{risk_amharic}"
    )

    return message


# ============================================================
# 24. SIMPLE TERMINAL TEST
# ============================================================
#
# This lets us test the engine BEFORE app.py or the USSD demo
# exists.
#
# IMPORTANT:
#     This test actually contacts GEE/Open-Meteo, so it requires
#     internet access and valid GEE authentication.
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("AGRIGUARD ETHIOPIA - SHARED INTELLIGENCE ENGINE")
    print("=" * 70)

    print()
    print("Loading models...")

    load_models()

    print("All three models loaded successfully.")

    print()
    print("Running test analysis...")

    try:

        result = analyze_farm(
            region="Oromiya",
            crop_type="Maize",
            area_cultivated_ha=2.0,
            year=2022,
        )

        print()
        print("=" * 70)
        print("AGRIGUARD TEST RESULT")
        print("=" * 70)

        print()
        print("Farmer Input:")
        print(
            result["farmer_input"]
        )

        print()
        print("Environmental Features:")
        print(
            result[
                "environmental_features"
            ]
        )

        print()
        print("Model 1 - Yield:")
        print(
            result[
                "yield_prediction"
            ]
        )

        print()
        print("Model 2 - Crop Performance:")
        print(
            result[
                "crop_performance"
            ]
        )

        print()
        print("Model 3 - Yield Risk:")
        print(
            result[
                "yield_risk"
            ]
        )

        print()
        print("Amharic Summary:")
        print(
            result[
                "amharic_summary"
            ]
        )

        print()
        print("SMS Version:")
        print(
            create_sms_summary(
                result
            )
        )

        print()
        print("=" * 70)
        print("ENGINE TEST COMPLETED")
        print("=" * 70)

    except Exception as error:

        print()
        print("=" * 70)
        print("ENGINE TEST FAILED")
        print("=" * 70)

        print()
        print(str(error))
