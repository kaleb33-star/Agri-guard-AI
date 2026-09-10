"""
AgriGuard Ethiopia - Real Dataset Preparation

Purpose:
    Build a real machine-learning dataset by combining:

    1. Ethiopian agricultural ground-truth data
       from the Etho-Agri CSV.

    2. Real Sentinel-2 satellite NDVI data
       obtained through Google Earth Engine.

    3. Real historical weather data
       obtained from Open-Meteo.

The final dataset will contain observations matched by:
    Region + Crop + Year

IMPORTANT:
    No synthetic NDVI, rainfall, temperature, or yield values
    are created by this script.

Only observations for which the required real satellite and
weather information can be obtained are kept.
"""

import os
import time
import requests
import numpy as np
import pandas as pd
import ee


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

INPUT_FILE = "data/Etho-Agri Dataset.csv"
OUTPUT_FILE = "data/etho_agri_real_matched.csv"
CACHE_DIR = "data/real_data_cache"

# Sentinel-2 became useful for this project from 2015 onward.
# We will not invent satellite data for older years.
MIN_YEAR = 2015
MAX_YEAR = 2022

# Google Earth Engine project
GEE_PROJECT = "agriguard-ethiopia"


# ============================================================
# 2. ETHIOPIAN REGION REPRESENTATIVE LOCATIONS
# ============================================================
#
# The CSV contains region names, but not field polygons.
# Therefore, these coordinates represent regional sampling
# locations rather than individual farms.
#
# Sentinel-2 values generated from these locations should be
# interpreted as regional/crop-season indicators.
#

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
# 3. CROP GROWING SEASONS
# ============================================================
#
# These are approximate Ethiopian growing-season windows.
# They are used consistently to calculate the satellite and
# weather indicators.
#

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
# 4. CREATE CACHE DIRECTORY
# ============================================================

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


# ============================================================
# 5. LOAD THE ETHO-AGRI DATASET
# ============================================================

print("=" * 70)
print("AGriGuard Ethiopia - REAL DATA PREPARATION")
print("=" * 70)

print("\n[1/6] Loading Etho-Agri dataset...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nCould not find:\n{INPUT_FILE}\n\n"
        "Make sure your CSV is named:\n"
        "Etho-Agri Dataset.csv\n"
        "and is inside the data folder."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} rows.")
print("Columns found:")
print(df.columns.tolist())


# ============================================================
# 6. STANDARDIZE COLUMN NAMES
# ============================================================

df = df.rename(
    columns={
        "Region": "region",
        "crop type": "crop_type",
        "Year": "year",
        " Area cultivated(Ha)": "area_cultivated_ha",
        "Production(kg)": "production_kg",
        "Yeild (kg/ha)": "yield_kg_ha",
    }
)

required_columns = [
    "region",
    "crop_type",
    "year",
    "area_cultivated_ha",
    "production_kg",
    "yield_kg_ha",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing required columns: {missing_columns}\n"
        f"Columns found: {df.columns.tolist()}"
    )


# ============================================================
# 7. CLEAN BASIC DATA
# ============================================================

print("\n[2/6] Cleaning agricultural data...")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["area_cultivated_ha"] = pd.to_numeric(
    df["area_cultivated_ha"],
    errors="coerce"
)
df["production_kg"] = pd.to_numeric(
    df["production_kg"],
    errors="coerce"
)
df["yield_kg_ha"] = pd.to_numeric(
    df["yield_kg_ha"],
    errors="coerce"
)

df["region"] = df["region"].astype(str).str.strip()
df["crop_type"] = df["crop_type"].astype(str).str.strip()


# ============================================================
# 8. KEEP ONLY YEARS WITH SENTINEL-2
# ============================================================

original_rows = len(df)

df = df[
    (df["year"] >= MIN_YEAR)
    & (df["year"] <= MAX_YEAR)
].copy()

print(
    f"Kept {len(df):,} rows from "
    f"{MIN_YEAR}-{MAX_YEAR}."
)

print(
    f"Removed {original_rows - len(df):,} rows "
    "because Sentinel-2 data will not be fabricated "
    "for older years."
)


# ============================================================
# 9. REMOVE ROWS WITHOUT AREA
# ============================================================
#
# Area is not allowed to be fabricated.
# We need real cultivated area for the final yield model
# because it is one of the intended model inputs.
#

before_area = len(df)

df = df.dropna(
    subset=[
        "area_cultivated_ha"
    ]
).copy()

print(
    f"Removed {before_area - len(df):,} rows "
    "with missing cultivated area."
)


# ============================================================
# 10. REMOVE INVALID TARGET VALUES
# ============================================================

df = df.dropna(
    subset=[
        "yield_kg_ha"
    ]
).copy()

df = df[
    df["yield_kg_ha"] >= 0
].copy()


# ============================================================
# 11. CHECK REGIONS AND CROPS
# ============================================================

unknown_regions = sorted(
    set(df["region"]) - set(REGION_COORDS)
)

if unknown_regions:
    print(
        "\nWARNING: These regions do not have coordinates:"
    )
    print(unknown_regions)

    df = df[
        df["region"].isin(REGION_COORDS)
    ].copy()


unknown_crops = sorted(
    set(df["crop_type"]) - set(CROP_SEASONS)
)

if unknown_crops:
    print(
        "\nWARNING: These crops do not have growing-season "
        "definitions:"
    )
    print(unknown_crops)

    df = df[
        df["crop_type"].isin(CROP_SEASONS)
    ].copy()


print("\nRegions:")
print(sorted(df["region"].unique()))

print("\nCrops:")
print(sorted(df["crop_type"].unique()))

print("\nYears:")
print(sorted(df["year"].unique()))


# ============================================================
# 12. INITIALIZE GOOGLE EARTH ENGINE
# ============================================================

print("\n[3/6] Connecting to Google Earth Engine...")

try:
    ee.Initialize(project=GEE_PROJECT)
    print("Google Earth Engine initialized successfully.")

except Exception as e:
    print("\nGoogle Earth Engine could not be initialized.")
    print("\nRun this command first:")
    print("    earthengine authenticate")
    print("\nThen run this script again.")
    raise e


# ============================================================
# 13. SENTINEL-2 NDVI FUNCTION
# ============================================================

def get_sentinel_ndvi(region, crop, year):
    """
    Calculate real Sentinel-2 NDVI indicators.

    Returns:
        NDVI_mean
        NDVI_peak

    NDVI = (NIR - RED) / (NIR + RED)

    Sentinel-2:
        B8 = Near Infrared
        B4 = Red
    """

    cache_file = os.path.join(
        CACHE_DIR,
        f"ndvi_{region}_{crop}_{year}.csv"
    )

    if os.path.exists(cache_file):
        cached = pd.read_csv(cache_file)

        if len(cached) > 0:
            return (
                float(cached.iloc[0]["NDVI_mean"]),
                float(cached.iloc[0]["NDVI_peak"])
            )

    if region not in REGION_COORDS:
        return None, None

    latitude, longitude = REGION_COORDS[region]

    start_month_day, end_month_day = CROP_SEASONS[crop]

    start_date = f"{year}-{start_month_day}"
    end_date = f"{year}-{end_month_day}"

    # Create a small regional analysis area around
    # the representative coordinate.
    point = ee.Geometry.Point(
        [longitude, latitude]
    )

    region_geometry = point.buffer(5000)

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterDate(start_date, end_date)
        .filterBounds(region_geometry)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                70
            )
        )
        .map(
            lambda image:
            image.normalizedDifference(
                ["B8", "B4"]
            ).rename("NDVI")
        )
    )

    count = collection.size().getInfo()

    if count == 0:
        return None, None

    # Mean NDVI across all usable images.
    mean_image = collection.mean()

    mean_result = mean_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region_geometry,
        scale=10,
        bestEffort=True,
        maxPixels=1e8
    ).getInfo()

    # Peak NDVI:
    # First calculate the spatial mean NDVI for each image,
    # then take the maximum seasonal value.
    def image_mean(image):
        value = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region_geometry,
            scale=10,
            bestEffort=True,
            maxPixels=1e8
        ).get("NDVI")

        return image.set(
            "regional_mean_ndvi",
            value
        )

    with_means = collection.map(image_mean)

    peak = (
        with_means
        .aggregate_max("regional_mean_ndvi")
        .getInfo()
    )

    mean_ndvi = mean_result.get("NDVI")

    if mean_ndvi is None or peak is None:
        return None, None

    mean_ndvi = float(mean_ndvi)
    peak = float(peak)

    result = pd.DataFrame(
        [{
            "NDVI_mean": mean_ndvi,
            "NDVI_peak": peak
        }]
    )

    result.to_csv(
        cache_file,
        index=False
    )

    return mean_ndvi, peak


# ============================================================
# 14. OPEN-METEO WEATHER FUNCTION
# ============================================================

def get_weather(region, crop, year):
    """
    Retrieve real historical weather from Open-Meteo.

    Returns:
        temperature_mean
        rainfall

    temperature_mean:
        Mean daily temperature during the crop season.

    rainfall:
        Total precipitation during the crop season.
    """

    cache_file = os.path.join(
        CACHE_DIR,
        f"weather_{region}_{crop}_{year}.csv"
    )

    if os.path.exists(cache_file):
        cached = pd.read_csv(cache_file)

        if len(cached) > 0:
            return (
                float(
                    cached.iloc[0]["temperature_mean"]
                ),
                float(
                    cached.iloc[0]["rainfall"]
                )
            )

    if region not in REGION_COORDS:
        return None, None

    latitude, longitude = REGION_COORDS[region]

    start_month_day, end_month_day = CROP_SEASONS[crop]

    start_date = f"{year}-{start_month_day}"
    end_date = f"{year}-{end_month_day}"

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
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        daily = data.get("daily")

        if not daily:
            return None, None

        temperatures = daily.get(
            "temperature_2m_mean"
        )

        rainfall = daily.get(
            "precipitation_sum"
        )

        if temperatures is None or rainfall is None:
            return None, None

        temperatures = pd.to_numeric(
            pd.Series(temperatures),
            errors="coerce"
        )

        rainfall = pd.to_numeric(
            pd.Series(rainfall),
            errors="coerce"
        )

        temperature_mean = temperatures.mean()
        rainfall_total = rainfall.sum()

        if pd.isna(temperature_mean):
            return None, None

        if pd.isna(rainfall_total):
            return None, None

        result = pd.DataFrame(
            [{
                "temperature_mean": float(
                    temperature_mean
                ),
                "rainfall": float(
                    rainfall_total
                )
            }]
        )

        result.to_csv(
            cache_file,
            index=False
        )

        return (
            float(temperature_mean),
            float(rainfall_total)
        )

    except Exception as e:
        print(
            f"\nWeather request failed for "
            f"{region}, {crop}, {year}: {e}"
        )

        return None, None


# ============================================================
# 15. MATCH AGRICULTURAL + SATELLITE + WEATHER DATA
# ============================================================

print("\n[4/6] Matching real satellite and weather data...")

results = []

total = len(df)

for index, row in df.iterrows():

    region = row["region"]
    crop = row["crop_type"]
    year = int(row["year"])

    print(
        f"\rProcessing "
        f"{len(results) + 1}/{total}: "
        f"{region} | {crop} | {year}",
        end=""
    )

    # --------------------------------------------------------
    # Sentinel-2
    # --------------------------------------------------------

    ndvi_mean, ndvi_peak = get_sentinel_ndvi(
        region,
        crop,
        year
    )

    if ndvi_mean is None or ndvi_peak is None:
        continue

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    temperature_mean, rainfall = get_weather(
        region,
        crop,
        year
    )

    if (
        temperature_mean is None
        or rainfall is None
    ):
        continue

    # --------------------------------------------------------
    # Combine everything
    # --------------------------------------------------------

    results.append(
        {
            "region": region,
            "crop_type": crop,
            "year": year,
            "area_cultivated_ha":
                float(row["area_cultivated_ha"]),
            "production_kg":
                float(row["production_kg"]),
            "yield_kg_ha":
                float(row["yield_kg_ha"]),
            "NDVI_mean": ndvi_mean,
            "NDVI_peak": ndvi_peak,
            "temperature_mean":
                temperature_mean,
            "rainfall": rainfall,
        }
    )

    # Small delay to avoid unnecessarily aggressive
    # requests to external services.
    time.sleep(0.1)


print("\n")


# ============================================================
# 16. CREATE FINAL DATASET
# ============================================================

print("[5/6] Creating final matched dataset...")

if len(results) == 0:
    raise RuntimeError(
        "\nNo observations were successfully matched.\n\n"
        "Possible causes:\n"
        "1. GEE authentication/project configuration.\n"
        "2. Sentinel-2 coverage/cloud filtering.\n"
        "3. Region coordinate problems.\n"
        "4. Open-Meteo request failure.\n"
    )

final_df = pd.DataFrame(results)


# ============================================================
# 17. REMOVE DUPLICATES
# ============================================================

final_df = final_df.drop_duplicates(
    subset=[
        "region",
        "crop_type",
        "year"
    ]
).copy()


# ============================================================
# 18. SAVE FINAL DATASET
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 19. FINAL REPORT
# ============================================================

print("\n[6/6] DATA PREPARATION COMPLETE")
print("=" * 70)

print(
    f"Original agricultural rows: {original_rows:,}"
)

print(
    f"Final matched rows: {len(final_df):,}"
)

print(
    f"Output file:\n{OUTPUT_FILE}"
)

print("\nFinal columns:")
print(final_df.columns.tolist())

print("\nRegions:")
print(
    sorted(
        final_df["region"].unique()
    )
)

print("\nCrops:")
print(
    sorted(
        final_df["crop_type"].unique()
    )
)

print("\nYear range:")
print(
    int(final_df["year"].min()),
    "to",
    int(final_df["year"].max())
)

print("\nMissing values:")
print(
    final_df.isna().sum()
)

print("\nExample rows:")
print(
    final_df.head(10).to_string(
        index=False
    )
)

print("\nSaved successfully.")
print("=" * 70)
