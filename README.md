# 🇪🇹 AgriGuard Ethiopia

## AI-Powered Crop Intelligence for Ethiopian Agriculture

AgriGuard Ethiopia is an agricultural intelligence platform that combines real Ethiopian agricultural data, satellite imagery, weather information, and machine learning to provide crop-related predictions and risk insights.

The goal is to turn complex agricultural data into information that can be understood and used by farmers and agricultural decision-makers.

---

## 🌱 What AgriGuard Does

A farmer provides simple information:

- Region
- Crop
- Cultivated area
- Year

AgriGuard then automatically obtains environmental information and runs three machine-learning models.

### 🛰️ Satellite Intelligence

Google Earth Engine and Sentinel-2 imagery are used to calculate vegetation information such as:

- NDVI Mean
- NDVI Peak

The farmer does not need to enter NDVI manually.

### 🌦️ Weather Intelligence

Weather information is obtained from Open-Meteo, including:

- Mean temperature
- Total rainfall

---

# 🧠 Three AI Models

## Model 1 — Yield Prediction

Predicts expected crop yield in:

**kg/ha**

It also calculates expected total production:

**predicted yield × cultivated area**

---

## Model 2 — Crop Performance Intelligence

Classifies expected crop performance into:

- Usually Low
- Usually Normal
- Usually Above

The classification is based on historical regional and crop performance.

---

## Model 3 — Yield Risk Intelligence

Classifies yield risk into:

- Low Risk
- Moderate Risk
- High Risk

---

# 🔄 System Architecture

```text
                 FARMER
                   │
                   ▼
       Region + Crop + Area + Year
                   │
                   ▼
          AGRIGUARD ENGINE
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 Google Earth Engine     Open-Meteo
        │                     │
        ▼                     ▼
      NDVI              Temperature
      Mean              Rainfall
      Peak
        │                     │
        └──────────┬──────────┘
                   ▼
              AI MODELS
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
      Yield   Performance    Risk
        │          │          │
        └──────────┼──────────┘
                   ▼
          FARMER-FRIENDLY
              RESULTS
```
