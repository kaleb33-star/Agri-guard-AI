"""
AgriGuard Ethiopia
===================

Main Streamlit application.

This app does NOT train the models.
It uses the already-trained models through agriguard_engine.py.

Farmer/user enters:
    - Region
    - Crop
    - Cultivated area
    - Year

AgriGuard then obtains:
    - NDVI from Google Earth Engine
    - Temperature from Open-Meteo
    - Rainfall from Open-Meteo

and sends the complete feature set through the three AI models.
"""

from pathlib import Path
import json

import pandas as pd
import streamlit as st

import agriguard_engine


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriGuard Ethiopia",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7faf7;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #0b5d3b,
            #16824f
        );
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        margin-bottom: 5px;
        font-size: 2.3rem;
    }

    .hero p {
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e3e9e4;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }

    .result-title {
        font-size: 0.9rem;
        color: #6c756f;
        font-weight: 600;
        text-transform: uppercase;
    }

    .result-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #123b29;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        background: #eef7f1;
        border-left: 5px solid #16824f;
        margin: 12px 0;
    }

    .warning-box {
        padding: 18px;
        border-radius: 12px;
        background: #fff8e6;
        border-left: 5px solid #e0a21a;
        margin: 12px 0;
    }

    .risk-low {
        color: #16824f;
        font-weight: 800;
    }

    .risk-medium {
        color: #c58913;
        font-weight: 800;
    }

    .risk-high {
        color: #c43d35;
        font-weight: 800;
    }

    .footer {
        text-align: center;
        color: #737c76;
        margin-top: 40px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🇪🇹 AgriGuard Ethiopia</h1>
        <p>
            AI-powered crop intelligence using real agricultural,
            satellite, and weather data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌱 AgriGuard")

st.sidebar.markdown(
    """
    ### How it works

    👨‍🌾 Farmer information
    ↓

    🛰️ Satellite NDVI
    ↓

    🌦️ Weather data
    ↓

    🧠 Three AI models
    ↓

    📊 Agricultural intelligence
    """
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🌾 Crop Analysis",
        "🧠 How It Works",
        "📊 Model Information",
    ],
)


# ============================================================
# AVAILABLE OPTIONS
# ============================================================

REGIONS = [
    "Afar",
    "Amhara",
    "Benushangul Gumuz",
    "DIRE DAWA",
    "Gambela",
    "Harari",
    "Oromiya",
    "S.N.N.P.R",
    "Somalia",
    "Tigray",
]

CROPS = [
    "Teff",
    "Barely",
    "Wheat",
    "Maize",
    "Millet",
    "Sorghum",
    "Oats",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_value(data, possible_names, default=None):
    """
    Safely find a value in a dictionary.

    This makes the interface more tolerant if the engine
    uses slightly different names for the same result.
    """

    if not isinstance(data, dict):
        return default

    for name in possible_names:
        if name in data:
            return data[name]

    return default


def format_number(value, decimals=2):
    """Format numerical values safely."""

    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def risk_class(value):
    """Return CSS class for risk level."""

    text = str(value).lower()

    if "high" in text:
        return "risk-high"

    if "moderate" in text or "medium" in text:
        return "risk-medium"

    return "risk-low"


def get_result_summary(result):
    """
    Extract the main values from agriguard_engine output.

    The engine returns several values inside nested dictionaries,
    so this function extracts them safely for the Streamlit UI.
    """

    # ============================================================
    # MODEL 1 — YIELD + PRODUCTION
    # ============================================================

    yield_data = result.get("yield_prediction", {})

    predicted_yield = yield_data.get("yield_kg_ha")
    predicted_production = yield_data.get("production_kg")

    # ============================================================
    # MODEL 2 — CROP PERFORMANCE
    # ============================================================

    performance = result.get("crop_performance", {})

    if isinstance(performance, dict):
        performance_name = performance.get("class_name", "N/A")
        performance_confidence = performance.get("confidence")
    else:
        performance_name = str(performance)
        performance_confidence = None

    # ============================================================
    # MODEL 3 — YIELD RISK
    # ============================================================

    risk = result.get("yield_risk", {})

    if isinstance(risk, dict):
        risk_name = risk.get("class_name", "N/A")
        risk_confidence = risk.get("confidence")
    else:
        risk_name = str(risk)
        risk_confidence = None

    # ============================================================
    # ENVIRONMENTAL FEATURES
    # ============================================================

    environmental = result.get("environmental_features", {})

    ndvi_mean = environmental.get("NDVI_mean")
    ndvi_peak = environmental.get("NDVI_peak")
    temperature = environmental.get("temperature_mean")
    rainfall = environmental.get("rainfall")

    # ============================================================
    # FARMER SUMMARIES
    # ============================================================

    amharic_summary = result.get("amharic_summary", "")
    sms_summary = result.get("sms_summary", "")

    return {
        "yield": predicted_yield,
        "production": predicted_production,
        "performance": performance_name,
        "performance_confidence": performance_confidence,
        "risk": risk_name,
        "risk_confidence": risk_confidence,
        "ndvi_mean": ndvi_mean,
        "ndvi_peak": ndvi_peak,
        "temperature": temperature,
        "rainfall": rainfall,
        "amharic_summary": amharic_summary,
        "sms_summary": sms_summary,
    }


# ============================================================
# CROP ANALYSIS PAGE
# ============================================================

if page == "🌾 Crop Analysis":

    st.subheader("🌾 Analyze a Crop")

    st.write(
        "Enter the information available to the farmer. "
        "AgriGuard automatically obtains satellite and weather features."
    )

    col1, col2 = st.columns(2)

    with col1:

        region = st.selectbox(
            "🇪🇹 Region",
            REGIONS,
        )

        crop = st.selectbox(
            "🌱 Crop",
            CROPS,
        )

    with col2:

        area = st.number_input(
            "📐 Cultivated area (hectares)",
            min_value=0.1,
            max_value=100000.0,
            value=2.0,
            step=0.1,
        )

        year = st.number_input(
            "📅 Year",
            min_value=2015,
            max_value=2100,
            value=2022,
            step=1,
        )

    st.info(
        "🛰️ You do not need to enter NDVI. "
        "AgriGuard retrieves it automatically."
    )

    analyze = st.button(
        "🔍 ANALYZE CROP",
        type="primary",
        use_container_width=True,
    )

    if analyze:

        if area <= 0:
            st.error("Cultivated area must be greater than zero.")
            st.stop()

        with st.spinner(
            "AgriGuard is collecting satellite and weather data "
            "and running the AI models..."
        ):

            try:

                result = agriguard_engine.analyze_farm(
                    region=region,
                    crop_type=crop,
                    area_cultivated_ha=float(area),
                    year=int(year),
                )

            except Exception as error:

                st.error(
                    "AgriGuard could not complete the analysis."
                )
                st.exception(error)
                st.stop()

        values = get_result_summary(result)

        st.success("✅ Analysis completed.")

        # ----------------------------------------------------
        # MAIN RESULTS
        # ----------------------------------------------------

        st.subheader("🧠 AI Results")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Expected Yield",
                (
                    f"{format_number(values['yield'])} kg/ha"
                    if values["yield"] is not None
                    else "N/A"
                ),
            )

        with c2:
            st.metric(
                "Expected Production",
                (
                    f"{format_number(values['production'])} kg"
                    if values["production"] is not None
                    else "N/A"
                ),
            )

        with c3:
            st.metric(
                "Performance",
                str(values["performance"]),
            )
            if values["performance_confidence"] is not None:
                st.caption(
                    f"Confidence: {float(values['performance_confidence']):.0%}"
                )

        with c4:

            risk = str(values["risk"])

            st.markdown(
                f"""
                <div class="card">
                    <div class="result-title">
                        Yield Risk
                    </div>
                    <div class="result-value {risk_class(risk)}">
                        {risk}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if values["risk_confidence"] is not None:
                st.caption(
                    f"Confidence: {float(values['risk_confidence']):.0%}"
                )

        # ----------------------------------------------------
        # ENVIRONMENTAL DATA
        # ----------------------------------------------------

        st.subheader("🛰️ Environmental Intelligence")

        e1, e2, e3, e4 = st.columns(4)

        with e1:
            st.metric(
                "NDVI Mean",
                format_number(values["ndvi_mean"], 3),
            )

        with e2:
            st.metric(
                "NDVI Peak",
                format_number(values["ndvi_peak"], 3),
            )

        with e3:
            st.metric(
                "Mean Temperature",
                (
                    f"{format_number(values['temperature'])} °C"
                    if values["temperature"] is not None
                    else "N/A"
                ),
            )

        with e4:
            st.metric(
                "Rainfall",
                (
                    f"{format_number(values['rainfall'])} mm"
                    if values["rainfall"] is not None
                    else "N/A"
                ),
            )

        # ----------------------------------------------------
        # FARMER SUMMARY
        # ----------------------------------------------------

        st.subheader("👨‍🌾 Farmer-Friendly Result")

        if values["amharic_summary"]:

            st.markdown(
                f"""
                <div class="info-box">
                    <h4>🇪🇹 የአማርኛ ማጠቃለያ</h4>
                    <p>{values["amharic_summary"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.info(
                "The engine did not return an Amharic summary "
                "for this analysis."
            )

        if values["sms_summary"]:

            st.subheader("📱 SMS-style Message")

            st.code(
                str(values["sms_summary"]),
                language=None,
            )

        # ----------------------------------------------------
        # TECHNICAL DETAILS
        # ----------------------------------------------------

        with st.expander("🔬 Technical analysis details"):

            st.write(
                "These are the values passed to the AI system."
            )

            technical_data = {
                "Region": region,
                "Crop": crop,
                "Year": year,
                "Cultivated area (ha)": area,
                "NDVI mean": values["ndvi_mean"],
                "NDVI peak": values["ndvi_peak"],
                "Temperature mean (°C)": values["temperature"],
                "Rainfall (mm)": values["rainfall"],
            }

            st.dataframe(
                pd.DataFrame(
                    technical_data.items(),
                    columns=["Feature", "Value"],
                ),
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # RAW ENGINE RESULT
        # ----------------------------------------------------

        with st.expander("🧪 Engine output"):

            st.json(result)


# ============================================================
# HOW IT WORKS PAGE
# ============================================================

elif page == "🧠 How It Works":

    st.subheader("🧠 How AgriGuard Works")

    st.markdown(
        """
        ### 1️⃣ Farmer input

        The farmer provides simple information:

        - Region
        - Crop
        - Cultivated area
        - Year

        ### 2️⃣ Satellite intelligence

        AgriGuard uses Google Earth Engine and Sentinel-2
        imagery to obtain vegetation information.

        **NDVI is calculated automatically.**

        The farmer does not need to understand or enter NDVI.

        ### 3️⃣ Weather intelligence

        Historical weather information is obtained from
        Open-Meteo.

        AgriGuard uses:

        - Mean temperature
        - Total rainfall

        ### 4️⃣ AI analysis

        The complete feature set is sent through three
        trained models:

        **Model 1 — Yield Prediction**

        Predicts crop yield in kg/ha.

        **Model 2 — Crop Performance**

        Classifies performance as:

        - Usually Low
        - Usually Normal
        - Usually Above

        **Model 3 — Yield Risk**

        Classifies risk as:

        - Low Risk
        - Moderate Risk
        - High Risk

        ### 5️⃣ Farmer-friendly result

        The technical model outputs are transformed into
        understandable agricultural information.

        The system can also produce an Amharic and
        SMS-style summary.
        """
    )

    st.markdown(
        """
        ### 🔄 Complete pipeline

        ```text
        👨‍🌾 Farmer
             ↓
        Region + Crop + Area + Year
             ↓
        🛰️ Google Earth Engine
             ↓
        NDVI Mean + NDVI Peak
             ↓
        🌦️ Open-Meteo
             ↓
        Temperature + Rainfall
             ↓
        🧠 Three AI Models
             ↓
        🌾 Yield
        📊 Performance
        ⚠️ Risk
             ↓
        🇪🇹 Farmer-friendly information
        ```
        """
    )


# ============================================================
# MODEL INFORMATION PAGE
# ============================================================

elif page == "📊 Model Information":

    st.subheader("📊 AgriGuard Model Information")

    st.markdown(
        """
        AgriGuard uses three separate machine-learning layers.

        The models are trained independently and are loaded
        from the `models/` directory.
        """
    )

    # --------------------------------------------------------
    # MODEL 1
    # --------------------------------------------------------

    st.markdown("### 🌾 Model 1 — Real Yield Prediction")

    st.write(
        "Regression model that predicts crop yield in kg/ha."
    )

    model1_metadata = (
        Path("models") /
        "real_yield_metadata.json"
    )

    if model1_metadata.exists():

        try:

            metadata = json.loads(
                model1_metadata.read_text(
                    encoding="utf-8"
                )
            )

            st.json(metadata)

        except Exception as error:

            st.warning(
                f"Could not read Model 1 metadata: {error}"
            )

    else:

        st.info(
            "Model 1 metadata file was not found."
        )

    # --------------------------------------------------------
    # MODEL 2
    # --------------------------------------------------------

    st.markdown("### 📊 Model 2 — Crop Performance")

    st.write(
        """
        Classification model that evaluates expected
        crop performance relative to historical observations.
        """
    )

    model2_metadata = (
        Path("models") /
        "crop_performance_metadata.json"
    )

    if model2_metadata.exists():

        try:

            metadata = json.loads(
                model2_metadata.read_text(
                    encoding="utf-8"
                )
            )

            st.json(metadata)

        except Exception as error:

            st.warning(
                f"Could not read Model 2 metadata: {error}"
            )

    else:

        st.info(
            "Model 2 metadata file was not found."
        )

    # --------------------------------------------------------
    # MODEL 3
    # --------------------------------------------------------

    st.markdown("### ⚠️ Model 3 — Yield Risk")

    st.write(
        """
        Classification model that estimates the level
        of yield risk.
        """
    )

    model3_metadata = (
        Path("models") /
        "yield_risk_metadata.json"
    )

    if model3_metadata.exists():

        try:

            metadata = json.loads(
                model3_metadata.read_text(
                    encoding="utf-8"
                )
            )

            st.json(metadata)

        except Exception as error:

            st.warning(
                f"Could not read Model 3 metadata: {error}"
            )

    else:

        st.info(
            "Model 3 metadata file was not found."
        )

    # --------------------------------------------------------
    # DATA CREDIBILITY
    # --------------------------------------------------------

    st.divider()

    st.subheader("🔎 Data credibility")

    st.success(
        "The current real-data yield pipeline uses "
        "real matched agricultural, satellite, and weather data."
    )

    st.markdown(
        """
        **Core data sources**

        - Ethiopian agricultural observations
        - Sentinel-2 satellite imagery
        - Open-Meteo weather data

        **The model training dataset contains 317 usable
        matched records covering 2015–2022.**

        The final yield-model test year was 2022, while
        earlier years were used for training.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🌱 <b>AgriGuard Ethiopia</b><br>
        AI + Satellite + Weather Intelligence for Agriculture
    </div>
    """,
    unsafe_allow_html=True,
)
