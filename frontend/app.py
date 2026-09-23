"""
Animal Disease Outbreak Prediction - Streamlit Application
A modern, rich frontend for veterinary epidemiologists, livestock managers,
and researchers to predict disease outbreaks using machine learning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

from inference import (
    get_predictor,
    load_country_coordinates,
    load_species_list,
    CAUSAL_AGENTS,
    EPI_UNIT_TYPES,
    MONTH_MAP,
    SEASON_LOOKUP,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="VETSHIELD | Epizootic Disease Prediction Engine",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient header styling */
    .hero-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(99, 102, 241, 0.12) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10b981, #6366f1, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        max-width: 800px;
        line-height: 1.5;
    }

    .badge-container {
        display: flex;
        gap: 10px;
        margin-top: 14px;
        flex-wrap: wrap;
    }

    .stat-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #cbd5e1;
    }

    .stat-badge-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
    }

    /* Result Card Styles */
    .result-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        position: relative;
    }

    .result-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .disease-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 12px;
        line-height: 1.2;
    }

    .confidence-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 16px;
    }

    .section-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Button glow effect */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.39);
        transition: all 0.2s ease-in-out;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.55);
        transform: translateY(-1px);
    }

    .advisory-box {
        background: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-top: 14px;
        font-size: 0.9rem;
        color: #fde68a;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Resources & Singletons
# ---------------------------------------------------------
@st.cache_resource
def get_cached_predictor():
    return get_predictor()

@st.cache_data
def get_cached_country_coords():
    return load_country_coordinates()

@st.cache_data
def get_cached_species():
    return load_species_list()


predictor = get_cached_predictor()
country_coords = get_cached_country_coords()
species_list = get_cached_species()
all_countries = sorted(list(country_coords.keys()))


# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🐾 VETSHIELD Epizootic Intelligence</div>
    <div class="hero-subtitle">
        AI-Powered Animal Disease Outbreak Diagnostic & Surveillance Platform.
        Evaluates species vulnerabilities, bioclimatic seasonality, spatial geolocations, and host epidemiology
        across 60 high-impact global livestock & wildlife pathogens.
    </div>
    <div class="badge-container">
        <span class="stat-badge"><span class="stat-badge-dot"></span> CatBoost Ensemble Classifier</span>
        <span class="stat-badge">🎯 77.7% Stratified CV Accuracy</span>
        <span class="stat-badge">🔬 60 Pathogen Categories</span>
        <span class="stat-badge">🌍 187 Tracked Countries</span>
        <span class="stat-badge">⚡ Real-time Inference</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_predict, tab_batch, tab_analytics = st.tabs([
    "🔍 Outbreak Diagnostic Predictor",
    "📁 Batch Outbreak Assessment (CSV)",
    "📊 Model Performance & Epizootic Directory"
])


# =========================================================
# TAB 1: OUTBREAK DIAGNOSTIC PREDICTOR
# =========================================================
with tab_predict:
    # Sidebar: Model Architecture Info
    with st.sidebar:
        st.subheader("🛠️ Model Architecture")
        st.markdown("""
        - **Model**: CatBoost 1000 Trees
        - **Engineered Features**: 73 dimensions
        - **Input Variables**: 12 core features
        - **Encodings**: Cyclical Months, Log-Susceptible, Target Frequency
        """)

    # Main Form Container
    st.markdown("### 📋 Enter Outbreak Epidemiological Parameters")

    # Helper to retrieve session values if scenario loaded
    def get_val(key, default):
        return st.session_state.get(f"input_{key}", default)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # Card 1: Host & Pathogen Profile
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🧬 1. Host Animal & Pathogen Profile</div>
        </div>
        """, unsafe_allow_html=True)

        # Species selection
        default_species = get_val("species_name", "Swine")
        species_idx = species_list.index(default_species) if default_species in species_list else 0
        species_name = st.selectbox(
            "Animal Species Name",
            options=species_list,
            index=species_idx,
            help="Select the animal species afflicted in this outbreak incident."
        )

        # Habitat toggles
        st.markdown("**Habitat & Domestication Status**")
        hab_col1, hab_col2, hab_col3 = st.columns(3)
        with hab_col1:
            is_domestic_bool = st.checkbox("Domestic", value=bool(get_val("is_domestic", 1)))
        with hab_col2:
            is_wild_bool = st.checkbox("Wild", value=bool(get_val("is_wild", 0)))
        with hab_col3:
            is_aquatic_bool = st.checkbox("Aquatic", value=bool(get_val("is_aquatic", 0)))

        # Causal Agent Type
        default_agent = get_val("causal_agent_type", "Virus")
        agent_idx = CAUSAL_AGENTS.index(default_agent) if default_agent in CAUSAL_AGENTS else 0
        causal_agent_type = st.selectbox(
            "Suspected Causal Agent Type",
            options=CAUSAL_AGENTS,
            index=agent_idx,
            help="Etiological agent category identified via clinical examination or preliminary microscopy."
        )

        # Card 2: Population at Risk & Temporal Settings
        st.markdown("""
        <div class="section-card">
            <div class="section-title">⏱️ 2. Temporal & Population Dynamics</div>
        </div>
        """, unsafe_allow_html=True)

        temp_col1, temp_col2 = st.columns(2)
        with temp_col1:
            months = list(MONTH_MAP.keys())
            default_month = get_val("month", "August")
            month_idx = months.index(default_month) if default_month in months else 0
            month = st.selectbox("Outbreak Month", options=months, index=month_idx)

        # Auto-compute recommended season based on month
        month_num = MONTH_MAP[month]
        auto_season = SEASON_LOOKUP.get(month_num, "Monsoon")
        
        with temp_col2:
            season_options = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
            default_season = get_val("season", auto_season)
            season_idx = season_options.index(default_season) if default_season in season_options else 0
            season = st.selectbox(
                "Bioclimatic Season",
                options=season_options,
                index=season_idx,
                help=f"Auto-suggested for {month}: {auto_season}"
            )

        susceptible_count = st.number_input(
            "Susceptible Animal Population",
            min_value=0,
            max_value=200000000,
            value=int(get_val("susceptible", 1461)),
            step=10,
            help="Total head count of susceptible livestock or wildlife at risk within the epidemiological zone."
        )

    with col_right:
        # Card 3: Location & Epidemiological Facility
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📍 3. Geographic & Facility Epidemiology</div>
        </div>
        """, unsafe_allow_html=True)

        default_country = get_val("country_name", "Bolivia")
        country_idx = all_countries.index(default_country) if default_country in all_countries else 0
        
        country_name = st.selectbox(
            "Country / Jurisdiction",
            options=all_countries,
            index=country_idx,
            help="Country where outbreak occurred (calibrates historical disease frequency distributions)."
        )

        # Country default coords
        country_default_lat = country_coords.get(country_name, {}).get("latitude", 0.0)
        country_default_lon = country_coords.get(country_name, {}).get("longitude", 0.0)

        coord_col1, coord_col2 = st.columns(2)
        with coord_col1:
            lat_val = float(get_val("latitude", country_default_lat))
            latitude = st.number_input(
                "Latitude (°)",
                min_value=-90.0,
                max_value=90.0,
                value=lat_val,
                format="%.4f",
                help="Geographic latitude coordinate."
            )
        with coord_col2:
            lon_val = float(get_val("longitude", country_default_lon))
            longitude = st.number_input(
                "Longitude (°)",
                min_value=-180.0,
                max_value=180.0,
                value=lon_val,
                format="%.4f",
                help="Geographic longitude coordinate."
            )

        # Epidemiological Unit Type
        default_unit = get_val("epi_unit_type", "Farm")
        unit_idx = EPI_UNIT_TYPES.index(default_unit) if default_unit in EPI_UNIT_TYPES else 0
        epi_unit_type = st.selectbox(
            "Epidemiological Facility Unit Type",
            options=EPI_UNIT_TYPES,
            index=unit_idx,
            help="Setting where infection transmission or mortality was detected."
        )

        # Quick map preview of coordinates
        st.caption("📍 Outbreak Coordinate Verification Map:")
        map_df = pd.DataFrame([{"lat": latitude, "lon": longitude}])
        st.map(map_df, zoom=3, height=180)

    # Prediction Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Analyze & Predict Disease Outbreak", use_container_width=True)

    # ---------------------------------------------------------
    # Inference Execution & Results
    # ---------------------------------------------------------
    if predict_clicked or "last_prediction" in st.session_state:
        # Prepare input payload
        input_data = {
            "species_name": species_name,
            "is_wild": int(is_wild_bool),
            "is_domestic": int(is_domestic_bool),
            "is_aquatic": int(is_aquatic_bool),
            "causal_agent_type": causal_agent_type,
            "country_name": country_name,
            "latitude": latitude,
            "longitude": longitude,
            "month": month,
            "season": season,
            "susceptible": susceptible_count,
            "epi_unit_type": epi_unit_type
        }

        with st.spinner("Executing CatBoost inference across 60 pathogen classes..."):
            result = predictor.predict_single(input_data)
            st.session_state["last_prediction"] = result

        res = st.session_state["last_prediction"]
        pred_disease = res["predicted_disease"]
        conf_pct = res["confidence_percentage"]
        conf_level = res["confidence_level"]
        badge_color = res["badge_color"]

        st.markdown("---")
        st.markdown("## 🎯 Diagnostic Results & Differential Analysis")

        res_col1, res_col2 = st.columns([1.1, 1], gap="large")

        with res_col1:
            # Hero Prediction Card
            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">Primary Differential Diagnosis</div>
                <div class="disease-title">{pred_disease}</div>
                <div class="confidence-pill" style="background-color: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color};">
                    Confidence: {conf_pct}% ({conf_level})
                </div>
                <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
                    <strong>Pathogen Class:</strong> {causal_agent_type}<br>
                    <strong>Identified Host:</strong> {species_name} ({'Domestic' if is_domestic_bool else ''} {'Wild' if is_wild_bool else ''} {'Aquatic' if is_aquatic_bool else ''})<br>
                    <strong>Transmission Environment:</strong> {epi_unit_type} in {country_name}<br>
                    <strong>Population Exposure:</strong> {susceptible_count:,} head at risk
                </div>
                <div class="advisory-box">
                    ⚠️ <strong>Immediate Biosecurity Advisory:</strong><br>
                    Initiate strict quarantine perimeter around {epi_unit_type}. Restrict animal transit within a 10 km containment radius. Collect mandatory serum and tissue swabs for molecular PCR confirmation. Notify competent national veterinary authority and submit WOAH early warning report.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            # Differential Candidate Ranking Chart (Plotly)
            st.markdown("#### 📊 Top 5 Differential Pathogen Candidates")
            top_5_df = pd.DataFrame(res["top_5"])
            
            # Reverse order for horizontal chart so top is on top
            top_5_df = top_5_df.iloc[::-1]

            fig = go.Figure(go.Bar(
                x=top_5_df["percentage"],
                y=top_5_df["disease"],
                orientation="h",
                text=[f"{p:.1f}%" for p in top_5_df["percentage"]],
                textposition="auto",
                marker=dict(
                    color=top_5_df["percentage"],
                    colorscale="Tealgrn",
                    line=dict(color="rgba(255,255,255,0.2)", width=1)
                )
            ))

            fig.update_layout(
                xaxis_title="Predicted Probability (%)",
                yaxis_title="",
                margin=dict(l=10, r=20, t=10, b=40),
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                xaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.15)", range=[0, max(100, top_5_df["percentage"].max() + 10)]),
                yaxis=dict(showgrid=False)
            )

            st.plotly_chart(fig, use_container_width=True)

        # Feature Transformation Audit Accordion
        with st.expander("🔍 View Machine Learning Feature Transformations"):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                st.metric("Cleaned Species", input_data["species_name"])
                st.metric("Common Species Grouping", input_data["species_name"] if input_data["species_name"] in predictor.common_species else "Other")
            with f_col2:
                country_f = predictor.country_freq.get(input_data["country_name"], 0.0)
                st.metric("Country Historical Frequency", f"{country_f:.4f}")
                st.metric("Log1p Susceptible", f"{np.log1p(input_data['susceptible']):.3f}")
            with f_col3:
                m_num = MONTH_MAP[input_data["month"]]
                st.metric("Month Cyclical Sin", f"{np.sin(2 * np.pi * m_num / 12):.3f}")
                st.metric("Month Cyclical Cos", f"{np.cos(2 * np.pi * m_num / 12):.3f}")


# =========================================================
# TAB 2: BATCH PREDICTION (CSV)
# =========================================================
with tab_batch:
    st.markdown("### 📁 Batch Outbreak Assessment & Triage")
    st.markdown("""
    Upload a CSV file containing multiple outbreak observation records. The platform will batch-transform
    all entries and return predicted disease classifications along with confidence ratings.
    """)

    # Download Template Button
    sample_batch_df = pd.DataFrame([
        {
            "species_name": "Swine",
            "is_wild": 0,
            "is_domestic": 1,
            "is_aquatic": 0,
            "causal_agent_type": "Virus",
            "country_name": "Bolivia",
            "latitude": -19.7444,
            "longitude": -64.1010,
            "month": "August",
            "season": "Monsoon",
            "susceptible": 1461,
            "epi_unit_type": "Farm"
        },
        {
            "species_name": "House Crow",
            "is_wild": 1,
            "is_domestic": 0,
            "is_aquatic": 0,
            "causal_agent_type": "Virus",
            "country_name": "India",
            "latitude": 22.5726,
            "longitude": 88.3639,
            "month": "January",
            "season": "Winter",
            "susceptible": 150,
            "epi_unit_type": "Not applicable"
        },
        {
            "species_name": "Cattle",
            "is_wild": 0,
            "is_domestic": 1,
            "is_aquatic": 0,
            "causal_agent_type": "Virus",
            "country_name": "France",
            "latitude": 46.2276,
            "longitude": 2.2137,
            "month": "September",
            "season": "Monsoon",
            "susceptible": 340,
            "epi_unit_type": "Farm"
        }
    ])

    csv_template = sample_batch_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sample Outbreak Batch CSV Template",
        data=csv_template,
        file_name="sample_outbreaks_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Outbreak Records CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df)} records!")
            st.markdown("#### Raw Uploaded Records Preview:")
            st.dataframe(batch_df.head(), use_container_width=True)

            required_cols = [
                "species_name", "is_wild", "is_domestic", "is_aquatic",
                "causal_agent_type", "country_name", "latitude", "longitude",
                "month", "season", "susceptible", "epi_unit_type"
            ]

            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            if missing_cols:
                st.error(f"Uploaded CSV is missing mandatory columns: {missing_cols}")
            else:
                if st.button("⚡ Run Batch Prediction Pipeline", use_container_width=True):
                    with st.spinner("Processing batch predictions..."):
                        processed_df = predictor.predict_batch(batch_df)
                    
                    st.success("Batch classification completed!")
                    st.markdown("#### 🎯 Prediction Results:")
                    st.dataframe(
                        processed_df[[
                            "species_name", "country_name", "predicted_disease",
                            "confidence_pct", "causal_agent_type", "susceptible"
                        ]],
                        use_container_width=True
                    )

                    # Outbreak Map for batch
                    if "latitude" in processed_df.columns and "longitude" in processed_df.columns:
                        st.markdown("#### 🗺️ Batch Outbreak Geographic Distribution:")
                        geo_clean = processed_df.dropna(subset=["latitude", "longitude"]).rename(
                            columns={"latitude": "lat", "longitude": "lon"}
                        )
                        st.map(geo_clean, zoom=1, height=350)

                    # Export results
                    processed_csv = processed_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions Report (CSV)",
                        data=processed_csv,
                        file_name="outbreak_predictions_report.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")


# =========================================================
# TAB 3: MODEL PERFORMANCE & EPIZOOTIC DIRECTORY
# =========================================================
with tab_analytics:
    st.markdown("### 📊 Model Architecture & Epizootic Directory")
    
    st.markdown("""
    The **CatBoost Classifier** was trained on cross-border disease outbreak reporting data, combining
    demographic, geographic, pathogen, and seasonal markers to diagnose epizootic occurrences.
    """)

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("CV Accuracy", "77.66%", delta="±0.66%")
    with m_col2:
        st.metric("Macro F1-Score", "60.21%", delta="Balanced")
    with m_col3:
        st.metric("Weighted F1-Score", "77.34%", delta="±0.67%")
    with m_col4:
        st.metric("Output Classes", "60 Diseases", delta="Multiclass")

    st.markdown("---")
    st.markdown("#### 🧬 Tracked Pathogens Directory (60 Classes)")
    
    classes_list = predictor.label_encoder.classes_
    cols_dir = st.columns(3)
    chunk_size = (len(classes_list) + 2) // 3
    
    for i, col in enumerate(cols_dir):
        with col:
            sub_classes = classes_list[i * chunk_size : (i + 1) * chunk_size]
            for d in sub_classes:
                st.markdown(f"- 🦠 **{d}**")

    st.markdown("---")
    st.markdown("""
    #### ⚙️ Feature Pipeline Technical Summary:
    1. **Text Normalization**: Regex stripping of parenthetical taxonomies and formatting artifacts.
    2. **Species Cardinality Compression**: Rare species below frequency threshold grouped into `Other`.
    3. **Spatial Frequency Encoding**: Target country probability mapping reflecting historical epizootic incidence.
    4. **Harmonic Month Trigonometrics**: Continuous circular representation using $\\sin(2\\pi m / 12)$ and $\\cos(2\\pi m / 12)$.
    5. **Stabilized Exposure Scaling**: Skew-resistant transformation via $\\log(1 + x)$.
    6. **One-Hot & Standard Scaler ColumnTransformer**: Encodes high-dimensional sparse representations for CatBoost GPU/CPU evaluation.
    """)
