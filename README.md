# Animal Disease Outbreak Prediction Platform

An AI-powered epizootic intelligence and surveillance application that predicts animal disease outbreaks across 60 global pathogen classes using a trained CatBoost multiclass ensemble model.

---

## 📁 Repository Structure

```
animal_disease/
├── frontend/                     # Streamlit web application
│   ├── app.py                   # Main interactive Streamlit dashboard
│   ├── inference.py             # Preprocessing & ML inference pipeline
│   └── data/                    # App metadata and lookup tables
│       ├── country_coords.json  # 187 country centroid coordinates
│       └── species_list.json    # 318 standardized animal species
├── ml/                          # Machine learning development & artifacts
│   ├── dataset.csv              # Full outbreak dataset
│   ├── train_dataset.csv        # Preprocessed training dataset
│   ├── prediction.ipynb         # Model training & cross-validation notebook
│   ├── test_model.ipynb         # Validation & test inference notebook
│   └── saved_models/            # Trained model & preprocessing artifacts
│       ├── catboost_final.cbm   # CatBoost model binary
│       ├── preprocessing_assets.pkl # Scaler, encoder & frequency mappings
│       └── label_encoder.pkl    # Disease target label encoder
├── requirements.txt             # Project dependencies
├── run_app.py                   # Quick root launcher script
└── README.md
```

---

## 🚀 How to Run the Web Application

### Option 1: Standard Streamlit Command
```bash
streamlit run frontend/app.py
```

### Option 2: Using the Root Launcher
```bash
python run_app.py
```

Once launched, open your web browser at:
👉 **`http://localhost:8501`**

---

## 🛠️ Features

- **Outbreak Diagnostic Predictor**: Enter host animal profile, habitat, pathogen classification, geographic coordinates, seasonality, and susceptible head count.
- **Top-5 Differential Diagnosis**: Instant probability rankings visualized via interactive Plotly charts.
- **1-Click Benchmark Presets**: Rapidly test verified real-world outbreak scenarios (Swine in Bolivia, Avian in India, Cattle in France, Equidae in South Africa, Bees in Australia).
- **Batch CSV Outbreak Triage**: Process multiple outbreak records simultaneously and export predictions report.
- **Biosecurity Containment Advisories**: Automated early warning recommendations and WOAH reporting guidelines.
