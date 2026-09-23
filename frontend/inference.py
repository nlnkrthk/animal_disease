"""
Inference engine for Animal Disease Outbreak Prediction.
Handles loading model artifacts, preprocessors, label encoders, feature transformations,
and predictions (both single instance and batch).
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
SAVED_MODELS_DIR = PROJECT_ROOT / "ml" / "saved_models"
MODEL_PATH = SAVED_MODELS_DIR / "catboost_final.cbm"
ASSETS_PATH = SAVED_MODELS_DIR / "preprocessing_assets.pkl"
LABEL_ENCODER_PATH = SAVED_MODELS_DIR / "label_encoder.pkl"

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}

SEASON_LOOKUP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon"
}

CAUSAL_AGENTS = ["Virus", "Bacterium", "Parasite", "Prion", "Fungus"]

EPI_UNIT_TYPES = [
    "Farm", "Backyard", "Forest", "Village", "Apiary", "Slaughterhouse",
    "River system", "Body of water", "Natural park", "Coastal area", "Cage",
    "Zoo", "Pond", "Lake", "Reservoir", "Livestock market", "Shell fish bed",
    "Estuary", "Other", "Not applicable"
]

SAMPLE_SCENARIOS = {
    "Swine in Bolivia (Aujeszky's Disease)": {
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
        "epi_unit_type": "Farm",
        "description": "Commercial swine herd exhibiting respiratory distress and neurologic symptoms in central Bolivia."
    },
    "House Crow in India (Avian Influenza)": {
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
        "epi_unit_type": "Not applicable",
        "description": "Wild avian mortality spike observed in an urban wetlands perimeter during winter migration."
    },
    "Cattle in France (Bluetongue)": {
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
        "epi_unit_type": "Farm",
        "description": "Dairy herd displaying facial edema, coronitis, and hyperthermia in late summer / early autumn."
    },
    "Equidae in South Africa (African Horse Sickness)": {
        "species_name": "Equidae",
        "is_wild": 0,
        "is_domestic": 1,
        "is_aquatic": 0,
        "causal_agent_type": "Virus",
        "country_name": "South Africa",
        "latitude": -30.5595,
        "longitude": 22.9375,
        "month": "April",
        "season": "Summer",
        "susceptible": 85,
        "epi_unit_type": "Farm",
        "description": "Sudden onset of pulmonary distress, coughing, and fever in stabled equines following midge proliferation."
    },
    "Bees in Australia (Small Hive Beetle Infestation)": {
        "species_name": "Bees",
        "is_wild": 0,
        "is_domestic": 1,
        "is_aquatic": 0,
        "causal_agent_type": "Parasite",
        "country_name": "Australia",
        "latitude": -25.2744,
        "longitude": 133.7751,
        "month": "December",
        "season": "Winter",
        "susceptible": 5000,
        "epi_unit_type": "Apiary",
        "description": "Honey comb fermentation and structural hive collapse identified in commercial apiary hives."
    }
}


def load_country_coordinates() -> Dict[str, Dict[str, float]]:
    coords_file = FRONTEND_DIR / "data" / "country_coords.json"
    if coords_file.exists():
        with open(coords_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_species_list() -> List[str]:
    species_file = FRONTEND_DIR / "data" / "species_list.json"
    if species_file.exists():
        with open(species_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["Swine", "Cattle", "Birds", "Wild boar", "Equidae", "Sheep", "Dogs", "Goats", "Bees", "Cats"]


class DiseasePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.country_freq = None
        self.common_species = None
        self.label_encoder = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads model, preprocessor assets, and label encoder."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        logger.info(f"Loading CatBoost model from {MODEL_PATH}...")
        self.model = CatBoostClassifier()
        self.model.load_model(str(MODEL_PATH))

        if not ASSETS_PATH.exists() or not LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError(
                f"Assets missing at {ASSETS_PATH} or {LABEL_ENCODER_PATH}. "
                "Ensure saved_models contains preprocessing_assets.pkl and label_encoder.pkl"
            )

        logger.info("Loading preprocessing assets and label encoder...")
        assets = joblib.load(str(ASSETS_PATH))
        self.preprocessor = assets["preprocessor"]
        self.country_freq = assets["country_freq"]
        self.common_species = assets["common_species"]
        self.label_encoder = joblib.load(str(LABEL_ENCODER_PATH))
        logger.info("All model artifacts loaded successfully.")

    def preprocess_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms raw input dataframe (with 12 feature columns) into the 73 encoded features
        expected by the CatBoost model.
        """
        data = df.copy()

        # 1. Clean species name
        data["species_name"] = (
            data["species_name"]
            .astype(str)
            .str.replace(r"\s*\([^)]*\)", "", regex=True)
            .str.replace("\n", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # 2. Filter common species
        data["species_name"] = data["species_name"].where(
            data["species_name"].isin(self.common_species),
            "Other"
        )

        # 3. Country frequency encoding
        data["country_freq"] = data["country_name"].map(self.country_freq).fillna(0)
        data = data.drop(columns=["country_name"])

        # 4. Cyclical month encoding
        data["month_num"] = data["month"].map(MONTH_MAP).fillna(1).astype(int)
        data["month_sin"] = np.sin(2 * np.pi * data["month_num"] / 12)
        data["month_cos"] = np.cos(2 * np.pi * data["month_num"] / 12)
        data = data.drop(columns=["month", "month_num"])

        # 5. Susceptible log transform
        data["susceptible"] = pd.to_numeric(data["susceptible"], errors="coerce").fillna(0)
        data["susceptible"] = np.log1p(np.maximum(data["susceptible"], 0))

        # Ensure correct column ordering for ColumnTransformer
        # Categorical: species_name, causal_agent_type, season, epi_unit_type
        # Numeric: is_wild, is_domestic, is_aquatic, latitude, longitude, susceptible, country_freq, month_sin, month_cos
        cols = [
            "species_name", "causal_agent_type", "season", "epi_unit_type",
            "is_wild", "is_domestic", "is_aquatic", "latitude", "longitude",
            "susceptible", "country_freq", "month_sin", "month_cos"
        ]
        data = data[cols]

        # Apply preprocessor
        encoded = self.preprocessor.transform(data)
        return encoded

    def predict_single(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs inference on a single outbreak input dictionary.
        Returns top prediction, probability, top-5 candidates, and severity indicator.
        """
        df = pd.DataFrame([input_dict])
        encoded = self.preprocess_df(df)

        # Predicted probabilities
        probs = self.model.predict_proba(encoded)[0]
        top_indices = np.argsort(probs)[::-1]

        top_class_idx = top_indices[0]
        top_disease = self.label_encoder.inverse_transform([int(top_class_idx)])[0]
        top_prob = float(probs[top_class_idx])

        # Top 5 candidates
        top_5 = []
        for idx in top_indices[:5]:
            disease_name = self.label_encoder.inverse_transform([int(idx)])[0]
            prob_val = float(probs[idx])
            top_5.append({
                "disease": disease_name,
                "probability": prob_val,
                "percentage": round(prob_val * 100, 2)
            })

        # Risk Assessment
        if top_prob >= 0.70:
            confidence_level = "High"
            badge_color = "#10b981"  # Emerald
        elif top_prob >= 0.40:
            confidence_level = "Moderate"
            badge_color = "#f59e0b"  # Amber
        else:
            confidence_level = "Low (Differential Investigation Urged)"
            badge_color = "#ef4444"  # Rose/Red

        return {
            "predicted_disease": top_disease,
            "confidence": top_prob,
            "confidence_percentage": round(top_prob * 100, 1),
            "confidence_level": confidence_level,
            "badge_color": badge_color,
            "top_5": top_5,
            "input_summary": input_dict
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs batch prediction on a dataframe.
        Appends 'predicted_disease' and 'confidence_percent'.
        """
        encoded = self.preprocess_df(df)
        predictions = self.model.predict(encoded)
        probs = self.model.predict_proba(encoded)

        predicted_diseases = []
        confidences = []

        for i, pred in enumerate(predictions):
            disease = self.label_encoder.inverse_transform([int(pred[0])])[0]
            prob = float(np.max(probs[i])) * 100
            predicted_diseases.append(disease)
            confidences.append(round(prob, 2))

        result_df = df.copy()
        result_df["predicted_disease"] = predicted_diseases
        result_df["confidence_pct"] = confidences
        return result_df


# Singleton predictor cache
_predictor_instance: Optional[DiseasePredictor] = None


def get_predictor() -> DiseasePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = DiseasePredictor()
    return _predictor_instance
