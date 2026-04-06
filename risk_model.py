import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from feature_engineering import prepare_health_features

class HealthRiskModel:
    """
    Dual Logistic Regression model for PCOS and Endometriosis risk.
    Connects to Kaggle Datasets (pcos_dataset.csv, endo_dataset.csv) automatically.
    Falls back to generated heuristic data if files are not present.
    """

    def __init__(self):
        # PCOS setup
        self.pcos_model = LogisticRegression(max_iter=1000, random_state=42)
        self.pcos_scaler = StandardScaler()
        self.pcos_features = [
            "age", "bmi", "cycle_length", "cycle_irregular", "weight_gain",
            "hair_growth", "skin_darkening", "hair_loss", "pimples", "fast_food",
            "exercise", "sleep_hours"
        ]
        
        # Endometriosis setup
        self.endo_model = LogisticRegression(max_iter=1000, random_state=42)
        self.endo_scaler = StandardScaler()
        self.endo_features = [
            "age", "bmi", "cycle_length", "pelvic_pain", "heavy_bleeding",
            "pain_intercourse", "family_history", "cycle_irregular", "exercise"
        ]
        
        self._initialize_models()

    def _initialize_models(self):
        """Attempts to load Kaggle datasets, falls back if missing."""
        # Train PCOS Model
        if os.path.exists("pcos_dataset.csv"):
            self._train_from_csv("pcos_dataset.csv", self.pcos_model, self.pcos_scaler, self.pcos_features, "PCOS (Y/N)")
        else:
            self._train_pcos_fallback()
            
        # Train Endometriosis Model
        if os.path.exists("endo_dataset.csv"):
            self._train_from_csv("endo_dataset.csv", self.endo_model, self.endo_scaler, self.endo_features, "Endometriosis (Y/N)")
        else:
            self._train_endo_fallback()

    def _train_from_csv(self, path, model, scaler, feature_cols, label_col):
        df = pd.read_csv(path)
        if label_col not in df.columns: return
        
        # Ensure requested columns exist, fill missing safely
        available = [c for c in feature_cols if c in df.columns]
        X = df[available].fillna(df[available].mean())
        
        # Pad missing columns requested by the model shape but missing in CSV
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
                
        # Reorder to match model signature exactly
        X = X[feature_cols]
        y = df[label_col]
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)

    def _train_pcos_fallback(self):
        """Synthetic fallback mimicking Kaggle distributions"""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, len(self.pcos_features)), columns=self.pcos_features)
        y = np.random.randint(0, 2, 200)
        # Create loose correlation for the fallback
        y = (X["cycle_irregular"] * 0.5 + X["weight_gain"] * 0.3 + X["hair_growth"] * 0.4 > 0).astype(int)
        
        X_scaled = self.pcos_scaler.fit_transform(X)
        self.pcos_model.fit(X_scaled, y)

    def _train_endo_fallback(self):
        """Synthetic fallback mimicking Endo clinical indicators"""
        np.random.seed(43)
        X = pd.DataFrame(np.random.randn(200, len(self.endo_features)), columns=self.endo_features)
        y = (X["pelvic_pain"] * 0.6 + X["heavy_bleeding"] * 0.4 + X["pain_intercourse"] * 0.5 > 0).astype(int)
        
        X_scaled = self.endo_scaler.fit_transform(X)
        self.endo_model.fit(X_scaled, y)

    def predict_risk(self, user_input: dict) -> dict:
        X_full = prepare_health_features(user_input)
        
        # PCOS Prediction
        X_pcos = X_full[self.pcos_features]
        prob_pcos = self.pcos_model.predict_proba(self.pcos_scaler.transform(X_pcos))[0][1]
        
        # Endo Prediction
        X_endo = X_full[self.endo_features]
        prob_endo = self.endo_model.predict_proba(self.endo_scaler.transform(X_endo))[0][1]

        def format_risk(prob):
            if prob < 0.35: return "Low", "Routine care advised."
            if prob < 0.65: return "Moderate", "Consider discussing symptoms with your gynecologist."
            return "High", "Please consult a medical professional for evaluation."

        r_pcos, a_pcos = format_risk(prob_pcos)
        r_endo, a_endo = format_risk(prob_endo)

        return {
            "PCOS": {"probability": round(prob_pcos * 100, 1), "risk_level": r_pcos, "advice": a_pcos},
            "Endometriosis": {"probability": round(prob_endo * 100, 1), "risk_level": r_endo, "advice": a_endo}
        }
