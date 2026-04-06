import numpy as np
import pandas as pd
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from feature_engineering import prepare_health_features


class HealthRiskModel:
    """
    Dual calibrated ML model for PCOS and Endometriosis risk detection.
    Uses:
    - Logistic Regression + Calibration
    - Feature interactions
    - Domain-aware fallback data
    """

    def __init__(self):
        # ───── PCOS MODEL ─────
        self.pcos_model = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000),
            method="sigmoid"
        )
        self.pcos_scaler = StandardScaler()

        self.pcos_features = [
            "age", "bmi", "cycle_length", "cycle_irregular",
            "weight_gain", "hair_growth", "skin_darkening",
            "hair_loss", "pimples", "fast_food",
            "exercise", "sleep_hours",
            "cycle_variability", "hormonal_score", "lifestyle_risk"
        ]

        # ───── ENDO MODEL ─────
        self.endo_model = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000),
            method="sigmoid"
        )
        self.endo_scaler = StandardScaler()

        self.endo_features = [
            "age", "bmi", "cycle_length",
            "pelvic_pain", "heavy_bleeding",
            "pain_intercourse", "family_history_endo",
            "cycle_irregular", "exercise",
            "pain_score"
        ]

        self._initialize_models()

    # ───────────────────────────────────────────────
    # INITIALIZATION
    # ───────────────────────────────────────────────
    def _initialize_models(self):

        if os.path.exists("pcos_dataset.csv"):
            self._train_from_csv(
                "pcos_dataset.csv",
                self.pcos_model,
                self.pcos_scaler,
                self.pcos_features,
                "PCOS (Y/N)"
            )
        else:
            self._train_pcos_fallback()

        if os.path.exists("endo_dataset.csv"):
            self._train_from_csv(
                "endo_dataset.csv",
                self.endo_model,
                self.endo_scaler,
                self.endo_features,
                "Endometriosis (Y/N)"
            )
        else:
            self._train_endo_fallback()

    # ───────────────────────────────────────────────
    # TRAIN FROM CSV
    # ───────────────────────────────────────────────
    def _train_from_csv(self, path, model, scaler, feature_cols, label_col):

        df = pd.read_csv(path)
        if label_col not in df.columns:
            return

        available = [c for c in feature_cols if c in df.columns]
        X = df[available].fillna(df[available].mean())

        # Add missing columns
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0

        X = X[feature_cols]

        # ─── FEATURE INTERACTIONS ───
        if "cycle_irregular" in X.columns and "hormonal_score" in X.columns:
            X["pcos_interaction"] = X["cycle_irregular"] * X["hormonal_score"]

        if "pain_score" in X.columns and "heavy_bleeding" in X.columns:
            X["endo_interaction"] = X["pain_score"] * X["heavy_bleeding"]

        y = df[label_col]

        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)

    # ───────────────────────────────────────────────
    # FALLBACK MODELS
    # ───────────────────────────────────────────────
    def _train_pcos_fallback(self):
    np.random.seed(42)

    X = pd.DataFrame({
        "cycle_irregular": np.random.binomial(1, 0.4, 500),
        "weight_gain": np.random.binomial(1, 0.5, 500),
        "hair_growth": np.random.binomial(1, 0.4, 500),
        "pimples": np.random.binomial(1, 0.5, 500),
        "skin_darkening": np.random.binomial(1, 0.3, 500),
        "hair_loss": np.random.binomial(1, 0.3, 500),
        "fast_food": np.random.binomial(1, 0.5, 500),
        "exercise": np.random.randint(0, 7, 500),
        "sleep_hours": np.random.randint(4, 9, 500),
        "cycle_length": np.random.randint(24, 40, 500),
        "age": np.random.randint(16, 40, 500),
        "bmi": np.random.normal(25, 4, 500),
    })

    X["cycle_variability"] = abs(X["cycle_length"] - 28)
    X["hormonal_score"] = X["hair_growth"] + X["pimples"] + X["hair_loss"] + X["skin_darkening"]
    X["lifestyle_risk"] = X["fast_food"] + (X["exercise"] < 2) + (X["sleep_hours"] < 6)

    score = (
        0.4 * X["cycle_irregular"] +
        0.3 * X["weight_gain"] +
        0.4 * X["hormonal_score"] +
        0.2 * (X["bmi"] > 25)
    )

    y = (score > 1.2).astype(int)

    X = X[self.pcos_features]

    X_scaled = self.pcos_scaler.fit_transform(X)
    self.pcos_model.fit(X_scaled, y)

   def _train_endo_fallback(self):
    np.random.seed(43)

    X = pd.DataFrame({
        "pelvic_pain": np.random.binomial(1, 0.5, 500),
        "heavy_bleeding": np.random.binomial(1, 0.4, 500),
        "pain_intercourse": np.random.binomial(1, 0.3, 500),
        "cycle_irregular": np.random.binomial(1, 0.3, 500),
        "exercise": np.random.randint(0, 7, 500),
        "age": np.random.randint(16, 40, 500),
        "bmi": np.random.normal(24, 4, 500),
    })

    X["pain_score"] = (
        X["pelvic_pain"] +
        X["pain_intercourse"] +
        X["heavy_bleeding"]
    )

    score = (
        0.6 * X["pain_score"] +
        0.2 * X["cycle_irregular"]
    )

    y = (score > 1.0).astype(int)

    X = X[self.endo_features]

    X_scaled = self.endo_scaler.fit_transform(X)
    self.endo_model.fit(X_scaled, y)

    # ───────────────────────────────────────────────
    # RISK PREDICTION
    # ───────────────────────────────────────────────
    def predict_risk(self, user_input: dict) -> dict:

        X_full = prepare_health_features(user_input)

        # ─── PCOS ───
        X_pcos = X_full[self.pcos_features]
        prob_pcos = self.pcos_model.predict_proba(
            self.pcos_scaler.transform(X_pcos)
        )[0][1]

        # ─── ENDO ───
        X_endo = X_full[self.endo_features]
        prob_endo = self.endo_model.predict_proba(
            self.endo_scaler.transform(X_endo)
        )[0][1]

        print(X_full.columns)
        print("PCOS prob:", prob_pcos)
        print("Endo prob:", prob_endo)

        # ─── ADJUSTMENT ───
        def adjust_probability(prob, features, condition):
            if condition == "PCOS":
                prob += 0.02 * features["hormonal_score"]
                prob += 0.01 * features["lifestyle_risk"]

            if condition == "Endometriosis":
                prob += 0.03 * features["pain_score"]

            return max(0.05, min(prob, 0.95))

        features = X_full.iloc[0]

        prob_pcos = adjust_probability(prob_pcos, features, "PCOS")
        prob_endo = adjust_probability(prob_endo, features, "Endometriosis")

        # ─── RISK LEVEL ───
        def format_risk(prob):
            if prob < 0.35:
                return "Low", "Routine care advised."
            elif prob < 0.65:
                return "Moderate", "Monitor symptoms and consider medical advice."
            else:
                return "High", "Consult a medical professional."

        r_pcos, a_pcos = format_risk(prob_pcos)
        r_endo, a_endo = format_risk(prob_endo)

        # ─── INTERPRETATION ───
        if prob_pcos > 0.7 and prob_endo > 0.7:
            note = "Indicators of both PCOS and Endometriosis detected."
        elif prob_pcos > prob_endo:
            note = "Pattern more consistent with PCOS."
        else:
            note = "Pattern more consistent with Endometriosis."

        return {
            "PCOS": {
                "probability": round(prob_pcos * 100, 1),
                "risk_level": r_pcos,
                "advice": a_pcos
            },
            "Endometriosis": {
                "probability": round(prob_endo * 100, 1),
                "risk_level": r_endo,
                "advice": a_endo
            },
            "note": note
        }
