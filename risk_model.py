import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from feature_engineering import prepare_health_features


class HealthRiskModel:

    def __init__(self):

        self.pcos_model = CalibratedClassifierCV(LogisticRegression(max_iter=1000))
        self.endo_model = CalibratedClassifierCV(LogisticRegression(max_iter=1000))

        self.pcos_scaler = StandardScaler()
        self.endo_scaler = StandardScaler()

        self.pcos_features = [
            "age", "bmi", "cycle_length", "cycle_irregular",
            "weight_gain", "hair_growth", "skin_darkening",
            "hair_loss", "pimples", "fast_food",
            "exercise", "sleep_hours",
            "cycle_variability", "hormonal_score", "lifestyle_risk"
        ]

        self.endo_features = [
            "age", "bmi", "cycle_length",
            "pelvic_pain", "heavy_bleeding",
            "pain_intercourse", "family_history_endo",
            "cycle_irregular", "exercise", "pain_score"
        ]

        self._train_models()

    def _train_models(self):

        np.random.seed(42)

        # PCOS
        X_pcos = pd.DataFrame({
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

        X_pcos["cycle_variability"] = abs(X_pcos["cycle_length"] - 28)
        X_pcos["hormonal_score"] = X_pcos["hair_growth"] + X_pcos["pimples"]
        X_pcos["lifestyle_risk"] = X_pcos["fast_food"]

        y_pcos = (X_pcos["hormonal_score"] + X_pcos["cycle_irregular"] > 2).astype(int)

        X_pcos = X_pcos[self.pcos_features]

        self.pcos_model.fit(self.pcos_scaler.fit_transform(X_pcos), y_pcos)

        # Endo
        X_endo = pd.DataFrame({
            "pelvic_pain": np.random.binomial(1, 0.5, 500),
            "heavy_bleeding": np.random.binomial(1, 0.4, 500),
            "pain_intercourse": np.random.binomial(1, 0.3, 500),
            "cycle_irregular": np.random.binomial(1, 0.3, 500),
            "exercise": np.random.randint(0, 7, 500),
            "age": np.random.randint(16, 40, 500),
            "bmi": np.random.normal(24, 4, 500),
            "cycle_length": np.random.randint(24, 40, 500),
            "family_history_endo": np.random.binomial(1, 0.3, 500),
        })

        X_endo["pain_score"] = (
            X_endo["pelvic_pain"] +
            X_endo["pain_intercourse"] +
            X_endo["heavy_bleeding"]
        )

        y_endo = (X_endo["pain_score"] > 1).astype(int)

        X_endo = X_endo[self.endo_features]

        self.endo_model.fit(self.endo_scaler.fit_transform(X_endo), y_endo)

    def predict_risk(self, user_input: dict):

        X = prepare_health_features(user_input)

        pcos = self.pcos_model.predict_proba(
            self.pcos_scaler.transform(X[self.pcos_features])
        )[0][1]

        endo = self.endo_model.predict_proba(
            self.endo_scaler.transform(X[self.endo_features])
        )[0][1]

        def format(p):
            pct = round(p * 100, 1)
            if pct < 35:
                return {"probability": pct, "risk_level": "Low", "advice": "Low risk"}
            elif pct < 65:
                return {"probability": pct, "risk_level": "Moderate", "advice": "Monitor symptoms"}
            else:
                return {"probability": pct, "risk_level": "High", "advice": "Consult doctor"}

        return {
            "PCOS": format(pcos),
            "Endometriosis": format(endo)
        }
