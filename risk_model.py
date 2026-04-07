import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from feature_engineering import prepare_health_features


class HealthRiskModel:

    def __init__(self):
        # Models
        self.pcos_model = CalibratedClassifierCV(
            LogisticRegression(max_iter=2000), method="sigmoid"
        )
        self.endo_model = CalibratedClassifierCV(
            LogisticRegression(max_iter=2000), method="sigmoid"
        )

        # Scalers
        self.pcos_scaler = StandardScaler()
        self.endo_scaler = StandardScaler()

        # Feature lists
        self.pcos_features = [
            "age", "bmi", "cycle_length", "cycle_irregular",
            "weight_gain", "hair_growth", "skin_darkening",
            "hair_loss", "pimples", "fast_food",
            "exercise", "sleep_hours",
            "cycle_variability", "hormonal_score", "lifestyle_risk",
        ]

        self.endo_features = [
            "age", "bmi", "cycle_length",
            "pelvic_pain", "heavy_bleeding",
            "pain_intercourse", "family_history_endo",
            "cycle_irregular", "exercise", "pain_score",
        ]

        self._train_models()

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _clean_df(df):
        df.columns = df.columns.str.strip()
        df = df.replace({"Y": 1, "N": 0, "Yes": 1, "No": 0})
        df = df.fillna(0)
        return df

    # ── Training ────────────────────────────────────────────────────────────
    def _train_models(self):
        self._train_pcos()
        self._train_endo()

    def _train_pcos(self):
        """
        Uses only PCOS_data_without_infertility.csv which contains all the
        clinical symptom columns.  pcos.csv only has HCG/AMH and is merged
        in by Patient File No. to add the PCOS label where available,
        but the main feature source is the infertility file.
        """
        df = self._clean_df(
            pd.read_csv("data/PCOS_data_without_infertility.csv")
        )

        # Derived cycle flag
        df["cycle_irregular"] = df["Cycle(R/I)"].apply(
            lambda x: 1 if str(x).strip() == "I" else 0
        )

        # BMI is already present; compute cycle_variability
        df["cycle_variability"] = abs(df["Cycle length(days)"] - 28)
        df["hormonal_score"] = (
            df["hair growth(Y/N)"] +
            df["Pimples(Y/N)"] +
            df["Hair loss(Y/N)"] +
            df["Skin darkening (Y/N)"]
        )
        df["lifestyle_risk"] = df["Fast food (Y/N)"]

        X_pcos = pd.DataFrame({
            "age":              df["Age (yrs)"],
            "bmi":              df["BMI"],
            "cycle_length":     df["Cycle length(days)"],
            "cycle_irregular":  df["cycle_irregular"],
            "weight_gain":      df["Weight gain(Y/N)"],
            "hair_growth":      df["hair growth(Y/N)"],
            "skin_darkening":   df["Skin darkening (Y/N)"],
            "hair_loss":        df["Hair loss(Y/N)"],
            "pimples":          df["Pimples(Y/N)"],
            "fast_food":        df["Fast food (Y/N)"],
            "exercise":         df["Reg.Exercise(Y/N)"],
            "sleep_hours":      7,                        # not in dataset; use default
            "cycle_variability": df["cycle_variability"],
            "hormonal_score":   df["hormonal_score"],
            "lifestyle_risk":   df["lifestyle_risk"],
        })

        X_pcos = X_pcos[self.pcos_features]
        X_pcos = X_pcos.apply(pd.to_numeric, errors="coerce").fillna(0)

        y_pcos = df["PCOS (Y/N)"].apply(pd.to_numeric, errors="coerce").fillna(0)

        X_scaled = self.pcos_scaler.fit_transform(X_pcos)
        self.pcos_model.fit(X_scaled, y_pcos)

    def _train_endo(self):
        """
        endo.csv columns: Age, Menstrual_Irregularity, Chronic_Pain_Level,
                          Hormone_Level_Abnormality, Infertility, BMI, Diagnosis
        We map them to the features we need.
        """
        df = self._clean_df(pd.read_csv("data/endo.csv"))

        # Map available columns to expected feature names
        X_endo = pd.DataFrame({
            "age":                df["Age"],
            "bmi":                df["BMI"],
            # No cycle_length in this dataset — use 28 as neutral default
            "cycle_length":       28,
            # Menstrual irregularity → cycle_irregular proxy
            "cycle_irregular":    df["Menstrual_Irregularity"],
            # Chronic_Pain_Level (continuous 0–10) used as pelvic_pain proxy
            "pelvic_pain":        (df["Chronic_Pain_Level"] > 5).astype(int),
            # Heavy bleeding not directly available; use 0
            "heavy_bleeding":     0,
            # Pain intercourse not available; use 0
            "pain_intercourse":   0,
            # Family history not available; use 0
            "family_history_endo": 0,
            # Exercise not available; use neutral
            "exercise":           3,
            # pain_score derived below
        })

        X_endo["pain_score"] = (
            X_endo["pelvic_pain"] +
            X_endo["pain_intercourse"] +
            X_endo["heavy_bleeding"]
        )

        X_endo = X_endo[self.endo_features]
        X_endo = X_endo.apply(pd.to_numeric, errors="coerce").fillna(0)

        y_endo = df["Diagnosis"].apply(pd.to_numeric, errors="coerce").fillna(0)

        X_scaled = self.endo_scaler.fit_transform(X_endo)
        self.endo_model.fit(X_scaled, y_endo)

    # ── Prediction ──────────────────────────────────────────────────────────
    def predict_risk(self, user_input: dict) -> dict:
        X = prepare_health_features(user_input)

        pcos_prob = self.pcos_model.predict_proba(
            self.pcos_scaler.transform(X[self.pcos_features])
        )[0][1]

        endo_prob = self.endo_model.predict_proba(
            self.endo_scaler.transform(X[self.endo_features])
        )[0][1]

        # Small manual adjustments based on symptom scores
        f = X.iloc[0]
        pcos_prob += 0.01 * f["hormonal_score"]
        endo_prob  += 0.015 * f["pain_score"]

        pcos_prob = float(np.clip(pcos_prob, 0.05, 0.95))
        endo_prob  = float(np.clip(endo_prob,  0.05, 0.95))

        def format_output(p):
            pct = round(p * 100, 1)
            if pct < 35:
                return {"probability": pct, "risk_level": "Low",      "advice": "Maintain healthy lifestyle"}
            elif pct < 65:
                return {"probability": pct, "risk_level": "Moderate", "advice": "Monitor symptoms"}
            else:
                return {"probability": pct, "risk_level": "High",     "advice": "Consult a doctor"}

        return {
            "PCOS":            format_output(pcos_prob),
            "Endometriosis":   format_output(endo_prob),
        }
