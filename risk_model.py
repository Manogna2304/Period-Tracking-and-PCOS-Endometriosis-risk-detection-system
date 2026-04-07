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
            LogisticRegression(max_iter=2000),
            method="sigmoid"
        )
        self.endo_model = CalibratedClassifierCV(
            LogisticRegression(max_iter=2000),
            method="sigmoid"
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
            "cycle_variability", "hormonal_score", "lifestyle_risk"
        ]

        self.endo_features = [
            "age", "bmi", "cycle_length",
            "pelvic_pain", "heavy_bleeding",
            "pain_intercourse", "family_history_endo",
            "cycle_irregular", "exercise", "pain_score"
        ]

        self._train_models()

    # ================= CLEANING =================
    def _clean_df(self, df):
        df.columns = df.columns.str.strip()
        df = df.replace({"Y": 1, "N": 0, "Yes": 1, "No": 0})
        df = df.fillna(0)
        return df

    # ================= TRAINING =================
    def _train_models(self):

        # -------- PCOS --------
        df1 = self._clean_df(pd.read_csv("data/pcos.csv"))
        df2 = self._clean_df(pd.read_csv("data/PCOS_data_without_infertility.csv"))
        df = pd.concat([df1, df2], ignore_index=True)

        # Convert cycle type
        df["cycle_irregular"] = df["Cycle(R/I)"].apply(
            lambda x: 1 if str(x).strip() == "I" else 0
        )

        X_pcos = pd.DataFrame({
            "age": df["Age (yrs)"],
            "bmi": df["BMI"],
            "cycle_length": df["Cycle length(days)"],
            "cycle_irregular": df["cycle_irregular"],
            "weight_gain": df["Weight gain(Y/N)"],
            "hair_growth": df["hair growth(Y/N)"],
            "pimples": df["Pimples(Y/N)"],
            "skin_darkening": df["Skin darkening (Y/N)"],
            "hair_loss": df["Hair loss(Y/N)"],
            "fast_food": df["Fast food (Y/N)"],
            "exercise": df["Reg.Exercise(Y/N)"],
            "sleep_hours": 7
        })

        # Derived features
        X_pcos["cycle_variability"] = abs(X_pcos["cycle_length"] - 28)
        X_pcos["hormonal_score"] = (
            X_pcos["hair_growth"] +
            X_pcos["pimples"] +
            X_pcos["hair_loss"] +
            X_pcos["skin_darkening"]
        )
        X_pcos["lifestyle_risk"] = X_pcos["fast_food"]

        # Ensure correct feature set
        X_pcos = X_pcos[self.pcos_features]

        # Fix data types
        X_pcos = X_pcos.apply(pd.to_numeric, errors="coerce")
        X_pcos = X_pcos.fillna(0)

        y_pcos = df["PCOS (Y/N)"]

        X_scaled_pcos = self.pcos_scaler.fit_transform(X_pcos)
        self.pcos_model.fit(X_scaled_pcos, y_pcos)

        # -------- ENDOMETRIOSIS --------
        df_endo = self._clean_df(pd.read_csv("data/endo.csv"))

        X_endo = pd.DataFrame({
            "age": df_endo["age"],
            "bmi": df_endo["bmi"],
            "cycle_length": df_endo["cycle_length"],
            "cycle_irregular": df_endo["irregular_cycles"],
            "pelvic_pain": df_endo["pelvic_pain"],
            "heavy_bleeding": df_endo["heavy_bleeding"],
            "pain_intercourse": df_endo["pain_during_intercourse"],
            "family_history_endo": df_endo["family_history"],
            "exercise": df_endo["exercise_level"]
        })

        X_endo["pain_score"] = (
            X_endo["pelvic_pain"] +
            X_endo["pain_intercourse"] +
            X_endo["heavy_bleeding"]
        )

        X_endo = X_endo[self.endo_features]

        # Fix data types
        X_endo = X_endo.apply(pd.to_numeric, errors="coerce")
        X_endo = X_endo.fillna(0)

        y_endo = df_endo["endometriosis"]

        X_scaled_endo = self.endo_scaler.fit_transform(X_endo)
        self.endo_model.fit(X_scaled_endo, y_endo)

    # ================= PREDICTION =================
    def predict_risk(self, user_input):

        X = prepare_health_features(user_input)

        # PCOS
        pcos_prob = self.pcos_model.predict_proba(
            self.pcos_scaler.transform(X[self.pcos_features])
        )[0][1]

        # Endometriosis
        endo_prob = self.endo_model.predict_proba(
            self.endo_scaler.transform(X[self.endo_features])
        )[0][1]

        # Adjustments
        f = X.iloc[0]
        pcos_prob += 0.01 * f["hormonal_score"]
        endo_prob += 0.015 * f["pain_score"]

        pcos_prob = min(max(pcos_prob, 0.05), 0.95)
        endo_prob = min(max(endo_prob, 0.05), 0.95)

        def format_output(p):
            pct = round(p * 100, 1)
            if pct < 35:
                return {"probability": pct, "risk_level": "Low", "advice": "Maintain healthy lifestyle"}
            elif pct < 65:
                return {"probability": pct, "risk_level": "Moderate", "advice": "Monitor symptoms"}
            else:
                return {"probability": pct, "risk_level": "High", "advice": "Consult a doctor"}

        return {
            "PCOS": format_output(pcos_prob),
            "Endometriosis": format_output(endo_prob)
        }
