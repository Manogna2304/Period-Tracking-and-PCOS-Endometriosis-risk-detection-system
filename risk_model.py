import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from utils.feature_engineering import prepare_pcos_features


class PCOSRiskModel:
    """
    Logistic Regression model for PCOS risk assessment.
    In production, train on the Kaggle PCOS dataset (prasoonkottarathil/polycystic-ovary-syndrome-pcos).
    Here we use a synthetic dataset with realistic feature distributions for demonstration.
    Replace load_and_train() with real CSV loading when you have the dataset.
    """

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cols = [
            "age", "bmi", "cycle_length", "cycle_irregular", "weight_gain",
            "hair_growth", "skin_darkening", "hair_loss", "pimples",
            "fast_food", "exercise", "sleep_hours", "stress_level",
        ]
        self._train_on_synthetic()

    def _train_on_synthetic(self):
        """
        Synthetic training data mimicking PCOS dataset distributions.
        Replace this with real Kaggle CSV data in your final project.
        """
        np.random.seed(42)
        n = 500

        # PCOS positive cases (label=1)
        pcos = pd.DataFrame({
            "age": np.random.normal(26, 5, n // 2).clip(18, 45),
            "bmi": np.random.normal(27, 4, n // 2).clip(16, 45),
            "cycle_length": np.random.normal(35, 8, n // 2).clip(21, 60),
            "cycle_irregular": np.random.binomial(1, 0.85, n // 2),
            "weight_gain": np.random.binomial(1, 0.75, n // 2),
            "hair_growth": np.random.binomial(1, 0.7, n // 2),
            "skin_darkening": np.random.binomial(1, 0.65, n // 2),
            "hair_loss": np.random.binomial(1, 0.6, n // 2),
            "pimples": np.random.binomial(1, 0.72, n // 2),
            "fast_food": np.random.binomial(1, 0.68, n // 2),
            "exercise": np.random.normal(1.5, 1, n // 2).clip(0, 7),
            "sleep_hours": np.random.normal(6, 1.5, n // 2).clip(3, 10),
            "stress_level": np.random.normal(4, 1, n // 2).clip(1, 5),
            "label": 1,
        })

        # PCOS negative cases (label=0)
        no_pcos = pd.DataFrame({
            "age": np.random.normal(28, 6, n // 2).clip(18, 45),
            "bmi": np.random.normal(22, 3, n // 2).clip(16, 35),
            "cycle_length": np.random.normal(28, 3, n // 2).clip(21, 35),
            "cycle_irregular": np.random.binomial(1, 0.15, n // 2),
            "weight_gain": np.random.binomial(1, 0.2, n // 2),
            "hair_growth": np.random.binomial(1, 0.15, n // 2),
            "skin_darkening": np.random.binomial(1, 0.1, n // 2),
            "hair_loss": np.random.binomial(1, 0.12, n // 2),
            "pimples": np.random.binomial(1, 0.25, n // 2),
            "fast_food": np.random.binomial(1, 0.35, n // 2),
            "exercise": np.random.normal(4, 1.5, n // 2).clip(0, 7),
            "sleep_hours": np.random.normal(7.5, 1, n // 2).clip(5, 10),
            "stress_level": np.random.normal(2.5, 1, n // 2).clip(1, 5),
            "label": 0,
        })

        df = pd.concat([pcos, no_pcos]).sample(frac=1, random_state=42).reset_index(drop=True)
        X = df[self.feature_cols]
        y = df["label"]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def train_from_csv(self, csv_path: str):
        """
        Load and train from the real Kaggle PCOS CSV.
        Expected columns match self.feature_cols + 'PCOS (Y/N)' as label.
        """
        df = pd.read_csv(csv_path)
        label_col = "PCOS (Y/N)"
        if label_col not in df.columns:
            raise ValueError(f"CSV must have column: {label_col}")
        available = [c for c in self.feature_cols if c in df.columns]
        X = df[available].fillna(df[available].mean())
        y = df[label_col]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict_risk(self, user_input: dict) -> dict:
        """Return risk level and probability for a user's input."""
        X = prepare_pcos_features(user_input)[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        prob = self.model.predict_proba(X_scaled)[0][1]

        if prob < 0.35:
            risk_level = "Low"
            color = "green"
            advice = "Your inputs suggest a low risk profile. Maintain a balanced lifestyle."
        elif prob < 0.65:
            risk_level = "Moderate"
            color = "orange"
            advice = "Some risk indicators present. Consider consulting a gynecologist for a check-up."
        else:
            risk_level = "High"
            color = "red"
            advice = "Multiple risk indicators detected. Please consult a doctor for proper diagnosis."

        top_factors = self._get_top_factors(user_input)

        return {
            "probability": round(prob * 100, 1),
            "risk_level": risk_level,
            "color": color,
            "advice": advice,
            "top_factors": top_factors,
        }

    def _get_top_factors(self, user_input: dict) -> list:
        """Return the top contributing risk factors for the user."""
        risk_factors = []
        if user_input.get("cycle_irregular"):
            risk_factors.append("Irregular menstrual cycles")
        if user_input.get("weight_gain"):
            risk_factors.append("Unexplained weight gain")
        if user_input.get("hair_growth"):
            risk_factors.append("Excess hair growth")
        if user_input.get("bmi", 22) > 25:
            risk_factors.append(f"BMI of {user_input['bmi']} (above normal range)")
        if user_input.get("pimples"):
            risk_factors.append("Persistent acne/pimples")
        if user_input.get("stress_level", 3) >= 4:
            risk_factors.append("High stress levels")
        if user_input.get("exercise", 3) <= 1:
            risk_factors.append("Low physical activity")
        return risk_factors[:4]
