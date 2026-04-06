import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from feature_engineering import prepare_cycle_features


class CyclePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cols = ["prev_1", "prev_2", "prev_3", "avg_3", "avg_6", "std_3", "trend"]

    def train(self, cycle_lengths: list):
        """Train on user's own logged cycle lengths."""
        df = prepare_cycle_features(cycle_lengths)
        if df is None or len(df) < 2:
            return False
        X = df[self.feature_cols].fillna(df[self.feature_cols].mean())
        y = df["cycle_length"]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return True

    def predict_next(self, cycle_lengths: list) -> dict:
        """Predict the next cycle length given past lengths."""
        if len(cycle_lengths) < 2:
            avg = np.mean(cycle_lengths) if cycle_lengths else 28
            return {"predicted_length": round(avg), "confidence": "low", "method": "average"}

        if not self.is_trained:
            self.train(cycle_lengths)

        recent = cycle_lengths[-6:] if len(cycle_lengths) >= 6 else cycle_lengths
        prev_vals = recent[-3:] if len(recent) >= 3 else recent
        while len(prev_vals) < 3:
            prev_vals = [np.mean(recent)] + prev_vals

        features = {
            "prev_1": prev_vals[-1],
            "prev_2": prev_vals[-2],
            "prev_3": prev_vals[-3],
            "avg_3": np.mean(prev_vals[-3:]),
            "avg_6": np.mean(recent),
            "std_3": np.std(prev_vals[-3:]) if len(prev_vals) >= 3 else 0,
            "trend": prev_vals[-1] - prev_vals[-2] if len(prev_vals) >= 2 else 0,
        }

        X = pd.DataFrame([features])[self.feature_cols]
        if self.is_trained:
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]
            confidence = "high" if len(cycle_lengths) >= 6 else "medium"
        else:
            pred = np.mean(cycle_lengths)
            confidence = "low"

        pred = max(21, min(45, round(pred)))
        return {
            "predicted_length": int(pred),
            "confidence": confidence,
            "method": "linear_regression" if self.is_trained else "average",
            "avg_cycle": round(np.mean(cycle_lengths), 1),
            "std_dev": round(np.std(cycle_lengths), 1),
        }
