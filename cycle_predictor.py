import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from feature_engineering import prepare_cycle_features

class CyclePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cols = ["prev_1", "prev_2", "prev_3", "avg_3", "avg_6", "std_3", "trend"]

    def _get_cycle_lengths(self, dates: list) -> list:
        """Converts a list of period start dates into cycle lengths."""
        if len(dates) < 2:
            return []
        sorted_dates = sorted(dates)
        return [(sorted_dates[i] - sorted_dates[i-1]).days for i in range(1, len(sorted_dates))]

    def train(self, dates: list):
        """Train on user's logged period dates."""
        cycle_lengths = self._get_cycle_lengths(dates)
        df = prepare_cycle_features(cycle_lengths)
        if df is None or len(df) < 2:
            return False
        X = df[self.feature_cols].fillna(df[self.feature_cols].mean())
        y = df["cycle_length"]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return True

    def predict_next(self, dates: list) -> dict:
        """Predict the next period date given past start dates."""
        cycle_lengths = self._get_cycle_lengths(dates)
        last_date = max(dates) if dates else None
        
        if len(cycle_lengths) < 2:
            avg = np.mean(cycle_lengths) if cycle_lengths else 28
            pred_length = round(avg)
            return {
                "predicted_length": pred_length,
                "next_predicted_date": last_date + timedelta(days=pred_length) if last_date else None,
                "confidence": "low"
            }

        if not self.is_trained:
            self.train(dates)

        recent = cycle_lengths[-6:] if len(cycle_lengths) >= 6 else cycle_lengths
        prev_vals = recent[-3:] if len(recent) >= 3 else recent
        while len(prev_vals) < 3:
            prev_vals = [np.mean(recent)] + prev_vals

        features = {
            "prev_1": prev_vals[-1], "prev_2": prev_vals[-2], "prev_3": prev_vals[-3],
            "avg_3": np.mean(prev_vals[-3:]), "avg_6": np.mean(recent),
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

        pred = int(max(21, min(45, round(pred))))
        return {
            "predicted_length": pred,
            "next_predicted_date": last_date + timedelta(days=pred) if last_date else None,
            "confidence": confidence,
            "avg_cycle": round(np.mean(cycle_lengths), 1),
            "std_dev": round(np.std(cycle_lengths), 1),
        }
