import pandas as pd
from datetime import datetime, timedelta, date


class CyclePredictor:
    """
    Predicts the next cycle start date based on a user's past cycle history.

    cycle.csv columns:
        User ID, Age, BMI, Stress Level, Exercise Frequency, Sleep Hours,
        Diet, Cycle Start Date, Cycle Length, Period Length,
        Next Cycle Start Date, Symptoms
    """

    DATE_COL   = "Cycle Start Date"
    LENGTH_COL = "Cycle Length"
    USER_COL   = "User ID"

    def __init__(self, csv_path: str = "data/cycle.csv"):
        try:
            self.df = pd.read_csv(csv_path)
            self.df.columns = self.df.columns.str.strip()
            self.df[self.DATE_COL] = pd.to_datetime(
                self.df[self.DATE_COL], errors="coerce"
            )
        except Exception:
            self.df = pd.DataFrame()

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _to_datetime(d):
        """Convert date / str / Timestamp to datetime."""
        if isinstance(d, datetime):
            return d
        if isinstance(d, date):
            return datetime(d.year, d.month, d.day)
        return pd.Timestamp(d).to_pydatetime()

    def _get_cycle_lengths(self, dates: list) -> list:
        """
        Return a list of inter-period gap lengths (in days) from a list of
        period start dates. Required by app.py line 330.
        """
        if len(dates) < 2:
            return []
        dts = sorted([self._to_datetime(d) for d in dates])
        return [(dts[i] - dts[i - 1]).days for i in range(1, len(dts))]

    # ── Public API ──────────────────────────────────────────────────────────

    def predict_next(self, dates: list) -> dict:
        """
        Predict next cycle from a list of past start dates.

        Returns
        -------
        dict with keys:
            predicted_length, next_predicted_date, avg_cycle, std_dev
        """
        if len(dates) < 2:
            return {
                "predicted_length":    28,
                "avg_cycle":           28,
                "std_dev":             0,
            }

        lengths = self._get_cycle_lengths(dates)
        avg     = int(round(sum(lengths) / len(lengths)))
        std_dev = round(
            (sum((l - avg) ** 2 for l in lengths) / len(lengths)) ** 0.5, 1
        ) if len(lengths) > 1 else 0

        last = self._to_datetime(max(dates))

        return {
            "predicted_length":    avg,
            "next_predicted_date": last + timedelta(days=avg),
            "avg_cycle":           avg,
            "std_dev":             std_dev,
        }

    def predict_for_user(self, user_id: int) -> dict:
        """Look up a user's history from the dataset and predict next cycle."""
        if self.df.empty:
            return {"error": "Dataset not loaded"}

        user_df = (
            self.df[self.df[self.USER_COL] == user_id]
            .dropna(subset=[self.DATE_COL])
            .sort_values(self.DATE_COL)
        )

        if user_df.empty:
            return {"error": f"No records found for user_id={user_id}"}

        dates = user_df[self.DATE_COL].tolist()

        if self.LENGTH_COL in user_df.columns:
            lengths = pd.to_numeric(
                user_df[self.LENGTH_COL], errors="coerce"
            ).dropna()
            if not lengths.empty:
                avg = int(round(lengths.mean()))
                std_dev = round(float(lengths.std()), 1) if len(lengths) > 1 else 0
                last = self._to_datetime(dates[-1])
                return {
                    "predicted_length":    avg,
                    "next_predicted_date": last + timedelta(days=avg),
                    "avg_cycle":           avg,
                    "std_dev":             std_dev,
                }

        return self.predict_next(dates)

    def get_cycle_stats(self, user_id: int) -> dict:
        """Return basic statistics for a user's cycles from the dataset."""
        if self.df.empty:
            return {"error": "Dataset not loaded"}

        user_df = self.df[self.df[self.USER_COL] == user_id]
        if user_df.empty:
            return {"error": f"No records found for user_id={user_id}"}

        lengths  = pd.to_numeric(user_df[self.LENGTH_COL], errors="coerce").dropna()
        symptoms = user_df["Symptoms"].dropna().value_counts().to_dict()

        return {
            "user_id":      user_id,
            "num_cycles":   len(user_df),
            "avg_cycle":    round(float(lengths.mean()), 1) if not lengths.empty else None,
            "min_cycle":    int(lengths.min())              if not lengths.empty else None,
            "max_cycle":    int(lengths.max())              if not lengths.empty else None,
            "top_symptoms": symptoms,
        }
