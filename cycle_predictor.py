import pandas as pd
from datetime import datetime, timedelta


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
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        # Parse the date column, ignoring time component
        self.df[self.DATE_COL] = pd.to_datetime(
            self.df[self.DATE_COL], errors="coerce"
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def predict_next(self, dates: list) -> dict:
        """
        Predict next cycle from a list of past start dates (datetime objects).

        Parameters
        ----------
        dates : list[datetime]
            Chronologically ordered list of period start dates.

        Returns
        -------
        dict with keys: predicted_length, next_predicted_date, avg_cycle
        """
        if len(dates) < 2:
            return {"predicted_length": 28}

        # Sort to be safe
        dates = sorted(dates)
        lengths = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
        avg = int(round(sum(lengths) / len(lengths)))

        return {
            "predicted_length":     avg,
            "next_predicted_date":  dates[-1] + timedelta(days=avg),
            "avg_cycle":            avg,
        }

    def predict_for_user(self, user_id: int) -> dict:
        """
        Convenience method: look up a user's history from the dataset
        and predict their next cycle.

        Parameters
        ----------
        user_id : int

        Returns
        -------
        dict (same as predict_next) or error dict if user not found.
        """
        user_df = self.df[self.df[self.USER_COL] == user_id].dropna(
            subset=[self.DATE_COL]
        ).sort_values(self.DATE_COL)

        if user_df.empty:
            return {"error": f"No records found for user_id={user_id}"}

        dates = user_df[self.DATE_COL].tolist()

        # If we have recorded cycle lengths, use their average directly
        if self.LENGTH_COL in user_df.columns:
            lengths = pd.to_numeric(
                user_df[self.LENGTH_COL], errors="coerce"
            ).dropna()
            if not lengths.empty:
                avg = int(round(lengths.mean()))
                return {
                    "predicted_length":    avg,
                    "next_predicted_date": dates[-1] + timedelta(days=avg),
                    "avg_cycle":           avg,
                }

        return self.predict_next(dates)

    def get_cycle_stats(self, user_id: int) -> dict:
        """
        Return basic statistics for a user's cycles from the dataset.
        """
        user_df = self.df[self.df[self.USER_COL] == user_id]
        if user_df.empty:
            return {"error": f"No records found for user_id={user_id}"}

        lengths = pd.to_numeric(
            user_df[self.LENGTH_COL], errors="coerce"
        ).dropna()

        symptoms = user_df["Symptoms"].dropna().value_counts().to_dict()

        return {
            "user_id":        user_id,
            "num_cycles":     len(user_df),
            "avg_cycle":      round(lengths.mean(), 1) if not lengths.empty else None,
            "min_cycle":      int(lengths.min())        if not lengths.empty else None,
            "max_cycle":      int(lengths.max())        if not lengths.empty else None,
            "top_symptoms":   symptoms,
        }
