import pandas as pd
from datetime import timedelta

class CyclePredictor:

    def __init__(self):
        self.df = pd.read_csv("data/cycle.csv")

    def predict_next(self, dates):

        if len(dates) < 2:
            return {"predicted_length": 28}

        lengths = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        avg = int(sum(lengths) / len(lengths))

        return {
            "predicted_length": avg,
            "next_predicted_date": max(dates) + timedelta(days=avg),
            "avg_cycle": avg
        }
