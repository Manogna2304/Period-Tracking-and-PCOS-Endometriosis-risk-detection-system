import numpy as np
from datetime import timedelta


class CyclePredictor:

    def _get_cycle_lengths(self, dates):
        return [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]

    def predict_next(self, dates):

        if len(dates) < 2:
            return {"predicted_length": 28}

        lengths = self._get_cycle_lengths(dates)
        avg = int(np.mean(lengths))

        return {
            "predicted_length": avg,
            "next_predicted_date": max(dates) + timedelta(days=avg),
            "avg_cycle": avg,
            "std_dev": round(np.std(lengths), 1)
        }
