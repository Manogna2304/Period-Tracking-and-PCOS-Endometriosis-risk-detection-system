import pandas as pd
import numpy as np


def prepare_cycle_features(cycle_lengths: list) -> pd.DataFrame:
    """
    From a list of past cycle lengths, create features for linear regression.
    Features: last cycle, avg of last 3, avg of last 6, std dev, trend.
    """
    if len(cycle_lengths) < 2:
        return None

    df = pd.DataFrame({"cycle_length": cycle_lengths})
    df["prev_1"] = df["cycle_length"].shift(1)
    df["prev_2"] = df["cycle_length"].shift(2)
    df["prev_3"] = df["cycle_length"].shift(3)
    df["avg_3"] = df["cycle_length"].shift(1).rolling(3).mean()
    df["avg_6"] = df["cycle_length"].shift(1).rolling(6).mean()
    df["std_3"] = df["cycle_length"].shift(1).rolling(3).std()
    df["trend"] = df["cycle_length"].shift(1).diff()
    df = df.dropna()
    return df


def prepare_pcos_features(user_input: dict) -> pd.DataFrame:
    """
    Convert user input dict into a feature vector for logistic regression.
    """
    features = {
        "age": user_input.get("age", 25),
        "bmi": user_input.get("bmi", 22),
        "cycle_length": user_input.get("cycle_length", 28),
        "cycle_irregular": int(user_input.get("cycle_irregular", False)),
        "weight_gain": int(user_input.get("weight_gain", False)),
        "hair_growth": int(user_input.get("hair_growth", False)),
        "skin_darkening": int(user_input.get("skin_darkening", False)),
        "hair_loss": int(user_input.get("hair_loss", False)),
        "pimples": int(user_input.get("pimples", False)),
        "fast_food": int(user_input.get("fast_food", False)),
        "exercise": user_input.get("exercise", 3),
        "sleep_hours": user_input.get("sleep_hours", 7),
        "stress_level": user_input.get("stress_level", 3),
    }
    return pd.DataFrame([features])


def prepare_symptom_features(symptom_logs: list) -> pd.DataFrame:
    """
    Convert list of daily symptom dicts into a matrix for K-Means clustering.
    Each row = one day's log.
    """
    if not symptom_logs:
        return None
    df = pd.DataFrame(symptom_logs)
    # Encode boolean/categorical fields numerically
    bool_cols = ["cramps", "bloating", "headache", "fatigue",
                 "mood_swings", "acne", "back_pain", "nausea"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    numeric_cols = ["flow_intensity", "pain_level", "energy_level", "mood_score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df
