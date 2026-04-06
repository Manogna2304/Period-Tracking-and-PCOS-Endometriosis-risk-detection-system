import pandas as pd
import numpy as np

def prepare_cycle_features(cycle_lengths: list) -> pd.DataFrame:
    """Create features for linear regression from cycle lengths."""
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

def compute_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate BMI from weight (kg) and height (cm)."""
    if height_cm <= 0:
        return 22.0
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m ** 2), 1)

def prepare_health_features(user_input: dict) -> pd.DataFrame:
    """Convert user input dict into a feature vector for dual models.
    Accepts weight_kg + height_cm and computes BMI internally.
    Supports separate family_history_pcos and family_history_endo fields.
    """
    # Compute BMI from weight/height if provided, else fall back to bmi key
    if "weight_kg" in user_input and "height_cm" in user_input:
        bmi = compute_bmi(user_input["weight_kg"], user_input["height_cm"])
    else:
        bmi = user_input.get("bmi", 22.0)

    # family_history is used as a shared field; prefer specific ones if available
    family_history_pcos = int(user_input.get("family_history_pcos", user_input.get("family_history", False)))
    family_history_endo = int(user_input.get("family_history_endo", user_input.get("family_history", False)))

    features = {
        "age": user_input.get("age", 25),
        "bmi": bmi,
        "cycle_length": user_input.get("cycle_length", 28),
        "cycle_irregular": int(user_input.get("cycle_irregular", False)),
        "weight_gain": int(user_input.get("weight_gain", False)),
        "hair_growth": int(user_input.get("hair_growth", False)),
        "skin_darkening": int(user_input.get("skin_darkening", False)),
        "hair_loss": int(user_input.get("hair_loss", False)),
        "pimples": int(user_input.get("pimples", False)),
        "fast_food": int(user_input.get("fast_food", False)),
        "pelvic_pain": int(user_input.get("pelvic_pain", False)),
        "heavy_bleeding": int(user_input.get("heavy_bleeding", False)),
        "pain_intercourse": int(user_input.get("pain_intercourse", False)),
        "family_history": family_history_pcos,        # used by PCOS model
        "family_history_endo": family_history_endo,   # used by Endo model
        "exercise": user_input.get("exercise", 3),
        "sleep_hours": user_input.get("sleep_hours", 7),
        "cycle_variability": abs(user_input.get("cycle_length", 28) - 28),
        "hormonal_score": ( int(user_input.get("hair_growth", False)) + int(user_input.get("acne", False)) +  int(user_input.get("hair_loss", False)) + int(user_input.get("skin_darkening", False))),
        "pain_score": (int(user_input.get("pelvic_pain", False)) +int(user_input.get("pain_intercourse", False)) +int(user_input.get("heavy_bleeding", False))),
        "lifestyle_risk": (int(user_input.get("fast_food", False)) +    (1 if user_input.get("exercise", 3) < 2 else 0) +    (1 if user_input.get("sleep_hours", 7) < 6 else 0))
    }
    return pd.DataFrame([features])

def prepare_symptom_features(symptom_logs: list) -> pd.DataFrame:
    """Convert list of daily symptom dicts into a matrix for K-Means clustering."""
    if not symptom_logs:
        return None
    df = pd.DataFrame(symptom_logs)
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
