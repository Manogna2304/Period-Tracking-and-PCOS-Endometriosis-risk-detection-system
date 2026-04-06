import pandas as pd
import numpy as np


def compute_bmi(weight_kg: float, height_cm: float) -> float:
    if height_cm <= 0:
        return 22.0
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m ** 2), 1)


def prepare_health_features(user_input: dict) -> pd.DataFrame:

    # BMI
    if "weight_kg" in user_input and "height_cm" in user_input:
        bmi = compute_bmi(user_input["weight_kg"], user_input["height_cm"])
    else:
        bmi = user_input.get("bmi", 22.0)

    features = {
        "age": user_input.get("age", 25),
        "bmi": bmi,

        "cycle_length": user_input.get("cycle_length", 28),
        "cycle_irregular": int(user_input.get("cycle_irregular", False)),

        # PCOS
        "weight_gain": int(user_input.get("weight_gain", False)),
        "hair_growth": int(user_input.get("hair_growth", False)),
        "pimples": int(user_input.get("pimples", False)),
        "skin_darkening": int(user_input.get("skin_darkening", False)),
        "hair_loss": int(user_input.get("hair_loss", False)),

        # Endo
        "pelvic_pain": int(user_input.get("pelvic_pain", False)),
        "heavy_bleeding": int(user_input.get("heavy_bleeding", False)),
        "pain_intercourse": int(user_input.get("pain_intercourse", False)),
        "family_history_endo": int(user_input.get("family_history_endo", False)),

        # Lifestyle
        "fast_food": int(user_input.get("fast_food", False)),
        "exercise": user_input.get("exercise", 3),
        "sleep_hours": user_input.get("sleep_hours", 7),
    }

    df = pd.DataFrame([features])

    # 🔥 DERIVED FEATURES
    df["cycle_variability"] = abs(df["cycle_length"] - 28)

    df["hormonal_score"] = (
        df["hair_growth"] +
        df["pimples"] +
        df["hair_loss"] +
        df["skin_darkening"]
    )

    df["pain_score"] = (
        df["pelvic_pain"] +
        df["pain_intercourse"] +
        df["heavy_bleeding"]
    )

    df["lifestyle_risk"] = (
        df["fast_food"] +
        (df["exercise"] < 2).astype(int) +
        (df["sleep_hours"] < 6).astype(int)
    )

    return df
