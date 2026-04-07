import pandas as pd


def compute_bmi(weight, height):
    return round(weight / ((height / 100) ** 2), 1)


def prepare_health_features(user_input):
    """
    Accepts either:
      - a dict  (single user health input)
      - a list of dicts (symptom logs for clustering)
    Returns a DataFrame with all engineered features.
    """

    # Normalise input to a list of dicts
    if isinstance(user_input, dict):
        records = [user_input]
    elif isinstance(user_input, list):
        records = user_input
    else:
        raise ValueError("user_input must be a dict or list of dicts")

    rows = []
    for rec in records:
        bmi = compute_bmi(
            rec.get("weight_kg", 60),
            rec.get("height_cm", 160),
        )
        rows.append({
            # Demographics
            "age":               rec.get("age", 25),
            "bmi":               bmi,
            # Cycle info
            "cycle_length":      rec.get("cycle_length", 28),
            "cycle_irregular":   int(rec.get("cycle_irregular", False)),
            # PCOS-related symptoms
            "weight_gain":       int(rec.get("weight_gain", False)),
            "hair_growth":       int(rec.get("hair_growth", False)),
            "pimples":           int(rec.get("pimples", False)),
            "skin_darkening":    int(rec.get("skin_darkening", False)),
            "hair_loss":         int(rec.get("hair_loss", False)),
            # Endometriosis-related symptoms
            "pelvic_pain":       int(rec.get("pelvic_pain", False)),
            "heavy_bleeding":    int(rec.get("heavy_bleeding", False)),
            "pain_intercourse":  int(rec.get("pain_intercourse", False)),
            "family_history_endo": int(rec.get("family_history_endo", False)),
            # Lifestyle
            "fast_food":         int(rec.get("fast_food", False)),
            "exercise":          rec.get("exercise", 3),
            "sleep_hours":       rec.get("sleep_hours", 7),
            # Daily symptom log fields (used by clustering)
            "cramps":            int(rec.get("cramps", False)),
            "bloating":          int(rec.get("bloating", False)),
            "headache":          int(rec.get("headache", False)),
            "fatigue":           int(rec.get("fatigue", False)),
            "mood_swings":       int(rec.get("mood_swings", False)),
            "acne":              int(rec.get("acne", False)),
            "back_pain":         int(rec.get("back_pain", False)),
            "nausea":            int(rec.get("nausea", False)),
            "flow_intensity":    rec.get("flow_intensity", 0),
            "pain_level":        rec.get("pain_level", 0),
            "energy_level":      rec.get("energy_level", 3),
            "mood_score":        rec.get("mood_score", 3),
        })

    df = pd.DataFrame(rows)

    # Derived / engineered features
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
