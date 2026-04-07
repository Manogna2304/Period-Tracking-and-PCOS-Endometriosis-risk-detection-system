import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from feature_engineering import prepare_health_features


# ✅ RESTORED (IMPORTANT)
CLUSTER_PROFILES = {
    0: {"name": "High Pain", "emoji": "🤕"},
    1: {"name": "Hormonal", "emoji": "😓"},
    2: {"name": "Normal", "emoji": "⚡"},
    3: {"name": "Bloating", "emoji": "😮"},
}


FEATURE_COLS = [
    "cramps", "bloating", "headache", "fatigue",
    "mood_swings", "acne", "back_pain", "nausea",
    "flow_intensity", "pain_level", "energy_level", "mood_score",
]


class SymptomClusterer:

    def __init__(self, n_clusters=4):
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.is_trained = False

    def _generate_base(self):
        np.random.seed(42)
        return pd.DataFrame(
            np.random.randint(0, 5, (120, len(FEATURE_COLS))),
            columns=FEATURE_COLS
        )

    def fit(self, logs=None):
        base = self._generate_base()

        if logs and len(logs) >= 5:
            user_df = prepare_health_features(logs)

            if user_df is not None and not user_df.empty:
                user_df = user_df.reindex(columns=FEATURE_COLS, fill_value=0)
                df = pd.concat([base, user_df])
            else:
                df = base
        else:
            df = base

        X = df[FEATURE_COLS].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled)
        self.pca.fit(X_scaled)
        self.is_trained = True

    def predict_day(self, log):
        if not self.is_trained:
            self.fit()

        df = pd.DataFrame([log]).reindex(columns=FEATURE_COLS, fill_value=0)
        X = self.scaler.transform(df)

        cluster = int(self.model.predict(X)[0])

        profile = CLUSTER_PROFILES.get(cluster, CLUSTER_PROFILES[2])

        return {
            "cluster_id": cluster,
            "name": profile["name"],
            "emoji": profile["emoji"]
        }

    def get_pattern_summary(self, logs):
        if not logs or len(logs) < 3:
            return None

        if not self.is_trained:
            self.fit(logs)

        df = prepare_health_features(logs).reindex(columns=FEATURE_COLS, fill_value=0)

        X = self.scaler.transform(df)
        labels = self.model.predict(X)
        coords = self.pca.transform(X)

        counts = Counter(labels)
        dominant = counts.most_common(1)[0][0]

        return {
            "labels": labels.tolist(),
            "dominant": CLUSTER_PROFILES[dominant],
            "coords": coords.tolist(),
        }
