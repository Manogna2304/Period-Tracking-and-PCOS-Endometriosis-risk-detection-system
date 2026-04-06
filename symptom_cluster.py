import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from feature_engineering import prepare_symptom_features


CLUSTER_PROFILES = {
    0: {
        "name": "High Pain Days",
        "emoji": "🔴",
        "description": "Days with intense cramps, back pain, and low energy. Typical of early period days.",
        "tip": "Rest, use a heating pad, and stay hydrated. Avoid strenuous exercise.",
        "color": "#e74c3c",
    },
    1: {
        "name": "Hormonal Surge Days",
        "emoji": "🟡",
        "description": "Mood swings, acne, and fatigue dominate. Often pre-menstrual or ovulation days.",
        "tip": "Track food triggers. Magnesium-rich foods may ease mood swings.",
        "color": "#f39c12",
    },
    2: {
        "name": "Energetic Days",
        "emoji": "🟢",
        "description": "Low symptom load, good energy and mood. Likely follicular or mid-cycle phase.",
        "tip": "Great time for high-intensity workouts and social activities.",
        "color": "#2ecc71",
    },
    3: {
        "name": "Bloating & Discomfort Days",
        "emoji": "🟠",
        "description": "Bloating, nausea, and mild fatigue. Common in late luteal phase.",
        "tip": "Reduce sodium intake and eat smaller meals throughout the day.",
        "color": "#e67e22",
    },
}


class SymptomClusterer:
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.is_trained = False
        self.feature_cols = [
            "cramps", "bloating", "headache", "fatigue",
            "mood_swings", "acne", "back_pain", "nausea",
            "flow_intensity", "pain_level", "energy_level", "mood_score",
        ]

    def _generate_synthetic_base(self, n_per_cluster=30):
        """Generate synthetic base data to pre-train clusters with meaningful profiles."""
        np.random.seed(42)
        records = []

        # Cluster 0: High pain days
        for _ in range(n_per_cluster):
            records.append({"cramps": 1, "bloating": np.random.binomial(1, 0.6), "headache": np.random.binomial(1, 0.5),
                "fatigue": 1, "mood_swings": np.random.binomial(1, 0.4), "acne": 0,
                "back_pain": 1, "nausea": np.random.binomial(1, 0.5),
                "flow_intensity": np.random.randint(3, 5), "pain_level": np.random.randint(3, 5),
                "energy_level": np.random.randint(1, 3), "mood_score": np.random.randint(1, 3)})

        # Cluster 1: Hormonal days
        for _ in range(n_per_cluster):
            records.append({"cramps": np.random.binomial(1, 0.3), "bloating": np.random.binomial(1, 0.5),
                "headache": np.random.binomial(1, 0.5), "fatigue": np.random.binomial(1, 0.6),
                "mood_swings": 1, "acne": 1, "back_pain": 0, "nausea": 0,
                "flow_intensity": np.random.randint(1, 3), "pain_level": np.random.randint(2, 4),
                "energy_level": np.random.randint(2, 4), "mood_score": np.random.randint(1, 3)})

        # Cluster 2: Energetic days
        for _ in range(n_per_cluster):
            records.append({"cramps": 0, "bloating": 0, "headache": 0, "fatigue": 0,
                "mood_swings": 0, "acne": 0, "back_pain": 0, "nausea": 0,
                "flow_intensity": np.random.randint(0, 2), "pain_level": np.random.randint(0, 2),
                "energy_level": np.random.randint(4, 6), "mood_score": np.random.randint(4, 6)})

        # Cluster 3: Bloating days
        for _ in range(n_per_cluster):
            records.append({"cramps": np.random.binomial(1, 0.3), "bloating": 1,
                "headache": np.random.binomial(1, 0.3), "fatigue": np.random.binomial(1, 0.5),
                "mood_swings": np.random.binomial(1, 0.4), "acne": 0, "back_pain": 0, "nausea": 1,
                "flow_intensity": np.random.randint(0, 3), "pain_level": np.random.randint(1, 3),
                "energy_level": np.random.randint(2, 4), "mood_score": np.random.randint(2, 4)})

        return pd.DataFrame(records)

    def fit(self, symptom_logs: list = None):
        """Train K-Means. Uses user logs combined with synthetic base data."""
        base_df = self._generate_synthetic_base()
        if symptom_logs and len(symptom_logs) >= 5:
            user_df = prepare_symptom_features(symptom_logs)
            if user_df is not None:
                user_df = user_df.reindex(columns=self.feature_cols, fill_value=0)
                df = pd.concat([base_df, user_df], ignore_index=True)
            else:
                df = base_df
        else:
            df = base_df

        X = df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.pca.fit(X_scaled)
        self.is_trained = True

    def predict_day(self, symptom_log: dict) -> dict:
        """Predict cluster for a single day's symptoms."""
        if not self.is_trained:
            self.fit()
        df = pd.DataFrame([symptom_log]).reindex(columns=self.feature_cols, fill_value=0)
        X_scaled = self.scaler.transform(df)
        cluster_id = int(self.model.predict(X_scaled)[0])
        profile = CLUSTER_PROFILES.get(cluster_id, CLUSTER_PROFILES[2])
        return {"cluster_id": cluster_id, **profile}

    def get_pattern_summary(self, symptom_logs: list) -> dict:
        """Cluster all logged days and return summary statistics."""
        if not self.is_trained:
            self.fit(symptom_logs)
        if not symptom_logs or len(symptom_logs) < 3:
            return None

        df = prepare_symptom_features(symptom_logs)
        df = df.reindex(columns=self.feature_cols, fill_value=0)
        X_scaled = self.scaler.transform(df)
        labels = self.model.predict(X_scaled)
        coords = self.pca.transform(X_scaled)

        results = []
        for i, log in enumerate(symptom_logs):
            cid = int(labels[i])
            results.append({
                "day": i + 1,
                "cluster_id": cid,
                "cluster_name": CLUSTER_PROFILES[cid]["name"],
                "color": CLUSTER_PROFILES[cid]["color"],
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
            })

        from collections import Counter
        counts = Counter(labels)
        dominant_cluster = counts.most_common(1)[0][0]

        return {
            "days": results,
            "dominant_cluster": int(dominant_cluster),
            "dominant_profile": CLUSTER_PROFILES[dominant_cluster],
            "cluster_counts": {CLUSTER_PROFILES[k]["name"]: v for k, v in counts.items()},
        }
