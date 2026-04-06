import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from feature_engineering import prepare_symptom_features

CLUSTER_PROFILES = {
    0: {"name": "High Pain Days", "emoji": "🤒", "description": "Intense cramps, back pain, and low energy.", "tip": "Rest, use a heating pad.", "color": "#e74c3c"},
    1: {"name": "Hormonal Surge Days", "emoji": "😓", "description": "Mood swings, acne, and fatigue.", "tip": "Track food triggers.", "color": "#f39c12"},
    2: {"name": "Energetic Days", "emoji": "⚡", "description": "Low symptom load, good energy.", "tip": "Great time for workouts.", "color": "#2ecc71"},
    3: {"name": "Bloating & Discomfort Days", "emoji": "😮", "description": "Bloating and mild fatigue.", "tip": "Reduce sodium intake.", "color": "#e67e22"},
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

    def _generate_synthetic_base(self):
        np.random.seed(42)
        records = []
        for _ in range(30): records.append({"cramps": 1, "bloating": 0, "headache": 0, "fatigue": 1, "mood_swings": 0, "acne": 0, "back_pain": 1, "nausea": 0, "flow_intensity": 4, "pain_level": 4, "energy_level": 1, "mood_score": 2})
        for _ in range(30): records.append({"cramps": 0, "bloating": 0, "headache": 0, "fatigue": 0, "mood_swings": 1, "acne": 1, "back_pain": 0, "nausea": 0, "flow_intensity": 1, "pain_level": 2, "energy_level": 3, "mood_score": 2})
        for _ in range(30): records.append({"cramps": 0, "bloating": 0, "headache": 0, "fatigue": 0, "mood_swings": 0, "acne": 0, "back_pain": 0, "nausea": 0, "flow_intensity": 0, "pain_level": 0, "energy_level": 5, "mood_score": 5})
        for _ in range(30): records.append({"cramps": 0, "bloating": 1, "headache": 0, "fatigue": 1, "mood_swings": 0, "acne": 0, "back_pain": 0, "nausea": 1, "flow_intensity": 1, "pain_level": 1, "energy_level": 2, "mood_score": 3})
        return pd.DataFrame(records)

    def fit(self, symptom_logs: list = None):
        base_df = self._generate_synthetic_base()
        if symptom_logs and len(symptom_logs) >= 5:
            user_df = prepare_symptom_features(symptom_logs)
            df = pd.concat([base_df, user_df.reindex(columns=self.feature_cols, fill_value=0)], ignore_index=True) if user_df is not None else base_df
        else:
            df = base_df

        X = df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.pca.fit(X_scaled)
        self.is_trained = True

    def predict_day(self, symptom_log: dict) -> dict:
        if not self.is_trained: self.fit()
        df = pd.DataFrame([symptom_log]).reindex(columns=self.feature_cols, fill_value=0)
        X_scaled = self.scaler.transform(df)
        cluster_id = int(self.model.predict(X_scaled)[0])
        return {"cluster_id": cluster_id, **CLUSTER_PROFILES.get(cluster_id, CLUSTER_PROFILES[2])}

    def get_pattern_summary(self, symptom_logs: list) -> dict:
        if not symptom_logs or len(symptom_logs) < 3: return None
        if not self.is_trained: self.fit(symptom_logs)

        df = prepare_symptom_features(symptom_logs).reindex(columns=self.feature_cols, fill_value=0)
        X_scaled = self.scaler.transform(df)
        labels = self.model.predict(X_scaled)
        coords = self.pca.transform(X_scaled)

        results = [{"day": i+1, "cluster_id": int(l), "cluster_name": CLUSTER_PROFILES[int(l)]["name"], "color": CLUSTER_PROFILES[int(l)]["color"], "x": float(coords[i, 0]), "y": float(coords[i, 1])} for i, l in enumerate(labels)]
        
        from collections import Counter
        counts = Counter(labels)
        dom = counts.most_common(1)[0][0]

        return {"days": results, "dominant_profile": CLUSTER_PROFILES[dom], "cluster_counts": {CLUSTER_PROFILES[k]["name"]: v for k, v in counts.items()}}
