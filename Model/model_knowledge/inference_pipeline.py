import os
import json
import pandas as pd
from catboost import CatBoostClassifier

class FloodEarlyWarningPipeline:
    def __init__(self, artifacts_dir="."):
        self.artifacts_dir = artifacts_dir
        with open(os.path.join(artifacts_dir, "station_flood_thresholds.json"), "r") as f:
            self.station_thresholds = json.load(f)
        with open(os.path.join(artifacts_dir, "decision_policy.json"), "r") as f:
            self.policy = json.load(f)
        with open(os.path.join(artifacts_dir, "feature_schema.json"), "r") as f:
            self.schema = json.load(f)
        self.optimal_threshold = self.policy["optimal_threshold"]
        self.model = CatBoostClassifier()
        self.model.load_model(os.path.join(artifacts_dir, "champion_catboost_model.cbm"))

    def predict_risk(self, feature_df):
        features = feature_df[self.schema["feature_columns"]].copy()
        for col in self.schema["categorical_columns"]:
            features[col] = features[col].astype(str)
        probs = self.model.predict_proba(features)[:, 1]
        alerts = (probs >= self.optimal_threshold).astype(int)
        results = feature_df[["date", "location"]].copy() if "date" in feature_df.columns else pd.DataFrame()
        results["predicted_flood_probability"] = probs
        results["flood_alert_24h"] = alerts
        return results
