from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


class IsolationModel:
    def __init__(self, contamination="auto"):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.epsilon = 1e-9


    def train(self, df, label_column="label"):

        benign_df = df[df[label_column] == 3]
        X = benign_df.drop(columns=[label_column], errors="ignore")

        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled)


    def compute_scores(self, df, label_column="label"):

        X = df.drop(columns=[label_column], errors="ignore")

        X_scaled = self.scaler.transform(X)

        raw_scores = self.model.decision_function(X_scaled)


        scores = -raw_scores  
   
        scores = (scores - scores.min()) / (
            scores.max() - scores.min() + self.epsilon
        )

        df = df.copy()
        df["anomaly_score"] = scores

        return df
