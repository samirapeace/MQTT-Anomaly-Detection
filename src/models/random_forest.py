from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class RFModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,          # 🔥 أقوى
            max_depth=20,              # 🔥 يمنع overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",   # 🔥 adaptive
            random_state=42,
            n_jobs=-1
        )

    # =========================
    # Prepare Data
    # =========================
    def prepare_data(self, df, label_column="label"):

        X = df.drop(columns=[label_column], errors="ignore")

        y = df[label_column].apply(
            lambda x: 0 if x == 3 else 1        )

        return X, y

    # =========================
    # Train/Test Split
    # =========================
    def split(self, X, y):
        return train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

    # =========================
    # Train
    # =========================
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    # =========================
    # Predict Probability
    # =========================
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    # =========================
    # Feature Importance 🔥🔥🔥
    # =========================
    def get_feature_importance(self, feature_names):

        importances = self.model.feature_importances_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        return importance_df

    # =========================
    # Top Features (اختياري)
    # =========================
    def get_top_features(self, feature_names, top_n=10):
        importance_df = self.get_feature_importance(feature_names)
        return importance_df.head(top_n)