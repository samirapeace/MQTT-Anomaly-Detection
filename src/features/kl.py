import numpy as np
import pandas as pd

class KLDivergence:
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon
        self.reference_mean = None
        self.reference_var = None
        self.feature_weights = None

    def fit(self, df, label_column="label"):
        benign = df[df[label_column] == 3]
        X = benign.drop(columns=[label_column], errors="ignore")

        self.reference_mean = X.mean()
        self.reference_var = X.var() + self.epsilon


        self.feature_weights = 1 / (self.reference_var + self.epsilon)


    def compute(self, df, label_column="label"):
        X = df.drop(columns=[label_column, "anomaly_score"], errors="ignore")

        kl_scores = np.zeros(len(X))

        for col in X.columns:

            if col not in self.reference_mean:
                continue

            mean = self.reference_mean[col]
            var = self.reference_var[col]

            z = (X[col] - mean) / np.sqrt(var)

   
            kl = 0.5 * (z ** 2)


            weight = self.feature_weights[col]
            kl_scores += weight * kl


        kl_scores = (kl_scores - kl_scores.min()) / (
            kl_scores.max() - kl_scores.min() + self.epsilon
        )

        df = df.copy()
        df["kl_score"] = kl_scores

        return df
