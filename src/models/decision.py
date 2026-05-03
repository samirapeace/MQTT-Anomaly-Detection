import numpy as np

class DecisionModel:

    def __init__(self, w1=0.4, w2=0.4, w3=0.2):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.epsilon = 1e-9

    # =========================
    # Robust Normalization 
    # =========================
    def robust_normalize(self, series):
        series = np.array(series)

        median = np.median(series)
        q1 = np.percentile(series, 25)
        q3 = np.percentile(series, 75)

        iqr = q3 - q1 + self.epsilon

        normalized = (series - median) / iqr

        normalized = np.clip(normalized, -5, 5)

        normalized = (normalized - normalized.min()) / (
            normalized.max() - normalized.min() + self.epsilon
        )

        return normalized

    # =========================
    # Adaptive Weights 
    # =========================
    def adaptive_weights(self, anomaly, prob, kl):
        stds = np.array([
            np.std(anomaly),
            np.std(prob),
            np.std(kl)
        ]) + self.epsilon

        weights = stds / np.sum(stds)

        return weights

    # =========================
    # Safe Sigmoid
    # =========================
    def sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1 / (1 + np.exp(-x))

    # =========================
    # Main Decision Function 
    # =========================
    def compute_score(self, anomaly_score, prob_score, kl_score):

        # 🔹 Robust normalization
        anomaly_norm = self.robust_normalize(anomaly_score)
        prob_norm = self.robust_normalize(prob_score)
        kl_norm = self.robust_normalize(kl_score)

        # 🔹 Adaptive weights
        w_adaptive = self.adaptive_weights(anomaly_norm, prob_norm, kl_norm)

        w_final = np.array([self.w1, self.w2, self.w3]) * 0.5 + w_adaptive * 0.5

        # normalize weights
        w_final = w_final / np.sum(w_final)

        combined = (
            w_final[0] * anomaly_norm +
            w_final[1] * prob_norm +
            w_final[2] * kl_norm
        )

        final_score = self.sigmoid(combined)

        return final_score
