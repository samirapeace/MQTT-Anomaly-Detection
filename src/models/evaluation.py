import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report
)


class ModelEvaluator:

    def __init__(self):
        self.epsilon = 1e-9


    def evaluate(self, y_true, y_pred, y_prob):

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

  
        fpr_value = fp / (fp + tn + self.epsilon)
        tnr = tn / (tn + fp + self.epsilon)  
        balanced_acc = (rec + tnr) / 2

 
        if len(np.unique(y_true)) < 2:
            roc_auc = 0.5
            pr_auc = 0.5
        else:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "fpr": fpr_value,
            "specificity": tnr,
            "balanced_accuracy": balanced_acc,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm
        }


    def compute_roc(self, y_true, y_prob):
        return roc_curve(y_true, y_prob)


    def compute_pr(self, y_true, y_prob):
        return precision_recall_curve(y_true, y_prob)


    def find_best_threshold(self, y_true, y_prob):

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_index = tpr - fpr

        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]

        return best_threshold

    def print_results(self, results):

        print("\n===== EVALUATION =====")
        print(f"Accuracy:           {results['accuracy']:.4f}")
        print(f"Precision:          {results['precision']:.4f}")
        print(f"Recall:             {results['recall']:.4f}")
        print(f"F1 Score:           {results['f1_score']:.4f}")
        print(f"FPR:                {results['fpr']:.4f}")
        print(f"Specificity (TNR):  {results['specificity']:.4f}")
        print(f"Balanced Accuracy:  {results['balanced_accuracy']:.4f}")
        print(f"ROC AUC:            {results['roc_auc']:.4f}")
        print(f"PR AUC:             {results['pr_auc']:.4f}")

        print("\nConfusion Matrix:")
        print(results["confusion_matrix"])

    def detailed_report(self, y_true, y_pred):
        print("\n===== CLASSIFICATION REPORT =====")
        print(classification_report(y_true, y_pred, zero_division=0))
