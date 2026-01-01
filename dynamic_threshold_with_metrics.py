import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# =========================
# الإعدادات
# =========================
REFERENCE_SIZE = 100    # حجم المرجع الطبيعي
PERCENTILE = 5
RANDOM_STATE = 42
BINS = 50

# =========================
# تحميل البيانات
# =========================
df = pd.read_csv("mqtt_dataset.csv")
X = df.drop(columns=["label"])
y_true = df["label"].values

# =========================
# تدريب Isolation Forest (طبيعي فقط)
# =========================
X_train = X[y_true == 0]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=RANDOM_STATE
)
model.fit(X_train_scaled)

# =========================
# حساب anomaly scores
# =========================
X_scaled = scaler.transform(X)
scores = model.decision_function(X_scaled)

scores_normal = scores[y_true == 0]
scores_attack = scores[y_true == 1]

# =========================
# Dynamic Threshold الصحيح (مرجع طبيعي فقط)
# =========================
y_pred = np.zeros(len(scores))
reference_scores = []
thresholds_over_time = []

for i in range(len(scores)):
    # تحديث المرجع فقط من الترافيك الطبيعي
    if y_true[i] == 0:
        reference_scores.append(scores[i])
        if len(reference_scores) > REFERENCE_SIZE:
            reference_scores.pop(0)

    # حساب العتبة من المرجع الطبيعي
    if len(reference_scores) >= 20:
        threshold = np.percentile(reference_scores, PERCENTILE)
    else:
        threshold = np.percentile(reference_scores, PERCENTILE) if reference_scores else scores[i]

    thresholds_over_time.append(threshold)

    # القرار
    y_pred[i] = 1 if scores[i] < threshold else 0

thresholds_over_time = np.array(thresholds_over_time)

# =========================
# Confusion Matrix + Metrics
# =========================
cm = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.ravel()

cm_percent = cm / cm.sum() * 100

attack_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0

print("=== Correct Dynamic Threshold Results ===")
print("Confusion Matrix (counts):")
print(cm)
print("\nConfusion Matrix (%):")
print(cm_percent)
print(f"\nAttack Recall: {attack_recall*100:.2f}%")
print(f"False Positive Rate: {false_positive_rate*100:.2f}%")

# =========================
# رسم (1): Confusion Matrix (%)
# =========================
labels = ["Normal", "Attack"]

plt.figure(figsize=(6, 5))
plt.imshow(cm_percent)
plt.colorbar()

plt.xticks([0, 1], labels)
plt.yticks([0, 1], labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (%) - Correct Dynamic Threshold")

for i in range(2):
    for j in range(2):
        plt.text(
            j, i,
            f"{cm_percent[i, j]:.2f}%",
            ha="center",
            va="center",
            fontsize=11
        )

plt.tight_layout()
plt.show()

# =========================
# رسم (2): Score Distribution + Dynamic Threshold
# =========================
plt.figure(figsize=(8, 5))

plt.hist(scores_normal, bins=BINS, alpha=0.6, label="Normal")
plt.hist(scores_attack, bins=BINS, alpha=0.6, label="Attack")

# رسم مجموعة من العتبات عبر الزمن (كل 20 نقطة)
for th in thresholds_over_time[::20]:
    plt.axvline(th, linestyle="--", alpha=0.2)

plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.title("Score Distribution with Correct Dynamic Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
