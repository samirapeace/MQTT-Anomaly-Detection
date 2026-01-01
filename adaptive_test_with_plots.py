import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# =========================
# الإعدادات
# =========================
BATCH_SIZE = 50      # عدد النوافذ بكل مجموعة
PERCENTILE = 5       # عتبة ديناميكية
RANDOM_STATE = 42

# =========================
# تحميل البيانات
# =========================
df = pd.read_csv("mqtt_dataset.csv")

X = df.drop(columns=["label"])
y_true = df["label"].values

# =========================
# تدريب Isolation Forest (على الطبيعي فقط)
# =========================
X_train = X[df["label"] == 0]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=RANDOM_STATE
)
model.fit(X_train_scaled)

# =========================
# اختبار مع عتبة متغيرة
# =========================
X_scaled = scaler.transform(X)

y_pred = np.zeros(len(X_scaled))
thresholds = []
batch_centers = []

for start in range(0, len(X_scaled), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch_scores = model.decision_function(X_scaled[start:end])

    dynamic_threshold = np.percentile(batch_scores, PERCENTILE)

    y_pred[start:end] = (batch_scores < dynamic_threshold).astype(int)

    thresholds.append(dynamic_threshold)
    batch_centers.append(start + BATCH_SIZE / 2)

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_true, y_pred)
cm_percent = cm / cm.sum() * 100

# =========================
# رسم (1): تغيّر العتبة مع الزمن
# =========================
plt.figure()
plt.plot(batch_centers, thresholds, marker='o')
plt.xlabel("Time Windows")
plt.ylabel("Dynamic Threshold")
plt.title("Dynamic Threshold Variation Over Time")
plt.grid(True)
plt.show()

# =========================
# رسم (2): Confusion Matrix بالنسب المئوية
# =========================
labels = ["Normal", "Attack"]

plt.figure(figsize=(6, 5))
plt.imshow(cm_percent)
plt.colorbar()

plt.xticks([0, 1], labels)
plt.yticks([0, 1], labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (%)")

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
# طباعة النتائج النصية
# =========================
print("Confusion Matrix (counts):\n", cm)
print("\nConfusion Matrix (%):\n", cm_percent)
