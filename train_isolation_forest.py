import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# تحميل البيانات
df = pd.read_csv("mqtt_dataset.csv")

X = df.drop(columns=["label"])
y_true = df["label"]

# تدريب على الطبيعي فقط
X_train = X[y_true == 0]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)

model.fit(X_train_scaled)

# حساب عتبة ديناميكية
train_scores = model.decision_function(X_train_scaled)
threshold = np.percentile(train_scores, 5)

# اختبار على كل البيانات
X_test_scaled = scaler.transform(X)
test_scores = model.decision_function(X_test_scaled)

y_pred = (test_scores < threshold).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
cm_percent = cm / cm.sum() * 100

print("Confusion Matrix (counts):\n", cm)
print("\nConfusion Matrix (%):\n", cm_percent)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))
