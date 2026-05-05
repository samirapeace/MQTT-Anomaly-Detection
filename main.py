from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.features.engineer import FeatureEngineer
from src.features.selector import FeatureSelector
from src.models.isolation import IsolationModel
from src.models.random_forest import RFModel
from src.data.saver import DataSaver
from src.features.kl import KLDivergence
from src.models.decision import DecisionModel
from src.models.evaluation import ModelEvaluator
import pandas as pd
import time
import datetime
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib


DATA_PATH = "data/raw"

loader = DataLoader(DATA_PATH)
cleaner = DataCleaner(label_column="label")
engineer = FeatureEngineer()
selector = FeatureSelector(label_column="label")
iso_model = IsolationModel()
rf_model = RFModel()
kl_model = KLDivergence()
evaluator = ModelEvaluator()
saver = DataSaver("data/processed/final_dataset.csv")

print("Files found:", len(loader.get_files()))

print("\nBuilding TEST dataset...")
print("\nBuilding FULL dataset...")

all_data = []
max_samples = 300000  

for chunk in loader.dataset_generator():
    chunk = cleaner.clean_chunk(chunk)
    chunk = engineer.add_all_features(chunk)

    chunk = chunk.replace([float("inf"), float("-inf")], pd.NA)
    chunk = chunk.fillna(0)

    all_data.append(chunk)

    if sum(len(c) for c in all_data) >= max_samples:
        break

full_df = pd.concat(all_data, ignore_index=True)
full_df = full_df.rename(columns={"target": "label"})
train_df, test_df = train_test_split(
    full_df,
    test_size=0.3,
    stratify=full_df["label"],
    random_state=42
)

print("\nTrain size:", len(train_df))
print("Test size:", len(test_df))

print("Full dataset size:", len(full_df))
print(full_df["label"].value_counts())

print("Test dataset size:", len(test_df))
print(test_df["label"].value_counts())
train_df["label"] = train_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)
print("NaN count after cleaning:", train_df.isna().sum().sum())
train_df = train_df.replace([float("inf"), float("-inf")], pd.NA)
train_df = train_df.fillna(0).copy()
train_df, selected_features = selector.full_selection(train_df, top_k=25)
selected_columns = selected_features
print("\n🔥 SELECTED FEATURES:")
print(selected_features)

print("Before IF - label distribution:")
print(train_df["label"].value_counts())

print("Checking benign rows:")
print(len(train_df[train_df["label"] == 3]))

benign_train = train_df[train_df["label"] == 3]

if len(benign_train) == 0:
    raise ValueError("❌ No benign samples found after feature selection!")
print("Benign shape:", benign_train.shape)
iso_model.train(benign_train)
print("Benign used for IF:", len(benign_train))

kl_model.fit(train_df)
train_scored = iso_model.compute_scores(train_df)
train_scored = kl_model.compute(train_scored)

X_train, y_train = rf_model.prepare_data(train_scored)

rf_model.train(X_train, y_train)
rf_features = X_train.columns

X_rf_train = X_train.reindex(columns=rf_features, fill_value=0)
y_rf_train = y_train

rf_probs_train = rf_model.predict_proba(X_rf_train)

if len(rf_probs_train.shape) > 1:
    rf_probs_train = rf_probs_train[:, -1]
stack_X = pd.DataFrame({
    "rf": rf_probs_train,
    "iso": train_scored["anomaly_score"],
    "kl": train_scored["kl_score"]
})

stack_y = y_rf_train

scaler_stack = StandardScaler()
stack_X_scaled = scaler_stack.fit_transform(stack_X)

meta_model = LogisticRegression()
meta_model.fit(stack_X_scaled, stack_y) 

print("Stacking model trained!")
print("Models trained successfully!")


os.makedirs("models", exist_ok=True)

joblib.dump(meta_model, "models/meta_model.pkl")
joblib.dump(scaler_stack, "models/scaler.pkl")
joblib.dump(rf_model, "models/rf_model.pkl")
joblib.dump(iso_model, "models/iso_model.pkl")
joblib.dump(kl_model, "models/kl_model.pkl")
joblib.dump(selected_columns, "models/features.pkl")
threshold = 0.5  
print("\n🔍 Searching for optimal weights...")

best_auc = 0
best_weights = (0.5, 0.3, 0.2)


val_df, _ = train_test_split(
    train_scored,
    test_size=0.5,
    stratify=train_scored["label"],
    random_state=42
)
X_val, y_val = rf_model.prepare_data(val_df)

X_val = X_val.reindex(columns=rf_features, fill_value=0)

rf_probs_val_stack = rf_model.predict_proba(X_val)

if len(rf_probs_val_stack.shape) > 1:
    rf_probs_val_stack = rf_probs_val_stack[:, -1]
for w1 in [0.2, 0.3, 0.4, 0.5]:
    for w2 in [0.2, 0.3, 0.4]:
        w3 = 1 - (w1 + w2)
        if w3 <= 0:
            continue

        temp_model = DecisionModel(w1, w2, w3)

        scores = temp_model.compute_score(
            val_df["anomaly_score"],
            rf_probs_val_stack,
            val_df["kl_score"]
        )

        auc = roc_auc_score(y_val, scores)

        if auc > best_auc:
            best_auc = auc
            best_weights = (w1, w2, w3)

print("Best weights found:", best_weights)
decision_model = DecisionModel(
    w1=best_weights[0],
    w2=best_weights[1],
    w3=best_weights[2]
)



test_df = test_df[selected_columns.tolist() + ["label"]]
test_df = test_df.replace([float("inf"), float("-inf")], pd.NA)
test_df = test_df.fillna(0)


test_scored = iso_model.compute_scores(test_df)
test_scored = kl_model.compute(test_scored)


y_test = test_scored["label"].apply(lambda x: 0 if x == 3 else 1)
y_test = y_test.astype(int)
X_test = test_scored.drop(columns=["label"])
X_test = X_test.reindex(columns=rf_features, fill_value=0)
print("y_test distribution:")
print(y_test.value_counts())

rf_probs_test = rf_model.predict_proba(X_test)

if len(rf_probs_test.shape) > 1:
    rf_probs_test = rf_probs_test[:, -1]

stack_test_X = pd.DataFrame({
    "rf": rf_probs_test,
    "iso": test_scored["anomaly_score"],
    "kl": test_scored["kl_score"]
})

X_val, y_val = rf_model.prepare_data(val_df)
X_val = X_val.reindex(columns=rf_features, fill_value=0)

rf_probs_val_stack = rf_model.predict_proba(X_val)

if len(rf_probs_val_stack.shape) > 1:
    rf_probs_val_stack = rf_probs_val_stack[:, -1]

stack_val_X = pd.DataFrame({
    "rf": rf_probs_val_stack,
    "iso": val_df["anomaly_score"],
    "kl": val_df["kl_score"]
})

stack_val_X_scaled = scaler_stack.transform(stack_val_X)

probs = meta_model.predict_proba(stack_val_X_scaled)[:, 1]

probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)
fpr, tpr, thresholds = roc_curve(y_val, probs)
youden_index = tpr - fpr
threshold = thresholds[np.argmax(youden_index)]
threshold = float(threshold)
threshold = max(0.01, min(0.99, threshold))
print("Validation Threshold:", threshold)
print("Threshold type:", type(threshold))
stack_test_X_scaled = scaler_stack.transform(stack_test_X)
probs_test = meta_model.predict_proba(stack_test_X_scaled)[:, 1]
probs_test = (probs_test - probs_test.min()) / (probs_test.max() - probs_test.min() + 1e-8)
preds_test = (probs_test > threshold).astype(int)
print("\n===== REAL TEST EVALUATION =====")
results = evaluator.evaluate(y_test, preds_test, probs_test)
evaluator.print_results(results)

threshold = float(threshold)
threshold = max(0.01, min(0.99, threshold))
print("Clipped Threshold:", threshold)
for i, chunk in enumerate(loader.dataset_generator()):
    print(f"\nProcessing chunk {i}...")


    chunk = cleaner.clean_chunk(chunk)
 
    if len(chunk) > 1000:
        chunk = chunk.sample(frac=1, random_state=42)

    featured  = engineer.add_all_features(chunk)

    featured = featured.replace([float("inf"), float("-inf")], pd.NA)
    featured = featured.fillna(0)

    selected = featured.reindex(columns=selected_columns, fill_value=0)
    selected["label"] = featured["label"]


    scored = iso_model.compute_scores(selected)
    scored = kl_model.compute(scored)


    X_chunk, _ = rf_model.prepare_data(scored)

    X_chunk = X_chunk.reindex(columns=rf_features, fill_value=0)
    rf_probs_chunk = rf_model.predict_proba(X_chunk)

    if len(rf_probs_chunk.shape) > 1:
        rf_probs_chunk = rf_probs_chunk[:, -1]

    stack_chunk_X = pd.DataFrame({
        "rf": rf_probs_chunk,
        "iso": scored["anomaly_score"],
        "kl": scored["kl_score"]
    })

    stack_chunk_X_scaled = scaler_stack.transform(stack_chunk_X)
    probs = meta_model.predict_proba(stack_chunk_X_scaled)[:, 1]

    probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8) 
    probs = np.clip(probs, 0.001, 0.999)

    y_true_chunk = scored["label"].apply(lambda x: 0 if x == 3 else 1)
    if len(y_true_chunk.unique()) > 1:


        target_rate = max(0.02, min(0.08, probs.mean()))
        new_threshold = np.percentile(probs, 100 * (1 - target_rate))

        alpha = 0.2
        threshold = (1 - alpha) * threshold + alpha * new_threshold

        threshold = max(0.03, min(0.99, threshold))
    print("Updated Threshold:", threshold)


    preds = (probs > threshold).astype(int)


    if len(y_true_chunk.unique()) > 1:
        results = evaluator.evaluate(y_true_chunk, preds, probs)
        evaluator.print_results(results)
    else:
        print("Skipping evaluation (single class)")

   


    final_scores = decision_model.compute_score(
        scored["anomaly_score"],
        probs,
        scored["kl_score"]
    )

    print("\nFinal Decision Scores:")
    print(final_scores[:5])

    print("Prediction distribution:")
    print(pd.Series(preds).value_counts())


    log_entry = {
        "time": datetime.datetime.now(),
        "chunk": i,
        "threshold": threshold,
        "avg_score": probs.mean(),
        "num_samples": len(preds) ,
        "positive_rate": preds.mean(),  }

    os.makedirs("logs", exist_ok=True)

    pd.DataFrame([log_entry]).to_csv(
        "logs/system_log.csv",
        mode='a',
        header=not os.path.exists("logs/system_log.csv"),
        index=False
    )


    if i == 0:
        saver.save_chunk(scored, first=True)
    else:
        saver.save_chunk(scored, first=False)


    time.sleep(1)
    
