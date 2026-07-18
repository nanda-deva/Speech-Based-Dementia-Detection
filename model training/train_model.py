import os
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# ==========================================
# LOAD DATA
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "merged_features.csv")

df = pd.read_csv(CSV_PATH)

print("Dataset Shape:", df.shape)

# ==========================================
# PREPARE DATA
# ==========================================

X = df.drop(columns=["filename", "task", "label"])

y = df["label"].map({
    "Control": 0,
    "Dementia": 1
})

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# ==========================================
# TRAIN MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

# ==========================================
# CONFUSION MATRIX GRAPH
# ==========================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Control", "Dementia"],
    yticklabels=["Control", "Dementia"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, "confusion_matrix.png"),
    dpi=300
)

plt.close()

# ==========================================
# ROC CURVE
# ==========================================

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, "roc_curve.png"),
    dpi=300
)

plt.close()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10, 6))

plt.barh(
    feature_df["Feature"],
    feature_df["Importance"]
)

plt.gca().invert_yaxis()

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, "feature_importance.png"),
    dpi=300
)

plt.close()

# ==========================================
# SAVE MODEL
# ==========================================

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

joblib.dump(model, MODEL_PATH)

print("\nModel saved as model.pkl")
print("Confusion Matrix saved.")
print("ROC Curve saved.")
print("Feature Importance saved.")