import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
# GRID SEARCH
# ==========================================

param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False]
}

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

print("\nSearching for best parameters...\n")

grid_search.fit(X_train, y_train)

# ==========================================
# BEST MODEL
# ==========================================

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

# ==========================================
# TEST SET
# ==========================================

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# SAVE BEST MODEL
# ==========================================

MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")

joblib.dump(best_model, MODEL_PATH)

print("\nBest model saved as best_model.pkl")