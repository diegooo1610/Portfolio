# train_model.py
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Load dataset
df = pd.read_csv("final_5yr_stock_data.csv")
X = df.drop(["Return_Level", "Date", "Ticker", "Month"], axis=1)
y = df["Return_Level"]

label_mapping = {"Down": 0, "Neutral": 1, "Up": 2}
y = df["Return_Level"].map(label_mapping)


# Train/test split
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(eval_metric="mlogloss", random_state=42))
])

# Hyperparameter grid
param_grid = {
    "xgb__n_estimators": [200],
    "xgb__max_depth": [7],
    "xgb__learning_rate": [0.1],
    "xgb__subsample": [0.8]
}

# Grid search
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# Save best model
joblib.dump(grid.best_estimator_, "xgb_pipeline_model.pkl")
print("✅ Model trained and saved at 'xgb_pipeline_model.pkl'")
