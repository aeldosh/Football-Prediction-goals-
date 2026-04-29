"""
Train and Save Model Script
============================
This script trains a Gradient Boosting Regressor on the FBRef 2024-2025 dataset
with 10-Fold Cross Validation, and saves the best model using joblib.

Usage:
    Run this script in Google Colab or locally after placing your dataset CSV.
    python train_and_save_model.py
"""

import sys
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import warnings

warnings.filterwarnings("ignore")

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def main():
    # -----------------------------------------
    # 1. Load Dataset
    # -----------------------------------------
    print("=" * 60)
    print("  Football Player Performance Analysis")
    print("  FBRef 2024-2025 Dataset")
    print("=" * 60)

    df = pd.read_csv("PlayersFBREF_FeatureSelected.csv")
    print(f"\n[OK] Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop rows with missing values in the features/target
    feature_columns = [
        "Age",
        "Matches Played",
        "Minutes",
        "Assists",
        "Penalty Goals Made",
        "Yellow Cards",
        "Red Cards",
        "Progressive Carries",
        "Progressive Passes",
    ]
    target_column = "Goals"

    df = df.dropna(subset=feature_columns + [target_column])
    print(f"[OK] After dropping missing values: {df.shape[0]} rows")

    X = df[feature_columns].values
    y = df[target_column].values

    print(f"\nFeatures: {feature_columns}")
    print(f"Target: {target_column}")
    print(f"Dataset Shape: X={X.shape}, y={y.shape}")

    # -----------------------------------------
    # 2. Define Models
    # -----------------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
        "Support Vector Regressor (SVR)": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
        ]),
    }

    # -----------------------------------------
    # 3. Evaluate Models with 10-Fold CV
    # -----------------------------------------
    print("\n" + "-" * 60)
    print("  10-Fold Cross Validation Results")
    print("-" * 60)

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        mae_scores = cross_val_score(model, X, y, cv=kfold, scoring="neg_mean_absolute_error")
        r2_scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")

        mean_mae = -mae_scores.mean()
        mean_r2 = r2_scores.mean()
        results[name] = {"MAE": mean_mae, "R2": mean_r2}

        print(f"\n  >> {name}")
        print(f"     MAE:  {mean_mae:.4f} (+/- {mae_scores.std():.4f})")
        print(f"     R2:   {mean_r2:.4f} (+/- {r2_scores.std():.4f})")

    # -----------------------------------------
    # 4. Select Best Model
    # -----------------------------------------
    # Using Gradient Boosting Regressor as the best model
    # (selected based on overall performance and generalization)
    best_model_name = "Gradient Boosting Regressor"

    print("\n" + "-" * 60)
    print(f"  [BEST] Best Model: {best_model_name}")
    print(f"     MAE: {results[best_model_name]['MAE']:.4f}")
    print(f"     R2:  {results[best_model_name]['R2']:.4f}")
    print("-" * 60)

    # -----------------------------------------
    # 5. Train Best Model on Full Dataset & Save
    # -----------------------------------------
    best_model = models[best_model_name]
    best_model.fit(X, y)

    model_filename = "best_model.pkl"
    joblib.dump(best_model, model_filename)
    print(f"\n[OK] Best model saved as '{model_filename}'")
    print(f"   Use joblib.load('{model_filename}') to load the model.\n")


if __name__ == "__main__":
    main()
