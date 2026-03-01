#!/usr/bin/env python
# coding: utf-8

"""
Heart Disease Prediction - Production Training Script
This script trains the best model (Random Forest)
and saves the trained model + scaler for deployment.
"""

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# 2️⃣ Load Dataset
df = pd.read_csv("heart.csv")


# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5️⃣ Train Final Model (Random Forest)
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# 6️⃣ Save Model and Scaler
# 6️⃣ Save Model, Scaler and Columns
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("✅ Model, Scaler and Columns saved successfully!")

print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))
