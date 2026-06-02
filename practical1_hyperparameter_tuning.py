# ==============================
# PRACTICAL NO. 1
# AIM: Hyperparameter Tuning using Grid Search & Random Search
# Dataset: Breast Cancer (sklearn)
# ==============================

# 1. Import Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint

# 2. Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Base Model
rf = RandomForestClassifier(random_state=42)

# 4. Define Hyperparameter Space
grid_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

random_params = {
    'n_estimators': randint(50, 500),
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': randint(2, 20)
}

# 5. Grid Search
grid_search = GridSearchCV(
    rf,
    param_grid=grid_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_grid_model = grid_search.best_estimator_

# 6. Random Search
random_search = RandomizedSearchCV(
    rf,
    param_distributions=random_params,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
best_random_model = random_search.best_estimator_

# 7. Evaluate Models
rf.fit(X_train, y_train)
base_pred  = rf.predict(X_test)
base_acc   = accuracy_score(y_test, base_pred)

grid_pred  = best_grid_model.predict(X_test)
grid_acc   = accuracy_score(y_test, grid_pred)

random_pred = best_random_model.predict(X_test)
random_acc  = accuracy_score(y_test, random_pred)

# 8. Compare Results
print("Base Model Accuracy  :", base_acc)
print("Grid Search Accuracy :", grid_acc)
print("Random Search Accuracy:", random_acc)
print("\nBest Parameters (Grid)  :", grid_search.best_params_)
print("Best Parameters (Random):", random_search.best_params_)
print("\nClassification Report (Best Grid Model):")
print(classification_report(y_test, grid_pred))
