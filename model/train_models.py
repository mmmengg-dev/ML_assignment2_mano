# Model training script placeholder
"""
train_models.py

This script trains and saves ALL 6 classification models:

1. Logistic Regression
2. Decision Tree
3. KNN
4. Naive Bayes
5. Random Forest
6. XGBoost

Saved models are stored as .pkl files inside the model folder.
"""

import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Import model modules
import logistic_regression
import decision_tree
import knn
import naive_bayes
import random_forest
import xgboost_model


def load_dataset():
    """
    Load dataset
    """

    print("Loading dataset...")

    data = load_breast_cancer()

    X = data.data
    y = data.target

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def train_and_save_all_models():
    """
    Train and save all models
    """

    X_train, X_test, y_train, y_test = load_dataset()

    print("\nTraining Logistic Regression...")
    model = logistic_regression.train_model(X_train, y_train)
    logistic_regression.save_model(model)

    print("Training Decision Tree...")
    model = decision_tree.train_model(X_train, y_train)
    decision_tree.save_model(model)

    print("Training KNN...")
    model = knn.train_model(X_train, y_train)
    knn.save_model(model)

    print("Training Naive Bayes...")
    model = naive_bayes.train_model(X_train, y_train)
    naive_bayes.save_model(model)

    print("Training Random Forest...")
    model = random_forest.train_model(X_train, y_train)
    random_forest.save_model(model)

    print("Training XGBoost...")
    model = xgboost_model.train_model(X_train, y_train)
    xgboost_model.save_model(model)

    print("\nALL MODELS TRAINED AND SAVED SUCCESSFULLY!")


if __name__ == "__main__":

    train_and_save_all_models()
