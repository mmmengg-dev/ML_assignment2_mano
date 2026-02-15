import os
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "decision_tree.pkl")


def train_model(X_train, y_train):

    model = DecisionTreeClassifier()

    model.fit(X_train, y_train)

    return model


def save_model(model):

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def load_model():

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:,1]

    return {

        "Accuracy": accuracy_score(y_test, y_pred),

        "AUC": roc_auc_score(y_test, y_prob),

        "Precision": precision_score(y_test, y_pred),

        "Recall": recall_score(y_test, y_pred),

        "F1": f1_score(y_test, y_pred),

        "MCC": matthews_corrcoef(y_test, y_pred)
    }
