import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import seaborn as sns
import matplotlib.pyplot as plt

st.title("ML Classification Models Comparison App")

st.sidebar.header("Upload Dataset (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())
else:
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

model_name = st.selectbox("Select Model", list(models.keys()))

model = models[model_name]
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

st.subheader("Evaluation Metrics")

st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("AUC:", roc_auc_score(y_test, y_prob))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))
st.write("F1 Score:", f1_score(y_test, y_pred))
st.write("MCC:", matthews_corrcoef(y_test, y_pred))

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))
