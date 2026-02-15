# ML Classification Models Comparison using Streamlit

---

## a. Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models on a real-world dataset. The models are evaluated using various performance metrics including Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). An interactive web application is developed using Streamlit to demonstrate model performance and allow users to interact with the models.

---

## b. Dataset Description

The dataset used in this project is the Breast Cancer Wisconsin Dataset, available from the sklearn library.

Dataset characteristics:

- Number of instances: 569
- Number of features: 30
- Target variable: Binary classification
  - 0 → Malignant
  - 1 → Benign

Features include measurements such as:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry

The dataset satisfies assignment requirements:

- More than 500 instances
- More than 12 features
- Classification problem

---

## c. Models Used

The following machine learning classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN) Classifier
4. Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)
6. XGBoost Classifier (Ensemble)

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.97 | 0.99 | 0.97 | 0.98 | 0.97 | 0.94 |
| Decision Tree | 0.93 | 0.93 | 0.94 | 0.93 | 0.93 | 0.86 |
| kNN | 0.96 | 0.98 | 0.96 | 0.97 | 0.96 | 0.92 |
| Naive Bayes | 0.95 | 0.98 | 0.95 | 0.96 | 0.95 | 0.90 |
| Random Forest (Ensemble) | 0.98 | 0.99 | 0.98 | 0.99 | 0.98 | 0.96 |
| XGBoost (Ensemble) | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 | 0.98 |

---

## Observations about Model Performance

### Logistic Regression
Logistic Regression performed very well due to the linear separability of the dataset. It achieved high accuracy and AUC score, indicating strong classification performance.

### Decision Tree
Decision Tree achieved good performance but slightly lower accuracy compared to ensemble models. It tends to overfit when used alone.

### kNN
kNN performed well because the dataset has clear feature separation. However, it is computationally expensive during prediction.

### Naive Bayes
Naive Bayes achieved good performance despite its assumption of feature independence. It is computationally efficient.

### Random Forest
Random Forest provided excellent performance due to ensemble learning. It reduces overfitting and improves generalization.

### XGBoost
XGBoost achieved the best performance among all models. It uses gradient boosting, which improves prediction accuracy by correcting errors iteratively.

---

## Streamlit App Features

The Streamlit application provides:

- Interactive model selection
- Evaluation metrics display
- Confusion matrix visualization
- Classification report
- Dataset upload option

---

## Deployment

The application is deployed using Streamlit Community Cloud.

Streamlit App Link:
mlassignment2mano-2025aa05956.streamlit.app

---

## Conclusion

This project demonstrates the implementation and comparison of multiple classification models. Ensemble models such as Random Forest and XGBoost performed best. The Streamlit application provides an interactive interface for evaluating model performance.


