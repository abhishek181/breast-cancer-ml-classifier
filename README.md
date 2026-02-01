# 🤖 Machine Learning Classification Models Comparison

## Problem Statement

This project implements and compares six different machine learning classification models on the Breast Cancer Wisconsin dataset. The goal is to evaluate the performance of various algorithms for binary classification of breast cancer diagnosis (malignant vs benign) and provide an interactive web application for model comparison and testing.

## Dataset Description

**Dataset:** Breast Cancer Wisconsin Dataset
- **Source:** UCI Machine Learning Repository (via sklearn.datasets)
- **Features:** 30 numerical features (tumor characteristics)
- **Instances:** 569 samples
- **Target:** Binary classification (0: malignant, 1: benign)
- **Class Distribution:** 212 malignant, 357 benign cases

### Features:
The dataset contains 30 features computed from digitized images of breast mass:
1. **Mean features (10):** radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
2. **Standard error features (10):** Same measurements as above
3. **Worst features (10):** Largest values of the above measurements

## Models Used

The following six classification models were implemented and compared:

### Model Comparison Table

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
|---------------|----------|-----------|-----------|--------|----------|-----------|
| Logistic Regression | 0.9561 | 0.9957 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| K-Nearest Neighbors | 0.9123 | 0.9559 | 0.9429 | 0.9167 | 0.9296 | 0.8139 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9561 | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| XGBoost (Ensemble) | 0.9561 | 0.9901 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | **Best overall performer** with high accuracy (95.61%) and AUC (99.57%). Excellent for this linearly separable dataset with well-scaled features. Simple, interpretable, and highly effective. |
| Decision Tree | **Lowest performer** with 91.23% accuracy. Prone to overfitting despite good precision (95.59%). The tree structure may be too complex for this dataset, leading to poor generalization. |
| K-Nearest Neighbors | **Moderate performance** (91.23% accuracy) but good AUC (95.59%). Performance likely limited by curse of dimensionality with 30 features. Sensitive to feature scaling and local noise. |
| Naive Bayes | **Strong performance** (93.86% accuracy, 98.78% AUC) despite independence assumption. Works well due to the dataset's feature characteristics and good class separation in feature space. |
| Random Forest (Ensemble) | **Excellent performance** (95.61% accuracy, 99.37% AUC). Ensemble method effectively reduces overfitting. Good balance between performance and interpretability through feature importance. |
| XGBoost (Ensemble) | **Excellent performance** (95.61% accuracy, 99.01% AUC) with highest recall (98.61%). Gradient boosting handles feature interactions well. Slightly better at detecting malignant cases (higher recall). |

```


## License

This project is created for educational purposes.