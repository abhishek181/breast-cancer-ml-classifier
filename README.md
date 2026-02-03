# Machine Learning Classification Models Comparison

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
| Logistic Regression | 0.9808 | 0.9846 | 0.9846 | 0.9846 | 0.9846 | 0.9590 |
| Decision Tree | 0.9423 | 0.9385 | 0.9538 | 0.9538 | 0.9538 | 0.8769 |
| K-Nearest Neighbors | 0.9808 | 0.9850 | 0.9701 | 1.0000 | 0.9848 | 0.9594 |
| Naive Bayes | 0.9327 | 0.9866 | 0.9394 | 0.9538 | 0.9466 | 0.8559 |
| Random Forest (Ensemble) | 0.9615 | 0.9905 | 0.9841 | 0.9538 | 0.9688 | 0.9195 |
| XGBoost (Ensemble) | 0.9615 | 0.9893 | 0.9692 | 0.9692 | 0.9692 | 0.9179 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | **Excellent overall performer** with 98.08% accuracy and 98.46% AUC. Balanced performance across all metrics (98.46% precision, recall, F1). High MCC (95.90%) indicates strong correlation between predictions and actual values. Ideal for this linearly separable dataset. |
| Decision Tree | **Lowest performer** with 94.23% accuracy. Good precision (95.38%) but prone to overfitting. Lower MCC (87.69%) suggests moderate prediction reliability.|
| K-Nearest Neighbors | **Perfect recall** performer with 100% recall and 98.08% accuracy. Excellent at detecting all positive cases (no false negatives). High AUC (98.50%) and MCC (95.94%). Distance-based approach works well with scaled features. |
| Naive Bayes | **Strong performer** with highest AUC (98.66%) despite lower accuracy (93.27%). Good balance of precision (93.94%) and recall (95.38%). |
| Random Forest (Ensemble) | **Robust ensemble performer** with 96.15% accuracy and highest AUC (99.05%). Excellent precision (98.41%) with good generalization. High MCC (91.95%) shows strong predictive power. |
| XGBoost (Ensemble) | **Balanced ensemble performer** with 96.15% accuracy and 98.93% AUC. Consistent performance across precision, recall, and F1 (96.92%). Strong MCC (91.79%). |

```


## License

This project is created for educational purposes.
