import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import requests
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Breast Cancer ML Classifier",
    page_icon="",
    layout="wide"
)

st.title("Breast Cancer ML Classifier")
st.markdown("---")

# Download sample test data
st.subheader("Download Sample Test Data")

#Github URL
github_url = "https://raw.githubusercontent.com/abhishek181/breast-cancer-ml-classifier/refs/heads/main/model/fixed_breast_cancer_test_data_raw.csv"


response = requests.get(github_url)

st.download_button(
    label="Download fixed_breast_cancer_test_data_raw.csv",
    data=response.content,
    file_name="fixed_breast_cancer_test_data_raw.csv",
    mime="text/csv"
)

st.markdown("---")

# File upload
st.subheader("Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load uploaded data
        test_data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        st.subheader("Data Preview")
        st.dataframe(test_data.head())
        
        # Model selection
        st.subheader("Model Selection")
        model_files = {
            'Logistic Regression': 'model/logisticRegressionModel.pkl',
            'Decision Tree': 'model/decisionTreeClassifierModel.pkl',
            'K-Nearest Neighbors': 'model/knnModel.pkl',
            'Naive Bayes': 'model/naiveBayesModel.pkl',
            'Random Forest': 'model/randomForestmodel.pkl',
            'XGBoost': 'model/xgBoostModel.pkl'
        }
        
        selected_model = st.selectbox("Select a model:", list(model_files.keys()))
        
        if st.button("Evaluate Model"):
            try:
                # Load selected model and scalar 
                scaler = joblib.load('model/scaler.py')
                model = joblib.load(model_files[selected_model])

                if 'target' in test_data.columns:
                    y_true = test_data['target']
                    X_raw = test_data.drop('target', axis=1)
                    has_target = True
                else:
                    X_raw = test_data.copy()
                    has_target = False

                if X_raw.shape[1] != scaler.n_features_in_:
                    st.error(f"Error: The uploaded data has {X_raw.shape[1]} features, but the model expects {scaler.n_features_in_} features.")
                    st.stop()

                #Applying scaling
                X_scaled = scaler.transform(X_raw)
    
                
                # Make predictions
                predictions = model.predict(X_scaled)
                prediction_proba = model.predict_proba(X_scaled)[:, 1]
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Evaluation Metrics")
                    if has_target:
                        metrics = {
                            'Accuracy': accuracy_score(y_true, predictions),
                            'AUC Score': roc_auc_score(y_true, prediction_proba),
                            'Precision': precision_score(y_true, predictions),
                            'Recall': recall_score(y_true, predictions),
                            'F1 Score': f1_score(y_true, predictions),
                            'MCC Score': matthews_corrcoef(y_true, predictions)
                        }
                        
                        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                        metrics_df['Value'] = metrics_df['Value'].round(4)
                        st.dataframe(metrics_df, use_container_width=True)
                    else:
                        st.info("No target column found. Cannot calculate evaluation metrics.")
                
                with col2:
                    st.subheader("Confusion Matrix")
                    if has_target:
                        cm = confusion_matrix(y_true, predictions)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                   xticklabels=['Malignant', 'Benign'],
                                   yticklabels=['Malignant', 'Benign'])
                        ax.set_title(f'Confusion Matrix - {selected_model}')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)
                    else:
                        st.info("No target column found. Cannot display confusion matrix.")
                
                # Prediction results
                st.subheader("Results")
                results_df = pd.DataFrame({
                    'Prediction': predictions,
                    'Prediction_Label': ['Malignant' if p == 0 else 'Benign' for p in predictions],
                    'Confidence': prediction_proba.round(4)
                })
                st.dataframe(results_df, use_container_width=True)
                
                # Summary
                st.subheader("Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Predictions", len(predictions))
                
                with col2:
                    st.metric("Malignant Cases", sum(predictions == 0))
                
                with col3:
                    st.metric("Benign Cases", sum(predictions == 1))

                with col4:
                     st.metric("Average Confidence", f"{prediction_proba.mean():.3f}")
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure the uploaded file has the correct format and features.")
                
    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("Please upload a CSV file to get started.")
