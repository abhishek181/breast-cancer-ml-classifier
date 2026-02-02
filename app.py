import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification Models",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Breast Cancer Classification Models Comparison")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose an option:",
    ["Dataset Overview", "Model Training", "Model Evaluation", "Upload Test Data"]
)

# Load breast cancer dataset
@st.cache_data
def load_breast_cancer_dataset():
    breast_cancer = load_breast_cancer()
    data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    data['target'] = breast_cancer.target
    return data, breast_cancer.target_names

# Initialize models
def get_models():
    return {
        'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }

# Calculate all metrics
def calculate_all_metrics(y_test, y_pred, y_pred_probability):
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC Score': roc_auc_score(y_test, y_pred_probability),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC Score': matthews_corrcoef(y_test, y_pred)
    }

# Main content based on selection
if option == "Dataset Overview":
    st.header("📊 Dataset Overview")
    
    data, target_names = load_breast_cancer_dataset()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        st.write(f"**Shape:** {data.shape}")
        st.write(f"**Features:** {data.shape[1] - 1}")
        st.write(f"**Samples:** {data.shape[0]}")
        st.write(f"**Target classes:** {target_names}")
        
        st.write(f"**Target Distribution:**")
        target_counts = data['target'].value_counts()
        st.write(f"0 (malignant): {target_counts[0]}")
        st.write(f"1 (benign): {target_counts[1]}")
    
    with col2:
        st.subheader("Dataset Preview")
        st.dataframe(data.head())
    
    st.subheader("Statistical Summary")
    st.dataframe(data.describe())

elif option == "Model Training":
    st.header("🔧 Model Training & Results")
    
    data, target_names = load_breast_cancer_dataset()
    
    # Prepare data
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models and collect results
    models = get_models()
    all_metrics = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f'Training {name}...')
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba)
        metrics['Model'] = name
        all_metrics.append(metrics)
        
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text('Training completed!')
    
    # Display results table
    st.subheader("📈 Model Comparison Results")
    results_df = pd.DataFrame(all_metrics)
    results_df = results_df[['Model', 'Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']]
    results_df = results_df.round(4)
    st.dataframe(results_df)
    
    # Visualize results
    st.subheader("📊 Performance Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        results_df.set_index('Model')['Accuracy'].plot(kind='bar', ax=ax)
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylabel('Accuracy')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        results_df.set_index('Model')['F1 Score'].plot(kind='bar', ax=ax, color='orange')
        ax.set_title('Model F1 Score Comparison')
        ax.set_ylabel('F1 Score')
        plt.xticks(rotation=45)
        st.pyplot(fig)

elif option == "Model Evaluation":
    st.header("📋 Detailed Model Evaluation")
    
    data, target_names = load_breast_cancer_dataset()
    
    # Prepare data
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model selection
    models = get_models()
    selected_model = st.selectbox("Select a model for detailed evaluation:", list(models.keys()))
    
    if st.button("Evaluate Selected Model"):
        model = models[selected_model]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Malignant', 'Benign'],
                       yticklabels=['Malignant', 'Benign'])
            ax.set_title(f'Confusion Matrix - {selected_model}')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3))

elif option == "Upload Test Data":
    st.header("📤 Upload Test Data")
    
    # Download sample test data button
    st.subheader("📥 Download Sample Test Data")
    st.markdown("Download a sample CSV file with breast cancer features to test the models:")
    
    # Create download button for test data
    test_data_url = "https://raw.githubusercontent.com/your-username/breast-cancer-ml-classifier/main/notebooks/test_data.csv"
    st.markdown(f"[📥 Download test_data.csv]({test_data_url})")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            test_data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            st.subheader("Uploaded Data Preview")
            st.dataframe(test_data.head())
            
            st.subheader("Data Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Shape:** {test_data.shape}")
                st.write(f"**Columns:** {list(test_data.columns)}")
            
            with col2:
                st.write("**Data Types:**")
                st.write(test_data.dtypes)
            
            # Model selection for prediction
            model_files = {
                'Logistic Regression': 'notebooks/model/logisticRegressionModel.pkl',
                'Decision Tree': 'notebooks/model/decisionTreeClassifierModel.pkl',
                'K-Nearest Neighbors': 'notebooks/model/knnModel.pkl',
                'Naive Bayes': 'notebooks/model/naiveBayesModel.pkl',
                'Random Forest': 'notebooks/model/randomForestmodel.pkl',
                'XGBoost': 'notebooks/model/xgBoostModel.pkl'
            }
            
            selected_model = st.selectbox("Select model for prediction:", list(model_files.keys()))
            
            if st.button("Make Predictions"):
                try:
                    import joblib
                    
                    # Load the selected model
                    model = joblib.load(model_files[selected_model])
                    
                    # Remove target column if it exists
                    prediction_data = test_data.copy()
                    if 'target' in prediction_data.columns:
                        prediction_data = prediction_data.drop('target', axis=1)
                    
                    # Make predictions
                    predictions = model.predict(prediction_data)
                    prediction_proba = model.predict_proba(prediction_data)[:, 1]
                    
                    # Display results
                    st.subheader("Prediction Results")
                    results_df = test_data.copy()
                    results_df['Prediction'] = predictions
                    results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'Malignant', 1: 'Benign'})
                    results_df['Confidence'] = prediction_proba
                    
                    st.dataframe(results_df[['Prediction', 'Prediction_Label', 'Confidence']])
                    
                    # Summary
                    st.subheader("Prediction Summary")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Predictions", len(predictions))
                        st.metric("Malignant Cases", sum(predictions == 0))
                    
                    with col2:
                        st.metric("Benign Cases", sum(predictions == 1))
                        st.metric("Average Confidence", f"{prediction_proba.mean():.3f}")
                    
                except Exception as e:
                    st.error(f"Error making predictions: {e}")
                    st.info("Make sure the test data has the same features as the training data.")
                
        except Exception as e:
            st.error(f"Error reading file: {e}")