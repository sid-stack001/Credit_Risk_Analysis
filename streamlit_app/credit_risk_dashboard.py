import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_processing import preprocess_data, load_data
from utils.feature_engineering import create_new_features
from utils.visualization import plot_confusion_matrix, plot_feature_importances

# Set up the Streamlit app
st.title("Credit Risk Scoring Dashboard")

# Load and preprocess data
st.sidebar.header('Data Loading')
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview")
    st.dataframe(data.head())
    
    # Preprocess the data
    data = preprocess_data(data)
    data = create_new_features(data)

    # Select target column
    target = st.sidebar.selectbox('Select Target Column', data.columns)

    X = data.drop(columns=[target])
    y = data[target]

    # Train Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict on the training data
    predictions = model.predict(X)

    # Show Confusion Matrix
    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y, predictions)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # Feature Importance Plot
    st.subheader("Feature Importances")
    importances = model.feature_importances_
    feature_names = X.columns
    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=feature_names, ax=ax)
    ax.set_title('Feature Importances')
    st.pyplot(fig)
