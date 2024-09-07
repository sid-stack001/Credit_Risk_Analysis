# Credit_Risk_Analysis

This project uses machine learning models and neural networks to predict credit risk. The goal is to classify whether a borrower is likely to default based on financial and personal data. The project also includes data visualization through Jupyter Notebooks and a Streamlit app for interactive analysis.

## Features

- **Data Preprocessing**: Handles missing values, encoding categorical data, and feature scaling.
- **Feature Engineering**: Creates new features to enhance model performance.
- **Modeling**:
  - Random Forest Classifier
  - Neural Network (PyTorch-based)
- **Model Evaluation**: Measures performance using accuracy, precision, recall, and F1-score.
- **Visualization**: Confusion matrix, feature importances, and interactive data exploration via Streamlit.

## Project Structure

```bash
Credit-Risk-Scoring/
│
├── data/
│   └── sample_data.csv
│
├── model/
│   └── random_forest_model.py
│
├── networks/
│   └── neural_network.py
│
├── utils/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   └── visualization.py
│
├── test/
│   └── test_models.py
│
├── results/
│   └── feature_importances.csv
│
├── notebooks/
│   └── credit_risk_visualization.ipynb
│
├── streamlit_app/
│   └── credit_risk_dashboard.py
│
├── requirements.txt
└── README.md
```

Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Add your dataset in `data/sample_data.csv`.(Create data folder if it doesn't exist)
