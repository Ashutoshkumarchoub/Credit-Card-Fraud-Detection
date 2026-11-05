
# Credit Card Fraud Detection System

A machine learning-based application that identifies potentially fraudulent credit card transactions using Random Forest classification and provides interactive predictions through Streamlit.

## Project Overview

Credit card fraud is a major challenge in digital financial transactions. This project uses machine learning techniques to analyze transaction information and estimate the probability of fraudulent activity.

The application allows users to enter transaction details and receive a fraud-risk prediction through an interactive web interface.

## Objectives

- Analyze credit card transaction data.
- Perform exploratory data analysis.
- Preprocess transaction data.
- Handle imbalanced classification data.
- Train and compare machine learning models.
- Evaluate model performance.
- Predict potential fraudulent transactions.
- Develop an interactive Streamlit application.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Class Imbalance | Imbalanced-learn |
| Model Storage | Joblib |
| Web Application | Streamlit |

## Machine Learning Algorithms

- Logistic Regression
- Random Forest Classifier

## Dataset

The project uses the fraudTrain.csv and fraudTest.csv transaction datasets.

Place the downloaded datasets inside the data directory.

```text
data/
├── fraudTrain.csv
└── fraudTest.csv
```

Dataset source:
https://www.kaggle.com/datasets/kartik2112/fraud-detection

Refer to the dataset provider's licensing and attribution requirements before redistribution.

## Model Performance

Final Random Forest evaluation results:

| Metric | Result |
|---|---:|
| Accuracy | 99.65% |
| Precision | 53.53% |
| Recall | 65.36% |
| F1-Score | 58.86% |
| ROC-AUC | 0.9624 |
| PR-AUC | 0.6052 |

Selected validation threshold: 0.6410

These metrics are from the held-out test evaluation.

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
├── models/
├── reports/
│
├── app.py
├── check_dataset.py
├── eda.py
├── preprocess.py
├── train_model.py
├── threshold_tuning.py
├── feature_importance.py
├── validation.py
├── final_model.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project

```bash
cd Credit-Card-Fraud-Detection
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment on Windows

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Add the datasets

Place fraudTrain.csv and fraudTest.csv inside the data directory.

### 7. Run the application

```bash
streamlit run app.py
```

## Application Features

- Transaction information input.
- Fraud probability prediction.
- Configurable classification threshold.
- Fraud-risk classification.
- Model performance dashboard.
- Prediction history.
- CSV export.

## Limitations

- Predictions are estimates and are not verified fraud findings.
- The dataset is highly imbalanced.
- Model precision and recall require further improvement.
- The application is an academic demonstration, not a production payment-security system.

## Future Improvements

- Improve fraud detection recall and precision.
- Experiment with XGBoost and other ensemble models.
- Add persistent prediction storage.
- Improve dashboard visualizations.
- Deploy the application online.
- Add automated model monitoring.

## Author

Ashutosh Kumar Choubey

Credit Card Fraud Detection using Machine Learning and Streamlit.
