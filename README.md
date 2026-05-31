# Telco Churn Stabilizer

[![Deployment](https://img.shields.io/badge/Deployment-Local%20Preview-lightgrey)](https://placeholder.streamlitapp.com)

## Executive Summary

Customer churn erodes monthly contract revenue and increases acquisition costs for telecom providers. This project uses the Telco Customer Churn dataset to stabilize retention outcomes with production-ready machine learning classifiers, a reusable preprocessing pipeline, and an interactive Streamlit dashboard.

The solution demonstrates how model-driven churn scoring can help retention teams identify high-risk customers, reduce revenue leakage, and prioritize retention interventions.

## Project Structure

```
Telco-Churn-Stabilizer/
├── app.py
├── evaluate_models.py
├── generate_artifacts.py
├── requirements.txt
├── utils.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── data/                      # Optional dataset folder for deployment structure
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── champion_model.pkl
├── notebooks/
│   ├── 1_eda.ipynb
│   └── 2_modelling.ipynb
├── validation/
│   └── teleco_data_validation.py
└── visuals/
    ├── 01_roc_curves_comparison.png
    ├── 02_confusion_matrices_grid.png
    ├── 03_accuracy_vs_f1_comparison.png
    ├── 04_precision_recall_curves.png
    ├── 05_leaderboard_heatmap.png
    ├── 06_feature_importance.png
    ├── performance_leaderboard.csv
    ├── performance_metrics.json
    └── tuned_performance_leaderboard.csv
```

## Dependency Configuration

The pinned `requirements.txt` includes the exact dependency versions used for reproducible deployment:

- `streamlit==1.26.0`
- `scikit-learn==1.8.0`
- `xgboost==1.7.6`
- `pandas==2.3.3`
- `numpy==1.26.4`
- `plotly==6.5.2`
- `joblib==1.5.3`
- `imbalanced-learn==0.14.1`
- `matplotlib==3.10.9`
- `seaborn==0.13.2`
- `scipy==1.17.0`

## Validation Leaderboard

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.78 | 0.68 | 0.72 | 0.70 | 0.80 |
| Decision Tree | 0.74 | 0.65 | 0.70 | 0.67 | 0.76 |
| Random Forest | 0.82 | 0.75 | 0.77 | 0.76 | 0.85 |
| K-Nearest Neighbors | 0.77 | 0.67 | 0.69 | 0.68 | 0.78 |
| Support Vector Machine | 0.80 | 0.72 | 0.75 | 0.74 | 0.82 |
| XGBoost | 0.84 | 0.78 | 0.79 | 0.79 | 0.87 |

> This leaderboard is an example summary. Run the pipeline to generate exact benchmark results, tuned model summaries, and champion selection.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-org/telco-churn-stabilizer.git
cd telco-churn-stabilizer
```

### 2. Create an isolated Python environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

### 5. Optional: regenerate model artifacts and champion model

```bash
python generate_artifacts.py
```

## Application Features

- Sidebar navigation across six functional sections
- Dataset explorer with search and missing-value summaries
- EDA dashboard for interactive chart viewing
- Model training interface with background execution controls
- Model comparison leaderboard and benchmark visuals
- Prediction form powered by serialized pipeline and champion model artifact in `models/champion_model.pkl`

## Deployment Notes

- Ensure the dataset path is correct before running the app.
- For best production stability, serialize the fitted preprocessing pipeline together with the champion model asset.
- Use the provided `requirements.txt` for deterministic dependency pinning and GitHub validation.
