"""
minimal_artifacts.py - Generate essential artifacts without visualization dependencies
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.getcwd())

os.makedirs('models', exist_ok=True)
os.makedirs('visuals', exist_ok=True)

print("Generating minimal artifacts...")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
from joblib import dump

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

from utils import (
    load_and_clean_base_data,
    stratified_train_test_split,
    preprocess_features,
    apply_smote_balancing
)

# Load data
df = load_and_clean_base_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split
X_train_raw, X_test_raw, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess
X_train, X_test, feature_names = preprocess_features(X_train_raw, X_test_raw)

# SMOTE
X_train_bal, y_train_bal = apply_smote_balancing(X_train, y_train.values, random_state=42)

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, min_samples_split=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'SVM': SVC(kernel='linear', C=0.1, random_state=42, probability=True, max_iter=5000),
}

if XGBOOST_AVAILABLE:
    models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    print(f"✓ {name} trained")

# Evaluate
metrics_dict = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_proba = None
    
    metrics_dict[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    }

leaderboard = pd.DataFrame.from_dict(metrics_dict, orient='index')
leaderboard = leaderboard.sort_values('f1_score', ascending=False)

# Save leaderboard
leaderboard.to_csv('visuals/performance_leaderboard.csv')
leaderboard.to_json('visuals/performance_metrics.json', orient='index')

# Save champion model
champion_name = leaderboard['f1_score'].idxmax()
champion = {
    'model': models[champion_name],
    'model_name': champion_name,
    'metrics': leaderboard.loc[champion_name].to_dict(),
}
dump(champion, 'models/champion_model.pkl')

print("\n✓ Leaderboard: visuals/performance_leaderboard.csv")
print(f"✓ Champion ({champion_name}): models/champion_model.pkl")
print("\nLeaderboard:")
print(leaderboard)
