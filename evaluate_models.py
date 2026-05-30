"""evaluate_models.py

Comprehensive model evaluation pipeline for 6 baseline classifiers.

Generates:
- Performance leaderboard (all metrics)
- ROC curve comparison (multi-model)
- Confusion matrix grid
- Accuracy vs F1-score comparison
- Precision-Recall trade-off curves
- Exports all visualizations to visuals/ directory

Usage:
    python evaluate_models.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IMPORTS
# ============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    confusion_matrix, precision_recall_curve
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns

# Import data utilities
from utils import (
    load_and_clean_base_data,
    split_features_and_target,
    preprocess_features,
    stratified_train_test_split,
    apply_smote_balancing
)

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
VISUALS_DIR = 'visuals'

# Create visuals directory if not exists
os.makedirs(VISUALS_DIR, exist_ok=True)

# Style settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================
def load_and_preprocess_data():
    """Load, clean, split, preprocess, and balance data."""
    print("\n" + "=" * 80)
    print("DATA LOADING & PREPROCESSING")
    print("=" * 80)
    
    # Load & clean
    df = load_and_clean_base_data(DATA_PATH)
    print("[OK] Loaded: {} records x {} features".format(df.shape[0], df.shape[1]))
    
    # Extract features & target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Stratified split
    X_train_raw, X_test_raw, y_train, y_test = stratified_train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print("[OK] Stratified split: Train {} / Test {}".format(X_train_raw.shape[0], X_test_raw.shape[0]))
    
    # Feature preprocessing
    X_train_processed, X_test_processed, feature_names = preprocess_features(
        X_train_raw, X_test_raw
    )
    print("[OK] Features engineered: {} dimensions".format(X_train_processed.shape[1]))
    
    # SMOTE balancing
    X_train_balanced, y_train_balanced = apply_smote_balancing(
        X_train_processed, y_train.values, random_state=RANDOM_STATE
    )
    print("[OK] SMOTE applied: {} -> {} samples".format(X_train_processed.shape[0], X_train_balanced.shape[0]))
    
    return X_train_balanced, X_test_processed, y_train_balanced, y_test.values


# ============================================================================
# MODEL TRAINING
# ============================================================================
def train_baseline_models(X_train, y_train):
    """Instantiate and train 6 baseline classifiers."""
    print("\n" + "=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)
    
    models = {}
    
    # 1. Logistic Regression
    print("\n[1] Logistic Regression")
    models['Logistic Regression'] = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
    ).fit(X_train, y_train)
    print("    [OK] Trained")
    
    # 2. Decision Tree
    print("[2] Decision Tree Classifier")
    models['Decision Tree'] = DecisionTreeClassifier(
        max_depth=15, min_samples_split=5, random_state=RANDOM_STATE
    ).fit(X_train, y_train)
    print("    [OK] Trained")
    
    # 3. Random Forest
    print("[3] Random Forest Classifier")
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=-1
    ).fit(X_train, y_train)
    print("    [OK] Trained")
    
    # 4. KNN
    print("[4] K-Nearest Neighbors")
    models['KNN'] = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_train, y_train)
    print("    [OK] Trained")
    
    # 5. SVM (linear kernel for faster training)
    print("[5] Support Vector Machine")
    models['SVM'] = SVC(
        kernel='linear', C=0.1, random_state=RANDOM_STATE, probability=True, max_iter=5000
    ).fit(X_train, y_train)
    print("    [OK] Trained")
    
    # 6. XGBoost
    if XGBOOST_AVAILABLE:
        print("[6] XGBoost Classifier")
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss'
        ).fit(X_train, y_train)
        print("    [OK] Trained")
    else:
        print("[6] XGBoost - SKIPPED (not installed)")
    
    return models


# ============================================================================
# METRICS CALCULATION
# ============================================================================
def calculate_all_metrics(models, X_test, y_test):
    """Calculate all metrics for all models."""
    print("\n" + "=" * 80)
    print("METRICS CALCULATION")
    print("=" * 80)
    
    metrics_dict = {}
    predictions = {}
    probabilities = {}
    
    for model_name, model in models.items():
        print("\n{}:".format(model_name))
        
        # Predictions
        y_pred = model.predict(X_test)
        predictions[model_name] = y_pred
        
        # Probabilities
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            probabilities[model_name] = y_proba
        except:
            probabilities[model_name] = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        roc_auc = None
        if probabilities[model_name] is not None:
            try:
                roc_auc = roc_auc_score(y_test, probabilities[model_name])
            except:
                roc_auc = None
        
        metrics_dict[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc if roc_auc is not None else np.nan
        }
        
        print("  Accuracy:  {:.4f}".format(accuracy))
        print("  Precision: {:.4f}".format(precision))
        print("  Recall:    {:.4f}".format(recall))
        print("  F1-Score:  {:.4f}".format(f1))
        if roc_auc is not None:
            print("  ROC-AUC:   {:.4f}".format(roc_auc))
        else:
            print("  ROC-AUC:   N/A")
    
    # Create leaderboard DataFrame
    leaderboard = pd.DataFrame.from_dict(metrics_dict, orient='index')
    leaderboard = leaderboard.sort_values('f1_score', ascending=False)
    
    return leaderboard, predictions, probabilities


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_roc_curves(models, probabilities, y_test):
    """Generate multi-model ROC curve comparison."""
    print("\n[VIZ] Generating ROC Curve Comparison...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (model_name, proba) in enumerate(probabilities.items()):
        if proba is not None:
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2.5, label='{} (AUC={:.4f})'.format(model_name, roc_auc),
                   color=colors[idx % len(colors)])
    
    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve Comparison -- 6 Baseline Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, '01_roc_curves_comparison.png'), dpi=300, bbox_inches='tight')
    print("    [OK] Saved: 01_roc_curves_comparison.png")
    plt.close()


def plot_confusion_matrices(models, predictions, y_test):
    """Generate side-by-side confusion matrix grid."""
    print("[VIZ] Generating Confusion Matrix Grid...")
    
    n_models = len(models)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar=False, square=True, cbar_kws={'label': 'Count'})
        
        axes[idx].set_title('{}'.format(model_name), fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
        axes[idx].set_xticklabels(['No Churn', 'Churn'])
        axes[idx].set_yticklabels(['No Churn', 'Churn'])
    
    # Hide extra subplots
    for idx in range(n_models, 6):
        axes[idx].axis('off')
    
    fig.suptitle('Confusion Matrix Comparison Grid', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, '02_confusion_matrices_grid.png'), dpi=300, bbox_inches='tight')
    print("    [OK] Saved: 02_confusion_matrices_grid.png")
    plt.close()


def plot_accuracy_vs_f1(leaderboard):
    """Generate Accuracy vs F1-Score comparison bar chart."""
    print("[VIZ] Generating Accuracy vs F1-Score Comparison...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(leaderboard))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, leaderboard['accuracy'], width, label='Accuracy',
                   color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, leaderboard['f1_score'], width, label='F1-Score',
                   color='#ff7f0e', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(leaderboard.index, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   '{:.3f}'.format(height), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, '03_accuracy_vs_f1_comparison.png'), dpi=300, bbox_inches='tight')
    print("    [OK] Saved: 03_accuracy_vs_f1_comparison.png")
    plt.close()


def plot_precision_recall_curves(models, probabilities, y_test):
    """Generate Precision-Recall trade-off curves."""
    print("[VIZ] Generating Precision-Recall Trade-off Curves...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (model_name, proba) in enumerate(probabilities.items()):
        if proba is not None:
            precision, recall, _ = precision_recall_curve(y_test, proba)
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, lw=2.5, label='{} (AUC={:.4f})'.format(model_name, pr_auc),
                   color=colors[idx % len(colors)])
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Trade-off -- 6 Baseline Models', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, '04_precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    print("    [OK] Saved: 04_precision_recall_curves.png")
    plt.close()


def plot_leaderboard_heatmap(leaderboard):
    """Generate metric heatmap leaderboard."""
    print("[VIZ] Generating Metrics Heatmap Leaderboard...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    leaderboard_norm = leaderboard.copy()
    
    sns.heatmap(leaderboard_norm, annot=True, fmt='.4f', cmap='RdYlGn',
               cbar_kws={'label': 'Score'}, ax=ax, vmin=0, vmax=1)
    
    ax.set_title('Performance Leaderboard -- All Metrics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, '05_leaderboard_heatmap.png'), dpi=300, bbox_inches='tight')
    print("    [OK] Saved: 05_leaderboard_heatmap.png")
    plt.close()


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_leaderboard(leaderboard):
    """Export leaderboard to CSV."""
    leaderboard.to_csv(os.path.join(VISUALS_DIR, 'performance_leaderboard.csv'))
    print("\n[OK] Exported: performance_leaderboard.csv")


def export_metrics_json(leaderboard):
    """Export metrics to JSON."""
    metrics_json = leaderboard.to_dict('index')
    with open(os.path.join(VISUALS_DIR, 'performance_metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2, default=str)
    print("[OK] Exported: performance_metrics.json")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    print("\n" + "=" * 80)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 80)
    
    # Step 1: Load & preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Step 2: Train models
    models = train_baseline_models(X_train, y_train)
    
    # Step 3: Calculate metrics
    leaderboard, predictions, probabilities = calculate_all_metrics(models, X_test, y_test)
    
    # Step 4: Print leaderboard
    print("\n" + "=" * 80)
    print("PERFORMANCE LEADERBOARD (Sorted by F1-Score)")
    print("=" * 80)
    print(leaderboard.to_string())
    
    # Step 5: Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plot_roc_curves(models, probabilities, y_test)
    plot_confusion_matrices(models, predictions, y_test)
    plot_accuracy_vs_f1(leaderboard)
    plot_precision_recall_curves(models, probabilities, y_test)
    plot_leaderboard_heatmap(leaderboard)
    
    # Step 6: Export results
    print("\n" + "=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)
    
    export_leaderboard(leaderboard)
    export_metrics_json(leaderboard)
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    best_model = leaderboard.index[0]
    best_f1 = leaderboard['f1_score'].iloc[0]
    print("\nBest Model (by F1-Score): {} ({:.4f})".format(best_model, best_f1))
    print("All visualizations saved to: {}/".format(VISUALS_DIR))
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
