"""
generate_artifacts.py
Standalone script to generate all missing artifacts for the Streamlit app.
Runs evaluation pipeline and saves leaderboard, visuals, and champion model.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.getcwd())

# Create required directories
os.makedirs('models', exist_ok=True)
os.makedirs('visuals', exist_ok=True)

print("\n" + "="*80)
print("GENERATING PROJECT ARTIFACTS")
print("="*80)

try:
    from evaluate_models import (
        load_and_preprocess_data, train_baseline_models,
        calculate_all_metrics, tune_selected_models, plot_roc_curves,
        plot_confusion_matrices, plot_accuracy_vs_f1,
        plot_precision_recall_curves, plot_feature_importance,
        plot_leaderboard_heatmap, export_leaderboard, export_metrics_json
    )
    from joblib import dump
    import pandas as pd
    
    # Step 1: Load and preprocess
    print("\n[1/3] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = load_and_preprocess_data()
    
    # Step 2: Train models
    print("\n[2/3] Training baseline models...")
    models = train_baseline_models(X_train, y_train)
    
    # Step 2.5: Tune required ensemble models
    print("[2.5/3] Tuning Random Forest and Boosted Tree models...")
    tuned_models, tuning_summary = tune_selected_models(X_train, y_train)
    tuned_leaderboard, tuned_predictions, tuned_probabilities = calculate_all_metrics(tuned_models, X_test, y_test)
    pd.DataFrame(tuning_summary).T.to_csv(os.path.join('visuals', 'tuning_summary.csv'))
    tuned_leaderboard.to_csv(os.path.join('visuals', 'tuned_performance_leaderboard.csv'))

    # Step 3: Evaluate and generate visuals
    print("\n[3/3] Evaluating models and generating visuals...")
    leaderboard, predictions, probabilities = calculate_all_metrics(models, X_test, y_test)
    
    # Generate all visualizations
    plot_roc_curves(models, probabilities, y_test)
    plot_confusion_matrices(models, predictions, y_test)
    plot_accuracy_vs_f1(leaderboard)
    plot_precision_recall_curves(models, probabilities, y_test)
    best_tree_model = None
    for candidate in ['XGBoost', 'Gradient Boosting', 'Random Forest']:
        if candidate in models:
            best_tree_model = models[candidate]
            break
    plot_feature_importance(best_tree_model, feature_names, '06_feature_importance.png')
    plot_leaderboard_heatmap(leaderboard)
    
    # Export leaderboard and metrics
    export_leaderboard(leaderboard)
    export_metrics_json(leaderboard)
    
    # Serialize champion model (best by F1-score)
    best_model_name = leaderboard['f1_score'].idxmax()
    champion_model = models[best_model_name]
    
    champion_artifact = {
        'model': champion_model,
        'model_name': best_model_name,
        'metrics': leaderboard.loc[best_model_name].to_dict(),
        'preprocessor': preprocessor,
        'feature_names': feature_names,
    }
    
    dump(champion_artifact, 'models/champion_model.pkl')
    
    print("\n" + "="*80)
    print("SUCCESS: All artifacts generated!")
    print("="*80)
    print(f"\nChampion Model: {best_model_name}")
    print(f"F1-Score: {leaderboard.loc[best_model_name, 'f1_score']:.4f}")
    print(f"\nGenerated files:")
    print(f"  - models/champion_model.pkl")
    print(f"  - visuals/[5 PNG charts]")
    print(f"  - visuals/performance_leaderboard.csv")
    print(f"  - visuals/performance_metrics.json")
    print("\nThe Streamlit app is now fully functional!")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
