import os
import sys
import pandas as pd

sys.path.insert(0, os.getcwd())
from evaluate_models import load_and_preprocess_data, train_baseline_models, calculate_all_metrics

X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_data()
models = train_baseline_models(X_train, y_train)
leaderboard, _, _ = calculate_all_metrics(models, X_test, y_test)
print(leaderboard.to_markdown())
leaderboard.to_csv('leaderboard_snapshot.csv')
